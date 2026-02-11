"""
Prepare ImageNet-1k for drifting model training.

Streams ImageNet parquet shards from HF cache, encodes images to SD-VAE
latents (4×32×32), and precomputes FID reference statistics.

Storage layout:
  train/
    0/part-000000.npy  ...  999/part-NNNNNN.npy
    index.json
  val/
    ...
  fid_stats/
    val_inception_mu.npy
    val_inception_sigma.npy
  metadata.json

Usage:
  uv run python prepare_data.py
  uv run python prepare_data.py --output-dir /mnt/data
  uv run python prepare_data.py --batch-size 64
  uv run python prepare_data.py --splits train val
  uv run python prepare_data.py --skip-fid-stats
  uv run python prepare_data.py --verify-only
"""

import argparse
import io
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

SD_VAE_SCALE = 0.18215
NUM_CLASSES = 1000

# Expected sample counts per split (ImageNet-1k)
EXPECTED_SAMPLES = {"train": 1_281_167, "val": 50_000}

def _find_local_parquets(split: str) -> list[Path]:
    """Find local HF-cached parquet files for a split."""
    cache_dir = Path.home() / ".cache/huggingface/hub/datasets--ILSVRC--imagenet-1k/snapshots"
    if not cache_dir.exists():
        return []
    for snap in cache_dir.iterdir():
        data_dir = snap / "data"
        if not data_dir.exists():
            continue
        parquets = sorted(data_dir.glob(f"{split}-*.parquet"))
        if parquets:
            return parquets
    return []


@dataclass
class Example:
    image: "PIL.Image.Image"
    label: int


def iter_imagenet(split: str) -> Iterator[Example]:
    """Stream ImageNet from local parquet files with bounded memory.

    Uses ParquetFile.iter_batches — never loads a full shard into RAM.
    Yields Example(image, label). Skips corrupt rows silently.
    """
    import pyarrow.parquet as pq
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parquets = _find_local_parquets(split)
    if not parquets:
        raise FileNotFoundError(
            f"No local parquet files for '{split}'. Download first:\n"
            f'  python -c "from datasets import load_dataset; '
            f"load_dataset('ILSVRC/imagenet-1k', split='{split}')\""
        )

    total = sum(pq.read_metadata(p).num_rows for p in parquets)
    pbar = tqdm(total=total, desc=f"Reading {split}")

    for pf_path in parquets:
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
            img_col = batch.column("image")
            lbl_col = batch.column("label")
            for i in range(batch.num_rows):
                pbar.update(1)
                try:
                    raw = img_col[i].as_py()
                    img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
                    yield Example(
                        image=Image.open(io.BytesIO(img_bytes)),
                        label=int(lbl_col[i].as_py()),
                    )
                except Exception:
                    continue

    pbar.close()


def load_vae(device: torch.device):
    from diffusers import AutoencoderKL

    print("\nLoading SD-VAE (stabilityai/sd-vae-ft-ema)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device).half().eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


_preprocess_256 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor) -> np.ndarray:
    """[B,3,256,256] fp16 -> [B,4,32,32] float32 numpy."""
    return (vae.encode(images).latent_dist.mode() * SD_VAE_SCALE).float().cpu().numpy()


def _next_part_id(class_dir: Path) -> int:
    parts = sorted(class_dir.glob("part-*.npy"))
    if not parts:
        return 0
    return int(parts[-1].stem.split("-")[1]) + 1


def _flush(split_dir: Path, index: dict, class_latents: dict[int, list[np.ndarray]]):
    """Write accumulated latents as append-only chunk files, update index."""
    for cid, lats in class_latents.items():
        if not lats:
            continue
        cdir = split_dir / str(cid)
        cdir.mkdir(parents=True, exist_ok=True)
        arr = np.stack(lats)
        np.save(cdir / f"part-{_next_part_id(cdir):06d}.npy", arr)
        c = index["classes"][str(cid)]
        c["parts"] += 1
        c["samples"] += int(arr.shape[0])

    tmp = split_dir / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2) + "\n")
    tmp.replace(split_dir / "index.json")


def _split_is_complete(split_dir: Path, split_name: str) -> bool:
    """Check if a split was already fully encoded."""
    idx_path = split_dir / "index.json"
    if not idx_path.exists():
        return False
    index = json.loads(idx_path.read_text())
    total = sum(index["classes"][str(i)]["samples"] for i in range(NUM_CLASSES))
    expected = EXPECTED_SAMPLES.get(split_name, 0)
    return total >= expected * 0.999  # tolerance for skipped corrupt images


def encode_split(
    examples: Iterator[Example],
    vae,
    output_dir: Path,
    split_name: str,
    batch_size: int,
    device: torch.device,
    flush_every: int = 25_000,
):
    """Encode a full split to per-class chunk files.

    If already complete, skip. If partial data exists, wipe and re-encode
    from scratch — partial resume with streaming is unreliable, and
    encoding is fast relative to the initial download.
    """
    split_dir = output_dir / split_name

    if _split_is_complete(split_dir, split_name):
        print(f"\n{split_name} already complete, skipping.")
        return

    if split_dir.exists() and any(split_dir.iterdir()):
        print(f"\n{split_name} has partial data — wiping and re-encoding.")
        shutil.rmtree(split_dir)

    split_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "format": "per-class-chunks-v1",
        "classes": {str(i): {"parts": 0, "samples": 0} for i in range(NUM_CLASSES)},
    }

    class_latents: dict[int, list[np.ndarray]] = {}
    batch_imgs: list[torch.Tensor] = []
    batch_labs: list[int] = []
    n_encoded = 0
    last_flush = 0

    for ex in examples:
        img = ex.image
        if img.mode != "RGB":
            img = img.convert("RGB")
        batch_imgs.append(_preprocess_256(img))
        batch_labs.append(ex.label)

        if len(batch_imgs) >= batch_size:
            batch = torch.stack(batch_imgs).to(device).half()
            latents = encode_batch(vae, batch)
            for lat, lab in zip(latents, batch_labs):
                class_latents.setdefault(lab, []).append(lat)
            n_encoded += len(latents)
            batch_imgs.clear()
            batch_labs.clear()

            if n_encoded - last_flush >= flush_every:
                _flush(split_dir, index, class_latents)
                class_latents.clear()
                last_flush = n_encoded

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device).half()
        latents = encode_batch(vae, batch)
        for lat, lab in zip(latents, batch_labs):
            class_latents.setdefault(lab, []).append(lat)
        n_encoded += len(latents)

    _flush(split_dir, index, class_latents)

    total = sum(index["classes"][str(i)]["samples"] for i in range(NUM_CLASSES))
    print(f"\n  {split_name}: encoded {n_encoded} images, {total} stored across {NUM_CLASSES} classes")


# ---------------------------------------------------------------------------
# Stage 3: FID reference stats (streaming accumulation)
# ---------------------------------------------------------------------------


_inception_preprocess = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_inception(device: torch.device):
    from torchvision.models import Inception_V3_Weights, inception_v3

    print("\nLoading Inception v3...")
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
    model = model.to(device).half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _inception_pool3(model, images: torch.Tensor) -> np.ndarray:
    """Extract avgpool features [B, 2048] as float64."""
    feats = []

    def hook(_, __, out):
        feats.append(out.squeeze(-1).squeeze(-1).float().cpu().numpy().astype(np.float64))

    h = model.avgpool.register_forward_hook(hook)
    model(images)
    h.remove()
    return feats[0]


def compute_fid_stats(
    examples: Iterator[Example],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
):
    """Compute Inception pool3 mu/sigma via streaming accumulation.

    Accumulates S1 = Σx and S2 = Σ(xxᵀ) to avoid materializing the
    full [50000, 2048] feature matrix in RAM.
    """
    fid_dir = output_dir / "fid_stats"
    fid_dir.mkdir(parents=True, exist_ok=True)
    mu_path = fid_dir / "val_inception_mu.npy"
    sigma_path = fid_dir / "val_inception_sigma.npy"

    if mu_path.exists() and sigma_path.exists():
        print("\nFID stats already computed, skipping.")
        return

    model = load_inception(device)

    S1 = np.zeros(2048, dtype=np.float64)
    S2 = np.zeros((2048, 2048), dtype=np.float64)
    n = 0
    batch_imgs: list[torch.Tensor] = []

    print("\nComputing FID stats (streaming)...")
    for ex in tqdm(examples, desc="FID features"):
        img = ex.image
        if img.mode != "RGB":
            img = img.convert("RGB")
        batch_imgs.append(_inception_preprocess(img))

        if len(batch_imgs) >= batch_size:
            x = _inception_pool3(model, torch.stack(batch_imgs).to(device).half())
            batch_imgs.clear()
            S1 += x.sum(axis=0)
            S2 += x.T @ x
            n += x.shape[0]

    if batch_imgs:
        x = _inception_pool3(model, torch.stack(batch_imgs).to(device).half())
        S1 += x.sum(axis=0)
        S2 += x.T @ x
        n += x.shape[0]

    assert n >= 2, f"Need ≥2 images for covariance, got {n}"

    mu = S1 / n
    sigma = (S2 - n * np.outer(mu, mu)) / (n - 1)

    np.save(mu_path, mu.astype(np.float32))
    np.save(sigma_path, sigma.astype(np.float32))
    print(f"  Saved mu {mu.shape}, sigma {sigma.shape} to {fid_dir}/ (n={n})")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_data(output_dir: Path) -> bool:
    print(f"\nVerifying {output_dir}/")
    ok = True

    for split in ["train", "val"]:
        split_dir = output_dir / split
        idx_path = split_dir / "index.json"
        if not idx_path.exists():
            print(f"  {split}: MISSING")
            ok = False
            continue

        index = json.loads(idx_path.read_text())
        classes = index["classes"]
        total = sum(classes[str(i)]["samples"] for i in range(NUM_CLASSES))
        nonempty = sum(1 for i in range(NUM_CLASSES) if classes[str(i)]["samples"] > 0)

        # Spot-check shapes on a few classes
        shapes_ok = True
        for cid in [0, 500, 999]:
            cdir = split_dir / str(cid)
            parts = sorted(cdir.glob("part-*.npy")) if cdir.exists() else []
            if parts:
                arr = np.load(parts[0], mmap_mode="r")
                if arr.ndim != 4 or arr.shape[1:] != (4, 32, 32):
                    print(f"    BAD SHAPE: {split}/{cid}/{parts[0].name} -> {arr.shape}")
                    shapes_ok = False

        status = "OK" if (nonempty == NUM_CLASSES and shapes_ok) else "INCOMPLETE"
        if status != "OK":
            ok = False
        print(f"  {split}: {total} samples, {nonempty}/{NUM_CLASSES} classes — {status}")

    fid_dir = output_dir / "fid_stats"
    for name, shape in [("val_inception_mu.npy", (2048,)), ("val_inception_sigma.npy", (2048, 2048))]:
        p = fid_dir / name
        if not p.exists():
            print(f"  FID: {name} MISSING")
        else:
            arr = np.load(p, mmap_mode="r")
            if arr.shape != shape:
                print(f"  FID: {name} BAD SHAPE {arr.shape}")
                ok = False

    if ok:
        print("  All checks passed.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k for drifting model training")
    parser.add_argument("--output-dir", type=str, default="data/imagenet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--skip-fid-stats", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        raise SystemExit(0 if verify_data(output_dir) else 1)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    t0 = time.time()

    # Encode splits
    vae = load_vae(device)
    for split in args.splits:
        parquet_name = "validation" if split == "val" else split
        encode_split(iter_imagenet(parquet_name), vae, output_dir, split, args.batch_size, device)
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FID stats
    if not args.skip_fid_stats:
        compute_fid_stats(iter_imagenet("validation"), output_dir, args.batch_size, device)

    # Metadata
    elapsed = time.time() - t0
    meta = {
        "dataset": "ILSVRC/imagenet-1k",
        "vae": "stabilityai/sd-vae-ft-ema",
        "vae_scale": SD_VAE_SCALE,
        "latent_shape": [4, 32, 32],
        "image_size": 256,
        "splits": args.splits,
        "time_seconds": round(elapsed, 1),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone in {elapsed/3600:.1f}h.")
    verify_data(output_dir)


if __name__ == "__main__":
    main()