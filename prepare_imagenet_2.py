"""
Prepare ImageNet-1k for drifting model training (optimized).

Streams ImageNet parquet shards from HF cache, encodes images to SD-VAE
latents (4x32x32), and precomputes FID reference statistics.

Optimizations over prepare_imagenet.py:
  - Parallel CPU image decode via ThreadPoolExecutor + GPU prefetch queue
  - Bulk Arrow column extraction (to_pylist) instead of per-row as_py()
  - TF32 matmul, channels_last memory format, non_blocking transfers
  - Flat output format: split_dir/{class_id}.npy (matches train.py)
  - Persistent Inception hook for FID stats
  - Periodic flush to temp files to bound memory

Storage layout:
  train/
    0.npy  1.npy  ...  999.npy
  val/
    0.npy  1.npy  ...  999.npy
  fid_stats/
    val_inception_mu.npy
    val_inception_sigma.npy
  metadata.json

Usage:
  uv run python prepare_imagenet_2.py
  uv run python prepare_imagenet_2.py --output-dir /mnt/data
  uv run python prepare_imagenet_2.py --batch-size 128
  uv run python prepare_imagenet_2.py --splits train val
  uv run python prepare_imagenet_2.py --skip-fid-stats
  uv run python prepare_imagenet_2.py --verify-only
"""

import argparse
import io
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

SD_VAE_SCALE = 0.18215
NUM_CLASSES = 1000
EXPECTED_SAMPLES = {"train": 1_281_167, "val": 50_000}


# ---------------------------------------------------------------------------
# TF32 / precision setup
# ---------------------------------------------------------------------------

def _setup_precision():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Parquet discovery / download
# ---------------------------------------------------------------------------

def _find_local_parquets(split: str) -> list[Path]:
    """Find local HF-cached parquet files for a split.

    Handles the naming mismatch: HF stores the val split as 'validation-*.parquet'.
    """
    import os

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if not hf_hub_cache:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
        hf_hub_cache = str(Path(hf_home) / "hub")
    cache_dir = Path(hf_hub_cache) / "datasets--ILSVRC--imagenet-1k/snapshots"
    if not cache_dir.exists():
        return []

    # Try both the requested split name and 'validation' for val
    patterns = [f"{split}-*.parquet"]
    if split == "val":
        patterns.append("validation-*.parquet")

    for snap in cache_dir.iterdir():
        data_dir = snap / "data"
        if not data_dir.exists():
            continue
        for pattern in patterns:
            parquets = sorted(data_dir.glob(pattern))
            if parquets:
                return parquets
    return []


def _download_split(split: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    # HF uses 'validation' as the parquet prefix for the val split
    parquet_prefix = "validation" if split == "val" else split
    print(f"\nDownloading ILSVRC/imagenet-1k ({split}) from HuggingFace...")
    snapshot_download(
        "ILSVRC/imagenet-1k",
        repo_type="dataset",
        allow_patterns=[f"data/{parquet_prefix}-*.parquet"],
    )
    parquets = _find_local_parquets(split)
    return parquets


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_preprocess_256 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def _decode_and_preprocess(img_bytes: bytes) -> torch.Tensor | None:
    """Decode JPEG bytes to a preprocessed [3,256,256] tensor. Returns None on failure."""
    from PIL import Image
    try:
        raw = img_bytes
        if isinstance(raw, dict):
            raw = raw["bytes"]
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return _preprocess_256(img)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prefetch pipeline: Arrow -> thread pool decode -> bounded queue -> GPU
# ---------------------------------------------------------------------------

def _prefetch_worker(
    split: str,
    batch_size: int,
    num_decode_workers: int,
    out_queue: Queue,
):
    """Producer thread: reads Arrow batches, decodes images in parallel,
    pushes (tensor_batch, labels) onto out_queue. Sends None as sentinel."""
    import pyarrow.parquet as pq
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parquets = _find_local_parquets(split)
    if not parquets:
        parquets = _download_split(split)

    total = sum(pq.read_metadata(p).num_rows for p in parquets)
    pbar = tqdm(total=total, desc=f"Encoding {split}")

    pool = ThreadPoolExecutor(max_workers=num_decode_workers)
    pending_tensors: list[torch.Tensor] = []
    pending_labels: list[int] = []

    def _flush_batch():
        nonlocal pending_tensors, pending_labels
        if not pending_tensors:
            return
        batch_t = torch.stack(pending_tensors)
        batch_l = pending_labels.copy()
        out_queue.put((batch_t, batch_l))
        pending_tensors = []
        pending_labels = []

    for pf_path in parquets:
        pf = pq.ParquetFile(pf_path)
        for arrow_batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
            img_list = arrow_batch.column("image").to_pylist()
            lbl_list = arrow_batch.column("label").to_pylist()

            # Parallel decode across the arrow batch (PIL releases GIL for JPEG)
            decoded = list(pool.map(_decode_and_preprocess, img_list))

            for tensor, label in zip(decoded, lbl_list):
                pbar.update(1)
                if tensor is None:
                    continue
                pending_tensors.append(tensor)
                pending_labels.append(int(label))

                if len(pending_tensors) >= batch_size:
                    _flush_batch()

    _flush_batch()
    pbar.close()
    pool.shutdown(wait=True)
    out_queue.put(None)  # sentinel


def _start_prefetch(split: str, batch_size: int, num_decode_workers: int = 4) -> Queue:
    """Start the prefetch producer in a background thread, return the queue."""
    q: Queue = Queue(maxsize=4)
    t = Thread(
        target=_prefetch_worker,
        args=(split, batch_size, num_decode_workers, q),
        daemon=True,
    )
    t.start()
    return q


# ---------------------------------------------------------------------------
# VAE loading / encoding
# ---------------------------------------------------------------------------

def load_vae(device: torch.device):
    from diffusers import AutoencoderKL

    print("\nLoading SD-VAE (stabilityai/sd-vae-ft-ema)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device).half().eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor) -> np.ndarray:
    """[B,3,256,256] fp16 channels_last -> [B,4,32,32] float32 numpy."""
    return (vae.encode(images).latent_dist.mode() * SD_VAE_SCALE).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Flush / consolidation helpers
# ---------------------------------------------------------------------------

def _flush_parts(
    split_dir: Path,
    class_latents: dict[int, list[np.ndarray]],
    part_counter: dict[int, int],
):
    """Write accumulated latents to temporary part files."""
    tmp_dir = split_dir / "_tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for cid, lats in class_latents.items():
        if not lats:
            continue
        arr = np.stack(lats)
        pid = part_counter.get(cid, 0)
        np.save(tmp_dir / f"{cid}_part{pid:04d}.npy", arr)
        part_counter[cid] = pid + 1


def _consolidate_class(split_dir: Path, cid: int) -> int:
    """Concatenate all temp parts for a class into split_dir/{cid}.npy. Returns sample count."""
    tmp_dir = split_dir / "_tmp_parts"
    parts = sorted(tmp_dir.glob(f"{cid}_part*.npy"))
    if not parts:
        return 0
    arrays = [np.load(p) for p in parts]
    merged = np.concatenate(arrays)
    np.save(split_dir / f"{cid}.npy", merged)
    return merged.shape[0]


# ---------------------------------------------------------------------------
# Split encoding
# ---------------------------------------------------------------------------

def _split_is_complete(split_dir: Path, split_name: str) -> bool:
    """Check if a split was already fully encoded (flat format)."""
    npy_files = list(split_dir.glob("*.npy"))
    if len(npy_files) < NUM_CLASSES:
        return False
    total = 0
    for f in npy_files:
        try:
            arr = np.load(f, mmap_mode="r")
            total += arr.shape[0]
        except Exception:
            return False
    expected = EXPECTED_SAMPLES.get(split_name, 0)
    return total >= expected * 0.999


def encode_split(
    vae,
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    flush_every: int = 200_000,
    num_decode_workers: int = 4,
):
    """Encode a full split to flat per-class .npy files.

    Uses a prefetch pipeline: CPU threads decode images while the main thread
    runs VAE encoding on GPU. Periodically flushes to temp part files to bound
    memory, then consolidates at the end.
    """
    split_dir = output_dir / split

    if _split_is_complete(split_dir, split):
        print(f"\n{split} already complete, skipping.")
        return

    if split_dir.exists() and any(split_dir.iterdir()):
        print(f"\n{split} has partial data — wiping and re-encoding.")
        shutil.rmtree(split_dir)

    split_dir.mkdir(parents=True, exist_ok=True)

    # Start prefetch pipeline
    prefetch_q = _start_prefetch(split, batch_size, num_decode_workers)

    class_latents: dict[int, list[np.ndarray]] = {}
    part_counter: dict[int, int] = {}
    n_encoded = 0
    since_last_flush = 0

    while True:
        item = prefetch_q.get()
        if item is None:
            break

        batch_tensor, batch_labels = item
        # Send to GPU with channels_last and non_blocking
        batch_gpu = batch_tensor.to(
            device, dtype=torch.float16, non_blocking=True,
            memory_format=torch.channels_last,
        )
        latents = encode_batch(vae, batch_gpu)

        for lat, lab in zip(latents, batch_labels):
            class_latents.setdefault(lab, []).append(lat)

        n_encoded += len(latents)
        since_last_flush += len(latents)

        if since_last_flush >= flush_every:
            _flush_parts(split_dir, class_latents, part_counter)
            class_latents.clear()
            since_last_flush = 0

    # Final flush
    _flush_parts(split_dir, class_latents, part_counter)
    class_latents.clear()

    # Consolidate temp parts into flat files
    print(f"\n  Consolidating {split} into flat .npy files...")
    total_samples = 0
    nonempty = 0
    for cid in range(NUM_CLASSES):
        count = _consolidate_class(split_dir, cid)
        total_samples += count
        if count > 0:
            nonempty += 1

    # Clean up temp directory
    tmp_dir = split_dir / "_tmp_parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"  {split}: encoded {n_encoded} images, {total_samples} stored across {nonempty} classes")


# ---------------------------------------------------------------------------
# FID reference stats (streaming accumulation)
# ---------------------------------------------------------------------------

_inception_preprocess = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _decode_and_preprocess_inception(img_bytes: bytes) -> torch.Tensor | None:
    """Decode JPEG bytes to Inception-preprocessed [3,299,299] tensor."""
    from PIL import Image
    try:
        raw = img_bytes
        if isinstance(raw, dict):
            raw = raw["bytes"]
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return _inception_preprocess(img)
    except Exception:
        return None


class _InceptionFeatureExtractor:
    """Wraps Inception v3 with a persistent avgpool hook."""

    def __init__(self, device: torch.device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        print("\nLoading Inception v3...")
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
        self.model = self.model.to(device).half().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.features: list[np.ndarray] = []
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features.append(
            output.squeeze(-1).squeeze(-1).float().cpu().numpy().astype(np.float64)
        )

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """[B,3,299,299] fp16 -> [B,2048] float64."""
        self.features.clear()
        self.model(images)
        return self.features[0]


def _fid_prefetch_worker(
    split: str,
    batch_size: int,
    num_decode_workers: int,
    out_queue: Queue,
):
    """Producer for FID stats: decodes images at 299x299 for Inception."""
    import pyarrow.parquet as pq
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parquets = _find_local_parquets(split)
    if not parquets:
        parquets = _download_split(split)

    total = sum(pq.read_metadata(p).num_rows for p in parquets)
    pbar = tqdm(total=total, desc="FID features")

    pool = ThreadPoolExecutor(max_workers=num_decode_workers)
    pending: list[torch.Tensor] = []

    def _flush():
        nonlocal pending
        if not pending:
            return
        out_queue.put(torch.stack(pending))
        pending = []

    for pf_path in parquets:
        pf = pq.ParquetFile(pf_path)
        for arrow_batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
            img_list = arrow_batch.column("image").to_pylist()

            decoded = list(pool.map(_decode_and_preprocess_inception, img_list))

            for tensor in decoded:
                pbar.update(1)
                if tensor is None:
                    continue
                pending.append(tensor)
                if len(pending) >= batch_size:
                    _flush()

    _flush()
    pbar.close()
    pool.shutdown(wait=True)
    out_queue.put(None)


def compute_fid_stats(
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    num_decode_workers: int = 4,
):
    """Compute Inception pool3 mu/sigma via streaming accumulation."""
    fid_dir = output_dir / "fid_stats"
    fid_dir.mkdir(parents=True, exist_ok=True)
    mu_path = fid_dir / f"{split}_inception_mu.npy"
    sigma_path = fid_dir / f"{split}_inception_sigma.npy"

    if mu_path.exists() and sigma_path.exists():
        print("\nFID stats already computed, skipping.")
        return

    extractor = _InceptionFeatureExtractor(device)

    # Start prefetch pipeline for inception
    q: Queue = Queue(maxsize=4)
    t = Thread(
        target=_fid_prefetch_worker,
        args=(split, batch_size, num_decode_workers, q),
        daemon=True,
    )
    t.start()

    S1 = np.zeros(2048, dtype=np.float64)
    S2 = np.zeros((2048, 2048), dtype=np.float64)
    n = 0

    print("\nComputing FID stats (streaming)...")
    while True:
        item = q.get()
        if item is None:
            break
        batch_gpu = item.to(device, dtype=torch.float16, non_blocking=True)
        x = extractor.extract(batch_gpu)
        S1 += x.sum(axis=0)
        S2 += x.T @ x
        n += x.shape[0]

    assert n >= 2, f"Need >=2 images for covariance, got {n}"

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
        if not split_dir.exists():
            print(f"  {split}: MISSING")
            ok = False
            continue

        npy_files = sorted(split_dir.glob("*.npy"))
        class_ids = set()
        total = 0
        for f in npy_files:
            try:
                cid = int(f.stem)
                class_ids.add(cid)
                arr = np.load(f, mmap_mode="r")
                total += arr.shape[0]
            except Exception:
                continue

        nonempty = len(class_ids)

        # Spot-check shapes
        shapes_ok = True
        for cid in [0, 500, 999]:
            p = split_dir / f"{cid}.npy"
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                if arr.ndim != 4 or arr.shape[1:] != (4, 32, 32):
                    print(f"    BAD SHAPE: {split}/{cid}.npy -> {arr.shape}")
                    shapes_ok = False
            else:
                print(f"    MISSING: {split}/{cid}.npy")
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
            ok = False
        else:
            arr = np.load(p, mmap_mode="r")
            if arr.shape != shape:
                print(f"  FID: {name} BAD SHAPE {arr.shape}")
                ok = False

    if ok:
        print("  All checks passed.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k for drifting model training")
    parser.add_argument("--output-dir", type=str, default="data/imagenet")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--skip-fid-stats", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--flush-every", type=int, default=200_000,
                        help="Flush latents to disk every N samples to bound memory")
    parser.add_argument("--decode-workers", type=int, default=4,
                        help="Number of threads for parallel image decoding")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        raise SystemExit(0 if verify_data(output_dir) else 1)

    _setup_precision()

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
        encode_split(
            vae, output_dir, split, args.batch_size, device,
            flush_every=args.flush_every,
            num_decode_workers=args.decode_workers,
        )
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FID stats (always computed on val)
    if not args.skip_fid_stats:
        compute_fid_stats(output_dir, "val", args.batch_size, device, args.decode_workers)

    # Metadata
    elapsed = time.time() - t0
    meta = {
        "dataset": "ILSVRC/imagenet-1k",
        "vae": "stabilityai/sd-vae-ft-ema",
        "vae_scale": SD_VAE_SCALE,
        "latent_shape": [4, 32, 32],
        "image_size": 256,
        "splits": args.splits,
        "format": "flat-npy-v1",
        "time_seconds": round(elapsed, 1),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone in {elapsed/3600:.1f}h.")
    verify_data(output_dir)


if __name__ == "__main__":
    main()
