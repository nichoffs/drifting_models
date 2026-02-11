"""FID evaluation for drifting models."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from model import generator_b2, generator_l2


# --- FID computation ---


def compute_fid(mu1, sigma1, mu2, sigma2):
    """Frechet Inception Distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    # numerical fix: drop imaginary component from roundoff
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# --- Inception feature extraction ---


def load_inception(device):
    """Load Inception v3 and return a function that extracts pool3 features."""
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_inception_features(model, images, device):
    """Extract 2048-d pool3 features from Inception v3.

    Args:
        model: Inception v3 model
        images: [B, 3, H, W] tensor in [0, 1]
        device: torch device
    """
    # resize to 299×299, normalize for Inception
    x = torch.nn.functional.interpolate(images, size=299, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    features = []
    def hook_fn(module, input, output):
        features.append(output.squeeze(-1).squeeze(-1))

    handle = model.avgpool.register_forward_hook(hook_fn)
    model(x)
    handle.remove()
    return features[0]  # [B, 2048]


# --- VAE decoder ---


def load_vae_decoder(device):
    """Load SD-VAE for decoding latents to pixels."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_latents(vae, latents):
    """Decode SD-VAE latents → pixel images in [0, 1]."""
    x = latents / 0.18215
    x = vae.decode(x).sample
    return (x * 0.5 + 0.5).clamp(0, 1)


# --- main evaluation ---


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser(description="FID evaluation for drifting models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (for FID stats)")
    parser.add_argument("--n-samples", type=int, default=50000, help="Total samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Generation batch size")
    parser.add_argument("--cfg-alpha", type=float, default=1.0, help="CFG alpha for generation")
    parser.add_argument("--model", type=str, default="b2", choices=["b2", "l2"], help="Model config")
    parser.add_argument("--save-samples", type=str, default=None, help="Save generated images to dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load generator (EMA weights)
    model_fn = {"b2": generator_b2, "l2": generator_l2}[args.model]
    generator = model_fn().to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["ema"])
    print(f"Loaded EMA weights from {args.checkpoint} (step {ckpt.get('step', '?')})")

    # load VAE decoder
    vae = load_vae_decoder(device)

    # load Inception
    inception = load_inception(device)

    # generate and extract features
    n_classes = 1000
    samples_per_class = args.n_samples // n_classes  # 50 for 50K
    total = n_classes * samples_per_class
    print(f"Generating {total} samples ({samples_per_class} per class)...")

    all_features = []

    for class_id in tqdm(range(n_classes), desc="Generating"):
        for batch_start in range(0, samples_per_class, args.batch_size):
            bs = min(args.batch_size, samples_per_class - batch_start)

            noise = torch.randn(bs, 4, 32, 32, device=device)
            label = torch.full((bs,), class_id, dtype=torch.long, device=device)
            alpha = torch.full((bs,), args.cfg_alpha, device=device)
            style = torch.randint(0, 64, (bs, 32), device=device)

            latents = generator(noise, label, alpha, style)
            pixels = decode_latents(vae, latents)

            # optionally save
            if args.save_samples:
                save_dir = Path(args.save_samples)
                save_dir.mkdir(parents=True, exist_ok=True)
                from torchvision.utils import save_image
                for i in range(bs):
                    idx = class_id * samples_per_class + batch_start + i
                    save_image(pixels[i], save_dir / f"{idx:06d}.png")

            # extract Inception features
            feat = extract_inception_features(inception, pixels, device)
            all_features.append(feat.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)  # [total, 2048]
    print(f"Extracted features: {all_features.shape}")

    # compute generated statistics
    mu_gen = np.mean(all_features, axis=0)
    sigma_gen = np.cov(all_features, rowvar=False)

    # load reference statistics
    data_dir = Path(args.data_dir)
    mu_ref = np.load(data_dir / "fid_stats" / "val_inception_mu.npy")
    sigma_ref = np.load(data_dir / "fid_stats" / "val_inception_sigma.npy")
    print(f"Loaded reference stats from {data_dir / 'fid_stats'}")

    # compute FID
    fid = compute_fid(mu_gen, sigma_gen, mu_ref, sigma_ref)
    print(f"\nFID: {fid:.2f}")
    print(f"  (generated {total} samples, cfg_alpha={args.cfg_alpha})")

    return fid


if __name__ == "__main__":
    evaluate()
