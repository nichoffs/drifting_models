"""Training loop for drifting models on ImageNet latents."""

import argparse
import copy
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint

from loss import drifting_loss
from model import generator_b2, generator_l2

class LatentDataset:
    """Loads all per-class .npy latent files into memory."""

    def __init__(self, data_dir, split="train"):
        self.data = {}
        split_dir = Path(data_dir) / split
        for f in sorted(split_dir.glob("*.npy")):
            class_id = int(f.stem)
            self.data[class_id] = np.load(f)  # [N_c, 4, 32, 32]
        self.classes = sorted(self.data.keys())
        total = sum(v.shape[0] for v in self.data.values())
        print(f"Loaded {len(self.classes)} classes, {total} total samples from {split_dir}")

    def sample(self, class_id, n, device="cpu"):
        """Sample n random latents from a class, as a float32 tensor."""
        arr = self.data[class_id]
        idx = np.random.choice(len(arr), size=n, replace=len(arr) < n)
        return torch.from_numpy(arr[idx]).float().to(device)


class SampleQueue:
    """Per-class and global queues for caching generated/real samples."""

    def __init__(self, class_queue_size=128, global_queue_size=1000):
        self.class_queues = {}     # {class_id: deque of [4, 32, 32] tensors}
        self.global_queue = deque(maxlen=global_queue_size)
        self.class_queue_size = class_queue_size

    def push_class(self, latents, class_id):
        """Push latents [N, 4, 32, 32] into per-class queue."""
        if class_id not in self.class_queues:
            self.class_queues[class_id] = deque(maxlen=self.class_queue_size)
        q = self.class_queues[class_id]
        for i in range(latents.shape[0]):
            q.append(latents[i].detach())

    def push_global(self, latents):
        """Push latents [N, 4, 32, 32] into global unconditional queue."""
        for i in range(latents.shape[0]):
            self.global_queue.append(latents[i].detach())

    def sample_positives(self, class_id, n, device="cpu"):
        """Sample n from per-class queue (with replacement if needed)."""
        q = self.class_queues.get(class_id)
        if q is None or len(q) == 0:
            return None
        items = list(q)
        idx = np.random.choice(len(items), size=n, replace=len(items) < n)
        return torch.stack([items[i] for i in idx]).to(device)

    def sample_unconditional(self, n, device="cpu"):
        """Sample n from global queue."""
        if len(self.global_queue) < n:
            return None
        items = list(self.global_queue)
        idx = np.random.choice(len(items), size=n, replace=False)
        return torch.stack([items[i] for i in idx]).to(device)

    def class_queue_ready(self, class_id, min_size):
        q = self.class_queues.get(class_id)
        return q is not None and len(q) >= min_size


class FeatureEncoder(nn.Module):
    """Multi-scale feature extraction via pretrained ResNet-50 + VAE decoder.

    Extracts features at multiple spatial scales from ResNet-50 layers,
    then decomposes into per-location, 2×2 patch, 4×4 patch, and global stats.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        from diffusers import AutoencoderKL
        from torchvision.models import ResNet50_Weights, resnet50

        # VAE decoder (latent → pixel)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        self.vae = self.vae.to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae_scale = 0.18215

        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights).to(device).eval()
        for p in self.resnet.parameters():
            p.requires_grad_(False)

        # preprocessing (ImageNet normalization)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # hook storage
        self._features = {}
        self.resnet.layer1.register_forward_hook(self._make_hook("layer1"))
        self.resnet.layer2.register_forward_hook(self._make_hook("layer2"))
        self.resnet.layer3.register_forward_hook(self._make_hook("layer3"))
        self.resnet.layer4.register_forward_hook(self._make_hook("layer4"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def decode_latents(self, latents):
        x = latents / self.vae_scale

        # manually run the VAE decoder with checkpointing per block
        z = self.vae.decoder.conv_in(x)

        for up_block in self.vae.decoder.up_blocks:
            z = checkpoint(up_block, z, None, use_reentrant=False)

        z = self.vae.decoder.conv_norm_out(z)
        z = self.vae.decoder.conv_act(z)
        z = self.vae.decoder.conv_out(z)

        return (z * 0.5 + 0.5).clamp(0, 1)


    def extract_features(self, latents):
        """Extract multi-scale features from latents.

        Args:
            latents: [N, 4, 32, 32] SD-VAE latents

        Returns: list of [N, D_i] feature vectors (one per scale/stat combo)
        """
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pixels = self.decode_latents(latents)

            if pixels.shape[-1] != 224:
                pixels = nn.functional.interpolate(pixels, size=224, mode="bilinear", align_corners=False)

            pixels = (pixels - self.mean) / self.std
            self.resnet(pixels)

        result = []
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            feat = self._features[name].float()  
            N, C, H, W = feat.shape

            # global mean
            result.append(feat.mean(dim=(2, 3)))  # [N, C]

            # global std
            result.append(feat.std(dim=(2, 3)))   # [N, C]

            # 2×2 patch means (if spatial dim >= 2)
            if H >= 2 and W >= 2:
                ph, pw = H // 2, W // 2
                for i in range(2):
                    for j in range(2):
                        patch = feat[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        result.append(patch.mean(dim=(2, 3)))  # [N, C]

            # 4×4 patch means (if spatial dim >= 4)
            if H >= 4 and W >= 4:
                ph, pw = H // 4, W // 4
                for i in range(4):
                    for j in range(4):
                        patch = feat[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        result.append(patch.mean(dim=(2, 3)))  # [N, C]

        return result

def compute_multiscale_loss(x_features, ypos_features, yneg_features,
                            yneg_unc_features, w, taus):
    """Compute drifting loss across all feature scales.

    Args:
        x_features:        list of [N, D] feature vectors for generated
        ypos_features:     list of [N_pos, D] for positives
        yneg_features:     list of [N_neg, D] for negatives
        yneg_unc_features: list of [N_unc, D] for unconditional negatives (or None)
        w:                 scalar weight for unconditional negatives
        taus:              tuple of temperatures
    """
    total_loss = 0.0
    total_drift = 0.0
    n_scales = len(x_features)

    for i in range(n_scales):
        x_f = x_features[i]
        yp_f = ypos_features[i]
        yn_f = yneg_features[i]

        # concatenate unconditional negatives (weighted) if available
        if yneg_unc_features is not None and w > 0:
            yn_unc_f = yneg_unc_features[i]
            yn_full = torch.cat([yn_f, yn_unc_f], dim=0)
        else:
            yn_full = yn_f

        loss_i, drift_i = drifting_loss(x_f, yp_f, yn_full, taus=taus, mask_self=True)
        total_loss = total_loss + loss_i
        total_drift += drift_i

    return total_loss / n_scales, total_drift / n_scales


def sample_alpha(batch_size, device="cpu"):
    """Sample CFG alpha: 50% chance α=1, 50% power-law in [1, 4]."""
    alpha = torch.ones(batch_size, device=device)
    mask = torch.rand(batch_size, device=device) > 0.5
    # power-law p(α) ∝ α^{-3} → CDF inversion: α = (1 - u * (1 - 4^{-2}))^{-1/2}
    u = torch.rand(mask.sum().item(), device=device)
    alpha[mask] = (1 - u * (1 - 4.0**(-2))).pow(-0.5)
    return alpha


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1 - decay)

def get_lr(step, warmup_steps, total_steps, base_lr):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def setup_ddp():
    """Initialize DDP if launched via torchrun, else single-GPU."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

@torch.no_grad()
def save_viz(ema, encoder, dataset, step, viz_dir, device, n_per_class=8):
    """Generate a sample grid from the EMA model and save as PNG."""
    from torchvision.utils import save_image

    classes = dataset.classes
    if len(classes) > 10:
        idx = np.linspace(0, len(classes) - 1, 10, dtype=int)
        viz_classes = [classes[i] for i in idx]
    else:
        viz_classes = classes

    rng = torch.Generator(device=device).manual_seed(42)
    all_pixels = []

    for c in viz_classes:
        noise = torch.randn(n_per_class, 4, 32, 32, device=device, generator=rng)
        style = torch.randint(0, 64, (n_per_class, 32), device=device)
        label = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        alpha = torch.ones(n_per_class, device=device)
        latents = ema(noise, label, alpha, style)
        pixels = encoder.decode_latents(latents)
        all_pixels.append(pixels)

    grid = torch.cat(all_pixels, dim=0)
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    save_image(grid, f"{viz_dir}/step_{step:06d}.png", nrow=n_per_class, padding=2)
    print(f"  Saved viz: {viz_dir}/step_{step:06d}.png "
          f"({len(viz_classes)} classes x {n_per_class} samples)")

def train():
    parser = argparse.ArgumentParser(description="Train a generative drifting model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch-nc", type=int, default=64, help="Classes per batch")
    parser.add_argument("--n-pos", type=int, default=64, help="Positives per class")
    parser.add_argument("--n-neg", type=int, default=64, help="Negatives (generated) per class")
    parser.add_argument("--n-unc", type=int, default=16, help="Unconditional negatives for CFG")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=2.0, help="Gradient clip norm")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--model", type=str, default="b2", choices=["b2", "l2"], help="Model config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-every", type=int, default=50, help="Log interval (steps)")
    parser.add_argument("--save-every-epochs", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--viz-every", type=int, default=500, help="Save sample grid every N steps (0=off)")
    parser.add_argument("--viz-dir", type=str, default="plots", help="Directory for sample grids")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"Device: {device} (x{world_size})")

    dataset = LatentDataset(args.data_dir, split="train")
    n_classes = len(dataset.classes)

    model_fn = {"b2": generator_b2, "l2": generator_l2}[args.model]
    generator = model_fn(n_classes=n_classes).to(device)
    ema = copy.deepcopy(generator)
    for p in ema.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in generator.parameters())
    if is_main:
        print(f"Generator: {args.model.upper()} — {n_params / 1e6:.1f}M params")

    encoder = FeatureEncoder(device=device).to(device)

    start_step = 0
    start_epoch = 0
    ckpt = None
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        if is_main:
            print(f"  Resumed at epoch {start_epoch}, step {start_step}")

    if world_size > 1:
        generator = DDP(generator, device_ids=[local_rank])
    raw_generator = generator.module if world_size > 1 else generator

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.wd,
    )
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt

    queue = SampleQueue(class_queue_size=128, global_queue_size=1000)

    taus = (0.02, 0.05, 0.2)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    steps_per_epoch = n_classes // args.batch_nc
    total_steps = args.epochs * steps_per_epoch
    if is_main:
        print(f"Steps/epoch: {steps_per_epoch}, total steps: {total_steps}")

    for c in dataset.classes:
        real = dataset.sample(c, min(64, args.n_pos), device=device)
        queue.push_class(real, c)
        queue.push_global(real[:4])

    ckpt_dir = Path(args.ckpt_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        # deterministic class shuffle (same across all ranks)
        epoch_rng = random.Random(args.seed + epoch)
        class_order = list(dataset.classes)
        epoch_rng.shuffle(class_order)

        for batch_start in range(0, len(class_order), args.batch_nc):
            classes = class_order[batch_start:batch_start + args.batch_nc]
            if len(classes) < args.batch_nc:
                break

            if len(classes) >= world_size:
                local_nc = len(classes) // world_size
                my_classes = classes[rank * local_nc:(rank + 1) * local_nc]
            else:
                my_classes = classes

            lr = get_lr(step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            total_loss = 0.0
            total_drift = 0.0

            optimizer.zero_grad()
            for c in my_classes:
                alpha = sample_alpha(args.n_neg, device=device)

                # generate
                noise = torch.randn(args.n_neg, 4, 32, 32, device=device)
                style = torch.randint(0, 64, (args.n_neg, 32), device=device)
                label = torch.full((args.n_neg,), c, dtype=torch.long, device=device)
                x = generator(noise, label, alpha, style)

                # positives from queue (use real data from dataset as fallback)
                if queue.class_queue_ready(c, args.n_pos):
                    y_pos = queue.sample_positives(c, args.n_pos, device=device)
                else:
                    y_pos = dataset.sample(c, args.n_pos, device=device)

                # unconditional negatives
                alpha_mean = alpha.mean().item()
                w = (alpha_mean - 1) * (args.n_neg - 1) / max(args.n_unc, 1)

                y_unc = queue.sample_unconditional(args.n_unc, device=device)

                # extract features (decode using vae first)
                with torch.no_grad():
                    feat_pos = encoder.extract_features(y_pos)
                    feat_neg = encoder.extract_features(x.detach())
                    feat_unc = encoder.extract_features(y_unc) if y_unc is not None and w > 0 else None

                # generated -> vae decode (pixels) -> resnet for loss
                feat_x = encoder.extract_features(x)

                loss_c, drift_c = compute_multiscale_loss(feat_x, feat_pos, feat_neg, feat_unc, w, taus)
                (loss_c / len(my_classes)).backward()

                fresh = dataset.sample(c, min(64, args.n_pos), device=device)
                queue.push_class(fresh, c)

                total_loss += (loss_c.detach() / len(my_classes)).item()
                total_drift += drift_c / len(my_classes)

            grad_norm = clip_grad_norm_(generator.parameters(), args.grad_clip)
            optimizer.step()
            update_ema(ema, raw_generator, args.ema_decay)

            # push unconditional samples to global queue
            rand_c = random.choice(dataset.classes)
            queue.push_global(dataset.sample(rand_c, 4, device=device))

            step += 1

            if step % args.log_every == 0:
                if world_size > 1:
                    stats = torch.tensor([total_loss, total_drift], device=device)
                    dist.all_reduce(stats)
                    total_loss, total_drift = (stats / world_size).tolist()
                if is_main:
                    elapsed = time.time() - t0
                    steps_s = step / max(elapsed, 1)
                    print(f"epoch {epoch} | step {step}/{total_steps} | "
                          f"loss {total_loss:.4f} | drift {total_drift:.4f} | "
                          f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                          f"{steps_s:.1f} steps/s")

            if args.viz_every > 0 and step % args.viz_every == 0 and is_main:
                save_viz(ema, encoder, dataset, step, args.viz_dir, device)

        if is_main and ((epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1):
            ckpt_path = ckpt_dir / f"ckpt_epoch{epoch+1}.pt"
            torch.save({
                "generator": raw_generator.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if is_main:
        print(f"\nTraining complete. {step} steps in {(time.time() - t0) / 3600:.1f} hours.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
