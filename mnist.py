"""
Train a drifting model on MNIST (pixel space).

The generator maps noise [B, 1, 32, 32] → pixels [B, 1, 32, 32].
Drifting loss is computed on flattened pixels — no VAE, no feature encoder.

Usage:
  uv run python prepare_mnist.py               # one-time data prep
  uv run python mnist.py                        # train
  uv run python mnist.py --epochs 2000 --lr 5e-4
"""

import argparse
import copy
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image

from loss import drifting_loss
from model import generator_mnist

class PixelDataset:
    """Per-class .npy pixel tensors, all in memory."""

    def __init__(self, data_dir, split="train"):
        self.data = {}
        split_dir = Path(data_dir) / split
        for f in sorted(split_dir.glob("*.npy")):
            self.data[int(f.stem)] = np.load(f)  # [N_c, 1, 32, 32]
        self.classes = sorted(self.data.keys())
        total = sum(v.shape[0] for v in self.data.values())
        print(f"Loaded {len(self.classes)} classes, {total} samples from {split_dir}")

    def sample(self, class_id, n, device="cpu"):
        arr = self.data[class_id]
        idx = np.random.choice(len(arr), size=n, replace=len(arr) < n)
        return torch.from_numpy(arr[idx]).float().to(device)


class SampleQueue:
    """Per-class queue of real samples for positives."""

    def __init__(self, queue_size=256):
        self.queues = {}
        self.queue_size = queue_size

    def push(self, samples, class_id):
        if class_id not in self.queues:
            self.queues[class_id] = deque(maxlen=self.queue_size)
        q = self.queues[class_id]
        for i in range(samples.shape[0]):
            q.append(samples[i].detach())

    def sample(self, class_id, n, device="cpu"):
        q = self.queues.get(class_id)
        if q is None or len(q) < n:
            return None
        items = list(q)
        idx = np.random.choice(len(items), size=n, replace=False)
        return torch.stack([items[i] for i in idx]).to(device)


def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def save_viz(ema, dataset, step, viz_dir, device, n_per_class=8):
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device=device).manual_seed(42)
    rows = []
    for c in dataset.classes:
        noise = torch.randn(n_per_class, 1, 32, 32, device=device, generator=rng)
        label = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        alpha = torch.ones(n_per_class, device=device)
        style = torch.randint(0, 64, (n_per_class, 8), device=device)
        pixels = ema(noise, label, alpha, style)
        rows.append(pixels)
    grid = torch.cat(rows, dim=0)
    grid = grid.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] → [0,1]
    path = viz_dir / f"step_{step:06d}.png"
    save_image(grid, path, nrow=n_per_class, padding=1)
    print(f"  viz: {path}")

def train():
    parser = argparse.ArgumentParser(description="Train drifting model on MNIST")
    parser.add_argument("--data-dir", type=str, default="mnist_data")
    parser.add_argument("--n-pos", type=int, default=64, help="Positives per class")
    parser.add_argument("--n-neg", type=int, default=64, help="Negatives (generated) per class")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs (1 step/epoch)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_mnist")
    parser.add_argument("--viz-dir", type=str, default="plots_mnist")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--viz-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
    dataset = PixelDataset(args.data_dir, split="train")
    n_classes = len(dataset.classes)

    # model
    generator = generator_mnist(n_classes=n_classes).to(device)
    ema = copy.deepcopy(generator)
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator: {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
    )

    # resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # queue
    queue = SampleQueue(queue_size=256)
    for c in dataset.classes:
        queue.push(dataset.sample(c, 64, device=device), c)

    taus = (0.02, 0.05, 0.2)
    total_steps = args.epochs
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # training loop — all classes every step (batch_nc = n_classes)
    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup_steps, total_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        total_loss = 0.0
        total_drift = 0.0

        # shuffle class order
        class_order = list(dataset.classes)
        random.shuffle(class_order)

        for c in class_order:
            # generate
            noise = torch.randn(args.n_neg, 1, 32, 32, device=device)
            style = torch.randint(0, 64, (args.n_neg, 8), device=device)
            label = torch.full((args.n_neg,), c, dtype=torch.long, device=device)
            alpha = torch.ones(args.n_neg, device=device)
            x = generator(noise, label, alpha, style)

            # positives from queue
            y_pos = queue.sample(c, args.n_pos, device=device)
            if y_pos is None:
                y_pos = dataset.sample(c, args.n_pos, device=device)

            # features = flattened pixels
            x_feat = x.flatten(1)                        # [n_neg, 1024]
            y_pos_feat = y_pos.flatten(1).detach()       # [n_pos, 1024]
            y_neg_feat = x_feat.detach()                 # [n_neg, 1024]

            loss_c, drift_c = drifting_loss(x_feat, y_pos_feat, y_neg_feat, taus=taus, mask_self=True)
            (loss_c / n_classes).backward()

            total_loss += loss_c.item() / n_classes
            total_drift += drift_c / n_classes

            # refresh queue
            queue.push(dataset.sample(c, 32, device=device), c)

        grad_norm = clip_grad_norm_(generator.parameters(), args.grad_clip)
        optimizer.step()

        # ema update
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), generator.parameters()):
                ep.lerp_(p, 1 - args.ema_decay)

        # logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_s = max(1, step - start_step) / max(elapsed, 1)
            print(f"step {step}/{total_steps} | loss {total_loss:.4f} | "
                  f"drift {total_drift:.4f} | lr {lr:.2e} | "
                  f"grad {grad_norm:.2f} | {steps_s:.1f} steps/s")

        # viz
        if args.viz_every > 0 and step % args.viz_every == 0:
            save_viz(ema, dataset, step, args.viz_dir, device)

        # checkpoint
        if step > 0 and step % args.save_every == 0:
            path = Path(args.ckpt_dir) / f"step_{step:06d}.pt"
            torch.save({
                "generator": generator.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, path)
            print(f"  saved: {path}")

    print(f"\nDone. {total_steps} steps in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    train()
