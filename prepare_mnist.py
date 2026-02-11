"""
Prepare MNIST for drifting model training (pixel space).

Downloads MNIST, pads 28→32, normalizes to [-1,1], stores per-class .npy files.
No VAE — the generator works directly in pixel space for MNIST.

Storage layout:
  {output_dir}/
    train/
      0.npy  ...  9.npy   # [N_c, 1, 32, 32] float32
    val/
      0.npy  ...  9.npy

Usage:
  uv run python prepare_mnist.py
  uv run python prepare_mnist.py --output-dir mnist_data
"""

import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/mnist")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    output_dir = Path(args.output_dir)

    for split, is_train in [("train", True), ("val", False)]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        ds = datasets.MNIST(root=output_dir / "raw", train=is_train, download=True, transform=transform)

        # group by class
        by_class = {c: [] for c in range(10)}
        for img, label in ds:
            by_class[label].append(img.numpy())

        for c in range(10):
            arr = np.stack(by_class[c])  # [N_c, 1, 32, 32]
            np.save(split_dir / f"{c}.npy", arr)

        total = sum(len(v) for v in by_class.values())
        print(f"{split}: {total} images across 10 classes → {split_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
