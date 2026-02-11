#!/bin/bash
# Train drifting model with DDP across all available GPUs.
# Usage:
#   ./run.sh                              # defaults (B/2, 100 epochs, all GPUs)
#   ./run.sh --model l2 --epochs 50       # L/2, 50 epochs
#   ./run.sh --resume checkpoints/ckpt_epoch10.pt

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
NGPU=${NGPU:-1}

echo "Training with $NGPU GPU(s)"
uv run torchrun --nproc_per_node=$NGPU train.py "$@"