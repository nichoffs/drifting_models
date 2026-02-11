"""2D toy demo — reproduces drifting model training on simple distributions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# target distribution
def sample_spiral(n, device="cpu"):
    """Swiss-roll spiral normalized to [-1, 1]."""
    t = 0.5 * math.pi + 4.0 * math.pi * torch.rand(n, device=device)
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    pts = pts + 0.03 * torch.randn_like(pts)
    return pts



def compute_drift(gen, pos, temp=0.05):
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()

    # double normalization (geometric mean of row & col)
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)

    return pos_coeff @ targets[G:] - neg_coeff @ targets[:G]


class Generator(nn.Module):
    def __init__(self, in_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        return self.net(z)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    in_dim = 32
    gen = Generator(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

    N = 2048
    n_steps = 5001
    snapshot_iters = [0, 50, 100, 200, 500, 1000, 2000, 5000]
    snapshots = {}
    losses = []

    for step in range(n_steps):
        z = torch.randn(N, in_dim, device=device)
        x = gen(z)
        y_pos = sample_spiral(N, device=device)

        with torch.no_grad():
            V = compute_drift(x, y_pos)
            target = x + V

        loss = F.mse_loss(x, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if step in snapshot_iters:
            with torch.no_grad():
                z_vis = torch.randn(2048, in_dim, device=device)
                snapshots[step] = gen(z_vis).cpu()
            print(f"step {step:4d}  loss={loss.item():.6f}")

    return snapshots, losses, snapshot_iters

def plot(snapshots, losses, snapshot_iters):
    gt = sample_spiral(2048).numpy()
    titles = [f"iter {i}" for i in snapshot_iters] + ["ground truth"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 5),
                             gridspec_kw={"height_ratios": [3, 1.2]})

    n_cols = len(snapshot_iters) + 1
    gs = axes[0].get_subplotspec().subgridspec(1, n_cols, wspace=0.05)
    axes[0].remove()

    scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    lim = 1.5
    for i, ax in enumerate(scatter_axes):
        if i < len(snapshot_iters):
            pts = snapshots[snapshot_iters[i]].numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="tab:orange", alpha=0.6)
        else:
            ax.scatter(gt[:, 0], gt[:, 1], s=1.5, c="tab:blue", alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(titles[i], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_loss = axes[1]
    ax_loss.plot(range(len(losses)), losses, linewidth=0.7, color="gray", alpha=0.7)
    for it in snapshot_iters:
        if it < len(losses):
            ax_loss.plot(it, losses[it], "o", color="tab:orange", markersize=5, zorder=5)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("iteration", fontsize=9)
    ax_loss.set_ylabel("loss", fontsize=9)
    ax_loss.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig("plots/toy_spiral.png", dpi=150, bbox_inches="tight")
    print("saved plots/toy_spiral.png")


if __name__ == "__main__":
    snapshots, losses, snapshot_iters = train()
    plot(snapshots, losses, snapshot_iters)
