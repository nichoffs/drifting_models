"""DiT-like generator for drifting models — one-step latent generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.scale


def precompute_rope_2d(h, w, dim):
    """Precompute 2D rotary position embedding frequencies.

    Returns: [h*w, dim//2, 2] cos/sin pairs (half for row, half for col).
    """
    half = dim // 2
    freqs_h = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))  # [half//2]
    freqs_w = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))

    pos_h = torch.arange(h).float()
    pos_w = torch.arange(w).float()

    # [h, half//2] and [w, half//2]
    angles_h = torch.outer(pos_h, freqs_h)
    angles_w = torch.outer(pos_w, freqs_w)

    # expand to grid: [h, w, half//2] each
    angles_h = angles_h[:, None, :].expand(h, w, -1)
    angles_w = angles_w[None, :, :].expand(h, w, -1)

    # concat along freq dim → [h*w, half]
    angles = torch.cat([angles_h.reshape(h * w, -1), angles_w.reshape(h * w, -1)], dim=-1)

    # stack cos/sin → [h*w, half, 2]
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x, freqs):
    """Apply rotary position embeddings to x.

    Args:
        x: [B, n_heads, T, head_dim]
        freqs: [T, head_dim//2, 2]
    """
    B, H, T, D = x.shape
    x = x.float().reshape(B, H, T, D // 2, 2)
    cos = freqs[:, :, 0]  # [T, D//2]
    sin = freqs[:, :, 1]
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(B, H, T, D).to(x.dtype)


class DriftBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # adaLN modulation: cond → 6 * dim (γ1, β1, α1, γ2, β2, α2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # zero-init the final linear
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # attention
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # feedforward (SwiGLU)
        self.norm2 = RMSNorm(dim)
        ffn_dim = round(dim * 8 / 3 / 64) * 64
        self.ffn_gate_up = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, cond, rope_freqs, n_registers):
        """
        Args:
            x: [B, T, dim] — register tokens + spatial tokens
            cond: [B, dim] — conditioning vector
            rope_freqs: [n_spatial, head_dim//2, 2]
            n_registers: int — number of register tokens at the front
        """
        # adaLN modulation
        mod = self.adaLN_modulation(cond)  # [B, 6*dim]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # --- attention ---
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)

        B, T, D = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, n_heads, head_dim]
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on spatial tokens only (skip register tokens)
        if n_registers > 0:
            q_reg, q_spatial = q[:, :, :n_registers], q[:, :, n_registers:]
            k_reg, k_spatial = k[:, :, :n_registers], k[:, :, n_registers:]
            q_spatial = apply_rope(q_spatial, rope_freqs)
            k_spatial = apply_rope(k_spatial, rope_freqs)
            q = torch.cat([q_reg, q_spatial], dim=2)
            k = torch.cat([k_reg, k_spatial], dim=2)
        else:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, n_heads, T, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)

        x = x + alpha1.unsqueeze(1) * attn_out

        # --- feedforward ---
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)

        gate_up = self.ffn_gate_up(h)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = self.ffn_down(h)

        x = x + alpha2.unsqueeze(1) * h

        return x



class Generator(nn.Module):
    def __init__(self, depth=12, dim=768, patch_size=2, n_heads=12,
                 n_classes=1000, n_registers=16, n_style_tokens=32, codebook_size=64,
                 in_channels=4, img_size=32):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.n_registers = n_registers
        self.n_style_tokens = n_style_tokens
        self.in_channels = in_channels
        self.img_size = img_size

        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        n_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)

        self.register_tokens = nn.Parameter(torch.randn(1, n_registers, dim) * 0.02)

        # conditioning
        self.class_embed = nn.Embedding(n_classes, dim)
        self.alpha_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.style_codebook = nn.Embedding(codebook_size, dim)

        self.blocks = nn.ModuleList([DriftBlock(dim, n_heads) for _ in range(depth)])

        self.final_norm = RMSNorm(dim)
        self.final_linear = nn.Linear(dim, patch_size * patch_size * in_channels, bias=False)

        head_dim = dim // n_heads
        rope = precompute_rope_2d(self.grid_h, self.grid_w, head_dim)
        self.register_buffer("rope_freqs", rope)

    def forward(self, noise, class_label, alpha, style_indices=None):
        """
        Args:
            noise:         [B, C, H, W] Gaussian noise
            class_label:   [B] int class labels
            alpha:         [B] or scalar — CFG strength
            style_indices: [B, n_style_tokens] int indices into codebook, or None
        """
        B = noise.shape[0]

        x = self.patch_embed(noise)                          # [B, dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)                     # [B, n_patches, dim]
        x = x + self.pos_embed

        cond = self.class_embed(class_label)                  # [B, dim]

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor([alpha], device=noise.device, dtype=noise.dtype).expand(B)
        cond = cond + self.alpha_embed(alpha.unsqueeze(-1).float())

        if style_indices is not None:
            style = self.style_codebook(style_indices)        # [B, n_style_tokens, dim]
            cond = cond + style.sum(dim=1)                    # [B, dim]

        regs = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([regs, x], dim=1)                      # [B, n_reg + n_patches, dim]

        for block in self.blocks:
            x = checkpoint(block, x, cond, self.rope_freqs, self.n_registers, use_reentrant=False)

        x = x[:, self.n_registers:]                           # [B, n_patches, dim]
        x = self.final_norm(x)
        x = self.final_linear(x)                              # [B, n_patches, ps*ps*4]

        ps = self.patch_size
        C = self.in_channels
        x = x.reshape(B, self.grid_h, self.grid_w, ps, ps, C)
        x = x.permute(0, 5, 1, 3, 2, 4)                      # [B, C, grid_h, ps, grid_w, ps]
        x = x.reshape(B, C, self.img_size, self.img_size)

        return x

def generator_b2(**kwargs):
    """B/2 config: 768 dim, 12 depth, 12 heads, patch 2×2 — ~133M params."""
    return Generator(depth=12, dim=768, patch_size=2, n_heads=12, **kwargs)


def generator_l2(**kwargs):
    """L/2 config: 1024 dim, 24 depth, 16 heads, patch 2×2 — ~463M params."""
    return Generator(depth=24, dim=1024, patch_size=2, n_heads=16, **kwargs)


def generator_mnist(**kwargs):
    """Small config for MNIST: 256 dim, 6 depth, 4 heads, patch 4×4, 1ch — ~7M params."""
    defaults = dict(depth=6, dim=256, patch_size=4, n_heads=4, n_registers=4,
                    n_style_tokens=8, in_channels=1, img_size=32, n_classes=10)
    defaults.update(kwargs)
    return Generator(**defaults)