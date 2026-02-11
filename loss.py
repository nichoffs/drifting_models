"""Drifting field computation and loss — core of "Generative Modeling via Drifting"."""

import torch
import torch.nn.functional as F


def compute_V(x, y_pos, y_neg, tau, mask_self=True):
    """Compute drifting field V. Algorithm 2 from the paper.

    Args:
        x:      [N, D] generated samples (or features thereof)
        y_pos:  [N_pos, D] positive samples (data)
        y_neg:  [N_neg, D] negative samples (generated, often same as x)
        tau:    temperature scalar
        mask_self: if True, mask diagonal of dist_neg (when y_neg is x)

    Returns: V [N, D] — the drifting field
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    dist_pos = torch.cdist(x, y_pos, p=2)          # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg, p=2)          # [N, N_neg]

    # mask self-interactions (when y_neg is x, diagonal = self)
    if mask_self:
        dist_neg = dist_neg.clone()
        n = min(N, N_neg)
        dist_neg[range(n), range(n)] += 1e6

    # logits for softmax
    logit_pos = -dist_pos / tau                     # [N, N_pos]
    logit_neg = -dist_neg / tau                     # [N, N_neg]
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # double softmax normalization — geometric mean preserves anti-symmetry
    A_row = F.softmax(logit, dim=1)                 # normalize over y-samples
    A_col = F.softmax(logit, dim=0)                 # normalize over x-samples
    A = torch.sqrt(A_row * A_col)                   # [N, N_pos + N_neg]

    # split back into pos/neg
    A_pos = A[:, :N_pos]                            # [N, N_pos]
    A_neg = A[:, N_pos:]                            # [N, N_neg]

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]

    V = W_pos @ y_pos - W_neg @ y_neg               # [N, D]

    return V


def drifting_loss(x_feat, y_pos_feat, y_neg_feat, taus=(0.02, 0.05, 0.2), mask_self=True):
    """Full drifting loss with feature/drift normalization and multi-temperature.

    Args:
        x_feat:     [N, D] features of generated samples
        y_pos_feat: [N_pos, D] features of positive (data) samples
        y_neg_feat: [N_neg, D] features of negative samples
        taus:       tuple of temperature values
        mask_self:  mask self-interactions in compute_V

    Returns: scalar loss
    """
    D = x_feat.shape[1]

    # normalize so avg pairwise distance ≈ √D
    all_y = torch.cat([y_pos_feat, y_neg_feat], dim=0)
    pairwise_dist = torch.cdist(x_feat, all_y, p=2)    # [N, N_pos + N_neg]
    S = pairwise_dist.mean() / (D ** 0.5)
    S = S.detach().clamp(min=1e-6)

    x_norm = x_feat / S
    y_pos_norm = y_pos_feat / S
    y_neg_norm = y_neg_feat / S

    with torch.no_grad():
        V_total = torch.zeros_like(x_norm)
        raw_drift_norms = []
        for tau in taus:
            V_tau = compute_V(x_norm, y_pos_norm, y_neg_norm, tau, mask_self)
            raw_drift_norms.append(V_tau.pow(2).mean().sqrt())
            lam = (V_tau.pow(2).mean() / D).sqrt()
            V_total = V_total + V_tau / (lam + 1e-6)
        target = x_norm + V_total

    loss = F.mse_loss(x_norm, target)

    avg_raw_drift = sum(d.item() for d in raw_drift_norms) / len(raw_drift_norms)
    return loss, avg_raw_drift