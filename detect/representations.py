"""Penultimate-layer representation similarity (CKA-style features).

Distilled and fine-tuned stolen models often share the target's hidden
representation even when their output logits have drifted -- the avgpool
output (just before the 100-way classifier) is the most direct probe of
that shared representation. Independents trained on the same task can
have similar representations at a coarse level, but their per-neuron
activation patterns differ.

We use linear Centered Kernel Alignment (CKA), which is invariant to
orthogonal transformations -- a critical property since stolen-but-
permuted models would otherwise look unrelated.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def collect_penultimate(
    model: torch.nn.Module,
    probe_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run probe_x through `model` and capture the avgpool output (N, 512)."""
    out_chunks: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        out_chunks.append(out.detach().flatten(1).float().cpu())

    h = model.avgpool.register_forward_hook(hook)
    try:
        for i in range(0, probe_x.size(0), batch_size):
            chunk = probe_x[i:i + batch_size].to(device, non_blocking=True)
            model(chunk)
    finally:
        h.remove()
    return torch.cat(out_chunks, dim=0)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment between feature matrices X, Y.

    Both (N, d_*). Returns scalar in [0, 1]. Invariant to orthogonal
    transformations of either feature space (so permutations and rotations
    of suspect's penultimate features don't matter).
    """
    if X.numel() == 0 or Y.numel() == 0:
        return 0.0
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    # HSIC(X, Y) numerator / sqrt(HSIC(X, X) * HSIC(Y, Y))
    # In linear case: ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    XY = Xc.T @ Yc
    XX = Xc.T @ Xc
    YY = Yc.T @ Yc
    num = (XY ** 2).sum()
    den = XX.norm() * YY.norm() + 1e-12
    return float(num / den)


def representation_features(
    target_feat: torch.Tensor,
    suspect_feat: torch.Tensor,
) -> dict[str, float]:
    cka = linear_cka(target_feat, suspect_feat)
    # Per-sample cosine of penultimate vectors (sensitive to permutations)
    Tn = target_feat / (target_feat.norm(dim=1, keepdim=True) + 1e-12)
    Sn = suspect_feat / (suspect_feat.norm(dim=1, keepdim=True) + 1e-12)
    cos_per_sample = (Tn * Sn).sum(dim=1)
    return {
        "repr_cka_linear": cka,
        "repr_penult_cos_mean": float(cos_per_sample.mean().item()),
        "repr_penult_cos_std": float(cos_per_sample.std().item()),
    }
