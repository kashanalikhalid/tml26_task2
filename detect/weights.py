"""Weight-space features: compare suspect state_dict to the target's.

These signals are strong for direct copies and lightly fine-tuned suspects.
They are weak / uninformative for permutation-equivalent transforms and for
distillation (where the suspect's weights look unrelated even though its
behavior is identical). Always pair with behavioral features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _flat(sd: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([sd[k].detach().to(torch.float32).reshape(-1) for k in keys])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def weight_features(
    target_sd: dict[str, torch.Tensor],
    suspect_sd: dict[str, torch.Tensor],
) -> dict[str, float]:
    target_keys = sorted(target_sd.keys())
    suspect_keys = sorted(suspect_sd.keys())
    if target_keys != suspect_keys:
        # Architectures diverged; behavioral features carry the load instead.
        return {
            "weight_exact_tensor_frac": 0.0,
            "weight_cosine_full": 0.0,
            "weight_cosine_backbone": 0.0,
            "weight_cosine_head": 0.0,
            "weight_l2_relative": 1.0,
            "weight_max_abs_diff": float("inf"),
            "weight_keys_match": 0.0,
        }

    exact = 0
    max_abs_diff = 0.0
    for k in target_keys:
        t = target_sd[k]
        s = suspect_sd[k]
        if t.shape != s.shape:
            return {
                "weight_exact_tensor_frac": 0.0,
                "weight_cosine_full": 0.0,
                "weight_cosine_backbone": 0.0,
                "weight_cosine_head": 0.0,
                "weight_l2_relative": 1.0,
                "weight_max_abs_diff": float("inf"),
                "weight_keys_match": 0.5,
            }
        if torch.equal(t, s):
            exact += 1
        diff = (t.detach().to(torch.float32) - s.detach().to(torch.float32)).abs().max().item()
        if diff > max_abs_diff:
            max_abs_diff = float(diff)

    target_flat = _flat(target_sd, target_keys)
    suspect_flat = _flat(suspect_sd, target_keys)
    cos_full = _cosine(target_flat, suspect_flat)
    l2_rel = float((target_flat - suspect_flat).norm() / (target_flat.norm() + 1e-12))

    backbone_keys = [k for k in target_keys if not k.startswith("fc.")]
    head_keys = [k for k in target_keys if k.startswith("fc.")]
    cos_backbone = _cosine(_flat(target_sd, backbone_keys), _flat(suspect_sd, backbone_keys))
    cos_head = _cosine(_flat(target_sd, head_keys), _flat(suspect_sd, head_keys)) if head_keys else 0.0

    return {
        "weight_exact_tensor_frac": exact / max(len(target_keys), 1),
        "weight_cosine_full": cos_full,
        "weight_cosine_backbone": cos_backbone,
        "weight_cosine_head": cos_head,
        "weight_l2_relative": l2_rel,
        "weight_max_abs_diff": max_abs_diff,
        "weight_keys_match": 1.0,
    }
