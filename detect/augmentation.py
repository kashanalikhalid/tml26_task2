"""Augmentation-sensitivity probe.

The assignment PDF tells us target was trained with biased random crop:
  - pad 4 reflect, then 32x32 crop biased toward (bias_x=0.5, bias_y=-0.25)
  - jitter=0.25 around the bias point

This is target's specific training regime. Stolen and fine-tuned-from-target
models inherit sensitivity to crops from THIS region of the image. Independents
trained with a different crop policy (e.g. uniform random crop) handle the
biased crops differently.

We probe by generating biased-crop variants of each test image and measuring
the *delta* in suspect outputs vs target outputs between center-crop and
target-biased-crop. Stolen models have similar deltas.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reflect_pad(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    return F.pad(x, (pad, pad, pad, pad), mode="reflect")


def biased_crop(x_padded: torch.Tensor, bias_x: float = 0.5, bias_y: float = -0.25,
                jitter: float = 0.0, crop_size: int = 32) -> torch.Tensor:
    """Take a single deterministic biased crop. jitter=0 → deterministic centered-on-bias crop."""
    h_pad, w_pad = x_padded.shape[-2:]
    # bias=0 is image center (15.5 + pad). bias=+1 is far right edge. bias=-1 is far left.
    # We map bias ∈ [-1, +1] linearly to crop offset.
    center_x = w_pad / 2
    center_y = h_pad / 2
    # Maximum offset from center is (pad in either direction)
    pad_w = (w_pad - crop_size) // 2  # = 4 with crop=32, w_pad=40
    pad_h = (h_pad - crop_size) // 2
    off_x = int(round(center_x - crop_size / 2 + bias_x * pad_w))
    off_y = int(round(center_y - crop_size / 2 + bias_y * pad_h))
    off_x = max(0, min(w_pad - crop_size, off_x))
    off_y = max(0, min(h_pad - crop_size, off_y))
    return x_padded[..., off_y:off_y + crop_size, off_x:off_x + crop_size]


def make_biased_probe(probe_x: torch.Tensor, bias_x: float = 0.5, bias_y: float = -0.25) -> torch.Tensor:
    padded = reflect_pad(probe_x, pad=4)
    return biased_crop(padded, bias_x=bias_x, bias_y=bias_y)


def augmentation_features(
    target_center_logits: torch.Tensor,
    target_biased_logits: torch.Tensor,
    suspect_center_logits: torch.Tensor,
    suspect_biased_logits: torch.Tensor,
) -> dict[str, float]:
    """Compare how target and suspect respond to target's specific biased crop.

    Stolen models inherit target's specific (biased-crop) training regime, so
    their per-sample delta between biased and center crops is correlated with
    target's. Independents have uncorrelated deltas.
    """
    t_delta = target_biased_logits - target_center_logits
    s_delta = suspect_biased_logits - suspect_center_logits

    # Per-sample cosine of delta vectors
    cos = F.cosine_similarity(t_delta, s_delta, dim=1)
    delta_cos = float(cos.mean().item())

    # Pearson on flattened deltas
    tf = t_delta.flatten()
    sf = s_delta.flatten()
    tc = tf - tf.mean()
    sc = sf - sf.mean()
    pearson = float((tc * sc).sum() / (tc.norm() * sc.norm() + 1e-12))

    # Top-1 agreement on biased crops
    t_top1 = target_biased_logits.argmax(1)
    s_top1 = suspect_biased_logits.argmax(1)
    top1 = float((t_top1 == s_top1).float().mean().item())

    # JSD on biased-crop softmax
    eps = 1e-30
    tp = F.softmax(target_biased_logits, dim=1)
    sp = F.softmax(suspect_biased_logits, dim=1)
    mid = 0.5 * (tp + sp)
    ml = mid.clamp_min(eps).log()
    jsd = 0.5 * (
        (tp * ((tp + eps).log() - ml)).sum(1)
        + (sp * ((sp + eps).log() - ml)).sum(1)
    )
    jsd_mean = float(jsd.mean().item())

    return {
        "aug_delta_cos_mean": delta_cos,
        "aug_delta_pearson": pearson,
        "aug_biased_top1_agree": top1,
        "aug_biased_jsd": jsd_mean,
    }
