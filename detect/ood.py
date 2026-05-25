"""Out-of-distribution probe features.

Idea: target has model-specific weird responses to inputs far from its
training distribution (random noise, CIFAR-10 images). Stolen models
inherit these specific responses. Independents trained on the same
in-distribution data have different OOD response patterns -- different
random init, different decision boundary at the OOD edge.

Probes:
  - random_noise: N samples of Gaussian noise in normalized input space.
  - cifar10:      M CIFAR-10 test images (different distribution but similar low-level stats).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms


def build_gaussian_noise_probe(n: int = 2000, seed: int = 0) -> torch.Tensor:
    """Random Gaussian noise probe set, std=1 in normalized input space."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, 3, 32, 32), generator=g)


def build_cifar10_probe(cifar_root: Path | str, limit: int | None = 2000, download: bool = True) -> torch.Tensor:
    """CIFAR-10 test images normalized with CIFAR-100 stats (intentional mismatch)."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    ds = datasets.CIFAR10(root=str(cifar_root), train=False, download=download, transform=tfm)
    n = len(ds) if limit is None else min(int(limit), len(ds))
    return torch.stack([ds[i][0] for i in range(n)])


def ood_features(target_logits: torch.Tensor, suspect_logits: torch.Tensor, prefix: str) -> dict[str, float]:
    """Compare model outputs on OOD inputs.

    On OOD inputs target's argmax is a particular pattern of "spurious" labels.
    Stolen models replicate it; independents don't.
    """
    target_probs = F.softmax(target_logits, dim=1)
    suspect_probs = F.softmax(suspect_logits, dim=1)

    t_top1 = target_logits.argmax(1)
    s_top1 = suspect_logits.argmax(1)
    top1 = float((t_top1 == s_top1).float().mean().item())

    eps = 1e-30
    t_log = (target_probs + eps).log()
    s_log = (suspect_probs + eps).log()
    kl_ts = (target_probs * (t_log - s_log)).sum(1)
    kl_st = (suspect_probs * (s_log - t_log)).sum(1)
    sym_kl = float(((kl_ts + kl_st) / 2.0).mean().item())

    mid = 0.5 * (target_probs + suspect_probs)
    mid_log = mid.clamp_min(eps).log()
    jsd = 0.5 * (
        (target_probs * (t_log - mid_log)).sum(1)
        + (suspect_probs * (s_log - mid_log)).sum(1)
    )
    jsd_mean = float(jsd.mean().item())

    # Per-sample logit Pearson on raw logits (captures shape of output, not just argmax)
    tc = target_logits - target_logits.mean(dim=1, keepdim=True)
    sc = suspect_logits - suspect_logits.mean(dim=1, keepdim=True)
    pearson_per = (tc * sc).sum(dim=1) / (tc.norm(dim=1) * sc.norm(dim=1) + 1e-12)
    logit_pearson = float(pearson_per.mean().item())

    return {
        f"ood_{prefix}_top1_agree": top1,
        f"ood_{prefix}_sym_kl": sym_kl,
        f"ood_{prefix}_jsd": jsd_mean,
        f"ood_{prefix}_logit_pearson": logit_pearson,
    }
