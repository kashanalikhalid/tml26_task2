"""Probe-set construction for behavioral fingerprinting.

The probe set is a fixed tensor `(N, 3, 32, 32)` of normalized CIFAR-100 images.
We default to CIFAR-100 *test* because none of those images were in the target's
training set, so behavioural similarity on them isolates model-stealing signal
from data-overlap noise.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_probe_set(
    cifar_root: Path | str,
    limit: int | None = None,
    train: bool = False,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    dataset = datasets.CIFAR100(
        root=str(cifar_root),
        train=train,
        download=download,
        transform=transform,
    )
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    xs = torch.stack([dataset[i][0] for i in range(n)])
    ys = torch.tensor([dataset[i][1] for i in range(n)], dtype=torch.long)
    return xs, ys
