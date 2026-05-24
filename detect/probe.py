"""Probe-set construction for behavioral fingerprinting.

We use three probe sets:
  - test:       CIFAR-100 test split. None are training data for the target,
                so behavioural similarity here isolates stealing signal from
                data-overlap noise.
  - member:     The subset of CIFAR-100 *train* the target was trained on
                (given by train_main_idx.json). Stolen models inherit the
                target's specific memorization pattern on these samples.
  - nonmember:  CIFAR-100 train samples the target did NOT see. Independents
                trained on the full train split have low loss on these;
                stolen models inherit the target's higher loss pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_probe_set(
    cifar_root: Path | str,
    limit: int | None = None,
    train: bool = False,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(
        root=str(cifar_root),
        train=train,
        download=download,
        transform=_transform(),
    )
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    xs = torch.stack([dataset[i][0] for i in range(n)])
    ys = torch.tensor([dataset[i][1] for i in range(n)], dtype=torch.long)
    return xs, ys


def build_member_probes(
    cifar_root: Path | str,
    train_main_idx_path: Path | str,
    n_members: int | None = None,
    n_nonmembers: int | None = None,
    download: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Split CIFAR-100 train into target's training members and non-members.

    Returns ((member_x, member_y), (nonmember_x, nonmember_y)) tensors.
    Both subsets are downsampled to `n_members` / `n_nonmembers` if given,
    using deterministic ordering so the probe set is identical across runs.
    """
    with open(train_main_idx_path, "r") as fh:
        member_idx = sorted(set(int(i) for i in json.load(fh)))
    member_set = set(member_idx)

    dataset = datasets.CIFAR100(
        root=str(cifar_root),
        train=True,
        download=download,
        transform=_transform(),
    )
    nonmember_idx = [i for i in range(len(dataset)) if i not in member_set]

    if n_members is not None:
        member_idx = member_idx[: int(n_members)]
    if n_nonmembers is not None:
        nonmember_idx = nonmember_idx[: int(n_nonmembers)]

    def stack(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([dataset[i][0] for i in ids])
        ys = torch.tensor([dataset[i][1] for i in ids], dtype=torch.long)
        return xs, ys

    return stack(member_idx), stack(nonmember_idx)
