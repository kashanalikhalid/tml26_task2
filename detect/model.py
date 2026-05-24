"""Architecture and checkpoint loading for the TML 2026 Task 2 target / suspects.

All 360 suspect models and the target share the same architecture: a CIFAR-style
ResNet-18 (3x3 stride-1 stem, no maxpool, 100-way classifier). Every checkpoint
file is exactly 44,929,864 bytes, so `state_dict` loads with strict=True.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from torchvision.models import resnet18

NUM_CLASSES = 100
EXPECTED_CHECKPOINT_BYTES = 44_929_864


def make_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_state_dict(path: Path | str) -> dict[str, torch.Tensor]:
    return load_safetensors(str(path), device="cpu")


def load_model(
    path: Path | str,
    device: torch.device | str = "cpu",
    state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    sd = state_dict if state_dict is not None else load_state_dict(path)
    model = make_model()
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model
