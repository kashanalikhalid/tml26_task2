"""Forward pass utility used by experiments."""
from __future__ import annotations

import torch


@torch.no_grad()
def forward_logits(
    model: torch.nn.Module,
    probe_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for i in range(0, probe_x.size(0), batch_size):
        x = probe_x[i:i + batch_size].to(device, non_blocking=True)
        outputs.append(model(x).float().cpu())
    return torch.cat(outputs, dim=0)
