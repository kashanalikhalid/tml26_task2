"""Behavioral fingerprint features: compare suspect logits to target logits.

Behavioral signals catch every flavor of model stealing the assignment lists:
direct copies, function-preserving transforms (which preserve outputs), fine-tunes
from a target checkpoint, and distillation -- even though some of those leave the
weights looking ~random.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def forward_logits(
    model: torch.nn.Module,
    probe_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (N, num_classes) logits on CPU as float32."""
    outputs: list[torch.Tensor] = []
    for i in range(0, probe_x.size(0), batch_size):
        x = probe_x[i:i + batch_size].to(device, non_blocking=True)
        outputs.append(model(x).float().cpu())
    return torch.cat(outputs, dim=0)


def behavioral_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
) -> dict[str, float]:
    target_log_probs = F.log_softmax(target_logits, dim=1)
    suspect_log_probs = F.log_softmax(suspect_logits, dim=1)
    target_probs = target_log_probs.exp()
    suspect_probs = suspect_log_probs.exp()

    # Symmetric KL on the softmax distributions, averaged over the probe set.
    kl_ts = (target_probs * (target_log_probs - suspect_log_probs)).sum(dim=1)
    kl_st = (suspect_probs * (suspect_log_probs - target_log_probs)).sum(dim=1)
    sym_kl = float((kl_ts + kl_st).mean().item())

    # Jensen-Shannon divergence on the same distributions.
    mid = 0.5 * (target_probs + suspect_probs)
    mid_log = mid.clamp_min(1e-30).log()
    jsd = 0.5 * (
        (target_probs * (target_log_probs - mid_log)).sum(dim=1)
        + (suspect_probs * (suspect_log_probs - mid_log)).sum(dim=1)
    )
    jsd_mean = float(jsd.mean().item())

    # Top-k agreement.
    target_top1 = target_logits.argmax(dim=1)
    suspect_top1 = suspect_logits.argmax(dim=1)
    top1 = float((target_top1 == suspect_top1).float().mean().item())
    target_top5 = target_logits.topk(5, dim=1).indices
    top5 = float((target_top5 == suspect_top1.unsqueeze(1)).any(dim=1).float().mean().item())

    # Logit cosine / Pearson, per-sample then averaged.
    cos = float(F.cosine_similarity(target_logits, suspect_logits, dim=1).mean().item())
    tc = target_logits - target_logits.mean(dim=1, keepdim=True)
    sc = suspect_logits - suspect_logits.mean(dim=1, keepdim=True)
    pearson_per = (tc * sc).sum(dim=1) / (tc.norm(dim=1) * sc.norm(dim=1) + 1e-12)
    pearson = float(pearson_per.mean().item())

    # Cross-entropy of suspect logits against target's top-1 as a hard "soft label".
    # Low value = suspect predicts what target predicts, even on misclassified inputs.
    ce_to_target = float(F.cross_entropy(suspect_logits, target_top1, reduction="mean").item())

    return {
        "behavioral_sym_kl": sym_kl,
        "behavioral_jsd": jsd_mean,
        "behavioral_top1_agree": top1,
        "behavioral_top5_member": top5,
        "behavioral_logit_cosine": cos,
        "behavioral_logit_pearson": pearson,
        "behavioral_ce_to_target": ce_to_target,
    }
