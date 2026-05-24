"""Behavioral fingerprint features: compare suspect logits to target logits.

Three feature groups:
* output-only features on a test-set probe (KL/JSD, top-k agreement, logit
  cosine/Pearson, cross-entropy of suspect logits against target's top-1);
* label-aware features (per-sample loss correlation with target, agreement on
  target's misclassifications);
* membership-aware features (loss correlation and gap between target's
  training members and non-members — captures memorization the target
  specifically learned that independents would not share).
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

    kl_ts = (target_probs * (target_log_probs - suspect_log_probs)).sum(dim=1)
    kl_st = (suspect_probs * (suspect_log_probs - target_log_probs)).sum(dim=1)
    sym_kl = float((kl_ts + kl_st).mean().item())

    mid = 0.5 * (target_probs + suspect_probs)
    mid_log = mid.clamp_min(1e-30).log()
    jsd = 0.5 * (
        (target_probs * (target_log_probs - mid_log)).sum(dim=1)
        + (suspect_probs * (suspect_log_probs - mid_log)).sum(dim=1)
    )
    jsd_mean = float(jsd.mean().item())

    target_top1 = target_logits.argmax(dim=1)
    suspect_top1 = suspect_logits.argmax(dim=1)
    top1 = float((target_top1 == suspect_top1).float().mean().item())
    target_top5 = target_logits.topk(5, dim=1).indices
    top5 = float((target_top5 == suspect_top1.unsqueeze(1)).any(dim=1).float().mean().item())

    cos = float(F.cosine_similarity(target_logits, suspect_logits, dim=1).mean().item())
    tc = target_logits - target_logits.mean(dim=1, keepdim=True)
    sc = suspect_logits - suspect_logits.mean(dim=1, keepdim=True)
    pearson_per = (tc * sc).sum(dim=1) / (tc.norm(dim=1) * sc.norm(dim=1) + 1e-12)
    pearson = float(pearson_per.mean().item())

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


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denom = ac.norm() * bc.norm() + 1e-12
    return float((ac * bc).sum() / denom)


def label_aware_features(
    target_logits: torch.Tensor,
    suspect_logits: torch.Tensor,
    y: torch.Tensor,
    prefix: str = "test",
) -> dict[str, float]:
    """Features that need ground-truth labels.

    The two most-discriminating signals:
    * loss_corr: Pearson correlation of per-sample CE losses. Stolen models
      share the target's exact per-image loss landscape; independents have
      different memorization noise.
    * wrong_agree: among samples the target misclassifies, fraction where
      suspect predicts the same wrong class. Target's mistakes are largely
      seed-specific; only stolen models inherit them.
    """
    target_loss = F.cross_entropy(target_logits, y, reduction="none")
    suspect_loss = F.cross_entropy(suspect_logits, y, reduction="none")
    loss_corr = _pearson(target_loss, suspect_loss)
    loss_mean = float(suspect_loss.mean().item())
    loss_mean_target = float(target_loss.mean().item())
    loss_gap_abs = float(abs(loss_mean - loss_mean_target))

    target_top1 = target_logits.argmax(dim=1)
    suspect_top1 = suspect_logits.argmax(dim=1)
    target_wrong = target_top1 != y
    if target_wrong.any():
        wrong_agree = float((suspect_top1[target_wrong] == target_top1[target_wrong]).float().mean().item())
    else:
        wrong_agree = 0.0
    target_correct = ~target_wrong
    if target_correct.any():
        correct_agree = float((suspect_top1[target_correct] == target_top1[target_correct]).float().mean().item())
    else:
        correct_agree = 0.0

    return {
        f"behavioral_loss_corr_{prefix}": loss_corr,
        f"behavioral_loss_mean_{prefix}": loss_mean,
        f"behavioral_loss_gap_abs_{prefix}": loss_gap_abs,
        f"behavioral_wrong_agree_{prefix}": wrong_agree,
        f"behavioral_correct_agree_{prefix}": correct_agree,
    }


def member_gap_features(
    target_member_logits: torch.Tensor,
    target_nonmember_logits: torch.Tensor,
    suspect_member_logits: torch.Tensor,
    suspect_nonmember_logits: torch.Tensor,
    y_member: torch.Tensor,
    y_nonmember: torch.Tensor,
) -> dict[str, float]:
    """Membership-gap features.

    The target memorizes its training members (low loss) and generalizes to
    non-members (higher loss). Stolen models inherit this gap because they
    are derived from (or distilled from) the target. Independents trained on
    the full CIFAR-100 train split have low loss everywhere, so their member
    -minus-nonmember gap is much smaller than the target's.
    """
    t_member_loss = F.cross_entropy(target_member_logits, y_member, reduction="none")
    t_nm_loss = F.cross_entropy(target_nonmember_logits, y_nonmember, reduction="none")
    s_member_loss = F.cross_entropy(suspect_member_logits, y_member, reduction="none")
    s_nm_loss = F.cross_entropy(suspect_nonmember_logits, y_nonmember, reduction="none")

    t_gap = float(t_nm_loss.mean().item() - t_member_loss.mean().item())
    s_gap = float(s_nm_loss.mean().item() - s_member_loss.mean().item())
    gap_diff_abs = float(abs(s_gap - t_gap))
    gap_ratio = float(s_gap / (t_gap + 1e-12))

    # How well does suspect mimic target's confidence on members?
    member_loss_corr = _pearson(t_member_loss, s_member_loss)
    nm_loss_corr = _pearson(t_nm_loss, s_nm_loss)

    return {
        "behavioral_member_gap_suspect": s_gap,
        "behavioral_member_gap_diff_abs": gap_diff_abs,
        "behavioral_member_gap_ratio": gap_ratio,
        "behavioral_member_loss_corr": member_loss_corr,
        "behavioral_nonmember_loss_corr": nm_loss_corr,
    }
