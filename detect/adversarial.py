"""Adversarial-transfer fingerprint feature (ModelDiff-style).

Idea: craft adversarial examples against the *target* with PGD. Stolen
models (direct copy, function-preserving, fine-tune, distillation) inherit
the target's decision boundary, so target's adversaries transfer to them
at a much higher rate than to independents trained on the same data.

We use a small ε untargeted attack — large enough to flip target's
prediction reliably, small enough that an *independent* model on the same
distribution mostly resists.

We pre-compute the adversaries once against the target, then for each
suspect just measure how often its prediction agrees with target's
adversarial prediction on the same x_adv.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    preds = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i:i + batch_size].to(device, non_blocking=True)
        logits = model(chunk).float().cpu()
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def craft_pgd_adversaries(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    batch_size: int = 256,
    device: torch.device | str = "cuda",
    rand_init: bool = True,
) -> torch.Tensor:
    """PGD-L_inf attack on `model` against true labels `y`.

    Returns adversarial x of the same shape, on CPU. Epsilon/alpha are in
    the *normalized* input space (since our probe data is already mean/std
    normalized) -- 8/255 is a CIFAR-standard budget.
    """
    model = model.eval()
    out_chunks: list[torch.Tensor] = []
    for i in range(0, x.size(0), batch_size):
        x_b = x[i:i + batch_size].to(device)
        y_b = y[i:i + batch_size].to(device)
        if rand_init:
            delta = torch.empty_like(x_b).uniform_(-epsilon, epsilon)
        else:
            delta = torch.zeros_like(x_b)
        delta = delta.detach().requires_grad_(True)

        for _ in range(steps):
            logits = model(x_b + delta)
            loss = F.cross_entropy(logits, y_b)
            grad = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                delta = (delta + alpha * grad.sign()).clamp_(-epsilon, epsilon)
            delta = delta.detach().requires_grad_(True)

        out_chunks.append((x_b + delta).detach().cpu())
    return torch.cat(out_chunks, dim=0)


def adversarial_transfer_features(
    target_adv_preds: torch.Tensor,  # (N,) target's argmax on adv inputs
    target_clean_preds: torch.Tensor,  # (N,) target's argmax on clean inputs
    suspect_adv_preds: torch.Tensor,  # (N,)
    suspect_clean_preds: torch.Tensor,  # (N,)
    y: torch.Tensor,
) -> dict[str, float]:
    """Compute transfer-rate-style features from precomputed predictions.

    All inputs are (N,) tensors of class indices (int64) on CPU.
    """
    # Where target was successfully fooled (its clean and adv predictions differ)
    target_fooled = target_clean_preds != target_adv_preds
    if target_fooled.any():
        # Transfer rate: of target's successful adversarial examples, fraction
        # where suspect also predicts the *same wrong class* target predicts.
        adv_class_match = (suspect_adv_preds[target_fooled] == target_adv_preds[target_fooled]).float().mean()
        # Weaker: suspect is also fooled away from clean (without matching the same wrong class)
        suspect_clean_on_fooled = suspect_clean_preds[target_fooled]
        suspect_fooled = (suspect_adv_preds[target_fooled] != suspect_clean_on_fooled).float().mean()
    else:
        adv_class_match = torch.tensor(0.0)
        suspect_fooled = torch.tensor(0.0)

    # Robustness on adv inputs: fraction where suspect *still* predicts the true label.
    # Independents may "see through" adversaries crafted against target.
    suspect_robust = (suspect_adv_preds == y).float().mean()

    # Cross-clean agreement: just to provide a normalisation reference
    clean_agree = (suspect_clean_preds == target_clean_preds).float().mean()

    return {
        "adv_transfer_class_match": float(adv_class_match.item()),
        "adv_transfer_fooled_any": float(suspect_fooled.item()),
        "adv_suspect_robust_true": float(suspect_robust.item()),
        "adv_clean_agree": float(clean_agree.item()),
    }
