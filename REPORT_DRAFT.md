# TML 2026 Assignment 2 — Report Draft

> Working draft for the ≤2-page ICLR-template CMS report. Replace the
> placeholders before submission.

Matriculation number: _<fill in>_
CMS team id: _<fill in>_

## Introduction

We are given white-box access to a CIFAR-style ResNet-18 target trained from a
fixed CIFAR-100 subset, plus 360 suspect checkpoints with the same architecture.
Our task is to assign each suspect a continuous stealing-confidence score that
ranks direct copies, function-preserving transforms, fine-tunes from the target,
and distilled / extracted students above models trained independently --
including independents trained on the same data distribution. The leaderboard
metric is TPR @ 5 % FPR over the hidden stolen/not-stolen labels.

## Approach

All 360 suspect checkpoints are the same size as the target's
`weights.safetensors` (44,929,864 bytes) and load with `strict=True` into the
target's architecture, so weight-space and behavioural signals are both
well-defined for every suspect. Our score combines two families of features:

* **Behavioural fingerprint** — we forward the CIFAR-100 test split (10 000
  images, normalized with the assignment's stated mean/std) through the target
  and each suspect, then compute the Jensen–Shannon divergence and symmetric
  KL between the softmax outputs, top-1 / top-5 agreement, cross-entropy of
  suspect logits against the target's top-1, and the cosine / Pearson
  correlation of the logits. These signals fire for every stealing flavour the
  assignment lists, including distillation where the weights look unrelated.
* **Weight-space similarity** — for every shared parameter we compute exact
  equality, full-vector cosine, backbone-only cosine, classifier-only cosine,
  and relative L2 distance. These dominate for direct copies and
  lightly-fine-tuned suspects but cannot detect distillation on their own.

Features are combined with a Borda-style rank-mean ensemble (each feature is
mapped to its rank in [0, 1] with the appropriate sign and the average rank is
emitted as the final score). Rank fusion is robust to feature-scale outliers
and keeps the score interpretable: the final number is the average percentile
across all signals.

## Key Results

_To be filled in after the leaderboard score is recorded:_

* Public leaderboard score (TPR @ 5 % FPR): _<value>_
* Top suspect ids that all signals agree are stolen: _<list>_
* Feature ablations: behavioural-only vs weight-only vs combined.

## Conclusion

Detecting stolen models matters because foundation-style training pipelines
remain expensive and opaque, and the cheapest way for an adversary to deploy a
competitor product is to fine-tune or distil a leaked checkpoint. Our results
show that simple, low-cost similarity probes are enough to flag a wide variety
of theft modes from public artifacts alone, even when the suspect has been
modified post-hoc. For model owners this enables an evidence-based "did
someone copy us?" check before pursuing more expensive forensic work; for the
broader ecosystem it signals that releasing weights does not give attackers a
free pass on attribution.
