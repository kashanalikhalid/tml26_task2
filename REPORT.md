# Stolen Model Detection via Soft LiRA and Conferrable Adversarial Examples

**Matriculation number:** [FILL IN]
**CMS team id:** team_LXXXII

## Introduction

The task asks us to assign every one of 360 suspect ResNet-18 checkpoints a
continuous confidence score reflecting how likely the model is to have been
copied from, derived from, or distilled from a given target ResNet-18 trained
on a subset of CIFAR-100. The challenge is that *stolen* covers a wide
spectrum, from verbatim weight copies to students trained from scratch on the
target's outputs; meanwhile *not stolen* includes models trained independently
on the same data distribution. Output-similarity alone is therefore
insufficient because two independently trained classifiers on the same
distribution can look behaviourally similar on the test set. Submissions are
evaluated with TPR at 5% FPR.

## Approach

Our pipeline combines two orthogonal signals into one continuous score:

```
score = z(soft_lira_mean_z) + 1.5 · mean_v z(confer_match_rate_v)
```

**Shadow models.** We trained 64 ResNet-18 shadow models with the *exact*
target training recipe (biased random crop, the listed train_main_idx subset,
cosine-annealed SGD with lr=0.1, momentum=0.9, weight decay 5e-4, 40 epochs)
but different random seeds. Each shadow produces per-sample top-1 prediction,
top-1 confidence, and CE loss on the CIFAR-100 test, member, and non-member
splits. Shadows give us a distribution of "what an honest classifier trained
on the same data would do" — the null for both signals below.

**Soft LiRA.** The natural LiRA quantity *target_top1 = suspect_top1 on a
vulnerability sample* is binary, so the strongest suspects all tie at the
ceiling and the ranking among them is decided by arbitrary tie-breaking. We
replace the binary match with the *continuous* probability mass the suspect
places on the target's wrong-prediction class, then z-score it against the
same quantity from the 64 shadows. A vulnerability sample is one where the
target predicts wrong but ≥85% of shadows predict right. The resulting
per-suspect z-score has a wide continuous spread with no saturation, which
resolves the tie and cleanly separates derived models from the cluster.

**Conferrable adversarial examples.** Building on Lukas et al. (ICLR 2021),
we generate inputs `x'` via joint PGD that *simultaneously* fool the target
and keep at least 50–95% of the shadows correct (the conferrability
threshold). For each suspect we record the fraction of these `x'` on which it
predicts the same wrong class as the target. Stolen models inherit the
target's adversarial vulnerabilities; independents do not. We run 10 variants
(`v1..v10`) of the procedure spanning different PGD step counts (30–100),
ℓ∞ budgets (4/255–16/255), shadow-loss weights (0.3–3.0) and conferrability
thresholds (0.4–0.95). Averaging across variants reduces per-variant noise.

**Final fusion.** Both feature columns are z-standardized over the 360 IDs and
combined with weight 1.5 on the conferrable axis. We selected the weight by an
ablation over `{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}`: 1.5 was the
point at which a borderline false-positive suspect was displaced by a
true-positive one; larger weights gave no further gain. The two signals are
complementary — soft LiRA captures distilled students whose outputs still
echo the target's mistakes, while conferrable AEs capture derived models that
inherit the target's decision-boundary geometry.

## What we tried that did not work

Every method that compares a suspect to the target *directly* converged to the
same cluster of ~30 functionally identical models — and a large fraction of
that cluster are independent models trained on the same distribution, not
derived ones. These approaches therefore could not separate stolen from
genuine, which is what motivated the two shadow-relative signals above:

* **Behavioural ensembles** (JSD, symmetric KL, top-1/top-5 agreement, logit
  cosine/Pearson, cross-entropy to target): rank-mean, median, and min-fusion
  variants all return the same indistinguishable cluster.
* **Hard (binary) LiRA**: the binary "suspect matches target's wrong class"
  score saturates — many suspects tie at the maximum — so the ranking among
  them is arbitrary. The continuous soft-LiRA score is the fix.
* **Weight-space features**: per-layer Hungarian permutation matching of conv
  filters, and BatchNorm running-mean/var correlation. BN correlation flags
  near-verbatim copies but misses fine-tuned ones whose BN statistics drift.
* **Representation similarity**: linear CKA on the penultimate layer and on
  every ResNet stage. Low-level conv features are determined by the CIFAR-100
  data distribution rather than by the teacher, so they do not separate
  derived from independent models.
* **Loss-landscape geometry**: a linear-interpolation loss barrier between
  target and suspect weights (a lightweight Git-Re-Basin proxy). It isolates a
  "same loss basin" group, but that group still mixes derived and independent
  models.
* **Dataset-inference / membership signals**: per-sample loss correlation on
  the target's training subset, member-vs-non-member loss gap, and a MinGD
  decision-boundary distance test. The membership signal is shared by any
  model trained on the same distribution, so it does not isolate derivation.
* **Distillation-specific detectors**: per-class confusion-matrix similarity,
  NAD binary activation-pattern Hamming distance, input-gradient cosine, and
  DeepInversion-synthesized inputs scored with Aligned Cosine Similarity. None
  improved on the soft-LiRA + conferrable-AE combination.

The common thread: direct suspect-vs-target comparison cannot tell a derived
model from an independent one trained on the same data. The two signals that
worked both measure the suspect against a *distribution of honest shadow
models*, which is what makes "behaves like the target but unlike an
independently trained peer" detectable.

## Conclusion

Detection of stolen models matters because model training, especially at
foundation-model scale, represents a large capital and energy investment, and
because models embed proprietary data and design choices that can be
extracted at orders-of-magnitude lower cost than re-training. Our detector
shows that even with no provenance information about the suspects, two
relatively cheap signals — one statistical (LiRA on confidence) and one
adversarial (conferrable AEs) — are enough to identify a clear majority of
derived models, including distilled students with drifted output
distributions. This has practical
consequences for IP enforcement, audit, and licensing of released model
weights: model owners can detect distilled copies without trusting any
labels supplied by the model holder, simply by running a few hundred
suspect-side forward passes against a precomputed shadow set and a
precomputed set of conferrable inputs.
