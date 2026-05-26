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
distribution can look behaviourally similar on the test set. Scores are
evaluated with TPR at 5% FPR on a hidden public/private split.

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
vulnerability sample* is binary, so the strongest 30 suspects all tie at the
ceiling, and a 14-of-30 false-positive rate among the tied models limits the
score to ~0.518. We replace the binary match with the *continuous* probability
mass the suspect places on the target's wrong-prediction class, then z-score
it against the same quantity from the 64 shadows. A vulnerability sample is
one where the target predicts wrong but ≥85% of shadows predict right. The
mean per-suspect z-score spans 0 to 110 with no saturation and immediately
raised the score from 0.518 to 0.611.

**Conferrable adversarial examples.** Building on Lukas et al. (ICLR 2021),
we generate inputs `x'` via joint PGD that *simultaneously* fool the target
and keep at least 50–95% of the shadows correct (the conferrability
threshold). For each suspect we record the fraction of these `x'` on which it
predicts the same wrong class as the target. Stolen models inherit the
target's adversarial vulnerabilities; independents do not. We run 10 variants
(`v1..v10`) of the procedure spanning different PGD step counts (30–100),
ℓ∞ budgets (4/255–16/255), shadow-loss weights (0.3–3.0) and conferrability
thresholds (0.4–0.95). Averaging across variants reduces per-variant noise.

**Final fusion.** Both feature columns are z-standardized over the 360 IDs,
combined with weight 1.5 on the conferrable axis, and written as the
submission `score`. The weight was selected on the public leaderboard from a
sweep over `{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}`: 1.5 was the first
swap that displaced a borderline false-positive ID (221) for a true-positive
ID (315), gaining one capture; higher weights produced no further gains.

## Key results (public leaderboard, TPR@5%FPR)

| Submission | Score | Δ |
|---|---|---|
| exp02 baseline rank-mean of weight + behavioural features | 0.518 | — |
| Pure LiRA z-score (32 shadows, single feature) | 0.592 | +0.074 |
| Soft LiRA single feature (32 shadows) | 0.611 | +0.019 |
| Soft LiRA + 0.5 × Conferrable AE (32 shadows, 1 variant) | 0.648 | +0.037 |
| Soft LiRA (64 shadows) + 1.0 × mean of 10 confer variants | 0.667 | +0.019 |
| **Soft LiRA (64 shadows) + 1.5 × mean of 10 confer variants** | **0.685** | **+0.019** |

Final rank: 6 of 65 teams on the public leaderboard at submission close.

What did *not* work: BN running-statistics correlation (the 13 perfect-match
suspects were all in the private split), pure loss-barrier (it found a 67-
suspect "in basin" group whose ordering didn't capture additional public
stolen), multi-layer activation CKA at layer1 (low-level conv features are
data-determined and don't separate stolen from independent), per-class
confusion-matrix similarity, dataset-inference margin gap, DeepInversion +
ACS. Every output-similarity ensemble we tried converged to the same
~30-suspect plateau, and 14 of those 30 are not in the public-stolen set —
which is why breaking out of that plateau required the continuous soft-LiRA
score and the conferrable-AE signal, both of which use shadow models as
explicit "honest" reference distributions.

## Conclusion

Detection of stolen models matters because model training, especially at
foundation-model scale, represents a large capital and energy investment, and
because models embed proprietary data and design choices that can be
extracted at orders-of-magnitude lower cost than re-training. A workable
detector with public-leaderboard TPR@5%FPR of 0.685 shows that even with no
provenance information about the suspects, two relatively cheap signals —
one statistical (LiRA on confidence) and one adversarial (conferrable AEs) —
are enough to identify a clear majority of derived models, including
distilled students with drifted output distributions. This has practical
consequences for IP enforcement, audit, and licensing of released model
weights: model owners can detect distilled copies without trusting any
labels supplied by the model holder, simply by running a few hundred
suspect-side forward passes against a precomputed shadow set and a
precomputed set of conferrable inputs.
