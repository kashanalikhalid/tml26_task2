# TML 2026 Task 2 — Experiment Results

Tracking what each approach scored on the **public** TPR@5%FPR leaderboard.
Team is `team_LXXXII` (atml_team061 on conduit).

Public set: 30% of 360 suspects = ~108 models, of which ~27 are stolen
(score quantum ≈ 1/27 ≈ 0.037).

## Submissions

| # | Date/Time | Submission ID | Experiment | Score | Rank | Notes |
|---|-----------|---------------|------------|-------|------|-------|
| 1 | 25.05 ~21:42 | 1625 | exp01 baseline (all weight features + behavioral) | **0.407407** | 50/51 | First submission. weight_cosine_full is mostly noise (350/360 suspects >0.99). |
| 2 | 25.05 ~22:47 | 1634 | exp02 no_noisy_weights (variant C) | **0.518519** | 47/53 | Dropped weight_cosine_full, weight_cosine_backbone, weight_keys_match. **+0.111** improvement, **CURRENT BEST**. |
| 3 | 25.05 ~01:00 | 1656 | exp03 member_aware | **≤0.518** | unchanged | Added member/non-member loss correlation, wrong_agree, member_gap. Server kept variant C. Diagnosis: too many weak features dilute rank-mean. |
| 4 | 25.05 ~02:53 | 1667 | exp06 cka (penultimate-layer Linear CKA) | **≤0.518** | unchanged | CKA=1.0 for 30+ models (functionally identical to target) -- doesn't discriminate within the cluster. CKA brought in 5 new IDs to top-30 (8, 169, 244, 259, 295) and dropped 5 (4, 81, 83, 105, 109), but the swap didn't capture more stolen-in-public. |
| 5 | 25.05 ~03:54 | 1671 | exp07_plus_min (MIN-rank fusion 15 features, 18 new IDs in top-30) | **≤0.518** | unchanged | Aggressive MIN-fusion replacing 18 of top-30 IDs didn't help. **Strong evidence we've saturated this entire feature regime.** All 5+ test-set / member-aware / CKA / PGD experiments converge to top-30 = exp02 ∪ {169, 244, 295}, and that ceiling is 0.518 on public. |

## All cluster experiments completed by 02:54

| Cluster Job | Experiment | Output | Notes |
|---|---|---|---|
| 161587 | exp04 PGD-weak | (lost — disk full at write time) | |
| 161588 | exp05 PGD-strong (ε=16/255, 20 steps) | outputs/exp05/ | top-30 same as exp02 + {169, 244, 295} new |
| 161589 | exp06 CKA | outputs/exp06/ | already submitted |
| 161590 | exp07 CKA + PGD-strong combined | outputs/exp07/ | top-30 same as exp02 + {169, 244, 295} new |
| 161584/86 zombies | run_score.sh restarted from scratch | outputs/{features,submission}.csv | member+PGD-weak, current ensemble |

## Consensus diagnosis after 4 experiments

- **3 ID consensus**: 169, 244, 295 consistently appear in top-30 of every method that uses member+adversarial features. These are very likely true stolen models that variant C missed.
- **Exp06 (CKA) added 2 extra IDs (8, 259) on top of those 3, but didn't improve the leaderboard** — at least one of {8, 259} is a false positive that displaced a real stolen model.
- **The narrow rank-mean / median / q25 ensembles all converge to the same top-30** = exp02 ∪ {169, 244, 295}. Likely score: same 0.518 if those 3 are split between stolen/not in public, or up to 0.629 if all 3 are public-stolen.
- **`exp07_plus_min` is the high-variance bet** — adds 18 new IDs by demanding consistency across all features. Upside: huge if the conservative ones cluster around real stolen-in-public; Downside: no change (server keeps the best).

## Local laptop pipeline working

- 16 GB SprintML/tml26_task2 fully downloaded at ~02:49 (12.5 min).
- Smoke test on MPS (5 suspects): 17s, working correctly.
- exp07 local re-run started 02:51, ETA ~03:50 (PGD slows MPS — ~8s/suspect).
- All future iterations are ~15-50 min purely local, no cluster nonsense.

## In-flight Condor jobs

| Job ID | Experiment | Submitted | Status | Output dir |
|--------|-----------|-----------|--------|-----------|
| 161587 | exp04 pgd_weak (PGD ε=8/255, 10 steps) | 25.05 01:36 | running | `outputs/exp04/` |
| TBD | exp05 pgd_strong (PGD ε=16/255, 20 steps) | not yet | not submitted | `outputs/exp05/` |
| TBD | exp06 cka (penultimate-layer CKA) | not yet | not submitted | `outputs/exp06/` |
| TBD | exp07 cka_pgd (CKA + PGD combined) | not yet | not submitted | `outputs/exp07/` |

## Open hypotheses

- **Top 30 in exp02/03 are functionally identical to target** (jsd=0, top1=1.0, wrong_agree=1.0, member_loss_corr=1.0). Public score 14/27 means 13 stolen-in-public models are below our top-30 — likely fine-tuned or distilled variants with drifted outputs.
- **Adversarial transfer** (PGD on target → check transfer to suspects) should separate fine-tunes (inherit boundary) from independents (different boundary even on same data).
- **Penultimate-layer CKA** should separate distilled stolen models from independents, even when output behaviour has drifted.
- The "narrower ensemble" wins so far suggests **fewer, stronger features** are better than averaging many features.

## Leaderboard top context (public)

- 1st: team_XXXII @ 0.740741 (20/27)
- 2nd: team_XLVIII @ 0.666667 (18/27)
- 3rd: team_V @ 0.648148 (~17.5/27)
