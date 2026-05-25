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
