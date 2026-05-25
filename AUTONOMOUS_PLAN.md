# Autonomous Overnight Plan — TML 2026 Task 2

Started: 2026-05-25 ~03:05  ·  Deadline: 2026-05-26 23:59  ·  ~45 hours left.

## Current state
- **Leaderboard score: 0.518519** (rank 47/53, from exp02 / variant C, submission_id 1634)
- Public set: ~27 stolen / 108 total. Score quantum: 1/27 ≈ 0.037.
- Top of board: team_XXXII 0.740741 (20/27); team_V 0.740741; team_XLVIII 0.666667.
- Already submitted (4): exp01 (0.407), exp02 (0.518), exp03 (≤0.518), exp06 (≤0.518).
- Cooldown: last submission 02:53; next valid 03:53.

## Authorized overnight actions
- **Submit** to leaderboard every cooldown window (best-per-team kept; worst case stays at 0.518).
- **New local experiments** on MPS (~20 min each).
- **New cluster experiments** (~30-40 min; data now cached on NFS so no re-download).
- **Local re-rankings** of cached features.csv (seconds, free).
- **Self-replicating ensembles** (run same experiment with multiple seeds, average ranks).
- **No GitHub pushes** (remote needs PAT, can't auth).
- **No deletion** of cluster files (data, outputs, logs).
- **No `condor_rm`** without checking first.

## Iteration loop
Each wake-up (~30-60 min cadence):
1. Read `SCORES.md` + `git log --oneline -10`.
2. `curl http://34.63.153.158/leaderboard_page` → parse team_LXXXII score.
3. `ssh -n conduit "condor_q atml_team061 -nobatch"` → check cluster jobs.
4. Check local jobs: `pgrep -f experiments/`.
5. **If cooldown ≥ 60 min since last submission** → pick best fresh candidate from `outputs/*.csv` queue and submit:
   ```bash
   scp outputs/<candidate>.csv conduit:tml26_task2/outputs/<candidate>.csv
   ssh -n conduit "cd tml26_task2 && curl -sS -X POST -H 'X-API-Key: '\$(cat .submission_api_key) -F 'file=@outputs/<candidate>.csv' http://34.63.153.158/submit/19-stolen-model-detection"
   ```
6. **Plan & launch next experiment** (see queue below). Launch with `nohup ... &` to background.
7. **Update `SCORES.md`** with submission_id, score change, what's running, what's next.
8. `git add SCORES.md && git commit -m "..."` (local only).
9. `ScheduleWakeup` for the next cycle (~30-60 min depending on what's running).

## Submission candidates queue (priority order)

| # | File | Approach | Notes |
|---|---|---|---|
| 1 | `outputs/exp07_plus_min.csv` | MIN-rank over 15 features incl CKA+PGD+member-gap | 18 new IDs in top-30 vs exp02. Highest-variance bet. |
| 2 | `outputs/submission_merged_min.csv` | MIN-rank over 11 unified features | 15 new IDs vs exp02. Similar to #1. |
| 3 | `outputs/exp05/submission.csv` | exp05 PGD-strong cluster output | 2 new IDs vs exp02 (169, 295) |
| 4 | `outputs/exp07/submission.csv` | exp07 CKA+PGD-strong cluster output | 3 new IDs (169, 244, 295) |
| 5 | `outputs/exp07_local/submission.csv` | exp07 local (different PGD seed, in-flight) | when it finishes |

Pre-check before each submission: confirm 360 rows, ids 0..359, no NaN/Inf. Submission rejected by server otherwise.

## Experiments queue (to launch after each cooldown if cluster/local slot free)

A. **OOD probes** — forward random-noise + CIFAR-10 inputs, behavioural similarity on these.
   - Hypothesis: stolen models inherit target's specific high-confidence-on-noise pattern.
   - Cost: ~20 min local.
   - New file: `detect/ood.py` + `experiments/exp08_ood.py`.

B. **Augmentation-sensitive probe** — apply target's specific biased-crop (bias_x=0.5, bias_y=-0.25, jitter=0.25), measure how suspect's prediction changes vs target's prediction change.
   - Hypothesis: stolen models inherit target's specific augmentation regime.
   - Cost: ~20 min local.

C. **Self-replicating PGD ensemble** — run PGD 3x with different random seeds, average the per-suspect transfer scores.
   - Hypothesis: reduces variance from PGD random init; 3 perspectives on boundary.
   - Cost: 3 × current exp07 time = ~60 min local. Cluster much faster.

D. **Targeted PGD** — instead of untargeted (away from y), do targeted (toward fixed random class).
   - Hypothesis: targeted attack carves a more specific direction in target's boundary.

E. **Earlier-layer CKA** — hook layer3 / layer4 outputs (not just avgpool). Distilled models may share earlier representations more than penultimate.

F. **Per-class loss correlation** — vector of 100 per-class avg losses, correlate target vs suspect. Stolen models share target's per-class weakness profile.

G. **Logit magnitude / temperature** — compute mean |logit| per suspect. Distilled models often have flatter (lower-temp) logits.

H. **Stronger ensemble fusion** — try weighted rank-mean where the most-discriminating features get higher weight. Use the GAP in feature values between high and low scorers as a proxy for discriminability.

## Hard stop conditions
- Reached **12 successful submissions** overnight (preserve quota for morning).
- Score hits **0.70+** (top 5, mission likely accomplished).
- Local disk **<2 GB free**.
- Cluster disk **<2 GB free** AND we can't write outputs.
- Cluster SECMAN failures **3 times in a row** → cluster unusable, stay local.

## Files for the morning
- `SCORES.md` — chronological log of every submission + what each gave.
- `AUTONOMOUS_PLAN.md` — this file.
- `outputs/<exp_or_variant>.csv` — every candidate we generated.
- `outputs/<exp>/features.csv` — feature data for re-ranking experimentation.
- Git log — every commit explains what was added and why.
