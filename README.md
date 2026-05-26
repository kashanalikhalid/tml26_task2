# TML 2026 Task 2 — Stolen Model Detection

Reproduces our best leaderboard submission for detecting which of the 360
suspect ResNet-18 checkpoints are stolen or derived from the target.

The final score per suspect is

```
score = z(soft_lira_mean_z) + 1.5 * mean_v z(confer_match_rate_v)
```

* **soft_lira_mean_z** (`experiments/compute_soft_lira.py`): for each sample in
  the target's vulnerability set (target predicts wrong, ≥85% of shadows
  predict right), the suspect's softmax probability on the target's wrong
  class is z-scored against the same quantity from 64 shadow models, then
  averaged over the vulnerability samples.
* **confer_match_rate_v** (`experiments/conferrable_ae.py`): conferrable
  adversarial examples are crafted with joint PGD that simultaneously fools
  the target and keeps shadow models correct. For each suspect we record the
  fraction of these AEs on which the suspect predicts the same wrong class
  as the target. Ten variants `v1..v10` with different (eps, PGD steps,
  shadow weight, conferrability threshold) are averaged to reduce noise.

The two feature columns are z-standardized over the 360 IDs and summed with
weight 1.5 on the conferrable axis (step 6).

## Layout

```
detect/         ResNet-18 + CIFAR-100 probe helpers
experiments/    train_shadow, compute_soft_lira, conferrable_ae
submission.csv  the final leaderboard submission
```

## Reproducing the leaderboard score

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Download data
```bash
huggingface-cli download SprintML/tml26_task2 --repo-type model --local-dir data
```
Populates `data/` with `target_model/` and `suspect_models/`
from [SprintML/tml26_task2](https://huggingface.co/SprintML/tml26_task2).

### 3. Train 64 shadow models
Each takes ~30 minutes on a single GPU.

```bash
for seed in $(seq 1 64); do
  python experiments/train_shadow.py --seed "$seed" \
      --stats-out "outputs/shadow_stats/shadow_${seed}.npz" \
      --weights-out "outputs/shadow_weights/shadow_${seed}.safetensors"
done
```

### 4. Compute soft-LiRA features
```bash
python experiments/compute_soft_lira.py --batch-size 64 --n-test-probe 1000
```
Writes `outputs/soft_lira/features.csv`.

### 5. Compute 10 Conferrable-AE variants
```bash
# v1
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae/features.csv --submission-out outputs/conferrable_ae/submission.csv \
  --ae-cache outputs/conferrable_ae/aes.pt \
  --max-shadows 32 --n-seeds 300 --pgd-steps 40 --eps 0.0314 --pgd-alpha 0.00784 \
  --alpha-shadow 1.0 --conferrability-thr 0.5

# v2 — harder attack
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v2/features.csv --submission-out outputs/conferrable_ae_v2/submission.csv \
  --ae-cache outputs/conferrable_ae_v2/aes.pt \
  --max-shadows 32 --n-seeds 300 --pgd-steps 80 --eps 0.0627 --pgd-alpha 0.00784 \
  --alpha-shadow 1.0 --conferrability-thr 0.5

# v3 — stronger shadow preservation
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v3/features.csv --submission-out outputs/conferrable_ae_v3/submission.csv \
  --ae-cache outputs/conferrable_ae_v3/aes.pt \
  --max-shadows 32 --n-seeds 300 --pgd-steps 50 --eps 0.0314 --pgd-alpha 0.00784 \
  --alpha-shadow 2.0 --conferrability-thr 0.7

# v4 — more seeds, more shadows
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v4/features.csv --submission-out outputs/conferrable_ae_v4/submission.csv \
  --ae-cache outputs/conferrable_ae_v4/aes.pt \
  --max-shadows 48 --n-seeds 500 --pgd-steps 60 --eps 0.0314 --pgd-alpha 0.00392 \
  --alpha-shadow 1.5 --conferrability-thr 0.6

# v5 — 64 shadows, mild PGD, lenient confer filter
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v5/features.csv --submission-out outputs/conferrable_ae_v5/submission.csv \
  --ae-cache outputs/conferrable_ae_v5/aes.pt \
  --max-shadows 64 --n-seeds 500 --pgd-steps 50 --eps 0.0314 --pgd-alpha 0.00392 \
  --alpha-shadow 0.5 --conferrability-thr 0.7

# v6 — stronger PGD
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v6/features.csv --submission-out outputs/conferrable_ae_v6/submission.csv \
  --ae-cache outputs/conferrable_ae_v6/aes.pt \
  --max-shadows 64 --n-seeds 400 --pgd-steps 100 --eps 0.0392 --pgd-alpha 0.00392 \
  --alpha-shadow 1.5 --conferrability-thr 0.7

# v7 — weak attack, many seeds
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v7/features.csv --submission-out outputs/conferrable_ae_v7/submission.csv \
  --ae-cache outputs/conferrable_ae_v7/aes.pt \
  --max-shadows 64 --n-seeds 600 --pgd-steps 30 --eps 0.0157 --pgd-alpha 0.00196 \
  --alpha-shadow 1.0 --conferrability-thr 0.5

# v8 — very strict conferrability
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v8/features.csv --submission-out outputs/conferrable_ae_v8/submission.csv \
  --ae-cache outputs/conferrable_ae_v8/aes.pt \
  --max-shadows 64 --n-seeds 800 --pgd-steps 50 --eps 0.0314 --pgd-alpha 0.00392 \
  --alpha-shadow 3.0 --conferrability-thr 0.95

# v9 — weak shadow weight
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v9/features.csv --submission-out outputs/conferrable_ae_v9/submission.csv \
  --ae-cache outputs/conferrable_ae_v9/aes.pt \
  --max-shadows 64 --n-seeds 1000 --pgd-steps 60 --eps 0.0392 --pgd-alpha 0.00392 \
  --alpha-shadow 0.3 --conferrability-thr 0.4

# v10 — balanced, maximum samples
python experiments/conferrable_ae.py \
  --features-out outputs/conferrable_ae_v10/features.csv --submission-out outputs/conferrable_ae_v10/submission.csv \
  --ae-cache outputs/conferrable_ae_v10/aes.pt \
  --max-shadows 64 --n-seeds 1500 --pgd-steps 60 --eps 0.0314 --pgd-alpha 0.00196 \
  --alpha-shadow 1.0 --conferrability-thr 0.6
```

### 6. Build `submission.csv`
Z-standardize both feature columns over the 360 IDs and combine them with
weight 1.5 on the conferrable axis:

```python
import glob, numpy as np, pandas as pd

z = lambda x: (x - x.mean()) / (x.std() + 1e-12)

soft = pd.read_csv("outputs/soft_lira/features.csv").set_index("id")
ids = sorted(soft.index)
score = z(soft["soft_lira_mean_z"].reindex(ids).fillna(0).to_numpy())

confer = [
    z(pd.read_csv(f).set_index("id")["confer_match_rate"].reindex(ids).fillna(0).to_numpy())
    for f in sorted(glob.glob("outputs/conferrable_ae*/features.csv"))
]
score = score + 1.5 * np.mean(confer, axis=0)

pd.DataFrame({"id": ids, "score": score}).to_csv("submission.csv", index=False)
```

### 7. Upload to the leaderboard
```bash
curl -X POST -H "X-API-Key: <YOUR_API_KEY>" \
     -F file=@submission.csv \
     http://34.63.153.158/submit/19-stolen-model-detection
```
