# TML 2026 Assignment 2 — Stolen Model Detection

This repository reproduces the leaderboard submission for the **TML 2026 Stolen
Model Detection** task (CISPA, summer term 2026). Given a CIFAR-style ResNet-18
target trained on a subset of CIFAR-100 plus 360 suspect checkpoints of the
same architecture, the pipeline outputs a `submission.csv` with a continuous
stealing-confidence score per suspect.

## Reproduce the leaderboard submission

1. **Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download the assignment data (~16 GB) from HuggingFace**

   ```bash
   huggingface-cli download SprintML/tml26_task2 \
       --repo-type model \
       --local-dir data \
       --include "target_model/*" "suspect_models/*" "task_template.py"
   ```

   This populates `data/target_model/weights.safetensors`,
   `data/target_model/train_main_idx.json`, and
   `data/suspect_models/suspect_000.safetensors` … `suspect_359.safetensors`.

3. **Score the suspects** (downloads CIFAR-100 test set on first run):

   ```bash
   python detect_stolen_models.py \
       --target data/target_model/weights.safetensors \
       --suspects data/suspect_models \
       --cifar-root data/cifar100 \
       --features-out outputs/features.csv \
       --submission-out outputs/submission.csv
   ```

   GPU is auto-detected. On a single modern GPU the full 360-suspect pass is
   minutes; on CPU it is closer to an hour.

4. **Submit** by setting `FILE_PATH = "outputs/submission.csv"` and
   `API_KEY = "<your-key>"` in `submission.py` (copy from the SprintML HF repo)
   and running it. The API key must never be committed to this repo.

The submission file lives at `outputs/submission.csv` and follows the format
required by the assignment server (two columns: `id`, `score`; 360 rows;
`id ∈ [0, 359]`).
