# TML 2026 Assignment 2

Starter repository for the Trustworthy Machine Learning 2026 model stealing task.

## Task Summary

The goal is to identify which suspect image-classification models were stolen from, or derived from, a provided target model. The assignment brief says you receive:

- the target model weights and white-box training details
- white-box access to roughly 360 suspect models

The expected output is a continuous stealing-confidence score per suspect model, where larger scores indicate a higher chance that the suspect was stolen or derived from the target.

## Repository Layout

```text
tml26_task2/
|- data/
|  |- target/
|  `- suspects/
|- outputs/
|- detect_stolen_models.py
|- REPORT_DRAFT.md
`- requirements.txt
```

## Quick Start

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Place the released target checkpoint inside `data/target/`.

3. Place suspect checkpoints or suspect model directories inside `data/suspects/`.

   Supported weight formats:

   - `.pt`
   - `.pth`
   - `.bin`
   - `.safetensors`

4. Run the baseline detector:

   ```bash
   python detect_stolen_models.py \
     --target data/target \
     --suspects data/suspects \
     --output outputs/suspect_scores.csv
   ```

5. Inspect the ranked output CSV and iterate on the scoring logic.

## What The Baseline Does

`detect_stolen_models.py` is a first-pass heuristic, not a final solution. It:

- loads the target checkpoint
- scans the suspect directory for model files or model folders
- compares matching parameter names and tensor shapes
- measures exact tensor matches, sample-level agreement, and cosine similarity
- writes a ranked CSV you can use as a starting point

This is useful for catching direct copies, lightly modified copies, or closely related checkpoints. It will likely need stronger features for the final submission.

## Notes

- `torch.load(..., weights_only=False)` is used for flexibility because released checkpoints often wrap the state dict differently.
- Large model files are set up for Git LFS via `.gitattributes`, but the actual weights are ignored by default in `.gitignore`.
- No remote is configured yet. Add one whenever you are ready to push this repo to GitHub.
