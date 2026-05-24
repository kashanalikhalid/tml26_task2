#!/usr/bin/env bash
# Download SprintML/tml26_task2 from HuggingFace into $TASK_DIR/data/ using
# only curl. The HF repo is public so no auth is needed; we hit the resolve
# endpoints directly. Resumable, parallel.
#
# Usage:
#   TASK_DIR=~/tml26_task2 bash scripts/download_data.sh
# Or just:
#   cd ~/tml26_task2 && bash scripts/download_data.sh

set -euo pipefail

TASK_DIR="${TASK_DIR:-$(pwd)}"
DATA_DIR="$TASK_DIR/data"
BASE="https://huggingface.co/SprintML/tml26_task2/resolve/main"
PARALLEL="${PARALLEL:-8}"

mkdir -p "$DATA_DIR/target_model" "$DATA_DIR/suspect_models"
echo "::: TASK_DIR=$TASK_DIR  DATA_DIR=$DATA_DIR  PARALLEL=$PARALLEL"
echo "::: curl: $(curl --version | head -1)"

echo
echo "::: [1/3] target_model files"
curl -fSL --retry 5 --retry-delay 2 -C - \
    -o "$DATA_DIR/target_model/weights.safetensors" \
    "$BASE/target_model/weights.safetensors"
curl -fSL --retry 5 --retry-delay 2 -C - \
    -o "$DATA_DIR/target_model/train_main_idx.json" \
    "$BASE/target_model/train_main_idx.json"

echo
echo "::: [2/3] task_template.py + submission.py (small)"
curl -fSL --retry 5 -o "$TASK_DIR/task_template.py" "$BASE/task_template.py"
curl -fSL --retry 5 -o "$TASK_DIR/submission.py" "$BASE/submission.py"

echo
echo "::: [3/3] 360 suspect_models (parallel=$PARALLEL)"
# Build a curl config file with one url+output pair per line; curl --parallel
# downloads many files concurrently and resumes via -C -.
CFG="$(mktemp)"
trap 'rm -f "$CFG"' EXIT
for i in $(seq -f "%03g" 0 359); do
    {
        printf 'url = %s\n' "$BASE/suspect_models/suspect_${i}.safetensors"
        printf 'output = %s\n' "$DATA_DIR/suspect_models/suspect_${i}.safetensors"
    } >> "$CFG"
done

# curl --parallel arrived in 7.66 (June 2019); cluster has it.
curl -fSL --retry 5 --retry-delay 2 -C - --parallel --parallel-max "$PARALLEL" -K "$CFG"

echo
echo "::: verification"
echo "target weights: $(ls -la "$DATA_DIR/target_model/weights.safetensors" 2>&1)"
echo "train_main_idx: $(ls -la "$DATA_DIR/target_model/train_main_idx.json" 2>&1)"
N=$(ls "$DATA_DIR/suspect_models" 2>/dev/null | wc -l)
echo "suspect count: $N (expect 360)"
# Verify all suspects are the expected 44929864 bytes
WRONG_SIZE=$(find "$DATA_DIR/suspect_models" -name '*.safetensors' ! -size 44929864c 2>/dev/null | wc -l)
echo "wrong-size suspects: $WRONG_SIZE (expect 0)"
du -sh "$DATA_DIR"/target_model "$DATA_DIR"/suspect_models 2>/dev/null
echo
echo "::: download complete."
