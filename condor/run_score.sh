#!/usr/bin/env bash
# Wrapper executed inside the pytorch/pytorch docker image.
# - cd into the task dir
# - install our few extra Python deps to --user (safetensors/pandas/tqdm are
#   not all in the base image; safetensors definitely isn't)
# - run the scoring driver

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"

export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python

echo "::: host=$(hostname)  cwd=$(pwd)  python=$PY"
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Install the few packages the base image is missing. --user keeps these in
# /home so subsequent runs reuse the install.
$PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -5 || true
USER_SITE=$($PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"

$PY -u detect_stolen_models.py \
    --target data/target_model/weights.safetensors \
    --suspects data/suspect_models \
    --cifar-root data/cifar100 \
    --features-out outputs/features.csv \
    --submission-out outputs/submission.csv \
    --batch-size "${BATCH_SIZE:-256}" \
    ${PROBE_SIZE:+--probe-size $PROBE_SIZE} \
    ${LIMIT:+--limit $LIMIT}

echo "::: done. outputs/:"
ls -la outputs/
echo "::: submission preview:"
head -5 outputs/submission.csv
wc -l outputs/submission.csv
