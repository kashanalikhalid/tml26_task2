#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_$$}"
mkdir -p "$SCRATCH/cifar100" outputs/exp14
echo "::: [shadow_score] host=$(hostname) scratch=$SCRATCH"
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
USER_BASE="$SCRATCH/python_user"; mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
DATA_ROOT="$TASK_DIR/data"
SHADOWS=$(ls outputs/shadows/shadow_*.safetensors 2>/dev/null | tr '\n' ' ')
echo "::: shadows: $SHADOWS"
if [ -z "$SHADOWS" ]; then echo "ERROR: no shadows!"; exit 1; fi
$PY -u experiments/exp14_shadow_score.py \
    --target "$DATA_ROOT/target_model/weights.safetensors" \
    --shadows $SHADOWS \
    --suspects "$DATA_ROOT/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --train-main-idx "$DATA_ROOT/target_model/train_main_idx.json" \
    --features-out outputs/exp14/features.csv \
    --submission-out outputs/exp14/submission.csv
echo "::: [shadow_score] done"
head -3 outputs/exp14/submission.csv
wc -l outputs/exp14/submission.csv
