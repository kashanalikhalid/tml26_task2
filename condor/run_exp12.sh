#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_$$}"
mkdir -p "$SCRATCH/cifar100" outputs/exp12
echo "::: [exp12] host=$(hostname) scratch=$SCRATCH"
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
USER_BASE="$SCRATCH/python_user"; mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
DATA_ROOT="$TASK_DIR/data"
echo "::: using NFS cache at $DATA_ROOT"
$PY -u experiments/exp12_multi_layer_cka.py \
    --target "$DATA_ROOT/target_model/weights.safetensors" \
    --suspects "$DATA_ROOT/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --train-main-idx "$DATA_ROOT/target_model/train_main_idx.json" \
    --features-out outputs/exp12/features.csv \
    --submission-out outputs/exp12/submission.csv \
    --batch-size "${BATCH_SIZE:-256}"
echo "::: [exp12] done:"; ls -la outputs/exp12/
