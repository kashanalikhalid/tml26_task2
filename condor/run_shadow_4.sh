#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_$$}"
mkdir -p "$SCRATCH/cifar100" outputs/shadows
echo "::: [shadow_4] host=$(hostname) scratch=$SCRATCH"
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
USER_BASE="$SCRATCH/python_user"; mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
DATA_ROOT="$TASK_DIR/data"
# Train into SCRATCH first, then mv to NFS at end (one atomic write)
SCRATCH_OUT="$SCRATCH/shadow_4.safetensors"
$PY -u experiments/exp13_shadow_train.py \
    --out "$SCRATCH_OUT" --seed 4 \
    --cifar-root "$SCRATCH/cifar100" \
    --train-main-idx "$DATA_ROOT/target_model/train_main_idx.json" \
    --epochs 50 --batch-size 256 --lr 0.1 --wd 5e-4 --workers 4
echo "::: shadow trained, copying to NFS"
cp "$SCRATCH_OUT" "outputs/shadows/shadow_4.safetensors"
ls -la outputs/shadows/
