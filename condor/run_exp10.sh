#!/usr/bin/env bash
# Wrapper for exp10_pgd_seeds. Outputs -> outputs/exp10/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_$$}"
mkdir -p "$SCRATCH/cifar100" "$SCRATCH/cifar10" "outputs/exp10"
echo "::: [exp10] host=$(hostname) scratch=$SCRATCH"
df -h "$SCRATCH" | head -2
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

USER_BASE="$SCRATCH/python_user"; mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"

# NFS cache is now fully populated -> read directly from NFS.
DATA_ROOT="$TASK_DIR/data"
echo "::: using NFS cache at $DATA_ROOT"

EXTRA_ARGS=""
if [ "exp10_pgd_seeds" = "exp08_ood" ]; then
    EXTRA_ARGS="--cifar10-root $SCRATCH/cifar10"
fi

$PY -u experiments/exp10_pgd_seeds.py \
    --target "$DATA_ROOT/target_model/weights.safetensors" \
    --suspects "$DATA_ROOT/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --train-main-idx "$DATA_ROOT/target_model/train_main_idx.json" \
    --features-out outputs/exp10/features.csv \
    --submission-out outputs/exp10/submission.csv \
    --batch-size "${BATCH_SIZE:-256}" $EXTRA_ARGS

echo "::: [exp10] outputs:"
ls -la outputs/exp10/
head -3 outputs/exp10/submission.csv
wc -l outputs/exp10/submission.csv
