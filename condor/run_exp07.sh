#!/usr/bin/env bash
# Wrapper for experiment 07: CKA + PGD-strong combined, narrow ensemble.
# Outputs go to outputs/exp07/{features,submission}.csv on NFS.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python

SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_$$}"
mkdir -p "$SCRATCH/cifar100" outputs/exp07
echo "::: [exp07] host=$(hostname) scratch=$SCRATCH"
df -h "$SCRATCH" | head -2
$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

USER_BASE="$SCRATCH/python_user"; mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"

nfs_cache_complete() {
    local data="$TASK_DIR/data"
    local tgt="$data/target_model/weights.safetensors"
    [ -f "$tgt" ] || return 1
    [ "$(stat -c%s "$tgt" 2>/dev/null || echo 0)" = "44929864" ] || return 1
    local count
    count=$(find "$data/suspect_models" -maxdepth 1 -name 'suspect_*.safetensors' -size 44929864c 2>/dev/null | wc -l)
    [ "$count" = "360" ]
}
if nfs_cache_complete; then
    DATA_ROOT="$TASK_DIR/data"
    echo "::: using NFS cache at $DATA_ROOT"
else
    DATA_ROOT="$SCRATCH/data"
    echo "::: NFS cache incomplete; downloading to scratch"
    TASK_DIR="$SCRATCH" PARALLEL="${PARALLEL:-8}" $PY -u scripts/download_data.py
fi

$PY -u experiments/exp07_cka_pgd.py \
    --target "$DATA_ROOT/target_model/weights.safetensors" \
    --suspects "$DATA_ROOT/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --train-main-idx "$DATA_ROOT/target_model/train_main_idx.json" \
    --features-out outputs/exp07/features.csv \
    --submission-out outputs/exp07/submission.csv \
    --batch-size "${BATCH_SIZE:-256}"

echo "::: [exp07] outputs:"
ls -la outputs/exp07/
head -3 outputs/exp07/submission.csv
wc -l outputs/exp07/submission.csv
