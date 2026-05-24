#!/usr/bin/env bash
# Wrapper executed inside the pytorch/pytorch docker image as one HTCondor job.
#
# Data sourcing strategy:
#   1. If $TASK_DIR/data/ on NFS already has the target + all 360 suspects at
#      the expected 44,929,864 bytes each, read directly from NFS (reads are
#      fast even under load; only writes are contended).
#   2. Otherwise download into worker-local $SCRATCH/data and run from there.
#      After a successful scoring pass, attempt a best-effort scratch->NFS
#      copy so the next job can hit the cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"

export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python

SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_task2_$$}"
mkdir -p "$SCRATCH/cifar100"
echo "::: host=$(hostname)  task_dir=$TASK_DIR  scratch=$SCRATCH"
df -h "$SCRATCH" | head -2

$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

USER_BASE="$SCRATCH/python_user"
mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
echo "::: USER_SITE=$USER_SITE"
$PY -u -c "import safetensors, pandas, tqdm, numpy; print('deps ok')"

# Decide between NFS cache and a fresh scratch download.
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
    DOWNLOADED_FRESH=0
    echo "::: using NFS cache at $DATA_ROOT (all 360 suspects + target verified)"
else
    DATA_ROOT="$SCRATCH/data"
    DOWNLOADED_FRESH=1
    echo "::: NFS cache incomplete; downloading into $DATA_ROOT"
    TASK_DIR="$SCRATCH" PARALLEL="${PARALLEL:-8}" bash scripts/download_data.sh
fi

$PY -u detect_stolen_models.py \
    --target "$DATA_ROOT/target_model/weights.safetensors" \
    --suspects "$DATA_ROOT/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --features-out outputs/features.csv \
    --submission-out outputs/submission.csv \
    --batch-size "${BATCH_SIZE:-256}" \
    ${PROBE_SIZE:+--probe-size $PROBE_SIZE} \
    ${LIMIT:+--limit $LIMIT}

echo
echo "::: outputs:"
ls -la outputs/
echo "::: submission preview:"
head -5 outputs/submission.csv
wc -l outputs/submission.csv

# Best-effort scratch -> NFS warm-up so the next job hits the cache.
if [ "$DOWNLOADED_FRESH" = "1" ]; then
    echo "::: priming NFS cache (best-effort, may be slow under load)"
    mkdir -p "$TASK_DIR/data/target_model" "$TASK_DIR/data/suspect_models"
    # Use rsync if available, else cp -a. Both are interruptible safely.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --ignore-existing "$SCRATCH/data/target_model/" "$TASK_DIR/data/target_model/" || true
        rsync -a --ignore-existing "$SCRATCH/data/suspect_models/" "$TASK_DIR/data/suspect_models/" || true
    else
        cp -an "$SCRATCH/data/target_model/." "$TASK_DIR/data/target_model/" || true
        cp -an "$SCRATCH/data/suspect_models/." "$TASK_DIR/data/suspect_models/" || true
    fi
    echo "::: NFS cache size now: $(du -sh "$TASK_DIR/data" 2>/dev/null)"
fi
