#!/usr/bin/env bash
# Wrapper executed inside the pytorch/pytorch docker image as one HTCondor job.
#
# The cluster's shared NFS home is heavily contended (~10 KB/s under load),
# so we download the 16 GB of suspect models to the worker's local scratch
# disk first, run scoring there, and emit only the small CSVs back to NFS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"

export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python

# Scratch dir: Condor sets _CONDOR_SCRATCH_DIR; otherwise pick a TMPDIR-ish fallback.
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}/tml26_task2_$$}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cifar100"
echo "::: host=$(hostname)  task_dir=$TASK_DIR  scratch=$SCRATCH"
df -h "$SCRATCH" | head -2

$PY -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# Install the few packages the base image is missing. Keep under scratch so we
# don't write to the contended NFS for site-packages.
USER_BASE="$SCRATCH/python_user"
mkdir -p "$USER_BASE"
PYTHONUSERBASE="$USER_BASE" $PY -m pip install --quiet --user safetensors pandas tqdm numpy 2>&1 | tail -3 || true
USER_SITE=$(PYTHONUSERBASE="$USER_BASE" $PY -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
echo "::: USER_SITE=$USER_SITE"
$PY -u -c "import safetensors, pandas, tqdm, numpy; print('deps ok')"

# Download the 360 suspects + target into scratch. Resumable.
TASK_DIR="$SCRATCH" PARALLEL="${PARALLEL:-8}" bash scripts/download_data.sh

# Score using scratch-local data. CIFAR-100 also goes to scratch.
$PY -u detect_stolen_models.py \
    --target "$SCRATCH/data/target_model/weights.safetensors" \
    --suspects "$SCRATCH/data/suspect_models" \
    --cifar-root "$SCRATCH/cifar100" \
    --features-out outputs/features.csv \
    --submission-out outputs/submission.csv \
    --batch-size "${BATCH_SIZE:-256}" \
    ${PROBE_SIZE:+--probe-size $PROBE_SIZE} \
    ${LIMIT:+--limit $LIMIT}

echo
echo "::: done. outputs/ summary:"
ls -la outputs/
echo "::: submission preview:"
head -5 outputs/submission.csv
wc -l outputs/submission.csv
