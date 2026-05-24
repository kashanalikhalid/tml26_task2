#!/usr/bin/env bash
# NFS-cache stage job: downloads target_model + 360 suspect_models from
# SprintML/tml26_task2 directly into ~/tml26_task2/data on the shared
# home filesystem. Runs CPU-only, no GPU.
#
# We intentionally keep PARALLEL low because the shared NFS home is
# write-contended; many concurrent writers serialize on commits, killing
# throughput. A few streams with -C - resume cover network hiccups.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TASK_DIR"

export PYTHONUNBUFFERED=1
echo "::: host=$(hostname)  uid=$(id -u)  task_dir=$TASK_DIR"
df -h "$TASK_DIR" | head -2

echo "::: kicking off curl-based download into NFS at $TASK_DIR/data"
TASK_DIR="$TASK_DIR" PARALLEL="${PARALLEL:-4}" bash scripts/download_data.sh

echo
echo "::: stage finished. final data summary:"
du -sh "$TASK_DIR/data"
ls "$TASK_DIR/data/suspect_models" | wc -l
echo "verifying file sizes (expect 360 at 44929864 bytes):"
find "$TASK_DIR/data/suspect_models" -name '*.safetensors' -size 44929864c | wc -l
echo "wrong-size:"
find "$TASK_DIR/data/suspect_models" -name '*.safetensors' ! -size 44929864c | wc -l
