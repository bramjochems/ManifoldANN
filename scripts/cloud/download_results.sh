#!/bin/bash
# Download all blobs under runs/<run-id>/ to a local directory for
# post-processing.
#
# Usage: bash scripts/cloud/download_results.sh <run-id> [dest-dir]
#   default dest-dir: ./cloud-results/<run-id>

set -euo pipefail

RUN_ID="${1:?usage: download_results.sh <run-id> [dest-dir]}"
DEST="${2:-./cloud-results/$RUN_ID}"

SA="stbram"
CONTAINER="mai-thesis"

mkdir -p "$DEST"

echo "==> Downloading runs/$RUN_ID/ to $DEST"
az storage blob download-batch \
    --account-name "$SA" \
    --source "$CONTAINER" \
    --pattern "runs/$RUN_ID/*" \
    --destination "$DEST" \
    --auth-mode login \
    --no-progress 2>&1 | tail -3

echo
echo "==> Files downloaded:"
find "$DEST" -type f | wc -l
echo
echo "==> Per-shard results.csv files:"
find "$DEST" -name "results.csv"
