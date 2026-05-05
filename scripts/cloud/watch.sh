#!/bin/bash
# Watch a run's progress: list blobs every 30s + show running VMs.
# Usage: bash scripts/cloud/watch.sh <run-id>
#   ctrl-C to exit.

set -euo pipefail

RUN_ID="${1:?usage: watch.sh <run-id>}"
SA="stbram"
CONTAINER="mai-thesis"
RG="bram"

while true; do
    clear
    echo "=== $(date -u +%H:%M:%S) UTC | run: $RUN_ID ==="
    echo
    echo "--- VMs (tag run_id=$RUN_ID) ---"
    az vm list --resource-group "$RG" \
        --query "[?tags.run_id=='$RUN_ID'].{name:name, state:powerState, shard:tags.shard_id}" \
        -o table 2>/dev/null || echo "(none)"
    echo
    echo "--- Blobs under runs/$RUN_ID/ ---"
    az storage blob list --account-name "$SA" --container-name "$CONTAINER" \
        --prefix "runs/$RUN_ID/" --auth-mode login \
        --query "[].{name:name, size_kb:to_number(properties.contentLength)/\`1024\`}" \
        -o table 2>/dev/null || echo "(none yet)"
    echo
    echo "(refresh every 30s; ctrl-C to exit)"
    sleep 30
done
