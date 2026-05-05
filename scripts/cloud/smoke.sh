#!/bin/bash
# Smoke test: provision one D16s_v6 VM with the refactor-baseline config.
# Pushes current branch first (VM clones at the pushed SHA).
#
# Usage: bash scripts/cloud/smoke.sh [shards-file]
#   default shards-file: /tmp/smoke-shards.json (one fast shard)

set -euo pipefail

SHARDS_FILE="${1:-/tmp/smoke-shards.json}"
RUN_ID="smoke-$(date -u +%Y%m%d-%H%M)"

RG="bram"
LOCATION="westeurope"
SA="stbram"
CONTAINER="mai-thesis"
VM_SIZE="Standard_D16s_v6"
UAMI_ID="/subscriptions/db9f84bc-5025-425a-84b7-31e68913aa63/resourcegroups/bram/providers/Microsoft.ManagedIdentity/userAssignedIdentities/ann-bench-vm-identity"
REPO_URL="https://github.com/bramjochems/ManifoldANN.git"

cd "$(dirname "$0")/../.."

if [ ! -f "$SHARDS_FILE" ]; then
    echo "shards file not found: $SHARDS_FILE" >&2
    exit 1
fi

echo "==> Pushing current branch so VM can clone the latest commit..."
git push

GIT_SHA=$(git rev-parse HEAD)
echo "==> Run ID: $RUN_ID"
echo "==> Git SHA: $GIT_SHA"
echo "==> Shards: $SHARDS_FILE"

python3 scripts/cloud/run_shards.py \
    --resource-group "$RG" \
    --location "$LOCATION" \
    --storage-account "$SA" \
    --storage-container "$CONTAINER" \
    --vm-size "$VM_SIZE" \
    --git-sha "$GIT_SHA" \
    --shards-file "$SHARDS_FILE" \
    --run-id "$RUN_ID" \
    --user-assigned-identity-id "$UAMI_ID" \
    --repo-url "$REPO_URL"

echo
echo "==> Watch progress with:"
echo "  bash scripts/cloud/watch.sh $RUN_ID"
echo
echo "==> Or one-shot list:"
echo "  az storage blob list --account-name $SA --container-name $CONTAINER --prefix runs/$RUN_ID/ --auth-mode login -o table"
