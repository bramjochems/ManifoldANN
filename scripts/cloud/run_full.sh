#!/bin/bash
# Full thesis run: provisions 10 VMs from scripts/cloud/vms.json.
# Each VM runs its assigned shard list sequentially (multi-shard mode).
#
# Usage: bash scripts/cloud/run_full.sh
#
# The git SHA at HEAD must already be pushed to origin.

set -euo pipefail

VMS_FILE="$(dirname "$0")/vms.json"
RUN_ID="full-$(date -u +%Y%m%d-%H%M)"

RG="bram"
LOCATION="westeurope"
SA="stbram"
CONTAINER="mai-thesis"
VM_SIZE="Standard_D16s_v6"
UAMI_ID="/subscriptions/db9f84bc-5025-425a-84b7-31e68913aa63/resourcegroups/bram/providers/Microsoft.ManagedIdentity/userAssignedIdentities/ann-bench-vm-identity"
REPO_URL="https://github.com/bramjochems/ManifoldANN.git"

cd "$(dirname "$0")/../.."

echo "==> Pushing current branch so VMs can clone the latest commit..."
git push

GIT_SHA=$(git rev-parse HEAD)
N_VMS=$(jq 'length' "$VMS_FILE")
N_SHARDS=$(jq '[.[] | .shards | length] | add' "$VMS_FILE")

echo "==> Run ID: $RUN_ID"
echo "==> Git SHA: $GIT_SHA"
echo "==> Provisioning $N_VMS VMs for $N_SHARDS shards"

python3 scripts/cloud/run_shards.py \
    --resource-group "$RG" \
    --location "$LOCATION" \
    --storage-account "$SA" \
    --storage-container "$CONTAINER" \
    --vm-size "$VM_SIZE" \
    --git-sha "$GIT_SHA" \
    --shards-file "$VMS_FILE" \
    --run-id "$RUN_ID" \
    --user-assigned-identity-id "$UAMI_ID" \
    --repo-url "$REPO_URL"

echo
echo "==> RUN_ID=$RUN_ID"
echo "==> Watch progress: bash scripts/cloud/watch.sh $RUN_ID"
echo "==> When done:      bash scripts/cloud/download_results.sh $RUN_ID"
echo "==> Then teardown:  bash scripts/cloud/teardown.sh $RUN_ID"
