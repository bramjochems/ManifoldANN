#!/bin/bash
# Teardown all resources for a given run_id (or matching name prefix).
# `az vm delete` does NOT cascade — NIC, disk, public IP, NSG, VNET, and the
# DevTestLab auto-shutdown schedule all stay behind and accrue small charges
# unless explicitly removed.
#
# Usage:
#   bash scripts/cloud/teardown.sh <run-id>             # by run_id tag (preferred)
#   bash scripts/cloud/teardown.sh --prefix <prefix>    # by name prefix (fallback)
#   bash scripts/cloud/teardown.sh --dry-run <run-id>   # show what would be deleted

set -euo pipefail

RG="bram"
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

MODE="run_id"
if [ "${1:-}" = "--prefix" ]; then
    MODE="prefix"
    shift
fi
TARGET="${1:?usage: teardown.sh [--dry-run] [<run-id> | --prefix <prefix>]}"

# Collect resource IDs in the right delete order.
# Order matters: NIC before VNET/PublicIP/NSG; VM before disk; everything before VNET.
echo "==> Discovering resources for $MODE=$TARGET in RG=$RG"
if [ "$MODE" = "run_id" ]; then
    QUERY="[?tags.run_id=='$TARGET'] | [].id"
else
    QUERY="[?starts_with(name, '$TARGET')] | [].id"
fi
ALL=$(az resource list -g "$RG" --query "$QUERY" -o tsv)

if [ -z "$ALL" ]; then
    echo "No resources matched."
    exit 0
fi

# Sort into delete buckets by resource type.
declare -a VMS NICS PIPS NSGS VNETS DISKS SCHEDULES OTHERS
while IFS= read -r id; do
    case "$id" in
        */virtualMachines/*)               VMS+=("$id") ;;
        */networkInterfaces/*)             NICS+=("$id") ;;
        */publicIPAddresses/*)             PIPS+=("$id") ;;
        */networkSecurityGroups/*)         NSGS+=("$id") ;;
        */virtualNetworks/*)               VNETS+=("$id") ;;
        */disks/*)                         DISKS+=("$id") ;;
        */schedules/*)                     SCHEDULES+=("$id") ;;
        *)                                 OTHERS+=("$id") ;;
    esac
done <<< "$ALL"

delete_bucket() {
    local label=$1
    local arr_name=$2
    # Indirect array expansion (bash 4.3+) so empty arrays don't expand to "".
    local -n arr_ref="$arr_name"
    [ ${#arr_ref[@]} -eq 0 ] && return 0
    echo "==> [$label] deleting ${#arr_ref[@]} resource(s)"
    if [ "$DRY_RUN" = "1" ]; then
        for i in "${arr_ref[@]}"; do echo "  (dry-run) $i"; done
        return 0
    fi
    az resource delete --ids "${arr_ref[@]}" 2>&1 | tail -3 || true
}

# Delete in dependency order.
delete_bucket "VMs"       VMS
delete_bucket "NICs"      NICS
delete_bucket "PublicIPs" PIPS
delete_bucket "NSGs"      NSGS
delete_bucket "VNETs"     VNETS
delete_bucket "Disks"     DISKS
delete_bucket "Schedules" SCHEDULES
delete_bucket "Other"     OTHERS

echo "==> Verifying"
LEFTOVER=$(az resource list -g "$RG" --query "$QUERY" -o tsv)
if [ -n "$LEFTOVER" ]; then
    echo "WARNING: leftover resources:"
    echo "$LEFTOVER"
    exit 1
else
    echo "OK: all matching resources removed."
fi
