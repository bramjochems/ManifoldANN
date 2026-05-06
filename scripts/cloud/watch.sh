#!/bin/bash
# Per-VM progress dashboard for an in-flight multi-shard run.
#
# Reads scripts/cloud/vms.json to get each VM's assigned shard list,
# then probes blob storage per shard to classify state:
#   ✓ done:   has output/results_*/results.csv uploaded
#   …  active: has metadata.txt + run.log but no output yet (currently running)
#   ○ pending: nothing uploaded yet (queued behind earlier shards on this VM)
#
# Usage: bash scripts/cloud/watch.sh <run-id>
#        bash scripts/cloud/watch.sh <run-id> --watch   (refresh every 60s)

set -euo pipefail

RUN_ID="${1:?usage: watch.sh <run-id> [--watch]}"
WATCH=0
[ "${2:-}" = "--watch" ] && WATCH=1

SA="stbram"
CONTAINER="mai-thesis"
RG="bram"
VMS_FILE="$(dirname "$0")/vms.json"

show_progress() {
    # Pull the full blob list once (cheaper than 79 individual queries).
    local blobs
    blobs=$(az storage blob list --account-name "$SA" --container-name "$CONTAINER" \
        --auth-mode login --prefix "runs/$RUN_ID/" --query "[].name" -o tsv 2>/dev/null \
        | sort)

    # Build a sed-able list of shard_ids known to have results.csv vs only metadata.
    local done_shards active_shards
    done_shards=$(echo "$blobs" | awk -F'/' '/output\/results_.*\/results\.csv$/{print $3}' | sort -u)
    active_shards=$(echo "$blobs" | awk -F'/' '/\/metadata\.txt$/{print $3}' | sort -u)

    echo "=== $(date -u +%H:%M:%S) UTC | run: $RUN_ID ==="
    echo

    # Per-VM state
    local total_shards=0 total_done=0
    while IFS= read -r vm_id; do
        local shards
        shards=$(jq -r --arg v "$vm_id" '.[] | select(.vm_id == $v) | .shards[].shard_id' "$VMS_FILE")
        local n_total=0 n_done=0 active="" pending_first=""
        while IFS= read -r sid; do
            [ -z "$sid" ] && continue
            n_total=$((n_total + 1))
            if echo "$done_shards" | grep -qx "$sid"; then
                n_done=$((n_done + 1))
            elif echo "$active_shards" | grep -qx "$sid"; then
                [ -z "$active" ] && active="$sid"
            else
                [ -z "$pending_first" ] && pending_first="$sid"
            fi
        done <<< "$shards"
        total_shards=$((total_shards + n_total))
        total_done=$((total_done + n_done))

        # VM power state (if still listed)
        local vm_state
        vm_state=$(az vm get-instance-view -g "$RG" \
            -n "$RUN_ID-$vm_id" --query "instanceView.statuses[?starts_with(code,'PowerState')].displayStatus" \
            -o tsv 2>/dev/null || echo "?")
        local label
        if [ "$n_done" = "$n_total" ]; then
            label="✓ all $n_total done"
        elif [ -n "$active" ]; then
            label="$n_done/$n_total done | running: $active"
        elif [ -n "$pending_first" ]; then
            label="$n_done/$n_total done | next: $pending_first"
        else
            label="$n_done/$n_total done"
        fi
        printf "  %-35s [%s]  %s\n" "$vm_id" "$vm_state" "$label"
    done < <(jq -r '.[].vm_id' "$VMS_FILE")

    echo
    echo "Overall: $total_done / $total_shards shards complete"
}

if [ "$WATCH" = "1" ]; then
    while true; do
        clear
        show_progress
        echo
        echo "(refresh every 60s; ctrl-C to exit)"
        sleep 60
    done
else
    show_progress
fi
