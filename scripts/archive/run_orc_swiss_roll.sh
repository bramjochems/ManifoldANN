#!/usr/bin/env bash
# Run the ORC Swiss roll shortcut-detection experiment end-to-end.
#
# Usage:
#   ./scripts/run_orc_swiss_roll.sh            # full run
#   SMOKE=1 ./scripts/run_orc_swiss_roll.sh    # smoke test (n=200, k=10, standard only)
#   N_OVERRIDE=500 K_OVERRIDE=10 ./scripts/run_orc_swiss_roll.sh
#
# Results land in:
#   docs/thesis/results/orc_results/swiss_roll_<timestamp>/
#   Set RESUME_DIR=path to resume a previous run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JULIA_SCRIPT="$SCRIPT_DIR/experiment_orc_swiss_roll.jl"

# Forward any environment overrides to Julia
export SMOKE="${SMOKE:-}"
export N_OVERRIDE="${N_OVERRIDE:-}"
export K_OVERRIDE="${K_OVERRIDE:-}"
export SKIP_EDGES="${SKIP_EDGES:-}"
export SKIP_GEODERROR="${SKIP_GEODERROR:-}"
export RESUME_DIR="${RESUME_DIR:-}"

echo "========================================================"
echo "ORC Swiss Roll Experiment"
echo "Package : $PACKAGE_DIR"
echo "Script  : $JULIA_SCRIPT"
echo "Threads : ${JULIA_NUM_THREADS:-auto}"
echo "========================================================"

exec julia \
    --project="$PACKAGE_DIR" \
    -t "${JULIA_NUM_THREADS:-auto}" \
    "$JULIA_SCRIPT"
