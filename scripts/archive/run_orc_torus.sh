#!/usr/bin/env bash
# Run the ORC torus shortcut-detection experiment end-to-end.
#
# Usage:
#   ./scripts/run_orc_torus.sh                          # full run (all variants)
#   SMOKE=1 ./scripts/run_orc_torus.sh                  # smoke test (R4r1, n=200, k=10)
#   N_OVERRIDE=500 K_OVERRIDE=10 ./scripts/run_orc_torus.sh
#   VARIANT_OVERRIDE=R2r1 ./scripts/run_orc_torus.sh    # single torus variant
#
# Results land in:
#   docs/thesis/results/orc_results/torus_<timestamp>/
#   Set RESUME_DIR=path to resume a previous run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JULIA_SCRIPT="$SCRIPT_DIR/experiment_orc_torus.jl"

# Forward any environment overrides to Julia
export SMOKE="${SMOKE:-}"
export N_OVERRIDE="${N_OVERRIDE:-}"
export K_OVERRIDE="${K_OVERRIDE:-}"
export VARIANT_OVERRIDE="${VARIANT_OVERRIDE:-}"
export SKIP_EDGES="${SKIP_EDGES:-}"
export SKIP_GEODERROR="${SKIP_GEODERROR:-}"
export RESUME_DIR="${RESUME_DIR:-}"

echo "========================================================"
echo "ORC Torus Experiment"
echo "Package  : $PACKAGE_DIR"
echo "Script   : $JULIA_SCRIPT"
echo "Threads  : ${JULIA_NUM_THREADS:-auto}"
echo "Variant  : ${VARIANT_OVERRIDE:-all}"
echo "========================================================"

exec julia \
    --project="$PACKAGE_DIR" \
    -t "${JULIA_NUM_THREADS:-auto}" \
    "$JULIA_SCRIPT"
