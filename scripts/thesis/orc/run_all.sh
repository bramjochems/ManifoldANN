#!/usr/bin/env bash
# Run unified ORC experiment for both manifolds, then analysis.
#
# Usage:
#   ./scripts/thesis/orc/run_all.sh                    # full run
#   SMOKE=1 ./scripts/thesis/orc/run_all.sh            # smoke test
#   SKIP_PRUNING=1 ./scripts/thesis/orc/run_all.sh     # detection only
#
# Results land in:
#   docs/thesis/results/orc_results/swiss_roll_<ts>/
#   docs/thesis/results/orc_results/torus_<ts>/
#   docs/thesis/results/orc_results/thesis_*.csv
#   docs/thesis/figures/orc_*.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "========================================================"
echo "ORC Full Pipeline: Swiss Roll + Torus + Analysis"
echo "========================================================"
echo ""

echo "--- [1/3] Swiss roll experiment ---"
MANIFOLD=swiss julia --project="$PACKAGE_DIR" -t "${JULIA_NUM_THREADS:-auto}" \
    "$SCRIPT_DIR/experiment_orc.jl"

echo ""
echo "--- [2/3] Torus experiment ---"
MANIFOLD=torus julia --project="$PACKAGE_DIR" -t "${JULIA_NUM_THREADS:-auto}" \
    "$SCRIPT_DIR/experiment_orc.jl"

echo ""
echo "--- [3/3] Analysis ---"
python3 "$SCRIPT_DIR/analyze_orc.py"

echo ""
echo "========================================================"
echo "All done."
echo "========================================================"
