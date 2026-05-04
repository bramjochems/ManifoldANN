#!/usr/bin/env bash
# Run pruning-only experiment (SKIP_DETECTION=1) for both manifolds.
# Still computes ORC (needed for rank-based pruning) but skips F1/edge output.
#
# Usage:
#   ./scripts/thesis/orc/run_pruning.sh
#   SMOKE=1 ./scripts/thesis/orc/run_pruning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export SKIP_DETECTION=1

echo "========================================================"
echo "ORC Pruning Only: Swiss Roll + Torus"
echo "========================================================"
echo ""

echo "--- [1/2] Swiss roll ---"
MANIFOLD=swiss julia --project="$PACKAGE_DIR" -t "${JULIA_NUM_THREADS:-auto}" \
    "$SCRIPT_DIR/experiment_orc.jl"

echo ""
echo "--- [2/2] Torus ---"
MANIFOLD=torus julia --project="$PACKAGE_DIR" -t "${JULIA_NUM_THREADS:-auto}" \
    "$SCRIPT_DIR/experiment_orc.jl"

echo ""
echo "Done."
