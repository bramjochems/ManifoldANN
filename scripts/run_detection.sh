#!/usr/bin/env bash
# Run detection-only experiment (SKIP_PRUNING=1) for both manifolds.
#
# Usage:
#   ./scripts/run_detection.sh
#   SMOKE=1 ./scripts/run_detection.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export SKIP_PRUNING=1

echo "========================================================"
echo "ORC Detection Only: Swiss Roll + Torus"
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
