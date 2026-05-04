#!/usr/bin/env bash
# Run the ORC sampling diagnostic experiment and generate visualisations.
#
# Usage:
#   ./scripts/thesis/orc/run_orc_sampling_diagnostic.sh        # full run then plot
#   SMOKE=1 ./scripts/thesis/orc/run_orc_sampling_diagnostic.sh   # quick smoke test
#   SKIP_PLOT=1 ./scripts/thesis/orc/run_orc_sampling_diagnostic.sh   # Julia only
#
# Results land in:
#   docs/thesis/results/orc_results/sampling_diag_<timestamp>/
#   Set RESUME_DIR=path to resume a previous run.
# Figures land in:
#   docs/thesis/figures/orc_sampling_diagnostic_{heatmap,scatter,delta_kappa}.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
JULIA_SCRIPT="$SCRIPT_DIR/experiment_orc_sampling_diagnostic.jl"
PLOT_SCRIPT="$SCRIPT_DIR/plot_sampling_diagnostic.py"

export SMOKE="${SMOKE:-}"
export RESUME_DIR="${RESUME_DIR:-}"

echo "========================================================"
echo "ORC Sampling Diagnostic Experiment"
echo "Package  : $PACKAGE_DIR"
echo "Script   : $JULIA_SCRIPT"
echo "Threads  : ${JULIA_NUM_THREADS:-auto}"
echo "========================================================"

# Run Julia experiment
julia \
    --project="$PACKAGE_DIR" \
    -t "${JULIA_NUM_THREADS:-auto}" \
    "$JULIA_SCRIPT"

# Run visualisation unless disabled
if [ "${SKIP_PLOT:-}" != "1" ]; then
    echo ""
    echo "========================================================"
    echo "Generating visualisations..."
    echo "========================================================"
    python3 "$PLOT_SCRIPT"
fi
