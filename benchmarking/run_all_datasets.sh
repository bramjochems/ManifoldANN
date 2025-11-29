#!/usr/bin/env bash

# Batch runner for ManifoldANN benchmarking datasets.
# Hardcoded paths so it can be moved around locally without relying on cwd.

set -euo pipefail

BENCH_DIR="/home/bram/projects/mai/thesis/code/ManifoldANN/benchmarking"
RESULTS_BASE="/home/bram/projects/mai/thesis/docs/thesis/results/ann_results"
CONFIG_DIR="$BENCH_DIR/configs"
OUTPUT_BASE="$BENCH_DIR/output"
DATA_DIR="$BENCH_DIR/data"
PYTHON_BIN="$BENCH_DIR/venv/bin/python"
VENV_ACTIVATE="$BENCH_DIR/venv/bin/activate"

# Override with e.g. N_TRAIN_OVERRIDE=100 to smoke test.
N_TRAIN="${N_TRAIN_OVERRIDE:-0}"
N_TEST="${N_TEST_OVERRIDE:-}"
K_VALUE="${K_OVERRIDE:-10}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python venv not found at $PYTHON_BIN" >&2
  exit 1
fi

# shellcheck source=benchmarking/venv/bin/activate
source "$VENV_ACTIVATE"

# Static list of datasets (kept small, easy to update).
DATASETS=(
  fashion-mnist
  mnist
  sift
  gist
  glove-25
  glove-50
  glove-100
  nytimes
  lastfm
)

for dataset in "${DATASETS[@]}"; do
  echo "=== Running dataset: $dataset (n_train=$N_TRAIN) ==="

  run_start=$(date +%s)

  cmd=(
    "$PYTHON_BIN" "$BENCH_DIR/benchmark.py" "$dataset"
    "--n-train" "$N_TRAIN"
    "--save-output"
    "--data-dir" "$DATA_DIR"
    "-k" "$K_VALUE"
  )

  if [ -n "$N_TEST" ]; then
    cmd+=( "--n-test" "$N_TEST" )
  fi

  "${cmd[@]}"

  latest_output=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name 'results_*' -printf '%T@ %p\n' \
    | awk -v start="$run_start" '$1 >= start {print $0}' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-)

  if [ -z "$latest_output" ]; then
    latest_output=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name 'results_*' -printf '%T@ %p\n' \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-)
  fi

  if [ -z "$latest_output" ]; then
    echo "Could not locate output directory for $dataset" >&2
    exit 1
  fi

  dest="$RESULTS_BASE/$dataset"
  mkdir -p "$dest"
  cp -r "$latest_output"/* "$dest"/

  echo "Saved results for $dataset to $dest"
done

echo "All datasets processed."
