#!/bin/bash
# Dispatch for kind=ann-benchmark. Sourced by bootstrap.sh.

run_shard() {
    local shard_json="$1"
    local config_name threads reps
    config_name=$(echo "$shard_json" | jq -r .config_name)
    threads=$(echo "$shard_json" | jq -r .threads)
    reps=$(echo "$shard_json" | jq -r .reps)

    cd benchmarking
    JULIA_NUM_THREADS="$threads" .venv/bin/python benchmark.py \
        "$config_name" --threads "$threads" --reps "$reps" --save-output 2>&1 \
        | tee ../run.log
}
