#!/bin/bash
# bootstrap.sh - runs on each Azure VM after cloud-init clones the repo.
# Env vars expected (set by cloud-init): SHARD_JSON, RUN_ID, BLOB_CONTAINER_URL,
# UAMI_RESOURCE_ID, KIND.
# For local Docker testing: set BOOTSTRAP_TEST=1 to mock all `az` calls as echo.

set -eo pipefail

# Cloud-init runcmd doesn't set $HOME — running as root with $HOME unset
# means $HOME/.juliaup/bin etc. resolve to "/.juliaup/bin", which doesn't
# exist. Pin HOME and USER explicitly before anything depends on them.
export HOME="${HOME:-/root}"
export USER="${USER:-root}"

# Register the deallocate trap FIRST, before anything that could fail.
# This is the safety net: even if a downstream step crashes, we still
# deallocate the VM so the user doesn't pay for an idle host.
deallocate_self() {
    if [ "${BOOTSTRAP_TEST:-0}" = "1" ]; then
        echo "[bootstrap] BOOTSTRAP_TEST=1, skipping deallocate"
        return
    fi
    local rid
    rid=$(timeout 30 curl -s -H Metadata:true \
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01" \
        | jq -r .compute.resourceId 2>/dev/null || true)
    if [ -n "$rid" ]; then
        timeout 60 az vm deallocate --ids "$rid" --no-wait || true
    fi
}
cleanup_minimal() {
    local rc=$?
    set +e
    echo "[bootstrap] cleanup_minimal (rc=$rc) - deallocate only"
    deallocate_self
}
trap cleanup_minimal EXIT

# ---------- az shim (mock under BOOTSTRAP_TEST=1) ----------
if [ "${BOOTSTRAP_TEST:-0}" = "1" ]; then
    az() { echo "[mock-az] $*"; }
    export -f az
    AZ() { timeout 60 bash -c 'az "$@"' _ "$@" || true; }
else
    AZ() { timeout 60 az "$@"; }
fi

# Parse the container URL up front (no jq needed for sed parsing).
WORK_DIR=$(pwd)
SA_ACCOUNT=$(echo "$BLOB_CONTAINER_URL" | sed -E 's|https://([^.]+)\..*|\1|')
SA_CONTAINER=$(echo "$BLOB_CONTAINER_URL" | sed -E 's|https://[^/]+/([^/]+).*|\1|')
# SHARD_ID needs jq — set later after apt installs it. Default to a safe
# placeholder so the upload paths under the early-exit trap don't break.
SHARD_ID="unknown"

cleanup() {
    local rc=$?
    set +e
    echo "[bootstrap] cleanup (rc=$rc), uploading partials..."
    cd "$WORK_DIR" || true
    upload_artifacts || true
    deallocate_self
}
# Replace the minimal trap with the full one now that we have the
# account info needed to upload partials.
trap cleanup EXIT

upload_artifacts() {
    local prefix="runs/$RUN_ID/$SHARD_ID"
    if [ -f metadata.txt ]; then
        AZ storage blob upload --auth-mode login \
            --account-name "$SA_ACCOUNT" --container-name "$SA_CONTAINER" \
            --name "$prefix/metadata.txt" --file metadata.txt --overwrite || true
    fi
    if [ -f run.log ]; then
        AZ storage blob upload --auth-mode login \
            --account-name "$SA_ACCOUNT" --container-name "$SA_CONTAINER" \
            --name "$prefix/run.log" --file run.log --overwrite || true
    fi
    if [ -d benchmarking/output ]; then
        AZ storage blob upload-batch --auth-mode login \
            --account-name "$SA_ACCOUNT" \
            --destination "$SA_CONTAINER" \
            --destination-path "$prefix/output" \
            --source benchmarking/output --overwrite || true
    fi
}

# ---------- system deps ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    git curl jq ca-certificates build-essential \
    python3 python3-venv python3-pip \
    pkg-config

# Install latest available libopenblas-dev. We don't pin a specific version
# because pinning to a not-yet-existing-on-the-image version means every
# fresh VM apt-fails. Reproducibility comes from dpkg -l snapshot in
# metadata.txt; if BLAS drift is observed across runs, pin retroactively.
apt-get install -y libopenblas-dev

# Now that jq is installed, resolve SHARD_ID from the JSON env var.
# Falls back to "unknown" if the env var is unset/malformed.
SHARD_ID=$(echo "${SHARD_JSON:-}" | jq -r '.shard_id // "unknown"' 2>/dev/null || echo "unknown")
echo "[bootstrap] SHARD_ID=$SHARD_ID"

# ---------- wait-and-retry first az blob op (defensive RBAC propagation) ----------
if [ "${BOOTSTRAP_TEST:-0}" != "1" ]; then
    for i in 1 2 3 4 5; do
        if AZ storage blob list --auth-mode login \
              --account-name "$SA_ACCOUNT" --container-name "$SA_CONTAINER" \
              --num-results 1 >/dev/null 2>&1; then
            echo "[bootstrap] blob auth OK on try $i"
            break
        fi
        echo "[bootstrap] blob auth try $i failed, sleeping 30s"
        sleep 30
    done
fi

# ---------- juliaup + Julia 1.10.5 ----------
# install.julialang.org modifies shell rc files but does NOT add to the
# current shell's PATH. We add absolute path explicitly and verify the
# install actually succeeded before depending on it.
JULIAUP_BIN="$HOME/.juliaup/bin"
if [ ! -x "$JULIAUP_BIN/juliaup" ]; then
    # The Azure Ubuntu image sometimes ships with a stale ~/.julia/juliaup
    # config from prior provisioning, which makes the installer abort with
    # "Please remove the existing Juliaup configuration file". Wipe it.
    rm -rf "$HOME/.julia/juliaup" "$HOME/.juliaup"
    echo "[bootstrap] installing juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.10.5
fi
if [ ! -x "$JULIAUP_BIN/juliaup" ]; then
    echo "[bootstrap] FATAL: juliaup install failed; $JULIAUP_BIN/juliaup not found"
    ls -la "$HOME/.juliaup" 2>&1 || true
    exit 1
fi
export PATH="$JULIAUP_BIN:$PATH"
juliaup add 1.10.5 || true
juliaup default 1.10.5 || true
echo "[bootstrap] julia version: $(julia --version 2>&1 || echo unknown)"

# ---------- uv + Python deps ----------
UV_BIN="$HOME/.local/bin"
if [ ! -x "$UV_BIN/uv" ]; then
    echo "[bootstrap] installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
fi
if [ ! -x "$UV_BIN/uv" ]; then
    echo "[bootstrap] FATAL: uv install failed; $UV_BIN/uv not found"
    ls -la "$UV_BIN" 2>&1 || true
    exit 1
fi
export PATH="$UV_BIN:$PATH"
echo "[bootstrap] uv version: $(uv --version 2>&1 || echo unknown)"

cd benchmarking
# `--extra ann-complete` pulls hnswlib, faiss-cpu, scipy, annoy, pynndescent
# (all the comparison libraries the harness's wrappers expect). Without
# this, every ANN wrapper's is_available() returns False and the harness
# reports "Skipped (library not available)" for everything.
uv sync --extra ann-complete
cd ..

# ---------- metadata snapshot ----------
{
    echo "=== lscpu ==="; lscpu
    echo "=== lsb_release ==="; lsb_release -a 2>/dev/null || cat /etc/os-release
    echo "=== uname ==="; uname -a
    echo "=== cpuinfo (head 30) ==="; head -30 /proc/cpuinfo
    echo "=== blas/lapack/gomp packages ==="; dpkg -l | grep -E "blas|lapack|gomp" || true
    echo "=== Julia BLAS config ==="
    julia -e 'using LinearAlgebra; println(BLAS.get_config())' 2>&1 || true
    echo "=== uv pip list ==="
    (cd benchmarking && uv pip list) 2>&1 || true
    echo "=== shard json ==="; echo "$SHARD_JSON"
    echo "=== run id ==="; echo "$RUN_ID"
    echo "=== git sha ==="; git rev-parse HEAD 2>&1 || echo "(not a git repo)"
} > metadata.txt 2>&1 || true

# ---------- dataset download via datasets.lock ----------
DATASETS_LOCK="scripts/cloud/datasets.lock"
CONFIG_NAME=$(echo "$SHARD_JSON" | jq -r .config_name)
CONFIG_YAML="benchmarking/configs/${CONFIG_NAME}.yaml"

if [ -f "$CONFIG_YAML" ]; then
    DATASET_NAME=$(grep -E "^dataset:" "$CONFIG_YAML" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$DATASET_NAME" ]; then
        echo "=== dataset ===" >> metadata.txt
        echo "name=$DATASET_NAME source=blob://$SA_ACCOUNT/datasets/${DATASET_NAME}.hdf5" >> metadata.txt
        # Fetch the HDF5 from our own blob (uploaded once from laptop) into
        # benchmarking/data/<name>.hdf5 so the harness short-circuits its
        # built-in downloader. ann-benchmarks.com rate-limits/403s Azure
        # outbound IPs; HuggingFace ann-benchmarks repo requires auth. Blob
        # storage is the only path we control end-to-end.
        if [ "${BOOTSTRAP_TEST:-0}" != "1" ]; then
            mkdir -p benchmarking/data
            DEST="benchmarking/data/${DATASET_NAME}.hdf5"
            for i in 1 2 3; do
                if AZ storage blob download --auth-mode login \
                       --account-name "$SA_ACCOUNT" --container-name datasets \
                       --name "${DATASET_NAME}.hdf5" --file "$DEST" \
                       --no-progress; then
                    echo "[bootstrap] dataset cached: $DEST"
                    break
                fi
                echo "[bootstrap] dataset download try $i failed"
                sleep 15
            done
            if [ ! -f "$DEST" ]; then
                echo "[bootstrap] FATAL: dataset $DATASET_NAME not found in blob after retries"
                exit 1
            fi
        fi
    fi
fi

# ---------- dispatch ----------
KIND_SCRIPT="scripts/cloud/kinds/${KIND}.sh"
if [ ! -f "$KIND_SCRIPT" ]; then
    echo "[bootstrap] unknown kind: $KIND" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$KIND_SCRIPT"
run_shard "$SHARD_JSON"

# Trap handles upload + deallocate.
echo "[bootstrap] shard complete"
