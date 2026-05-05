#!/bin/bash
# bootstrap.sh — runs on each Azure VM after cloud-init clones the repo.
# Env vars expected (set by cloud-init): SHARD_JSON, RUN_ID, BLOB_CONTAINER_URL,
# UAMI_RESOURCE_ID, KIND.
# For local Docker testing: set BOOTSTRAP_TEST=1 to mock all `az` calls as echo.

set -eo pipefail

# ---------- az shim (mock under BOOTSTRAP_TEST=1) ----------
if [ "${BOOTSTRAP_TEST:-0}" = "1" ]; then
    az() { echo "[mock-az] $*"; }
    export -f az
    AZ() { timeout 60 bash -c 'az "$@"' _ "$@" || true; }
else
    AZ() { timeout 60 az "$@"; }
fi

# ---------- self-deallocate trap (uploads partials first) ----------
SHARD_ID=$(echo "${SHARD_JSON:-{}}" | jq -r '.shard_id // "unknown"')
WORK_DIR=$(pwd)

# Parse account + container from BLOB_CONTAINER_URL once.
# Expected form: https://<account>.blob.core.windows.net/<container>
SA_ACCOUNT=$(echo "$BLOB_CONTAINER_URL" | sed -E 's|https://([^.]+)\..*|\1|')
SA_CONTAINER=$(echo "$BLOB_CONTAINER_URL" | sed -E 's|https://[^/]+/([^/]+).*|\1|')

cleanup() {
    local rc=$?
    set +e
    echo "[bootstrap] cleanup (rc=$rc), uploading partials..."
    cd "$WORK_DIR" || true
    upload_artifacts || true

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

# Pin libopenblas — TODO: update version before first real run.
# Look up with: apt-cache madison libopenblas-dev
OPENBLAS_VER="0.3.20+ds-1ubuntu0.1"
apt-get install -y "libopenblas-dev=$OPENBLAS_VER" || apt-get install -y libopenblas-dev

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
if ! command -v juliaup >/dev/null 2>&1; then
    curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.10.5
    export PATH="$HOME/.juliaup/bin:$PATH"
fi
juliaup add 1.10.5 || true
juliaup default 1.10.5 || true

# ---------- uv + Python deps ----------
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd benchmarking
uv sync
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
    echo "=== git sha ==="; git rev-parse HEAD
} > metadata.txt 2>&1

# ---------- dataset download via datasets.lock ----------
DATASETS_LOCK="scripts/cloud/datasets.lock"
CONFIG_NAME=$(echo "$SHARD_JSON" | jq -r .config_name)
CONFIG_YAML="benchmarking/configs/${CONFIG_NAME}.yaml"

if [ -f "$CONFIG_YAML" ]; then
    DATASET_NAME=$(grep -E "^dataset:" "$CONFIG_YAML" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$DATASET_NAME" ]; then
        REVISION=$(jq -r ".datasets[\"$DATASET_NAME\"].revision // \"main\"" "$DATASETS_LOCK")
        REPO=$(jq -r ".datasets[\"$DATASET_NAME\"].repo // \"ann-benchmarks/ann-benchmarks\"" "$DATASETS_LOCK")
        echo "=== dataset pin ===" >> metadata.txt
        echo "name=$DATASET_NAME repo=$REPO revision=$REVISION" >> metadata.txt
        # The benchmark harness downloads on demand; here we just pre-warm via
        # the HF hub if the repo wants. One retry, then proceed (harness will
        # also try). Skip in test mode.
        if [ "${BOOTSTRAP_TEST:-0}" != "1" ]; then
            for i in 1 2; do
                python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO', repo_type='dataset', revision='$REVISION', allow_patterns=['*${DATASET_NAME}*'])" && break
                echo "[bootstrap] dataset prefetch try $i failed"
                sleep 10
            done || true
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
