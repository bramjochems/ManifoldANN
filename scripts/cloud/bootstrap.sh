#!/bin/bash
# bootstrap.sh - runs on each Azure VM after cloud-init clones the repo.
#
# Env vars expected (set by cloud-init):
#   - SHARDS_JSON: JSON array of shard objects (multi-shard mode), OR
#   - SHARD_JSON:  single shard object (legacy single-shard mode, smoke tests)
#   - RUN_ID, BLOB_CONTAINER_URL, UAMI_RESOURCE_ID
#
# In multi-shard mode, the VM runs each shard's KIND dispatcher in turn,
# uploading per-shard artifacts before moving to the next. Install + Julia
# env + datasets are set up ONCE; subsequent shards reuse them.
#
# For local Docker testing: set BOOTSTRAP_TEST=1 to mock all `az` calls.

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
# SHARD_ID is set per-shard inside the dispatch loop. Default to "init"
# so any artifacts written before the first shard starts (e.g. from the
# trap firing during apt install) land somewhere identifiable.
SHARD_ID="init"

cleanup() {
    local rc=$?
    set +e
    echo "[bootstrap] cleanup (rc=$rc), uploading partials for SHARD_ID=$SHARD_ID..."
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
    python3 python3-venv python3-pip python3-dev \
    pkg-config
# python3-dev is required because annoy ships as sdist (no wheel) and
# compiles annoymodule.cc against Python.h at install time. Without
# python3-dev, `uv sync --extra ann-complete` fails with:
#   src/annoymodule.cc:17:10: fatal error: Python.h: No such file or directory

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

# The MANN Python wrappers do `Pkg.activate(/opt/repo)` and `using ManifoldANN`.
# This fails if the main Julia env hasn't been instantiated against the same
# Julia juliacall picks at runtime (juliapkg manages its own Julia, often
# different from the juliaup default we installed above).
#
# The committed Manifest.toml was generated under Julia 1.12 but juliapkg
# typically resolves to 1.11.x. Stdlib UUIDs (Statistics, etc.) differ across
# versions, so loading a 1.12 Manifest under 1.11 fails with
# "failed to find source of parent package: Statistics".
#
# Use whatever julia juliacall picked (it was added to ~/.juliaup by juliapkg)
# and regenerate the Manifest fresh against that version.
echo "[bootstrap] instantiating ManifoldANN Julia env (via juliacall's Julia)..."
JULIACALL_JULIA=$(find "$HOME/.julia/juliaup" -name julia -type f -executable 2>/dev/null | head -1)
if [ -z "$JULIACALL_JULIA" ]; then
    JULIACALL_JULIA=$(command -v julia)
fi
echo "[bootstrap] using $JULIACALL_JULIA for instantiate"
# Delete stale Manifest so Pkg resolves fresh against this Julia version.
rm -f /opt/repo/Manifest.toml
"$JULIACALL_JULIA" --project=/opt/repo -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    using ManifoldANN
    println("[manifoldann] loaded OK")
' 2>&1 | tail -30 || echo "[bootstrap] WARN: ManifoldANN instantiate had errors"

# Also instantiate benchmarking/julia/ which is the SEPARATE env that
# julia_external.py activates for HNSW.jl, NND.jl, NearestNeighbors.jl
# wrappers. Without this, those wrappers' is_available() fail silently
# and tier-2 head-to-head shards (HNSW-jl, NND.jl) skip every algorithm
# with "library not available". Same Manifest.toml drift mitigation:
# delete and let Pkg resolve fresh against juliapkg's Julia.
echo "[bootstrap] instantiating benchmarking/julia env (for HNSW.jl, NND.jl)..."
rm -f /opt/repo/benchmarking/julia/Manifest.toml
"$JULIACALL_JULIA" --project=/opt/repo/benchmarking/julia -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    using HNSW, NearestNeighborDescent, NearestNeighbors
    println("[external-julia] loaded OK")
' 2>&1 | tail -30 || echo "[bootstrap] WARN: benchmarking/julia instantiate had errors"

# Force-create the juliacall-managed env by triggering one juliacall import
# now, BEFORE any shard runs. This ensures the env exists at a known state
# and any package extensions (StructUtilsStaticArraysCoreExt etc.) get
# resolved while we're still in setup, not during the timed harness.
echo "[bootstrap] warming juliacall env..."
(cd benchmarking && .venv/bin/python -c "
from juliacall import Main as jl
jl.seval('using Pkg; Pkg.add(\"StaticArraysCore\"); Pkg.precompile()')
print('[juliacall-env] precompiled OK')
" 2>&1 | tail -20) || echo "[bootstrap] WARN: juliacall env warm had errors"

# ---------- (legacy single-shard metadata removed; per-shard metadata now
#            written inside the dispatch loop below) ----------

# ---------- shard list normalisation ----------
# Accept either SHARDS_JSON (array, multi-shard mode) or SHARD_JSON (single,
# legacy/smoke mode). Internally we always work with an array.
if [ -n "${SHARDS_JSON:-}" ]; then
    SHARDS_LIST="$SHARDS_JSON"
else
    SHARDS_LIST=$(echo "${SHARD_JSON:-{}}" | jq -c '[.]')
fi
N_SHARDS=$(echo "$SHARDS_LIST" | jq 'length')
echo "[bootstrap] received $N_SHARDS shard(s)"

# ---------- dataset prefetch (download all unique datasets up front) ----------
# Loop over the shards' configs and figure out which datasets are needed.
# Download each ONCE; subsequent shards using the same dataset reuse the
# cached file (the harness's "if exists" short-circuit kicks in).
DATASETS_TO_FETCH=$(
    for i in $(seq 0 $((N_SHARDS - 1))); do
        cfg=$(echo "$SHARDS_LIST" | jq -r ".[$i].config_name")
        yml="benchmarking/configs/${cfg}.yaml"
        if [ -f "$yml" ]; then
            grep -E "^dataset:" "$yml" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'"
        fi
    done | sort -u
)
echo "[bootstrap] datasets to fetch: $(echo "$DATASETS_TO_FETCH" | tr '\n' ' ')"

if [ "${BOOTSTRAP_TEST:-0}" != "1" ]; then
    mkdir -p benchmarking/data
    for ds in $DATASETS_TO_FETCH; do
        [ -z "$ds" ] && continue
        dest="benchmarking/data/${ds}.hdf5"
        if [ -f "$dest" ]; then
            echo "[bootstrap] dataset already cached: $dest"
            continue
        fi
        for try in 1 2 3; do
            if AZ storage blob download --auth-mode login \
                   --account-name "$SA_ACCOUNT" --container-name datasets \
                   --name "${ds}.hdf5" --file "$dest" --no-progress; then
                echo "[bootstrap] dataset cached: $dest"
                break
            fi
            echo "[bootstrap] dataset $ds try $try failed"
            sleep 15
        done
        if [ ! -f "$dest" ]; then
            echo "[bootstrap] FATAL: dataset $ds not found in blob after retries"
            exit 1
        fi
    done
fi

# ---------- shard execution loop ----------
# Each shard runs the harness once, uploads its results to a per-shard blob
# prefix, then we move on to the next. Failure of one shard does NOT abort
# the others — partial results still upload via the trap.
for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD=$(echo "$SHARDS_LIST" | jq -c ".[$i]")
    SHARD_ID=$(echo "$SHARD" | jq -r '.shard_id')
    KIND=$(echo "$SHARD" | jq -r '.kind')

    echo
    echo "============================================================"
    echo "[bootstrap] shard $((i+1))/$N_SHARDS: $SHARD_ID (kind=$KIND)"
    echo "============================================================"

    # Reset workspace state between shards: clear stale output + run.log
    # so the upload only picks up THIS shard's artifacts.
    rm -rf "$WORK_DIR/benchmarking/output"
    rm -f  "$WORK_DIR/run.log"

    # Per-shard metadata.txt — overwrite (not append) the global one so each
    # upload reflects this shard's context.
    {
        echo "=== shard ===";       echo "$SHARD"
        echo "=== run id ===";      echo "$RUN_ID"
        echo "=== git sha ===";     git rev-parse HEAD 2>&1 || echo "(not a git repo)"
        echo "=== started_at ==="; date -u +"%Y-%m-%dT%H:%M:%SZ"
        echo "=== lscpu ===";       lscpu
        echo "=== uname ===";       uname -a
        echo "=== blas pkgs ===";   dpkg -l | grep -E "blas|lapack|gomp" || true
    } > "$WORK_DIR/metadata.txt" 2>&1 || true

    KIND_SCRIPT="$WORK_DIR/scripts/cloud/kinds/${KIND}.sh"
    if [ ! -f "$KIND_SCRIPT" ]; then
        echo "[bootstrap] unknown kind: $KIND — skipping shard"
        SHARD_ID="$SHARD_ID-error"
        upload_artifacts || true
        continue
    fi
    # shellcheck disable=SC1090
    source "$KIND_SCRIPT"

    cd "$WORK_DIR"
    # Run the shard. Trap `set -e` for this iteration only — we don't want
    # one shard's crash to skip the others.
    set +e
    run_shard "$SHARD"
    shard_rc=$?
    set -e
    echo "[bootstrap] shard $SHARD_ID exit code: $shard_rc"

    # Upload this shard's artifacts before moving on.
    cd "$WORK_DIR"
    upload_artifacts || true
done

echo "[bootstrap] all shards complete"
# Trap will deallocate.
