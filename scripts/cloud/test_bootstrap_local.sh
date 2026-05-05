#!/bin/bash
# End-to-end local test of bootstrap.sh in Ubuntu 22.04 Docker.
#
# Runs bootstrap.sh with REAL apt/uv/julia/harness execution.
# Mocks ONLY what requires Azure: az calls and the blob dataset
# download (substitutes a local fashion-mnist.hdf5 copy).
#
# Catches everything the BOOTSTRAP_TEST=1 mocked path misses:
# - missing apt deps (python3-dev for annoy, etc.)
# - HOME/USER unset under cloud-init
# - uv sync extras
# - juliacall/Julia integration
# - dataset path the harness expects
# - actual benchmark execution
#
# Iterate on this until it prints a results.csv path. Only then
# trust an Azure run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="$REPO_ROOT/benchmarking/data/fashion-mnist-784-euclidean.hdf5"

if [ ! -f "$DATASET" ]; then
    echo "Need $DATASET to exist locally for the test." >&2
    exit 1
fi

# We mock az with a small wrapper that:
# - blob download <dataset>.hdf5 -> copy from /datasets-mock/<name>
# - vm deallocate -> no-op
# - blob upload -> log + copy to /tmp/blob-out/
# - blob list -> always succeeds
cat > /tmp/az-mock.sh <<'AZMOCK'
#!/bin/bash
# Mock subset of `az` that bootstrap.sh actually uses.
case "$1 $2" in
    "storage blob")
        case "$3" in
            download)
                # Find --name X --file Y
                NAME=""; FILE=""
                while [ $# -gt 0 ]; do
                    case "$1" in
                        --name) NAME="$2"; shift 2 ;;
                        --file) FILE="$2"; shift 2 ;;
                        *) shift ;;
                    esac
                done
                if [ -f "/datasets-mock/$NAME" ]; then
                    cp "/datasets-mock/$NAME" "$FILE"
                    echo "[az-mock] copied $NAME -> $FILE"
                    exit 0
                fi
                echo "[az-mock] $NAME not in /datasets-mock" >&2
                exit 1 ;;
            upload|upload-batch)
                echo "[az-mock] $@"
                exit 0 ;;
            list)
                echo '[]'  # empty list, JSON
                exit 0 ;;
        esac
        ;;
    "vm deallocate")
        echo "[az-mock] vm deallocate (no-op in test)"
        exit 0 ;;
    "login --identity"|"login")
        echo "[az-mock] login (no-op)"
        exit 0 ;;
esac
echo "[az-mock] UNHANDLED: $@" >&2
exit 0  # don't break bootstrap on unhandled calls
AZMOCK
chmod +x /tmp/az-mock.sh

echo "==> Building / pulling Ubuntu 22.04 image..."
docker pull ubuntu:22.04 >/dev/null

echo "==> Running bootstrap.sh end-to-end in container..."
docker run --rm \
    -v "$REPO_ROOT:/repo:ro" \
    -v "$(dirname "$DATASET"):/datasets-mock:ro" \
    -v /tmp/az-mock.sh:/usr/local/bin/az:ro \
    -e RUN_ID=local-e2e \
    -e BLOB_CONTAINER_URL=https://stbram.blob.core.windows.net/mai-thesis \
    -e UAMI_RESOURCE_ID=mock \
    -e KIND=ann-benchmark \
    -e SHARD_JSON='{"shard_id":"local","kind":"ann-benchmark","config_name":"refactor-baseline","threads":4,"reps":1}' \
    --entrypoint bash \
    ubuntu:22.04 \
    -c '
        set -e
        unset HOME USER  # mimic cloud-init runcmd

        # rsync the repo to a writable location, EXCLUDING benchmarking/output/
        # so the container does not see prior local runs.
        apt-get update -qq && apt-get install -y -qq rsync >/dev/null
        rsync -a --exclude="benchmarking/output" --exclude=".venv" \
              /repo/ /opt/repo/
        cd /opt/repo
        bash scripts/cloud/bootstrap.sh > /tmp/bootstrap.log 2>&1
        rc=$?
        echo "==> Bootstrap exit code: $rc"
        echo "==> Last 30 lines of bootstrap log:"
        tail -30 /tmp/bootstrap.log
        echo "==> Results CSVs produced:"
        if [ -d benchmarking/output ]; then
            find benchmarking/output -name "results.csv" | head -5
            echo "--- contents of first results.csv ---"
            find benchmarking/output -name "results.csv" -exec cat {} \; | head -20
        else
            echo "NO benchmarking/output DIRECTORY"
        fi
        exit $rc
    '
