#!/usr/bin/env bash
#
# Fetch or update the ann-benchmarks repository used by the benchmarking harness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/ann-benchmarks"
REPO_URL="https://github.com/erikbern/ann-benchmarks.git"
PINNED_COMMIT="f402b2cc17b980d7cd45241ab5a7a4cc0f965e55"

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required to fetch ann-benchmarks." >&2
    exit 1
fi

if [ -d "${TARGET_DIR}/.git" ]; then
    echo "Updating existing ann-benchmarks clone at ${TARGET_DIR}..."
    git -C "${TARGET_DIR}" fetch --tags origin
else
    if [ -e "${TARGET_DIR}" ]; then
        echo "Error: ${TARGET_DIR} exists but is not a git repository." >&2
        echo "Remove or rename it before running this script." >&2
        exit 1
    fi
    echo "Cloning ann-benchmarks into ${TARGET_DIR}..."
    git clone "${REPO_URL}" "${TARGET_DIR}"
fi

echo "Checking out pinned commit ${PINNED_COMMIT}..."
git -C "${TARGET_DIR}" checkout --quiet "${PINNED_COMMIT}"
echo "ann-benchmarks ready."
