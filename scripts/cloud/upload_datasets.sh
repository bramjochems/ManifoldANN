#!/bin/bash
# One-time: upload all local benchmarking/data/*.hdf5 files to blob.
# bootstrap.sh on each VM downloads from there at runtime.
#
# Run again whenever you add a new dataset locally that the cloud needs.

set -euo pipefail

SA="stbram"
CONTAINER="datasets"
DATA_DIR="$(dirname "$0")/../../benchmarking/data"

cd "$DATA_DIR"

for f in *.hdf5; do
    [ -f "$f" ] || continue
    echo "==> $f ($(du -h "$f" | cut -f1))"
    az storage blob upload --account-name "$SA" --container-name "$CONTAINER" \
        --auth-mode login --name "$f" --file "$f" --overwrite \
        --query "{name: name, size_mb: contentLength}" -o table 2>&1 | tail -2
done

echo
echo "==> All uploaded. Listing container:"
az storage blob list --account-name "$SA" --container-name "$CONTAINER" \
    --auth-mode login --query "[].{name:name, size_mb: properties.contentLength}" -o table
