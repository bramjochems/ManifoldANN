# Cloud benchmark runs

Fan out ANN benchmark shards across one Azure VM per shard, results to blob, VMs self-deallocate.

See `DESIGN.md` for full rationale.

## Pre-flight (one-time setup)

1. Create resource group, storage account, blob container.
2. Create a **user-assigned managed identity** and grant it:
   - `Storage Blob Data Contributor` on the storage account
   - `Virtual Machine Contributor` on the resource group
   Record its resource ID (`/subscriptions/.../userAssignedIdentities/<name>`).
3. Configure budget alerts on the RG ($50 / $100 / $200).
4. Make repo public so VMs can clone without credentials.
5. Pin dataset revisions in `scripts/cloud/datasets.lock` (replace `"main"` with concrete HF commit SHAs).
6. Pin the `libopenblas-dev` apt version in `bootstrap.sh` — search latest with:
   ```bash
   apt-cache madison libopenblas-dev   # on a jammy box
   ```
7. `git push` your run SHA before launching.

## Usage

```bash
python scripts/cloud/run_shards.py \
    --resource-group     ann-bench-thesis \
    --location           westeurope \
    --storage-account    annbenchresults \
    --storage-container  ann-bench-results \
    --vm-size            Standard_D16s_v6 \
    --git-sha            "$(git rev-parse HEAD)" \
    --shards-file        scripts/cloud/shards.example.json \
    --run-id             "$(date +%Y%m%d-%H%M)-thesis" \
    --user-assigned-identity-id /subscriptions/.../userAssignedIdentities/ann-bench-vm-identity \
    --repo-url           https://github.com/bramjochems/ManifoldANN.git
```

Add `--dry-run` to print cloud-init + `az vm create` commands without touching cloud.

## Local Docker test of bootstrap.sh

```bash
docker run --rm -it \
    -v "$PWD":/repo -w /repo \
    -e BOOTSTRAP_TEST=1 \
    -e RUN_ID=local-test \
    -e BLOB_CONTAINER_URL=https://example.blob.core.windows.net/test \
    -e UAMI_RESOURCE_ID=mock \
    -e KIND=ann-benchmark \
    -e SHARD_JSON='{"shard_id":"local","kind":"ann-benchmark","config_name":"fashion-mnist","threads":4,"reps":1}' \
    ubuntu:22.04 \
    bash scripts/cloud/bootstrap.sh
```

## Watching progress

```bash
az storage blob list \
    --account-name annbenchresults \
    --container-name ann-bench-results \
    --prefix runs/$RUN_ID/ -o table
```

## Post-run: concatenate shard CSVs

```python
import pandas as pd, glob
csvs = glob.glob("runs/*/output/results_*/*.csv")
pd.concat([pd.read_csv(c) for c in csvs]).to_csv("merged.csv", index=False)
```

## Teardown stale VMs

```bash
az vm list --tag run_id=<id> --query '[].id' -o tsv \
    | xargs -r az vm delete --yes --no-wait
```

## Adding a new workload kind

Drop `kinds/<kind>.sh` defining `run_shard()` (see `kinds/ann-benchmark.sh`). Add the kind name to `KNOWN_KINDS` in `run_shards.py`.
