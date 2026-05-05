# Cloud benchmark runs — design

Goal: run thesis ANN benchmarks across N Azure VMs in parallel, results to blob storage, no laptop dependency, no orphan billing.

## Components

```
scripts/cloud/
  DESIGN.md            (this file)
  run_shards.py        local CLI: dispatches shards to VMs
  bootstrap.sh         runs on each VM: env setup, dispatch, upload, deallocate
  kinds/
    ann-benchmark.sh   workload-specific dispatch (sourced by bootstrap.sh)
    # orc-compute.sh   added later for ORC compute, same pattern
  shards.example.json  example shard manifest
  datasets.lock        pinned HuggingFace dataset revisions
```

Per-kind logic lives in `kinds/<kind>.sh`. `bootstrap.sh` sources the right one based on the shard's `kind` field, so adding a new workload type is one new file plus a one-line dispatch entry.

## Shard manifest

`shards.json` — JSON array, one element per shard. Hand-written or generated.

```json
[
  {
    "shard_id": "sift-mann-hnsw",
    "kind": "ann-benchmark",
    "config_name": "sift",
    "threads": 16,
    "reps": 5
  },
  {
    "shard_id": "nytimes-hnsw-headtohead",
    "kind": "ann-benchmark",
    "config_name": "nytimes-hnsw-headtohead",
    "threads": 16,
    "reps": 3
  }
]
```

`kind` selects the dispatch target inside `bootstrap.sh`. `ann-benchmark` runs `python benchmark.py $config_name --threads $threads --reps $reps --save-output`. Other kinds (e.g. `orc-compute`) added later by extending the dispatch case.

## Local CLI: `run_shards.py`

```
python scripts/cloud/run_shards.py \
    --resource-group ann-bench-thesis \
    --location westeurope \
    --storage-account <name> \
    --storage-container ann-bench-results \
    --vm-size Standard_D16s_v6 \
    --image Ubuntu2204 \
    --git-sha $(git rev-parse HEAD) \
    --shards-file scripts/cloud/shards.json \
    --run-id $(date +%Y%m%d-%H%M)-thesis \
    [--dry-run]
```

Behaviour:
1. Validate that the git SHA is pushed to origin (else VM clone will fail).
2. Reject if `runs/<run_id>/manifest.json` already exists in blob (prevents silent overwrite of a previous run with the same id).
3. Validate shard manifest: each `shard_id` is unique and a valid filename.
4. Write `runs/<run_id>/manifest.json` to blob with the shard list, git SHA, VM size, region, started_at.
5. For each shard, fire `az vm create` in parallel with cloud-init that:
   - Tags the VM with `run_id` and `shard_id`.
   - Templates in: shard JSON inline, run_id, container URL, git SHA.
   - Assigns a **user-assigned** managed identity created once beforehand (granted `Storage Blob Data Contributor` on the storage account + `Virtual Machine Contributor` on its own RG). User-assigned avoids the system-assigned RBAC propagation race that can cause 403 on the first blob upload.
   - Sets `--auto-shutdown` to 60 minutes after creation as backstop in case bootstrap.sh's deallocate path fails.
6. Exit. Does not wait. User checks `az storage blob list` to see results trickle in.

`--dry-run` prints the per-shard `az vm create` commands and exits.

## VM bootstrap: `bootstrap.sh`

Cloud-init runs a small inline preamble:
```bash
#!/bin/bash
# Self-deallocate even if pre-bootstrap setup fails (forgot-to-push,
# apt mirror down, network glitch). The 60min auto-shutdown is the
# final backstop. Each az call wrapped in `timeout` so a hung IMDS
# can't block deallocation forever.
trap 'timeout 60 az vm deallocate --ids $(timeout 30 curl -s -H Metadata:true http://169.254.169.254/metadata/instance?api-version=2021-02-01 | jq -r .compute.resourceId) --no-wait || true' EXIT

set -e
apt update && apt install -y git jq curl
az login --identity --username $UAMI_RESOURCE_ID
git clone $REPO_URL repo
cd repo
git checkout $GIT_SHA
exec bash scripts/cloud/bootstrap.sh
```

`bootstrap.sh` (versioned in repo, takes over from cloud-init) reads env vars (`SHARD_JSON`, `RUN_ID`, `BLOB_CONTAINER_URL`, `UAMI_RESOURCE_ID`) and:

1. Re-arm the `trap` to also upload partial logs/CSVs before deallocating (cloud-init's trap only deallocates).
2. Wait-and-retry pattern for the first `az` blob op (5 tries, 30s backoff) — defensive even with user-assigned identity.
3. Install Julia (`juliaup add 1.10.5; juliaup default 1.10.5`), Python venv via `uv sync` from `benchmarking/uv.lock`.
4. Pin system BLAS: `apt install -y libopenblas-dev=<pinned version>` (version locked via `apt-cache madison` lookup committed in `bootstrap.sh`).
5. Capture host metadata once: `lscpu`, `lsb_release -a`, `uname -a`, `cat /proc/cpuinfo | head -30`, `dpkg -l | grep -E 'blas|lapack|gomp'`, `julia -e 'using LinearAlgebra; println(BLAS.get_config())'`, `pip list`. Write to `metadata.txt`.
6. Download dataset(s) using pinned revisions from `scripts/cloud/datasets.lock` (one HF revision SHA per dataset). Capture resolved revisions into `metadata.txt`.
7. Dispatch by `kind`: `source scripts/cloud/kinds/$KIND.sh` and call `run_shard "$SHARD_JSON"`.
8. Upload `output/results_*/` + `metadata.txt` + `run.log` to `runs/$RUN_ID/$SHARD_ID/`.
9. Deallocate self via IMDS-derived `az vm deallocate --no-wait` (the trap also covers this).

### `kinds/ann-benchmark.sh`
```bash
run_shard() {
    local shard_json="$1"
    local config_name=$(echo "$shard_json" | jq -r .config_name)
    local threads=$(echo "$shard_json" | jq -r .threads)
    local reps=$(echo "$shard_json" | jq -r .reps)
    cd benchmarking
    JULIA_NUM_THREADS=$threads .venv/bin/python benchmark.py \
        "$config_name" --threads "$threads" --reps "$reps" --save-output 2>&1 \
        | tee ../run.log
}
```

## Blob layout

```
ann-bench-results/
  runs/
    <run_id>/
      manifest.json
      <shard_id>/
        metadata.txt          (lscpu, BLAS config, kernel)
        run.log               (full stdout/stderr)
        results.csv           (the harness output)
        results.json
        config.yaml           (copy of the YAML the shard ran)
```

Concatenation script (post-processing, separate, simple): `glob runs/<run_id>/*/results.csv`, concat with pandas, write merged CSV.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Script crashes mid-shard | `trap` uploads partial CSV + log, deallocates |
| Script can't deallocate (network/auth fail) | VM-level auto-shutdown set to 60 min after creation as backstop; cloud-init trap covers pre-bootstrap.sh failures (forgot-to-push, apt mirror down) |
| Identity RBAC propagation race | User-assigned managed identity (created once with role pre-assigned) avoids the system-assigned 30-90s lag |
| HuggingFace dataset silently updated mid-run | `datasets.lock` pins HF revision SHA per dataset; resolved revisions captured in `metadata.txt` |
| System BLAS drift between VMs | `apt install -y libopenblas-dev=<pinned>`; `dpkg -l` snapshot in metadata for post-hoc drift detection |
| Re-running with same `run_id` overwrites prior results | `run_shards.py` rejects if `runs/<run_id>/manifest.json` already exists |
| Stale VMs from a prior run | One-liner: `az vm list --tag run_id=<id> -o tsv | xargs az vm delete --yes --no-wait` |
| Runaway billing | Budget alert on the RG at $50/$100/$200 |
| HuggingFace download flaky | One retry in bootstrap.sh; otherwise shard fails fast and uploads its log |
| Different CPU generations across VMs | Locked: D16s_v6 = Emerald Rapids 8573C (single CPU, no substitution) |
| Manifest.toml drift | Commit `benchmarking/julia/Manifest.toml` (already done) |
| Python dep drift | Use `uv.lock` (already in repo) — `uv sync` instead of pip |

## What's explicitly NOT in scope

- No queue / worker pool / autoscale
- No retry of failed shards (re-run manually)
- No live progress dashboard (SSH or `az storage blob list` is enough)
- No JSON schema validation on shards (1-line check that `shard_id` is unique is enough)
- No custom VM image (5-10 min startup overhead is fine for 2-3 thesis runs)

## Pre-conditions before first run

Set up once, manually:
1. Resource group + storage account + container created (user already has)
2. **User-assigned** managed identity created (e.g. `ann-bench-vm-identity`) and granted: `Storage Blob Data Contributor` on the SA, `Virtual Machine Contributor` on the RG. Record its `--id` for `run_shards.py` to attach to each VM at create time.
3. Budget alert configured on the RG
4. Repo public (so VMs can clone without credentials)
5. `Manifest.toml` committed (done)
6. `scripts/cloud/datasets.lock` populated with HuggingFace revision SHAs for each dataset used
7. `git push` before each run so the VM's `git checkout $SHA` finds the SHA

## Implementation order

1. `bootstrap.sh` (testable in a local Ubuntu 22.04 Docker container against one shard)
2. `run_shards.py` (with `--dry-run`)
3. End-to-end smoke: one shard, fashion-mnist, real VM
4. Then fan out
