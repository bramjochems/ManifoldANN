# Cloud benchmark runs — operations runbook

Fan out ANN benchmark shards across multiple Azure VMs in parallel,
each VM running a sequence of shards. Results upload to blob, VMs
self-deallocate, you download + merge afterwards.

For architecture and design rationale see `DESIGN.md`.

## Pre-flight (one-time, already done for this thesis)

| Resource | Value |
|---|---|
| Resource group | `bram` (westeurope) |
| Storage account | `stbram` |
| Results container | `mai-thesis` |
| Datasets container | `datasets` |
| User-assigned identity | `ann-bench-vm-identity` |
| Identity scope | Storage Blob Data Contributor on `stbram`, VM Contributor on `bram` |
| VM size | `Standard_D16s_v6` (locked Emerald Rapids 8573C) |
| Quota | 160 vCPU regional + 160 Dsv6 family |

If setting up fresh on a new subscription, see `DESIGN.md` for the
sequence of `az` commands.

## Concepts

- **Shard**: one (dataset, method-group) unit of work, defined by a YAML
  config in `benchmarking/configs/<shard-id>.yaml`.
- **VM (work unit)**: one Azure VM running a list of shards sequentially.
  Datasets are downloaded once and reused; the Julia/Python envs are
  installed once. Shards are ordered fast → slow inside each VM.
- **Run**: one fan-out invocation, identified by `run_id`. Results live
  at `blob://stbram/mai-thesis/runs/<run-id>/<shard-id>/...`.

## Files

| File | Role |
|---|---|
| `vms.json` | The "production" 10-VM allocation for the full thesis run. |
| `vms-recovery.json` | Recovery manifest for known-broken shards. Edit per situation. |
| `shards.json` | Flat list of all 79 shards (input to allocation, not used directly by run). |
| `bootstrap.sh` | Runs on each VM. Installs deps, instantiates Julia envs, loops over shards. |
| `run_shards.py` | Local CLI: provisions one VM per work-unit. Handles both single-shard and multi-shard schemas. |
| `run_full.sh` | Convenience wrapper around `run_shards.py` with the production constants pre-filled. Defaults to `vms.json`. |
| `kinds/ann-benchmark.sh` | The dispatcher invoked by bootstrap for each shard. |
| `watch.sh` | Per-VM progress dashboard. |
| `teardown.sh` | Cascade-delete VM + NIC + disk + VNET + NSG + PublicIP + auto-shutdown schedule. Filter by `run_id` tag or `--prefix <vm-name-prefix>`. |
| `download_results.sh` | Pull all blobs for a given run_id locally. |
| `upload_datasets.sh` | One-time: upload `benchmarking/data/*.hdf5` to the `datasets` container. |
| `test_bootstrap_local.sh` | End-to-end Docker test of bootstrap.sh against mocked az calls. |

## Standard run flow

```bash
# Push the SHA you want to run at (VMs will git checkout it)
git push origin main

# Fire off the run
bash scripts/cloud/run_full.sh
# → prints RUN_ID like "full-20260506-0915"

# Monitor (refresh every 60s, ctrl-C to exit)
bash scripts/cloud/watch.sh full-20260506-0915 --watch

# Once everything is deallocated:
bash scripts/cloud/teardown.sh full-20260506-0915

# Download results locally
bash scripts/cloud/download_results.sh full-20260506-0915

# Merge into builds.csv + queries.csv
python3 scripts/thesis/merge_results.py cloud-results/full-20260506-0915
```

## Recovery flow (when shards fail mid-run)

If some shards fail or VMs die, write a recovery manifest with just
those shards:

```bash
# Edit scripts/cloud/vms-recovery.json (or write a fresh /tmp/redo.json)
# Then fire it directly via run_shards.py with a NEW run_id

GIT_SHA=$(git rev-parse HEAD)  # ensure pushed first
python3 scripts/cloud/run_shards.py \
  --resource-group bram --location westeurope \
  --storage-account stbram --storage-container mai-thesis \
  --vm-size Standard_D16s_v6 --git-sha "$GIT_SHA" \
  --shards-file /tmp/redo.json \
  --run-id "rec-$(date -u +%Y%m%d-%H%M)" \
  --user-assigned-identity-id "/subscriptions/db9f84bc-5025-425a-84b7-31e68913aa63/resourcegroups/bram/providers/Microsoft.ManagedIdentity/userAssignedIdentities/ann-bench-vm-identity" \
  --repo-url "https://github.com/bramjochems/ManifoldANN.git"
```

Watch with the matching manifest:
```bash
bash scripts/cloud/watch.sh rec-20260506-1030 /tmp/redo.json --watch
```

## Merging multiple runs (e.g. main + recoveries)

The merge script walks any directory tree shaped like
`<root>/runs/<run_id>/<shard_id>/output/results_*/results.csv`. To merge
multiple runs, download them all into a shared root:

```bash
mkdir -p cloud-results/all
for r in full-20260506-0915 rec-20260506-1030 rec2-20260506-1053 rec3-20260506-1328; do
  az storage blob download-batch --account-name stbram --source mai-thesis \
    --pattern "runs/$r/*" --destination cloud-results/all \
    --auth-mode login --no-progress
done
python3 scripts/thesis/merge_results.py cloud-results/all
# → cloud-results/all/merged/builds.csv
# → cloud-results/all/merged/queries.csv
```

The merge dedups builds by `(run_id, shard_id, algorithm, build_params)`,
so duplicate shards across runs (e.g. lastfm appearing in both main and
recovery) produce two distinct build rows. In post-processing, decide
per shard which run to keep (typically the latest succeeding one).

## Quota management

- Each `Standard_D16s_v6` = 16 vCPU. Quota is 160, so up to 10 VMs in
  parallel.
- **Deallocated VMs still count against quota** until you also delete
  them (NIC, disk, etc.). If you want to fire a new run while the
  previous one's VMs are deallocated, run `teardown.sh` on those
  finished VMs first to free vCPUs.
- Selective per-VM teardown (without removing the whole run):
  ```bash
  bash scripts/cloud/teardown.sh --prefix "<run-id>-<vm-id>"
  ```

## Cost discipline

- VMs auto-shutdown 90 min after creation if bootstrap's trap doesn't
  fire (network failure etc.).
- A normal full run costs ~$5-10 in compute (10 VMs × ~2-3h each at
  $0.65/hr).
- Disk (~$0.01/GB-month) is negligible after teardown.
- **Verify clean state at the end**: `az resource list -g bram -o table`
  should show only `stbram`, `ann-bench-vm-identity`, and any baseline
  resources you started with.

## Troubleshooting

**A VM provisioned but never uploaded results.**
- Restart it: `az vm start -g bram -n <vm-name>`
- SSH in: `az ssh vm -g bram -n <vm-name>` (needs RG-level VM Login or use the user-assigned key)
- Read `/var/log/cloud-init-output.log` — bootstrap.sh writes here, harness writes to `/opt/repo/run.log`
- Common failures and fixes are documented in commit messages on `main` (search for `cloud:` prefix)

**`az vm create --no-wait` returns 0 but the VM never appears.**
- Transient ARM control-plane issue. `run_shards.py` retries up to 3×
  with backoff. If it still fails, fire the failed VM manually with the
  cloud-init file at `/tmp/cloud-init-<run-id>-<vm-id>.yaml`.

**`metric: dot` rejected by harness.**
- Use `metric: angular` for dot-product datasets (e.g. lastfm). The
  harness validator only accepts `euclidean | angular`.

**HNSW.jl / NND.jl wrapper "library not available".**
- Bootstrap must instantiate `benchmarking/julia/` env. See `bootstrap.sh`
  step "instantiating benchmarking/julia env". `ManifoldANN` is NOT a
  dep of that env — drop it if it appears in `Project.toml`.

## Updating the configuration

To add new shards: write a new `benchmarking/configs/<shard-id>.yaml`,
add an entry to `vms.json` under whichever VM should run it, and
re-fire `run_full.sh`.

To change a Tier 1 algorithm's sweep grid: edit the relevant config
YAML's `build:` and/or `query_sweep:` blocks. The harness expands list
values in `build:` into a Cartesian product of build configs.
