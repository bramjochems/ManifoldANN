#!/usr/bin/env python3
"""Local CLI: provision one Azure VM per shard, fan out in parallel."""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

KNOWN_KINDS = {"ann-benchmark"}
SHARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


def run(cmd, check=True, capture=True):
    r = subprocess.run(cmd, check=False, text=True,
                       capture_output=capture)
    if check and r.returncode != 0:
        sys.stderr.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
        sys.exit(r.returncode)
    return r


def validate_sha_pushed(sha):
    # `git ls-remote origin <sha>` doesn't actually work - ls-remote lists
    # refs, not arbitrary SHAs. Use `branch -r --contains` instead, which
    # reads from local refs/remotes/. Run `git fetch` first so this reflects
    # the actual remote state, not a stale local view.
    run(["git", "fetch", "origin", "--quiet"], check=False)
    r = run(["git", "branch", "-r", "--contains", sha], check=False)
    if not (r.stdout or "").strip():
        sys.exit(f"git SHA {sha} not found on any origin branch. Push first.")


def blob_url(account, container, path):
    return f"https://{account}.blob.core.windows.net/{container}/{path}"


def manifest_exists(account, container, run_id):
    r = run(["az", "storage", "blob", "exists",
             "--auth-mode", "login",
             "--account-name", account,
             "--container-name", container,
             "--name", f"runs/{run_id}/manifest.json",
             "-o", "tsv"], check=False)
    return "True" in (r.stdout or "")


def upload_manifest(account, container, run_id, payload):
    tmp = Path(f"/tmp/manifest-{run_id}.json")
    tmp.write_text(json.dumps(payload, indent=2))
    run(["az", "storage", "blob", "upload",
         "--auth-mode", "login",
         "--account-name", account,
         "--container-name", container,
         "--name", f"runs/{run_id}/manifest.json",
         "--file", str(tmp), "--overwrite"])


def cloud_init_for(shard, run_id, container_url, git_sha, uami_id, repo_url):
    shard_json = json.dumps(shard)
    # `packages:` runs before `runcmd`, so jq/curl/git/azure-cli are in PATH
    # by the time the trap can fire. azure-cli is in the official Microsoft
    # apt repo via the `apt-transport-https` trick - but cloud-init's
    # `apt:` source stanza handles this without us shelling out.
    return f"""#cloud-config
apt:
  sources:
    azure-cli:
      source: "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ jammy main"
      keyid: BC528686B50D79E339D3721CEB3E94ADBE1229CF
package_update: true
packages:
  - git
  - jq
  - curl
  - ca-certificates
  - azure-cli
write_files:
  - path: /var/lib/cloud/scripts/per-instance/run.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      set -e
      export RUN_ID={run_id!r}
      export BLOB_CONTAINER_URL={container_url!r}
      export GIT_SHA={git_sha!r}
      export UAMI_RESOURCE_ID={uami_id!r}
      export REPO_URL={repo_url!r}
      export KIND={shard['kind']!r}
      export SHARD_JSON='{shard_json}'

      # Pre-bootstrap deallocate trap. By the time runcmd executes, az/jq/curl
      # are installed (via cloud-init `packages:` stanza above), so this trap
      # can always reach them. Only fails to deallocate if `az login --identity`
      # itself fails - the auto-shutdown schedule is the final backstop.
      trap 'timeout 60 az vm deallocate --ids $(timeout 30 curl -s -H Metadata:true "http://169.254.169.254/metadata/instance?api-version=2021-02-01" | jq -r .compute.resourceId) --no-wait || true' EXIT

      timeout 120 az login --identity --resource-id "$UAMI_RESOURCE_ID"
      [ -d /opt/repo/.git ] || git clone "$REPO_URL" /opt/repo
      cd /opt/repo
      git fetch --quiet origin
      git checkout "$GIT_SHA"
      exec bash scripts/cloud/bootstrap.sh
runcmd:
  - [bash, /var/lib/cloud/scripts/per-instance/run.sh]
"""


def az_vm_create_cmd(shard, run_id, rg, location, vm_size, image, uami_id,
                     cloud_init_path):
    vm_name = f"{run_id}-{shard['shard_id']}"[:63].lower()
    return [
        "az", "vm", "create",
        "--resource-group", rg,
        "--name", vm_name,
        "--location", location,
        "--size", vm_size,
        "--image", image,
        "--admin-username", "azureuser",
        "--generate-ssh-keys",
        "--assign-identity", uami_id,
        "--custom-data", cloud_init_path,
        "--tags", f"run_id={run_id}", f"shard_id={shard['shard_id']}",
        "--no-wait",
    ], vm_name


def az_auto_shutdown_cmd(rg, vm_name, location, minutes_ahead=90):
    # auto-shutdown takes a daily HHMM UTC time of day. Set 90 min ahead
    # of now to avoid race against ~5 min VM provisioning lag where a 60
    # min ahead schedule could lapse before the VM finishes provisioning.
    # Trade-off: a successfully completing shard might bill an extra ~30
    # min if the trap-based deallocate fails - acceptable for the safety.
    from datetime import timedelta
    t = datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead)
    return [
        "az", "vm", "auto-shutdown",
        "--resource-group", rg,
        "--name", vm_name,
        "--time", t.strftime("%H%M"),
    ]


def provision(shard, args, container_url):
    cloud_init = cloud_init_for(shard, args.run_id, container_url,
                                args.git_sha, args.user_assigned_identity_id,
                                args.repo_url)
    ci_path = Path(f"/tmp/cloud-init-{args.run_id}-{shard['shard_id']}.yaml")
    ci_path.write_text(cloud_init)

    create_cmd, vm_name = az_vm_create_cmd(
        shard, args.run_id, args.resource_group, args.location,
        args.vm_size, args.image, args.user_assigned_identity_id, str(ci_path))
    shutdown_cmd = az_auto_shutdown_cmd(args.resource_group, vm_name, args.location)

    if args.dry_run:
        print(f"--- shard {shard['shard_id']} ---")
        print("# cloud-init:")
        print(cloud_init)
        print("# az vm create:")
        print(" ".join(create_cmd))
        print("# auto-shutdown:")
        print(" ".join(shutdown_cmd))
        return shard["shard_id"], True

    r = run(create_cmd, check=False)
    if r.returncode != 0:
        return shard["shard_id"], False
    # auto-shutdown best-effort
    run(shutdown_cmd, check=False)
    return shard["shard_id"], True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resource-group", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--storage-account", required=True)
    p.add_argument("--storage-container", required=True)
    p.add_argument("--vm-size", required=True)
    p.add_argument("--git-sha", required=True)
    p.add_argument("--shards-file", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--user-assigned-identity-id", required=True)
    p.add_argument("--repo-url", required=True)
    p.add_argument("--image",
                   default="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not SHARD_ID_RE.match(args.run_id):
        sys.exit("run_id must be alphanumeric+dash")

    shards = json.loads(Path(args.shards_file).read_text())
    if not isinstance(shards, list) or not shards:
        sys.exit("shards file must be a non-empty JSON array")

    seen = set()
    for s in shards:
        sid = s.get("shard_id", "")
        if not SHARD_ID_RE.match(sid):
            sys.exit(f"invalid shard_id: {sid!r}")
        if sid in seen:
            sys.exit(f"duplicate shard_id: {sid}")
        seen.add(sid)
        if s.get("kind") not in KNOWN_KINDS:
            sys.exit(f"unknown kind in shard {sid}: {s.get('kind')!r}")

    if not args.dry_run:
        validate_sha_pushed(args.git_sha)
        if manifest_exists(args.storage_account, args.storage_container, args.run_id):
            sys.exit(f"runs/{args.run_id}/manifest.json already exists in blob")

    container_url = f"https://{args.storage_account}.blob.core.windows.net/{args.storage_container}"
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": args.run_id,
        "git_sha": args.git_sha,
        "vm_size": args.vm_size,
        "location": args.location,
        "started_at": started,
        "shards": shards,
    }

    if not args.dry_run:
        upload_manifest(args.storage_account, args.storage_container,
                        args.run_id, manifest)

    results = []
    with ThreadPoolExecutor(max_workers=min(16, len(shards))) as ex:
        futs = [ex.submit(provision, s, args, container_url) for s in shards]
        for f in as_completed(futs):
            results.append(f.result())

    ok = sum(1 for _, success in results if success)
    print(f"RUN_ID: {args.run_id}")
    print(f"Provisioned {ok}/{len(shards)} VMs")
    print(f"Watch progress: az storage blob list "
          f"--account-name {args.storage_account} "
          f"--container-name {args.storage_container} "
          f"--prefix runs/{args.run_id}/ -o table")


if __name__ == "__main__":
    main()
