#!/usr/bin/env python3
"""Merge per-shard results.csv files into two unified CSVs:

    builds.csv  — one row per (dataset, shard, algorithm, build_params)
                  with build_time + index footprint
    queries.csv — one row per (dataset, shard, algorithm, build_params, query_params)
                  with qps, recall, reps quartiles

Walks the directory layout produced by `scripts/cloud/download_results.sh`:

    cloud-results/<run-id>/
      runs/<run-id>/
        manifest.json
        <shard-id>/
          metadata.txt
          run.log
          output/
            results_<timestamp>/
              results.csv
              results.json
              ...

Each input results.csv has the same schema (algorithm, qps, recall@10,
build_time, status, error, reps, qps_q25, qps_q75, recall_q25, recall_q75,
build_rss_delta_mb, query_rss_delta_mb_max, index_mb, build_params,
query_params). For runs that pre-date the build_params column, that field
is empty.

If a shard has multiple results_<timestamp> directories (e.g. two runs of
the same shard), the lexicographically largest timestamp wins. shard_id
is parsed from the path, not from the CSV.

Usage:
    python scripts/thesis/merge_results.py <results-root>
    # writes <results-root>/merged/builds.csv and queries.csv
"""

import argparse
import csv
import sys
from pathlib import Path


# Source columns we always carry (rest end up in build_params/query_params).
BUILD_COLS = [
    "run_id", "shard_id", "dataset", "algorithm",
    "status", "error",
    "build_time", "build_rss_delta_mb", "index_mb",
    "build_params",
]

QUERY_COLS = [
    "run_id", "shard_id", "dataset", "algorithm",
    "status", "error",
    "qps", "recall@10", "reps",
    "qps_q25", "qps_q75", "recall_q25", "recall_q75",
    "query_rss_delta_mb_max",
    "build_params", "query_params",
]


def _read_dataset_from_yaml(repo_configs_dir, shard_id):
    """Try to look up the dataset name from the shard's YAML config.

    Falls back to inferring from the shard_id prefix (the shard naming
    convention puts the dataset first: e.g. `glove-25-mann-hnsw-sweep`).
    """
    yaml_path = repo_configs_dir / f"{shard_id}.yaml"
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("dataset:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    # Fallback: infer dataset prefix from shard_id. Hard to get right
    # without ambiguity (e.g. "glove-25-mann-hnsw-sweep" — is the dataset
    # "glove-25" or "glove-25-mann-hnsw"?). Heuristic: the dataset is
    # whatever doesn't end in a method-group suffix.
    method_groups = (
        "mann-hnsw-sweep", "hnswlib-sweep", "mann-lsh-sweep", "faiss-ivf-sweep",
        "annoy-sweep", "mann-rpforest-sweep", "mann-nnd-sweep",
        "mann-ivf-flat-sweep", "mann-ivf-hnsw-sweep",
        "single-point-methods", "tier2-hnsw-mann", "tier2-hnsw-jl",
        "tier2-nnd-mann", "tier2-nnd-jl",
    )
    for suffix in method_groups:
        if shard_id.endswith("-" + suffix):
            return shard_id[: -len(suffix) - 1]
    return shard_id  # give up; caller can clean up.


def _shard_results_csv(shard_dir):
    """Return the path to the latest results.csv under this shard's
    `output/results_<ts>/` directories, or None if none found."""
    output = shard_dir / "output"
    if not output.exists():
        return None
    candidates = sorted(output.glob("results_*/results.csv"))
    return candidates[-1] if candidates else None


def merge(results_root: Path, configs_dir: Path):
    """Walk the results tree under results_root and produce builds.csv +
    queries.csv under results_root/merged/."""
    runs_dir = results_root / "runs"
    if not runs_dir.exists():
        # The user may have passed `cloud-results/<run-id>` directly OR
        # `cloud-results/<run-id>/runs/<run-id>`. Try the inner form.
        candidates = list(results_root.glob("*/runs/*"))
        if not candidates:
            sys.exit(f"No runs/ subdirectory found under {results_root}")
        runs_dir = candidates[0].parent

    # Iterate one level deeper: runs/<run_id>/<shard_id>/...
    builds = []   # one row per build (dedup by build_params within a shard)
    queries = []  # one row per (build, query_combo)

    seen_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        seen_runs.append(run_id)
        for shard_dir in sorted(run_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard_id = shard_dir.name
            csv_path = _shard_results_csv(shard_dir)
            if csv_path is None:
                continue
            dataset = _read_dataset_from_yaml(configs_dir, shard_id)
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    algorithm = row.get("algorithm", "")
                    status = row.get("status", "")
                    error = row.get("error", "")
                    build_params = row.get("build_params", "{}") or "{}"
                    query_params = row.get("query_params", "{}") or "{}"

                    queries.append({
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "status": status,
                        "error": error,
                        "qps": row.get("qps", ""),
                        "recall@10": row.get("recall@10", ""),
                        "reps": row.get("reps", ""),
                        "qps_q25": row.get("qps_q25", ""),
                        "qps_q75": row.get("qps_q75", ""),
                        "recall_q25": row.get("recall_q25", ""),
                        "recall_q75": row.get("recall_q75", ""),
                        "query_rss_delta_mb_max": row.get("query_rss_delta_mb_max", ""),
                        "build_params": build_params,
                        "query_params": query_params,
                    })

    # Build rows: one per (run_id, shard_id, algorithm, build_params).
    # build_time / index_mb / build_rss_delta_mb are repeated across query
    # rows of the same build, so the first row per group is enough.
    build_seen = set()
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for shard_dir in sorted(run_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard_id = shard_dir.name
            csv_path = _shard_results_csv(shard_dir)
            if csv_path is None:
                continue
            dataset = _read_dataset_from_yaml(configs_dir, shard_id)
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    algorithm = row.get("algorithm", "")
                    build_params = row.get("build_params", "{}") or "{}"
                    key = (run_id, shard_id, algorithm, build_params)
                    if key in build_seen:
                        continue
                    build_seen.add(key)
                    builds.append({
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "status": row.get("status", ""),
                        "error": row.get("error", ""),
                        "build_time": row.get("build_time", ""),
                        "build_rss_delta_mb": row.get("build_rss_delta_mb", ""),
                        "index_mb": row.get("index_mb", ""),
                        "build_params": build_params,
                    })

    out_dir = results_root / "merged"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "builds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BUILD_COLS)
        w.writeheader()
        w.writerows(builds)
    with (out_dir / "queries.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_COLS)
        w.writeheader()
        w.writerows(queries)

    print(f"runs found:    {sorted(set(seen_runs))}")
    print(f"shards merged: {len({(b['run_id'], b['shard_id']) for b in builds})}")
    print(f"builds rows:   {len(builds)} -> {out_dir / 'builds.csv'}")
    print(f"queries rows:  {len(queries)} -> {out_dir / 'queries.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path,
                   help="Path to a downloaded results dir, e.g. cloud-results/full-20260506-0915")
    p.add_argument("--configs-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent /
                            "benchmarking" / "configs",
                   help="Path to benchmarking/configs/ for dataset name lookup")
    args = p.parse_args()
    merge(args.results_root, args.configs_dir)


if __name__ == "__main__":
    main()
