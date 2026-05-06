#!/usr/bin/env python3
"""Alternative to merge_results.py: flatten build_params and query_params
JSON blobs into individual columns (prefixed `build_` and `query_`).

The union of all keys seen across all rows determines the columns; missing
values are left empty. Two outputs under <results-root>/merged/:

    builds_flat.csv   — one row per (run_id, shard_id, algorithm, build_params)
    queries_flat.csv  — one row per (run_id, shard_id, algorithm,
                                     build_params, query_params)

Usage:
    python scripts/thesis/merge_results_flat.py <results-root>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Reuse helpers from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_results import _read_dataset_from_yaml, _shard_results_csv  # noqa: E402


BASE_BUILD_COLS = [
    "run_id", "shard_id", "dataset", "algorithm",
    "status", "error",
    "build_time", "build_rss_delta_mb", "index_mb",
]

BASE_QUERY_COLS = [
    "run_id", "shard_id", "dataset", "algorithm",
    "status", "error",
    "qps", "recall@10", "reps",
    "qps_q25", "qps_q75", "recall_q25", "recall_q75",
    "query_rss_delta_mb_max",
]


def _parse_json_blob(s):
    s = (s or "").strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _stringify(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def merge(results_root: Path, configs_dir: Path):
    runs_dir = results_root / "runs"
    if not runs_dir.exists():
        candidates = list(results_root.glob("*/runs/*"))
        if not candidates:
            sys.exit(f"No runs/ subdirectory found under {results_root}")
        runs_dir = candidates[0].parent

    builds = []
    queries = []
    build_param_keys = set()
    query_param_keys = set()
    build_seen = set()
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
                    bp_raw = row.get("build_params", "") or ""
                    qp_raw = row.get("query_params", "") or ""
                    bp = _parse_json_blob(bp_raw)
                    qp = _parse_json_blob(qp_raw)
                    build_param_keys.update(bp.keys())
                    query_param_keys.update(qp.keys())

                    qrow = {
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "status": row.get("status", ""),
                        "error": row.get("error", ""),
                        "qps": row.get("qps", ""),
                        "recall@10": row.get("recall@10", ""),
                        "reps": row.get("reps", ""),
                        "qps_q25": row.get("qps_q25", ""),
                        "qps_q75": row.get("qps_q75", ""),
                        "recall_q25": row.get("recall_q25", ""),
                        "recall_q75": row.get("recall_q75", ""),
                        "query_rss_delta_mb_max": row.get("query_rss_delta_mb_max", ""),
                    }
                    for k, v in bp.items():
                        qrow[f"build_{k}"] = _stringify(v)
                    for k, v in qp.items():
                        qrow[f"query_{k}"] = _stringify(v)
                    queries.append(qrow)

                    key = (run_id, shard_id, algorithm, bp_raw)
                    if key in build_seen:
                        continue
                    build_seen.add(key)
                    brow = {
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "status": row.get("status", ""),
                        "error": row.get("error", ""),
                        "build_time": row.get("build_time", ""),
                        "build_rss_delta_mb": row.get("build_rss_delta_mb", ""),
                        "index_mb": row.get("index_mb", ""),
                    }
                    for k, v in bp.items():
                        brow[f"build_{k}"] = _stringify(v)
                    builds.append(brow)

    build_extra = [f"build_{k}" for k in sorted(build_param_keys)]
    query_extra_b = [f"build_{k}" for k in sorted(build_param_keys)]
    query_extra_q = [f"query_{k}" for k in sorted(query_param_keys)]

    build_cols = BASE_BUILD_COLS + build_extra
    query_cols = BASE_QUERY_COLS + query_extra_b + query_extra_q

    out_dir = results_root / "merged"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "builds_flat.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=build_cols, extrasaction="ignore")
        w.writeheader()
        for row in builds:
            w.writerow({c: row.get(c, "") for c in build_cols})
    with (out_dir / "queries_flat.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=query_cols, extrasaction="ignore")
        w.writeheader()
        for row in queries:
            w.writerow({c: row.get(c, "") for c in query_cols})

    print(f"runs found:        {sorted(set(seen_runs))}")
    print(f"build_param keys:  {sorted(build_param_keys)}")
    print(f"query_param keys:  {sorted(query_param_keys)}")
    print(f"builds rows:       {len(builds)} -> {out_dir / 'builds_flat.csv'}")
    print(f"queries rows:      {len(queries)} -> {out_dir / 'queries_flat.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    p.add_argument("--configs-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent /
                            "benchmarking" / "configs")
    args = p.parse_args()
    merge(args.results_root, args.configs_dir)


if __name__ == "__main__":
    main()
