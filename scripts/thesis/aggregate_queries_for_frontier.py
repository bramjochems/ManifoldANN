#!/usr/bin/env python3
"""Collapse queries_flat.csv across run_id so each
(dataset, algorithm, build_params, query_params) combination is one row,
with qps and recall averaged across however many runs measured it.

Input:  <results-root>/merged/queries_flat.csv  (from merge_results_flat.py)
Output: <results-root>/merged/queries_frontier.csv

Each row in the input already aggregates the per-shard `reps` repetitions
(qps is the median over reps, with q25/q75). What this script does on top
is collapse multiple runs of the *same* shard config — e.g. a shard that
appeared in both the main run and a recovery run gets averaged into one
row instead of two.

Aggregation:
  - qps, recall@10: mean across runs
  - qps_q25, qps_q75, recall_q25, recall_q75: mean (rough; not a true
    quantile of quantiles, but useful as a band)
  - reps: sum (total reps across runs)
  - n_runs: count of contributing rows
  - run_ids: comma-joined list
  - build_*/query_* columns: passed through (identical within a group)

Failed rows (status != "success") are dropped.

Usage:
    python scripts/thesis/aggregate_queries_for_frontier.py <results-root>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _flt(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def aggregate(results_root: Path):
    src = results_root / "merged" / "queries_flat.csv"
    if not src.exists():
        raise SystemExit(f"missing {src} — run merge_results_flat.py first")

    with src.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    build_cols = [c for c in cols if c.startswith("build_")]
    query_cols = [c for c in cols if c.startswith("query_")]

    # Group key: dataset + algorithm + (every build_* and query_* value).
    # Two rows with the same params from different runs collapse together.
    groups = defaultdict(list)
    for r in rows:
        if (r.get("status") or "").strip() != "success":
            continue
        key = (
            r.get("dataset", ""),
            r.get("algorithm", ""),
            tuple(r.get(c, "") for c in build_cols),
            tuple(r.get(c, "") for c in query_cols),
        )
        groups[key].append(r)

    out_cols = (
        ["dataset", "algorithm"]
        + build_cols
        + query_cols
        + ["qps", "recall@10",
           "qps_q25", "qps_q75", "recall_q25", "recall_q75",
           "reps", "n_runs", "run_ids", "shard_ids"]
    )

    out_rows = []
    for (dataset, algorithm, b_vals, q_vals), members in groups.items():
        def avg(field):
            vs = [_flt(m.get(field, "")) for m in members]
            vs = [v for v in vs if v is not None]
            return mean(vs) if vs else ""

        def total_int(field):
            tot = 0
            saw = False
            for m in members:
                v = _flt(m.get(field, ""))
                if v is not None:
                    tot += int(v)
                    saw = True
            return tot if saw else ""

        row = {"dataset": dataset, "algorithm": algorithm}
        for c, v in zip(build_cols, b_vals):
            row[c] = v
        for c, v in zip(query_cols, q_vals):
            row[c] = v
        row["qps"] = avg("qps")
        row["recall@10"] = avg("recall@10")
        row["qps_q25"] = avg("qps_q25")
        row["qps_q75"] = avg("qps_q75")
        row["recall_q25"] = avg("recall_q25")
        row["recall_q75"] = avg("recall_q75")
        row["reps"] = total_int("reps")
        row["n_runs"] = len(members)
        row["run_ids"] = ",".join(sorted({m.get("run_id", "") for m in members}))
        row["shard_ids"] = ",".join(sorted({m.get("shard_id", "") for m in members}))
        out_rows.append(row)

    out_rows.sort(key=lambda r: (r["dataset"], r["algorithm"],
                                 -(r["recall@10"] if isinstance(r["recall@10"], float) else 0.0)))

    dst = results_root / "merged" / "queries_frontier.csv"
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    n_collapsed = sum(1 for ms in groups.values() if len(ms) > 1)
    print(f"input rows (success):    {sum(len(m) for m in groups.values())}")
    print(f"output rows:             {len(out_rows)}")
    print(f"groups with >1 run:      {n_collapsed}")
    print(f"wrote: {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    args = p.parse_args()
    aggregate(args.results_root)


if __name__ == "__main__":
    main()
