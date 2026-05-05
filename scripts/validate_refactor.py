#!/usr/bin/env python3
"""Validate the query-sweep refactor against a pre-refactor baseline CSV.

Usage:
    python scripts/validate_refactor.py <new_results.csv>

Compares row by row against scripts/refactor_baseline/baseline.csv:
    - recall: must be bit-identical
    - qps: within 5% relative
    - build_time: within 10% relative

Exits non-zero on any failure.

IMPORTANT: capture the baseline AND run validation with --threads 1.
Threaded HNSW builds (both hnswlib and MANN) are non-deterministic across
runs (lock race-windows during neighbor list updates), so recall will not
be bit-identical between two threaded runs of the same config. This is a
property of the algorithm, not a bug, but it makes "bit-identical" too
strong a gate when threads > 1. The single-threaded gate validates that
the harness logic is correct independently of threading non-determinism.
"""
import csv
import sys
from pathlib import Path


BASELINE = (
    Path(__file__).resolve().parent / "refactor_baseline" / "baseline.csv"
)

QPS_TOL = 0.05
BUILD_TOL = 0.10


def _load_rows(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue
            # Key by (algorithm, query_params) when present; fall back to
            # algorithm only for legacy CSVs without the column.
            qp = row.get("query_params", "") or "{}"
            key = (row["algorithm"], qp)
            rows[key] = row
    return rows


def _rel_diff(a, b):
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    if len(sys.argv) != 2:
        print("usage: validate_refactor.py <new_results.csv>", file=sys.stderr)
        sys.exit(2)
    new_path = Path(sys.argv[1])
    if not BASELINE.exists():
        print(f"baseline not found: {BASELINE}", file=sys.stderr)
        sys.exit(2)
    if not new_path.exists():
        print(f"new results not found: {new_path}", file=sys.stderr)
        sys.exit(2)

    base = _load_rows(BASELINE)
    new = _load_rows(new_path)

    failures = []
    print(f"{'algorithm':<32} {'recall':>10} {'qps Δ':>10} {'build Δ':>10}")
    print("-" * 66)
    for key, b in base.items():
        n = new.get(key)
        algo_disp = key[0] if key[1] in ("", "{}") else f"{key[0]} {key[1]}"
        if n is None:
            failures.append(f"missing in new results: {algo_disp}")
            print(f"{algo_disp:<32} {'MISSING':>10}")
            continue
        recall_b = float(b["recall@10"]) if "recall@10" in b else float(
            next(b[c] for c in b if c.startswith("recall@"))
        )
        recall_n = float(n["recall@10"]) if "recall@10" in n else float(
            next(n[c] for c in n if c.startswith("recall@"))
        )
        qps_d = _rel_diff(b["qps"], n["qps"])
        bt_d = _rel_diff(b["build_time"], n["build_time"])
        recall_ok = recall_b == recall_n
        qps_ok = qps_d <= QPS_TOL
        bt_ok = bt_d <= BUILD_TOL

        flag = "" if (recall_ok and qps_ok and bt_ok) else "  FAIL"
        print(
            f"{algo_disp:<32} "
            f"{('=' if recall_ok else f'{recall_b:.4f}!={recall_n:.4f}'):>10} "
            f"{qps_d * 100:>9.2f}% "
            f"{bt_d * 100:>9.2f}%{flag}"
        )
        if not recall_ok:
            failures.append(
                f"recall mismatch {algo_disp}: baseline={recall_b}, new={recall_n}"
            )
        if not qps_ok:
            failures.append(
                f"qps drift {algo_disp}: {qps_d * 100:.2f}% > {QPS_TOL * 100:.0f}%"
            )
        if not bt_ok:
            failures.append(
                f"build_time drift {algo_disp}: {bt_d * 100:.2f}% > {BUILD_TOL * 100:.0f}%"
            )

    extras = set(new.keys()) - set(base.keys())
    for k in extras:
        print(f"  (new only) {k[0]} {k[1]}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nOK: baseline matches within tolerances.")


if __name__ == "__main__":
    main()
