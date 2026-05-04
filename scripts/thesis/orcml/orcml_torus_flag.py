"""Dump orcml's algorithmic shortcut flag for each torus cell.

For each torus run (R/r in {1.5, 2, 4}) and each (n, k, noise_std) cell,
regenerate the same point cloud the Julia pipeline used (via the existing
Julia helper), run the orcml ORCManL fit, and write a per-edge CSV
{a, b, shortcut} with a<b, 0-indexed, into

    benchmark_results/composite_shortcut_extended/orcml_flags/<label>_n<n>_k<k>_noise<noise>.csv

The CSV is consumed by composite_shortcut_extended_eval.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ORCML_ROOT = ROOT / "benchmarking" / "external" / "orcml"
sys.path.insert(0, str(ORCML_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.orcmanl import ORCManL  # noqa: E402

from composite_shortcut_full_eval import RUNS, regen_points_via_julia  # noqa: E402

OUT = ROOT / "benchmark_results" / "composite_shortcut_extended" / "orcml_flags"
OUT.mkdir(parents=True, exist_ok=True)


def run_cell(run, n, k, noise_std):
    out_path = OUT / f"{run['label']}_n{n}_k{k}_noise{noise_std:.2f}.csv"
    if out_path.exists():
        return out_path, "skip"
    pts = regen_points_via_julia(run, n, noise_std)
    coords = pts[["x", "y", "z"]].to_numpy()

    exp_params = {"mode": "nbrs", "n_neighbors": int(k), "lda": 0.01, "delta": 0.8}
    model = ORCManL(exp_params=exp_params, verbose=False, reattach=True)
    model.fit(coords)
    G = model.G_ann

    rows = []
    for i, j, d in G.edges(data=True):
        a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        rows.append({"a": a, "b": b, "shortcut": int(d.get("shortcut", 0))})
    df = pd.DataFrame(rows).drop_duplicates(["a", "b"])
    df.to_csv(out_path, index=False)
    return out_path, f"ok n_short={int(df.shortcut.sum())}/{len(df)}"


def main():
    torus_runs = [r for r in RUNS if r["kind"] == "torus"]
    t0 = time.time()
    for run in torus_runs:
        edges_path = run["edges"]
        if not edges_path.exists():
            print(f"!! missing {edges_path}, skip {run['label']}")
            continue
        print(f"\n=== {run['label']} ===")
        df_all = pd.read_csv(edges_path)
        cells = sorted(df_all.groupby(["n", "k", "noise_std"]).groups.keys())
        for (n, k, noise_std) in cells:
            t1 = time.time()
            try:
                path, status = run_cell(run, int(n), int(k), float(noise_std))
            except Exception as e:
                print(f"  n={n} k={k} noise={noise_std} FAILED: {e}")
                continue
            print(f"  n={n:5d} k={k:3d} noise={noise_std:.2f}  {status}  ({time.time()-t1:.1f}s)")
    print(f"\nDone in {time.time()-t0:.1f}s, output in {OUT}")


if __name__ == "__main__":
    main()
