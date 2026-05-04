"""
Generate the validation dataset and reference Python ORC-ManL curvatures
used by scripts/test_orcml_exact_match.jl.

Produces two files in benchmark_results/:
  - test_data.csv       : the swiss roll point cloud (N rows, 3 columns)
  - curvatures_orcml_python.csv : edge curvatures from the reference orcml package
                                  (columns: source, target, curvature; 0-indexed nodes)

Usage:
    cd benchmarking
    uv run python ../scripts/generate_orcml_validation_data.py \
        --n-points 500 --noise 0.05 --k 15 --seed 42
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Make the orcml package importable
HERE = Path(__file__).resolve().parent
ORCML_ROOT = HERE.parent / "benchmarking" / "external" / "orcml"
sys.path.insert(0, str(ORCML_ROOT))

from data.data import swiss_roll  # noqa: E402
from src.orcmanl import ORCManL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--noise-thresh", type=float, default=0.275)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--lda", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=str(HERE.parent / "benchmark_results"),
        help="Directory for output CSVs (default: ../benchmark_results/)",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate the swiss roll dataset.
    print(f"Generating swiss roll: n={args.n_points}, noise={args.noise}")
    bundle = swiss_roll(
        n_points=args.n_points,
        noise=args.noise,
        noise_thresh=args.noise_thresh,
        supersample=True,
        dim=3,
        hole=False,
    )
    data = bundle["data"]  # shape (N, 3)
    print(f"  shape: {data.shape}")

    # 2. Save the point cloud.
    data_path = out_dir / "test_data.csv"
    np.savetxt(data_path, data, delimiter=",")
    print(f"  saved {data_path}")

    # 3. Run ORC-ManL on the point cloud.
    exp_params = {
        "mode": "nbrs",
        "n_neighbors": args.k,
        "lda": args.lda,
        "delta": args.delta,
    }
    print(f"Running ORC-ManL: {exp_params}")
    model = ORCManL(exp_params=exp_params, verbose=False, reattach=True)
    model.fit(data)
    G = model.G  # graph with per-edge ricciCurvature attribute
    print(f"  n_nodes={G.number_of_nodes()}  n_edges={G.number_of_edges()}")

    # 4. Save curvatures.
    curv_path = out_dir / "curvatures_orcml_python.csv"
    with open(curv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "curvature"])
        for i, j, d in G.edges(data=True):
            w.writerow([i, j, d["ricciCurvature"]])
    print(f"  saved {curv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
