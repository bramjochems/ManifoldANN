#!/usr/bin/env python3
"""Quick-look Pareto frontier plots: recall@10 (x) vs qps (y, log scale)
per dataset, one colour per algorithm.

For each (dataset, algorithm) we:
  - drop rows that are dominated (lower recall AND lower qps than another
    point of the same algorithm),
  - sort the survivors by recall and connect them with a line,
  - scatter the original points underneath for context.

Input:  <results-root>/merged/queries_frontier.csv
Output: <results-root>/merged/pareto/<dataset>.png

Usage:
    python scripts/thesis/plot_pareto_frontiers.py <results-root>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _pareto(points):
    """Return the Pareto-optimal subset (max recall, max qps), sorted by recall."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    best_qps = -1.0
    for r, q in reversed(pts):
        if q > best_qps:
            frontier.append((r, q))
            best_qps = q
    frontier.reverse()
    return frontier


def plot(results_root: Path):
    src = results_root / "merged" / "queries_frontier.csv"
    if not src.exists():
        raise SystemExit(f"missing {src} — run aggregate_queries_for_frontier.py first")

    by_ds_algo = defaultdict(lambda: defaultdict(list))
    with src.open() as f:
        for r in csv.DictReader(f):
            try:
                recall = float(r["recall@10"])
                qps = float(r["qps"])
            except (TypeError, ValueError):
                continue
            if qps <= 0:
                continue
            by_ds_algo[r["dataset"]][r["algorithm"]].append((recall, qps))

    out_dir = results_root / "merged" / "pareto"
    out_dir.mkdir(exist_ok=True)

    cmap = plt.get_cmap("tab20")

    for dataset, algos in sorted(by_ds_algo.items()):
        fig, ax = plt.subplots(figsize=(9, 6))
        algo_names = sorted(algos.keys())
        for i, algo in enumerate(algo_names):
            color = cmap(i % cmap.N)
            pts = algos[algo]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=color, alpha=0.35, s=18)
            front = _pareto(pts)
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(fx, fy, color=color, label=algo, linewidth=1.8, marker="o", markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_ylabel("qps (log scale)")
        ax.set_title(f"{dataset} — Pareto frontier per algorithm")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)

        dst = out_dir / f"{dataset}.png"
        fig.tight_layout()
        fig.savefig(dst, dpi=130)
        plt.close(fig)
        print(f"wrote {dst}  ({len(algo_names)} algorithms)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    args = p.parse_args()
    plot(args.results_root)


if __name__ == "__main__":
    main()
