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


# Algorithms to omit from plots. Edit this list while iterating on what
# to show in the thesis; a dropped algorithm produces no scatter, no
# frontier line, and no legend entry, but its colour slot in the global
# palette stays reserved so colours stay stable across runs.
EXCLUDE_ALGORITHMS = {
    "MANN-LSH",
    "MANN-KDTree",
    "SciPy-KDTree",
}


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

    # Stable algorithm→color map across all plots: assign colors from a
    # 20-slot qualitative palette to the union of *all* algorithms seen
    # (including excluded ones), so a future change to EXCLUDE_ALGORITHMS
    # doesn't shift the colours of the remaining algorithms.
    all_algos = sorted({a for algos in by_ds_algo.values() for a in algos})
    cmap = plt.get_cmap("tab20")
    color_for = {a: cmap(i % cmap.N) for i, a in enumerate(all_algos)}

    def _draw(ax, dataset, algos, *, show_legend=True, show_labels=True):
        algo_names = sorted(a for a in algos if a not in EXCLUDE_ALGORITHMS)
        for algo in algo_names:
            color = color_for[algo]
            pts = algos[algo]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=color, alpha=0.35, s=18)
            front = _pareto(pts)
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(fx, fy, color=color, label=algo, linewidth=1.8, marker="o", markersize=5)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if show_labels:
            ax.set_xlabel("recall@10")
            ax.set_ylabel("qps (log scale)")
        ax.set_title(dataset, fontsize=10)
        if show_legend:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
        return algo_names

    # Per-dataset plots
    for dataset, algos in sorted(by_ds_algo.items()):
        fig, ax = plt.subplots(figsize=(9, 6))
        algo_names = _draw(ax, dataset, algos)
        ax.set_title(f"{dataset} — Pareto frontier per algorithm")
        dst = out_dir / f"{dataset}.png"
        fig.tight_layout()
        fig.savefig(dst, dpi=130)
        plt.close(fig)
        print(f"wrote {dst}  ({len(algo_names)} algorithms)")

    # Combined 5x2 grid with a single shared legend
    datasets_sorted = sorted(by_ds_algo.keys())
    fig, axes = plt.subplots(5, 2, figsize=(14, 24))
    flat_axes = axes.flatten()
    for i, ax in enumerate(flat_axes):
        if i < len(datasets_sorted):
            ds = datasets_sorted[i]
            _draw(ax, ds, by_ds_algo[ds], show_legend=False, show_labels=True)
        else:
            ax.axis("off")
    # Build a single legend from the union of plotted algorithms
    plotted = sorted({a for algos in by_ds_algo.values()
                       for a in algos if a not in EXCLUDE_ALGORITHMS})
    handles = [plt.Line2D([0], [0], color=color_for[a], marker="o",
                          linewidth=1.8, markersize=5, label=a)
               for a in plotted]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    fig.suptitle("Pareto frontiers per dataset", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    grid_dst = out_dir / "_grid.png"
    fig.savefig(grid_dst, dpi=130)
    plt.close(fig)
    print(f"wrote {grid_dst}  (5x2 grid, {len(datasets_sorted)} datasets)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    args = p.parse_args()
    plot(args.results_root)


if __name__ == "__main__":
    main()
