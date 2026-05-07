#!/usr/bin/env python3
"""Grouped bar charts comparing build times across implementations.

Two figures, both grouped by dataset:
  - HNSW family:     MANN-HNSW-diversified, HNSWlib, HNSW-jl
  - NN-Descent family: MANN-NNDescent-full-deferred, NearestNeighborDescent-jl,
                       PyNNDescent

For each (dataset, algorithm) pair the bar shows the **median** build time
across the algorithm's sweep grid. A vertical line through each bar marks
the [min, max] of the same set, so the spread of the parameter sweep is
visible. Missing combinations (algorithm not yet run on a dataset) are
silently skipped — the group narrows and the remaining bars stay aligned.

Input:  <results-root>/merged/builds_flat.csv
Output: <results-root>/merged/build_times_hnsw.png
        <results-root>/merged/build_times_nnd.png

Usage:
    python scripts/thesis/plot_build_times.py <results-root>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


HNSW_GROUP = [
    ("MANN-HNSW-diversified", "MANN-HNSW"),
    ("HNSWlib",                "hnswlib"),
    ("HNSW-jl",                "HNSW.jl"),
]

NND_GROUP = [
    ("MANN-NNDescent-full-deferred",  "MANN-NN-Descent"),
    ("NearestNeighborDescent-jl",     "NND.jl"),
    ("PyNNDescent",                   "PyNNDescent"),
]


def _load_build_times(results_root: Path):
    """Returns {(dataset, algorithm): [build_time, ...]} for status=success rows."""
    src = results_root / "merged" / "builds_flat.csv"
    if not src.exists():
        raise SystemExit(f"missing {src} — run merge_results_flat.py first")
    out = defaultdict(list)
    with src.open() as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() != "success":
                continue
            try:
                t = float(r["build_time"])
            except (TypeError, ValueError):
                continue
            if t <= 0:
                continue
            out[(r["dataset"], r["algorithm"])].append(t)
    return out


def _grouped_bar(ax, data, group_spec, datasets, title):
    """Draw a grouped bar chart on `ax`.

    `data` is {(dataset, algorithm): [times]}; `group_spec` is the list
    of (algorithm, label) pairs (ordered, controls bar order within a
    group); `datasets` is the dataset order on the x-axis.
    """
    n_algos = len(group_spec)
    n_ds = len(datasets)
    bar_w = 0.8 / n_algos
    x = np.arange(n_ds)

    # Reuse the same tab20 palette used by the Pareto plots so colours
    # are consistent across figures. We index into the global algorithm
    # ordering rather than per-figure ordering.
    cmap = plt.get_cmap("tab20")
    all_algos = sorted({a for (_, a) in data.keys()})
    color_for = {a: cmap(i % cmap.N) for i, a in enumerate(all_algos)}

    for k, (algo, label) in enumerate(group_spec):
        offset = (k - (n_algos - 1) / 2) * bar_w
        medians, mins, maxs, xs = [], [], [], []
        for j, ds in enumerate(datasets):
            ts = data.get((ds, algo))
            if not ts:
                continue
            medians.append(median(ts))
            mins.append(min(ts))
            maxs.append(max(ts))
            xs.append(j + offset)
        if not xs:
            continue
        medians = np.array(medians)
        yerr_lo = medians - np.array(mins)
        yerr_hi = np.array(maxs) - medians
        ax.bar(xs, medians, width=bar_w, color=color_for.get(algo, "gray"),
               label=label, edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, medians, yerr=[yerr_lo, yerr_hi], fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("build time (s, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_title(title)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def plot(results_root: Path):
    data = _load_build_times(results_root)
    datasets = sorted({ds for (ds, _) in data.keys()})

    out_dir = results_root / "merged"
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _grouped_bar(ax, data, HNSW_GROUP, datasets,
                 "HNSW build times (median; bars span min–max of sweep)")
    fig.tight_layout()
    dst = out_dir / "build_times_hnsw.png"
    fig.savefig(dst, dpi=130)
    plt.close(fig)
    print(f"wrote {dst}")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _grouped_bar(ax, data, NND_GROUP, datasets,
                 "NN-Descent build times (median; bars span min–max of sweep)")
    fig.tight_layout()
    dst = out_dir / "build_times_nnd.png"
    fig.savefig(dst, dpi=130)
    plt.close(fig)
    print(f"wrote {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    args = p.parse_args()
    plot(args.results_root)


if __name__ == "__main__":
    main()
