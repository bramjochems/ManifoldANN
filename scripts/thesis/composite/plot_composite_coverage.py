"""
Plot composite-label coverage across the $(n, k)$ grid.

Reads:
  benchmark_results/composite_shortcut_extended/extended_summary.csv

Produces:
  docs/thesis/figures/composite_label_coverage.pdf
  docs/thesis/figures/composite_label_coverage.csv

A grid of heatmaps showing the number of composite-label shortcuts per cell
under the min-chord guard ($c = 1$). Rows: manifolds. Columns: $\\sigma = 0$
(clean) vs $\\sigma = 0.5$ (noisy). Cells with zero shortcuts are shown in
grey: AUROC is undefined there. The figure makes visible that the composite
filter rejects nearly everything in clean torus cells (the headline of the
chapter) but leaves a meaningful population once noise is added.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "benchmark_results/composite_shortcut_extended/extended_summary.csv"
OUT = REPO.parent.parent / "docs/thesis/figures/composite_label_coverage.pdf"

MANIFOLD_ORDER = [
    ("swiss_roll", "Swiss roll"),
    ("swiss_roll_tight_s005", "Swiss roll (tight)"),
    ("torus_R1_5_r1", r"Torus $R/r=1.5$"),
    ("torus_R2_r1", r"Torus $R/r=2$"),
    ("torus_R4_r1", r"Torus $R/r=4$"),
]
NOISE_LEVELS = [(0.0, r"$\sigma = 0$"), (0.5, r"$\sigma = 0.5$")]


def main():
    df = pd.read_csv(SUMMARY)
    cell = (
        df.groupby(["manifold", "noise", "n", "k"], as_index=False)
        .agg(
            n_kept=("n_kept_guard_c1.0", "first"),
            n_composite=("n_composite_guard_c1.0", "first"),
        )
    )
    cell.to_csv(OUT.with_suffix(".csv"), index=False)

    n_vals = sorted(cell.n.unique())
    k_vals = sorted(cell.k.unique())
    vmax = max(2, cell.n_composite.max())

    fig, axes = plt.subplots(
        len(MANIFOLD_ORDER), len(NOISE_LEVELS),
        figsize=(6.5, 1.9 * len(MANIFOLD_ORDER)), sharex=True, sharey=True,
    )

    for row, (manifold, title) in enumerate(MANIFOLD_ORDER):
        for col, (noise, noise_label) in enumerate(NOISE_LEVELS):
            sub = cell[(cell.manifold == manifold) & (cell.noise == noise)]
            grid = np.full((len(n_vals), len(k_vals)), np.nan)
            for _, r in sub.iterrows():
                grid[n_vals.index(r.n), k_vals.index(r.k)] = r.n_composite

            ax = axes[row, col]
            ax.set_facecolor("#eeeeee")
            masked = np.ma.masked_where((grid == 0) | np.isnan(grid), grid)
            im = ax.imshow(masked, aspect="auto", origin="lower",
                           norm=LogNorm(vmin=1, vmax=vmax), cmap="viridis")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    if np.isnan(v):
                        continue
                    color = "white" if (v > 0 and np.log(max(v, 1)) > 0.5 * np.log(vmax)) else "black"
                    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7, color=color)
            ax.set_xticks(range(len(k_vals)), k_vals)
            ax.set_yticks(range(len(n_vals)), n_vals)
            if col == 0:
                ax.set_ylabel(f"{title}\n$n$", fontsize=9)
            if row == 0:
                ax.set_title(noise_label, fontsize=10)
            if row == len(MANIFOLD_ORDER) - 1:
                ax.set_xlabel("$k$")

    fig.suptitle(
        "Number of composite-label shortcuts per cell (grey = zero, AUROC undefined)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("composite shortcuts (log scale)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
