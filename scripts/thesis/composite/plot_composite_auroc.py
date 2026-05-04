"""
Plot AUROC versus k under the composite-guarded(c=1) shortcut label.

Reads:
  benchmark_results/composite_shortcut_extended/extended_summary.csv

Produces:
  docs/thesis/figures/orc_auroc_vs_k_composite.pdf

One panel per manifold (Swiss roll standard, Swiss roll tight, three torus
aspect ratios). One line per detection signal. Cells in which the composite
label has zero positives or zero negatives are dropped before averaging
(AUROC is undefined there).

Restricted to noise=0.0 to match the rest of the alt chapter's headline
numbers; toggle NOISE_LEVEL below to switch.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "benchmark_results/composite_shortcut_extended/extended_summary.csv"
OUT = REPO.parent.parent / "docs/thesis/figures/orc_auroc_vs_k_composite.pdf"

NOISE_LEVEL = 0.0
AUROC_COL = "auroc_composite_guard_c1.0"
N_POS_COL = "n_composite_guard_c1.0"
N_KEPT_COL = "n_kept_guard_c1.0"

MANIFOLD_ORDER = [
    ("swiss_roll", "Swiss roll"),
    ("swiss_roll_tight_s005", "Swiss roll (tight)"),
    ("torus_R1_5_r1", r"Torus $R/r=1.5$"),
    ("torus_R2_r1", r"Torus $R/r=2$"),
    ("torus_R4_r1", r"Torus $R/r=4$"),
]

SIGNAL_ORDER = [
    ("neg_kappa_orcml", r"$-\kappa_{\mathrm{ManL}}$", "tab:blue", "-"),
    ("neg_kappa_std", r"$-\kappa_{\mathrm{std}}$", "tab:orange", "-"),
    ("one_minus_jaccard", r"$1 - J$", "tab:green", "-"),
    ("tangent_angle", "tangent angle", "tab:red", "-"),
    ("neg_kappa_zscore", r"$\kappa$ z-score", "tab:blue", "--"),
    ("angle_zscore", "angle z-score", "tab:red", "--"),
]


def main():
    df = pd.read_csv(SUMMARY)
    df = df[df.noise == NOISE_LEVEL].copy()
    # Drop cells where the composite label is degenerate (no positives or no negatives).
    df = df[(df[N_POS_COL] > 0) & (df[N_POS_COL] < df[N_KEPT_COL])]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, (manifold, title) in zip(axes, MANIFOLD_ORDER):
        sub = df[df.manifold == manifold]
        for sig_key, sig_label, color, ls in SIGNAL_ORDER:
            ssig = sub[sub.signal == sig_key]
            if ssig.empty:
                continue
            # Mean AUROC over n at each k (cells already filtered for non-degeneracy).
            agg = ssig.groupby("k")[AUROC_COL].mean().sort_index()
            if agg.empty:
                continue
            ax.plot(agg.index, agg.values, label=sig_label, color=color, linestyle=ls, marker="o", ms=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$k$")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks([5, 10, 15, 20, 30])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC (composite label)")
    axes[-1].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"AUROC vs $k$ under composite-guarded ($c=1$) label, $\\sigma={NOISE_LEVEL}$, averaged over $n$",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    # Also dump the per-(manifold, signal, k) means as a CSV for inspection.
    out_csv = OUT.with_suffix(".csv")
    grp = (
        df.groupby(["manifold", "signal", "k"])[AUROC_COL]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "auroc_mean", "count": "n_cells"})
    )
    grp.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
