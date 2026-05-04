"""
ORC Experiment — Unified Analysis and Thesis CSV/Figure Generator (v2)

Reads experiment outputs from the unified experiment_orc.jl:
  swiss_roll_{ts}/  — Swiss roll outputs
  torus_{ts}/       — Torus (R2r1) outputs

Produces:

  Detection CSVs (results/orc_results/):
    thesis_auroc_summary.csv      — AUROC by (manifold, n, k, noise, variant, tau)
    thesis_detection_summary.csv  — F1, κ stats aggregated

  Pruning CSVs (results/orc_results/):
    thesis_pruning_oracle.csv
    thesis_pruning_rank.csv
    thesis_pruning_random.csv

  Detection figures (docs/thesis/figures/):
    orc_frac_shortcuts_heatmap.pdf
    orc_auroc_vs_k.pdf
    orc_auroc_roc_curves.pdf
    orc_kappa_separation.pdf
    orc_runtime.pdf

  Pruning figures (docs/thesis/figures/):
    orc_pruning_oracle.pdf
    orc_pruning_rank_comparison.pdf
    orc_pruning_mre_vs_fraction.pdf

Usage:
    python3 scripts/analyze_orc.py [--skip-figures] [--swiss-dir PATH] [--torus-dir PATH]
"""

import argparse
import sys
import os
import glob as globmod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "docs", "thesis", "results", "orc_results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "docs", "thesis", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Mutable container so command-line overrides take effect without rewriting every call site
_OUTPUT_OVERRIDES = {"figures_dir": None}


def _fig_out(name):
    """Return the full path for a figure; honours --figures-out-dir."""
    base = _OUTPUT_OVERRIDES["figures_dir"] or FIGURES_DIR
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

MANIFOLD_LABELS = {
    "swiss": "Swiss roll",
    "R2r1":  r"Torus $R/r=2$",
}
MANIFOLD_ORDER = ["swiss", "R2r1"]

ORC_VARIANTS = ["standard", "orcml"]
VAR_LABELS   = {"standard": "Standard ORC", "orcml": "ORC-ManL"}
VAR_COLORS   = {"standard": "#1f77b4", "orcml": "#d62728"}
VAR_LS       = {"standard": "-", "orcml": "--"}

METHOD_LABELS = {
    "orc_rank":             "ORC rank",
    "orc_rank_bridge_safe": "ORC rank (bridge-safe)",
    "tangent_angle_rank":   "Tangent angle",
    "jaccard_rank":         "Jaccard",
    "orc_zscore_rank":      r"$\kappa$ z-score",
    "tangent_zscore_rank":  "Angle z-score",
    "gabriel":              "Gabriel",
}
METHOD_COLORS = {
    "orc_rank":             "#1f77b4",
    "orc_rank_bridge_safe": "#17becf",
    "tangent_angle_rank":   "#2ca02c",
    "jaccard_rank":         "#ff7f0e",
    "orc_zscore_rank":      "#9467bd",
    "tangent_zscore_rank":  "#8c564b",
    "gabriel":              "#e377c2",
}
METHOD_LS = {
    "orc_rank":             "-",
    "orc_rank_bridge_safe": "--",
    "tangent_angle_rank":   "-.",
    "jaccard_rank":         "-",
    "orc_zscore_rank":      "--",
    "tangent_zscore_rank":  "-.",
    "gabriel":              ":",
}

PRIMARY_TAU = 0.5
TAU_GRID = [0.5, 0.67, 0.8]


def best_n_for_manifold(df_edges, manifold, n_grid, tau=None, min_frac=0.01):
    """Pick the largest n where shortcuts comprise >= min_frac of edges.

    Considers edges at moderate k (10 <= k <= 20) to ensure ROC curves,
    AUROC-vs-k, and pruning figures all have non-trivial shortcut
    populations.  Falls back to the largest n overall if none qualify.
    """
    if tau is None:
        tau = PRIMARY_TAU
    for n in sorted(n_grid, reverse=True):
        sub = df_edges[(df_edges["manifold"] == manifold) &
                       (df_edges["n"] == n) &
                       (df_edges["noise_std"] == 0.0) &
                       (df_edges["k"] >= 10) & (df_edges["k"] <= 20)]
        if len(sub) == 0:
            continue
        frac = (sub["ratio"] < tau).sum() / len(sub)
        if frac >= min_frac:
            return n
    return max(n_grid)


# ==============================================================================
# Auto-discovery of run directories
# ==============================================================================

def find_latest_run_dir(prefix):
    """Find the most recent run directory matching prefix in RESULTS_DIR."""
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_*")
    candidates = sorted(globmod.glob(pattern))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None
    return candidates[-1]


# ==============================================================================
# Load data
# ==============================================================================

def load_raw(run_dir, label):
    """Load raw.csv from a run directory."""
    path = os.path.join(run_dir, "raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} raw.csv not found: {path}")
    print(f"Loading {label}: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, n: {sorted(df['n'].unique())}, k: {sorted(df['k'].unique())}")
    return df


def load_edges(run_dir):
    """Load edges.csv from a run directory (for AUROC)."""
    path = os.path.join(run_dir, "edges.csv")
    if not os.path.exists(path):
        return None
    print(f"Loading edges: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}")
    return df


def load_csv_if_exists(run_dir, filename):
    """Load a CSV from a run directory if it exists."""
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return None
    print(f"Loading {filename}: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}")
    return df


# ==============================================================================
# AUROC computation (from edges.csv with multi-τ support)
# ==============================================================================

def compute_auroc(kappas, labels):
    """Compute AUROC from per-edge kappa values and binary labels.

    Convention: lower kappa -> more likely shortcut (positive class).
    So we negate kappa as the score for ROC computation.
    """
    scores = -kappas
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"auroc": np.nan, "n_pos": n_pos, "n_neg": n_neg,
                "fpr": np.array([]), "tpr": np.array([])}

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1
        if i == len(sorted_labels) - 1 or sorted_scores[i] != sorted_scores[i + 1]:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auroc = np.trapezoid(tpr_arr, fpr_arr)

    return {"auroc": auroc, "n_pos": int(n_pos), "n_neg": int(n_neg),
            "fpr": fpr_arr, "tpr": tpr_arr}


def compute_angle_auroc(angles, labels):
    """Compute AUROC from tangent angles (higher angle = more likely shortcut).

    Convention: higher angle -> more likely shortcut. Use angle directly as score.
    """
    scores = angles
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"auroc": np.nan, "n_pos": n_pos, "n_neg": n_neg,
                "fpr": np.array([]), "tpr": np.array([])}

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1
        if i == len(sorted_labels) - 1 or sorted_scores[i] != sorted_scores[i + 1]:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auroc = np.trapezoid(tpr_arr, fpr_arr)

    return {"auroc": auroc, "n_pos": int(n_pos), "n_neg": int(n_neg),
            "fpr": fpr_arr, "tpr": tpr_arr}


def _auroc_for_column(values, labels, higher_is_shortcut):
    """Compute AUROC for a single score column.

    If higher_is_shortcut=True, use the score directly (high = positive).
    If higher_is_shortcut=False, negate the score (low = positive).
    Returns NaN if the column is missing or all-NaN.
    """
    if values is None:
        return np.nan
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return np.nan
    v, lb = values[valid], labels[valid]
    if lb.sum() == 0 or (len(lb) - lb.sum()) == 0:
        return np.nan
    if higher_is_shortcut:
        result = compute_angle_auroc(v, lb)  # uses score directly
    else:
        result = compute_auroc(v, lb)  # negates internally
    return result["auroc"]


def compute_auroc_table(df_edges):
    """Compute AUROC for each (manifold, n, k, noise, orc_variant) at multiple τ.

    Uses the ratio column to define shortcuts at each τ threshold.
    Computes AUROC for: ORC κ, tangent angle, Jaccard, Gabriel, κ z-score, angle z-score.
    """
    groups = df_edges.groupby(["manifold", "n", "k", "noise_std", "orc_variant"])
    rows = []
    for (manifold, n, k, noise, orc_var), grp in groups:
        kappas = grp["kappa"].values
        ratios = grp["ratio"].values

        # Extract optional columns
        def _col(name):
            if name in grp.columns and not grp[name].isna().all():
                return grp[name].values
            return None

        angles = _col("tangent_angle")
        jaccards = _col("jaccard")
        gabriels = _col("gabriel")
        kappa_zs = _col("kappa_zscore")
        angle_zs = _col("angle_zscore")

        for tau in TAU_GRID:
            labels = (ratios < tau).astype(bool)
            n_pos = labels.sum()
            n_neg = len(labels) - n_pos

            if n_pos == 0 or n_neg == 0:
                continue

            orc_result = compute_auroc(kappas, labels)
            row = {
                "manifold": manifold,
                "n": n,
                "k": k,
                "noise_std": noise,
                "orc_variant": orc_var,
                "tau": tau,
                "orc_auroc": orc_result["auroc"],
                "n_pos": n_pos,
                "n_neg": n_neg,
                "frac_pos": n_pos / len(labels),
            }

            # Tangent angle: higher = more likely shortcut
            row["tangent_auroc"] = _auroc_for_column(angles, labels, higher_is_shortcut=True)

            # Jaccard: lower = more likely shortcut (negate)
            row["jaccard_auroc"] = _auroc_for_column(jaccards, labels, higher_is_shortcut=False)

            # Gabriel: 0 = non-Gabriel = shortcut candidate. Use (1 - gabriel) as score.
            if gabriels is not None:
                gab_score = 1.0 - gabriels
                row["gabriel_auroc"] = _auroc_for_column(gab_score, labels, higher_is_shortcut=True)
            else:
                row["gabriel_auroc"] = np.nan

            # κ z-score: lower = more likely shortcut (negate)
            row["kappa_zscore_auroc"] = _auroc_for_column(kappa_zs, labels, higher_is_shortcut=False)

            # Angle z-score: higher = more likely shortcut
            row["angle_zscore_auroc"] = _auroc_for_column(angle_zs, labels, higher_is_shortcut=True)

            rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# Detection figures
# ==============================================================================

def fig_frac_shortcuts(df, manifold_key, n_grid, k_grid):
    """Heatmap of shortcut fraction across (n, k)."""
    tau = PRIMARY_TAU
    noises = [0.0, 0.5]

    fig, axes = plt.subplots(1, len(noises), figsize=(8, 3),
                             constrained_layout=True)

    all_vals = []
    for noise in noises:
        sub = df[(df["noise_std"] == noise) & (df["tau"] == tau) &
                 (df["variant"] == "standard")]
        if len(sub) > 0:
            all_vals.extend(sub["frac_shortcuts"].values)
    vmin = 0.0
    vmax = min(1.0, np.nanpercentile(all_vals, 98)) if all_vals else 1.0

    for j, noise in enumerate(noises):
        ax = axes[j]
        sub = df[(df["noise_std"] == noise) & (df["tau"] == tau) &
                 (df["variant"] == "standard")]
        mat = np.full((len(n_grid), len(k_grid)), np.nan)
        for ni, n in enumerate(n_grid):
            for ki, k in enumerate(k_grid):
                cell = sub[(sub["n"] == n) & (sub["k"] == k)]
                if len(cell) == 1:
                    mat[ni, ki] = cell["frac_shortcuts"].values[0]

        im = ax.imshow(mat, aspect="auto", origin="upper",
                       vmin=vmin, vmax=vmax, cmap="YlOrRd")
        ax.set_xticks(range(len(k_grid)));  ax.set_xticklabels(k_grid)
        ax.set_yticks(range(len(n_grid)));  ax.set_yticklabels(n_grid)
        ax.set_xlabel("$k$");  ax.set_ylabel("$n$")

        for ni in range(len(n_grid)):
            for ki in range(len(k_grid)):
                v = mat[ni, ki]
                if not np.isnan(v):
                    ax.text(ki, ni, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 0.5 else "white")

        sig = f"$\\sigma={noise}$"
        mlbl = MANIFOLD_LABELS.get(manifold_key, manifold_key)
        ax.set_title(f"{mlbl}, {sig}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Shortcut fraction ($\\tau={tau}$, standard ORC)", fontsize=11)
    out = _fig_out("orc_frac_shortcuts_heatmap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_auroc_vs_k(auroc_df, n_grid, k_grid, manifold_n=None):
    """AUROC vs k for each manifold, comparing all detection signals."""
    tau = PRIMARY_TAU
    noise = 0.0
    sub = auroc_df[(auroc_df["tau"] == tau) & (auroc_df["noise_std"] == noise)]
    manifolds = [m for m in MANIFOLD_ORDER if m in sub["manifold"].unique()]
    default_n = max(n_grid)

    # Signal definitions: (column_name, label, color, marker, linestyle)
    SIGNALS = [
        ("orc_auroc",           r"ORC $\kappa$",    "#1f77b4", "o", "-"),
        ("tangent_auroc",       "Tangent angle",     "#2ca02c", "^", "-."),
        ("jaccard_auroc",       "Jaccard",           "#ff7f0e", "D", "-"),
        ("gabriel_auroc",       "Gabriel",           "#e377c2", "p", ":"),
        ("kappa_zscore_auroc",  r"$\kappa$ z-score", "#9467bd", "s", "--"),
        ("angle_zscore_auroc",  "Angle z-score",     "#8c564b", "v", "-."),
    ]

    fig, axes = plt.subplots(1, len(manifolds), figsize=(5.5 * len(manifolds), 4),
                             constrained_layout=True, sharey=True)
    if len(manifolds) == 1:
        axes = [axes]

    for mi, m in enumerate(manifolds):
        ax = axes[mi]
        sel_n = manifold_n.get(m, default_n) if manifold_n else default_n

        # Plot each signal for standard ORC variant
        for col, label, color, marker, ls in SIGNALS:
            if col not in sub.columns:
                continue
            grp = sub[(sub["manifold"] == m) & (sub["orc_variant"] == "standard") &
                      (sub["n"] == sel_n)].sort_values("k")
            if len(grp) == 0 or grp[col].isna().all():
                continue
            ax.plot(grp["k"], grp[col], marker=marker, color=color,
                    linestyle=ls, linewidth=1.5, markersize=5, label=label)

        # Also show ORC-ManL κ for comparison
        grp_ml = sub[(sub["manifold"] == m) & (sub["orc_variant"] == "orcml") &
                      (sub["n"] == sel_n)].sort_values("k")
        if len(grp_ml) > 0 and not grp_ml["orc_auroc"].isna().all():
            ax.plot(grp_ml["k"], grp_ml["orc_auroc"], marker="o", color="#d62728",
                    linestyle="--", linewidth=1.5, markersize=5, label=r"ORC-ManL $\kappa$")

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="Random (0.5)")
        ax.set_xlabel("$k$");  ax.set_ylabel("AUROC")
        ax.set_xlim(k_grid[0] - 1, k_grid[-1] + 1)
        ax.set_ylim(0.3, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{MANIFOLD_LABELS.get(m, m)}, $n={sel_n}$")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"AUROC vs $k$ ($\\tau={tau}$, $\\sigma=0$)", fontsize=11)
    out = _fig_out("orc_auroc_vs_k.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_roc_curves(df_edges, n_grid, k_grid, manifold_n=None):
    """ROC curves for select configs, noise=0, k=10 and k=20."""
    default_n = max(n_grid)
    k_vals = [k for k in [10, 20] if k in df_edges["k"].unique()]
    manifolds = [m for m in MANIFOLD_ORDER if m in df_edges["manifold"].unique()]

    if not k_vals:
        print("  [skip] No k=10 or k=20 in edges data — skipping ROC curves")
        return

    # Signal definitions for ROC: (col, label, color, ls, higher_is_shortcut)
    ROC_SIGNALS = [
        ("kappa",         r"ORC $\kappa$",    "#1f77b4", "-",  False),
        ("tangent_angle", "Tangent angle",     "#2ca02c", "-.", True),
        ("jaccard",       "Jaccard",           "#ff7f0e", "-",  False),
        ("gabriel",       "Gabriel",           "#e377c2", ":",  False),  # 0=shortcut
        ("kappa_zscore",  r"$\kappa$ z-score", "#9467bd", "--", False),
        ("angle_zscore",  "Angle z-score",     "#8c564b", "-.", True),
    ]

    fig, axes = plt.subplots(len(k_vals), len(manifolds),
                             figsize=(4.5 * len(manifolds), 4 * len(k_vals)),
                             constrained_layout=True)
    if len(k_vals) == 1:
        axes = axes[np.newaxis, :] if len(manifolds) > 1 else np.array([[axes]])
    if len(manifolds) == 1:
        axes = axes[:, np.newaxis] if len(k_vals) > 1 else np.array([[axes]])

    tau = PRIMARY_TAU

    for ki, k in enumerate(k_vals):
        for mi, m in enumerate(manifolds):
            ax = axes[ki, mi]
            sel_n = manifold_n.get(m, default_n) if manifold_n else default_n

            # Standard ORC variant for all signals
            grp = df_edges[(df_edges["manifold"] == m) &
                           (df_edges["n"] == sel_n) &
                           (df_edges["k"] == k) &
                           (df_edges["noise_std"] == 0.0) &
                           (df_edges["orc_variant"] == "standard")]
            if len(grp) > 0:
                labels = (grp["ratio"].values < tau).astype(bool)
                for col, label, color, ls, higher in ROC_SIGNALS:
                    if col not in grp.columns:
                        continue
                    vals = grp[col].values
                    valid = ~np.isnan(vals)
                    if valid.sum() == 0 or labels[valid].sum() == 0:
                        continue
                    if higher:
                        result = compute_angle_auroc(vals[valid], labels[valid])
                    else:
                        result = compute_auroc(vals[valid], labels[valid])
                    if np.isnan(result["auroc"]):
                        continue
                    ax.plot(result["fpr"], result["tpr"],
                            color=color, linewidth=1.5, linestyle=ls,
                            label=f"{label} ({result['auroc']:.3f})")

            # ORC-ManL κ
            grp_ml = df_edges[(df_edges["manifold"] == m) &
                              (df_edges["n"] == sel_n) &
                              (df_edges["k"] == k) &
                              (df_edges["noise_std"] == 0.0) &
                              (df_edges["orc_variant"] == "orcml")]
            if len(grp_ml) > 0:
                labels_ml = (grp_ml["ratio"].values < tau).astype(bool)
                if labels_ml.sum() > 0:
                    result = compute_auroc(grp_ml["kappa"].values, labels_ml)
                    if not np.isnan(result["auroc"]):
                        ax.plot(result["fpr"], result["tpr"],
                                color="#d62728", linewidth=1.5, linestyle="--",
                                label=f"ORC-ManL ({result['auroc']:.3f})")

            ax.plot([0, 1], [0, 1], "k:", linewidth=0.7, label="Random")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{MANIFOLD_LABELS.get(m, m)}, $n={sel_n}$, $k={k}$")
            ax.legend(fontsize=6, loc="lower right")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")

    fig.suptitle(r"ROC Curves — shortcut detection ($\tau=" + f"{tau}$, $\\sigma=0$)",
                 fontsize=11)
    out = _fig_out("orc_auroc_roc_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_kappa_separation(df, k_grid):
    """Mean kappa for shortcuts vs non-shortcuts."""
    tau = PRIMARY_TAU
    noise = 0.0

    fig, axes = plt.subplots(1, len(ORC_VARIANTS), figsize=(9, 3.5),
                             constrained_layout=True, sharey=True)

    for j, variant in enumerate(ORC_VARIANTS):
        ax = axes[j]
        sub = df[(df["variant"] == variant) & (df["noise_std"] == noise) & (df["tau"] == tau)]
        k_vals, sc_means, nsc_means = [], [], []
        for k in k_grid:
            ks = sub[sub["k"] == k]
            if len(ks) > 0:
                k_vals.append(k)
                sc_means.append(ks["mean_kappa_shortcuts"].mean())
                nsc_means.append(ks["mean_kappa_non_shortcuts"].mean())
        ax.plot(k_vals, nsc_means, color=VAR_COLORS[variant], linestyle="-",
                marker="s", linewidth=1.5, markersize=5, label="Non-shortcut")
        ax.plot(k_vals, sc_means,  color=VAR_COLORS[variant], linestyle="--",
                marker="o", linewidth=1.5, markersize=5, label="Shortcut")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.fill_between(k_vals, sc_means, nsc_means,
                        alpha=0.12, color=VAR_COLORS[variant])
        ax.set_xlabel("$k$");  ax.set_ylabel(r"Mean $\kappa$")
        ax.set_xlim(k_grid[0] - 1, k_grid[-1] + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{VAR_LABELS[variant]}")
        ax.legend(fontsize=8)

    fig.suptitle(r"Mean $\kappa$ for shortcut vs non-shortcut edges ($\sigma=0$, $\tau=" +
                 f"{tau}$)", fontsize=11)
    out = _fig_out("orc_kappa_separation.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_runtime(df_all, n_grid, k_grid):
    """ORC vs PCA computation time comparison."""
    noise = 0.0
    tau = PRIMARY_TAU

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors_n = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_grid)))

    # Prefer orcml data, fall back to standard
    preferred_var = "orcml" if "orcml" in df_all["variant"].unique() else "standard"

    # ORC time (left)
    ax = axes[0]
    sub = df_all[(df_all["noise_std"] == noise) & (df_all["tau"] == tau) &
                 (df_all["variant"] == preferred_var)]
    for ci, n_val in enumerate(n_grid):
        row = sub[sub["n"] == n_val].sort_values("k")
        if len(row) > 0:
            ax.plot(row["k"], row["orc_time_s"], marker="o", color=colors_n[ci],
                    label=f"$n={n_val}$", linewidth=1.5, markersize=4)
    ax.set_xlabel("$k$");  ax.set_ylabel("Time (s)")
    ax.set_title(f"{VAR_LABELS.get(preferred_var, preferred_var)} computation time")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # PCA time (right)
    ax = axes[1]
    if "pca_time_s" in df_all.columns:
        for ci, n_val in enumerate(n_grid):
            row = sub[sub["n"] == n_val].sort_values("k")
            if len(row) > 0:
                ax.plot(row["k"], row["pca_time_s"], marker="s", color=colors_n[ci],
                        label=f"$n={n_val}$", linewidth=1.5, markersize=4)
    ax.set_xlabel("$k$");  ax.set_ylabel("Time (s)")
    ax.set_title("PCA tangent-plane fitting time")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Computation time comparison", fontsize=11)
    out = _fig_out("orc_runtime.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ==============================================================================
# Pruning figures
# ==============================================================================

def fig_pruning_oracle(oracle_dfs, n_grid, k_grid, manifold_n=None):
    """Oracle pruning: MRE vs τ for each manifold, comparing to unpruned baseline."""
    manifolds = [m for m in MANIFOLD_ORDER if m in oracle_dfs["manifold"].unique()]
    noise = 0.0
    default_n = max(n_grid)
    k_vals = [k for k in k_grid if k >= 10]

    # Use sharey=False so each manifold panel can tighten its y-axis to its own
    # data range (Swiss-roll MRE is ~10x smaller than torus, so a shared axis
    # crushes the Swiss-roll curves and hides per-k differences).
    fig, axes = plt.subplots(1, len(manifolds), figsize=(5 * len(manifolds), 4),
                             constrained_layout=True, sharey=False)
    if len(manifolds) == 1:
        axes = [axes]

    for mi, m in enumerate(manifolds):
        ax = axes[mi]
        sel_n = manifold_n.get(m, default_n) if manifold_n else default_n
        panel_vals = []
        for var in ORC_VARIANTS:
            for k in k_vals:
                sub = oracle_dfs[(oracle_dfs["manifold"] == m) &
                                 (oracle_dfs["n"] == sel_n) &
                                 (oracle_dfs["k"] == k) &
                                 (oracle_dfs["noise_std"] == noise) &
                                 (oracle_dfs["variant"] == var)].sort_values("tau")
                if len(sub) == 0:
                    continue
                ax.plot(sub["tau"], sub["mean_rel_error"],
                        marker="o", markersize=4, linewidth=1.5,
                        color=VAR_COLORS[var], linestyle=VAR_LS[var],
                        label=f"{VAR_LABELS[var]}, $k={k}$")
                panel_vals.extend(sub["mean_rel_error"].values)

        # Tighten y-axis to this panel's data range plus ~10% margin so the
        # per-k curves are visually distinguishable.
        if panel_vals:
            lo, hi = float(np.nanmin(panel_vals)), float(np.nanmax(panel_vals))
            pad = max((hi - lo) * 0.10, 1e-4)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)

        ax.set_xlabel(r"Shortcut threshold $\tau$")
        ax.set_ylabel("Mean relative error (MRE)")
        ax.set_title(f"{MANIFOLD_LABELS.get(m, m)}, $n={sel_n}$, $\\sigma=0$")
        ax.legend(fontsize=7)

    fig.suptitle("Oracle pruning: MRE after removing ground-truth shortcuts", fontsize=11)
    out = _fig_out("orc_pruning_oracle.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_pruning_rank_comparison(rank_dfs, random_dfs, n_grid, k_grid, manifold_n=None):
    """Compare rank-based pruning methods vs random baseline at each fraction p."""
    manifolds = [m for m in MANIFOLD_ORDER if m in rank_dfs["manifold"].unique()]
    noise = 0.0
    default_n = max(n_grid)
    k_select = 15 if 15 in k_grid else (10 if 10 in k_grid else k_grid[0])

    METHOD_MARKERS = {
        "orc_rank": "o", "orc_rank_bridge_safe": "s", "tangent_angle_rank": "^",
        "jaccard_rank": "D", "orc_zscore_rank": "v", "tangent_zscore_rank": "p",
        "gabriel": "*",
    }

    # sharey=False: torus MRE band sits at ~0.05 while Swiss spans 0.05-0.25,
    # so a shared y-axis squashes the torus panel into invisibility.
    fig, axes = plt.subplots(1, len(manifolds), figsize=(6 * len(manifolds), 4.5),
                             constrained_layout=True, sharey=False)
    if len(manifolds) == 1:
        axes = [axes]

    for mi, m in enumerate(manifolds):
        ax = axes[mi]
        sel_n = manifold_n.get(m, default_n) if manifold_n else default_n

        # ORC-ManL methods
        for method, label in METHOD_LABELS.items():
            sub = rank_dfs[(rank_dfs["manifold"] == m) &
                           (rank_dfs["n"] == sel_n) &
                           (rank_dfs["k"] == k_select) &
                           (rank_dfs["noise_std"] == noise) &
                           (rank_dfs["variant"] == "orcml") &
                           (rank_dfs["method"] == method) &
                           (rank_dfs["fraction"] <= 0.21)].sort_values("fraction")
            if len(sub) == 0:
                continue
            marker = METHOD_MARKERS.get(method, "o")
            ax.plot(sub["fraction"] * 100, sub["mean_rel_error"],
                    marker=marker, markersize=5, linewidth=1.5,
                    color=METHOD_COLORS[method], linestyle=METHOD_LS[method],
                    label=label)

        # Random baseline (mean + std over replicates)
        if random_dfs is not None:
            rand_sub = random_dfs[(random_dfs["manifold"] == m) &
                                  (random_dfs["n"] == sel_n) &
                                  (random_dfs["k"] == k_select) &
                                  (random_dfs["noise_std"] == noise)]
            if len(rand_sub) > 0:
                rand_agg = rand_sub.groupby("fraction")["mean_rel_error"].agg(["mean", "std"]).reset_index()
                rand_agg = rand_agg.sort_values("fraction")
                ax.plot(rand_agg["fraction"] * 100, rand_agg["mean"],
                        marker="x", markersize=5, linewidth=1.5,
                        color="gray", linestyle=":", label="Random")
                ax.fill_between(rand_agg["fraction"] * 100,
                                rand_agg["mean"] - rand_agg["std"],
                                rand_agg["mean"] + rand_agg["std"],
                                alpha=0.08, color="gray")

        ax.set_xlabel("Fraction removed (%)")
        ax.set_ylabel("Mean relative error (MRE)")
        ax.set_title(f"{MANIFOLD_LABELS.get(m, m)}\n$k={k_select}$, $n={sel_n}$, $\\sigma=0$")
        ax.legend(fontsize=6, loc="best")
        # Some pruning methods (e.g. Gabriel single-shot) can remove ~80% of
        # edges; restrict the x-axis to the comparable rank/random regime so
        # the meaningful 0–20% range is not visually compressed.
        ax.set_xlim(0, 21)

    fig.suptitle("Pruning method comparison (ORC-ManL)", fontsize=11)
    out = _fig_out("orc_pruning_rank_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_pruning_mre_vs_fraction(rank_dfs, random_dfs, n_grid, k_grid, manifold_n=None):
    """MRE vs fraction removed for multiple k values, ORC rank only."""
    manifolds = [m for m in MANIFOLD_ORDER if m in rank_dfs["manifold"].unique()]
    noise = 0.0
    default_n = max(n_grid)
    prunable_k = [k for k in k_grid if k >= 10]

    fig, axes = plt.subplots(len(manifolds), len(prunable_k),
                             figsize=(3.5 * len(prunable_k), 3.5 * len(manifolds)),
                             constrained_layout=True, sharey="row")
    if len(manifolds) == 1:
        axes = axes[np.newaxis, :] if len(prunable_k) > 1 else np.array([[axes]])
    if len(prunable_k) == 1:
        axes = axes[:, np.newaxis] if len(manifolds) > 1 else np.array([[axes]])

    for mi, m in enumerate(manifolds):
        sel_n = manifold_n.get(m, default_n) if manifold_n else default_n
        for ki, k in enumerate(prunable_k):
            ax = axes[mi, ki]

            for var in ORC_VARIANTS:
                sub = rank_dfs[(rank_dfs["manifold"] == m) &
                               (rank_dfs["n"] == sel_n) &
                               (rank_dfs["k"] == k) &
                               (rank_dfs["noise_std"] == noise) &
                               (rank_dfs["variant"] == var) &
                               (rank_dfs["method"] == "orc_rank")].sort_values("fraction")
                if len(sub) == 0:
                    continue
                ax.plot(sub["fraction"] * 100, sub["mean_rel_error"],
                        marker="o", markersize=3, linewidth=1.5,
                        color=VAR_COLORS[var], linestyle=VAR_LS[var],
                        label=VAR_LABELS[var])

            # Random baseline
            if random_dfs is not None:
                rand_sub = random_dfs[(random_dfs["manifold"] == m) &
                                      (random_dfs["n"] == sel_n) &
                                      (random_dfs["k"] == k) &
                                      (random_dfs["noise_std"] == noise)]
                if len(rand_sub) > 0:
                    rand_agg = rand_sub.groupby("fraction")["mean_rel_error"].agg(["mean"]).reset_index()
                    rand_agg = rand_agg.sort_values("fraction")
                    ax.plot(rand_agg["fraction"] * 100, rand_agg["mean"],
                            marker="x", markersize=4, linewidth=1,
                            color="gray", linestyle=":", label="Random")

            ax.set_xlabel("Fraction removed (%)")
            ax.set_ylabel("MRE")
            ax.set_title(f"{MANIFOLD_LABELS.get(m, m)}, $n={sel_n}$, $k={k}$")
            ax.legend(fontsize=6)

    fig.suptitle(f"ORC rank-based pruning: MRE vs edge removal ($\\sigma=0$)",
                 fontsize=11)
    out = _fig_out("orc_pruning_mre_vs_fraction.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ==============================================================================
# Additional thesis figures (Chapter 5: torus-only and clean-Swiss-roll views)
# ==============================================================================

# Map manifold key -> R/r ratio for torus rows. Both `R3r2`/`R4r1` and the
# float-stringified naming `R1_5r1_0`/`R4_0r1_0` are recognised.
TORUS_RATIO_KEYS = {
    "R3r2":     1.5,   # R/r = 3/2
    "R1_5r1_0": 1.5,   # R/r = 1.5/1.0 (float-stringified naming)
    "R2r1":     2.0,
    "R4r1":     4.0,
    "R4_0r1_0": 4.0,   # R/r = 4.0/1.0 (float-stringified naming)
    "R8r2":     4.0,
}


def _torus_keys_present(df):
    """Return torus manifold keys present in df, sorted by their R/r ratio."""
    keys = [m for m in df["manifold"].unique() if m in TORUS_RATIO_KEYS or m.startswith("R")]
    return sorted(keys, key=lambda m: TORUS_RATIO_KEYS.get(m, 99.0))


def _torus_label(key):
    if key in TORUS_RATIO_KEYS:
        return rf"Torus $R/r={TORUS_RATIO_KEYS[key]:g}$"
    return f"Torus {key}"


def fig_torus_frac_shortcuts_heatmap(df_torus, n_grid, k_grid):
    """Per-(R/r) torus shortcut-fraction heatmap.

    One row per torus aspect ratio present in the data; columns: σ=0, σ=0.5.
    Mirrors the layout of fig_frac_shortcuts but stacks rows instead of
    multiplexing manifolds into a single row.
    """
    tau = PRIMARY_TAU
    noises = [0.0, 0.5]
    torus_keys = _torus_keys_present(df_torus)
    if not torus_keys:
        print("  [skip] No torus manifold keys present — skipping torus shortcut heatmap")
        return

    # Shared color scale across all panels
    all_vals = []
    for m in torus_keys:
        for noise in noises:
            sub = df_torus[(df_torus["manifold"] == m) &
                           (df_torus["noise_std"] == noise) & (df_torus["tau"] == tau) &
                           (df_torus["variant"] == "standard")]
            if len(sub) > 0:
                all_vals.extend(sub["frac_shortcuts"].values)
    vmin = 0.0
    vmax = min(1.0, np.nanpercentile(all_vals, 98)) if all_vals else 1.0

    n_rows = len(torus_keys)
    fig, axes = plt.subplots(n_rows, len(noises),
                             figsize=(8, 3 * n_rows),
                             constrained_layout=True, squeeze=False)

    for ri, m in enumerate(torus_keys):
        for j, noise in enumerate(noises):
            ax = axes[ri, j]
            sub = df_torus[(df_torus["manifold"] == m) &
                           (df_torus["noise_std"] == noise) & (df_torus["tau"] == tau) &
                           (df_torus["variant"] == "standard")]
            mat = np.full((len(n_grid), len(k_grid)), np.nan)
            for ni, n in enumerate(n_grid):
                for ki, k in enumerate(k_grid):
                    cell = sub[(sub["n"] == n) & (sub["k"] == k)]
                    if len(cell) >= 1:
                        mat[ni, ki] = cell["frac_shortcuts"].mean()

            im = ax.imshow(mat, aspect="auto", origin="upper",
                           vmin=vmin, vmax=vmax, cmap="YlOrRd")
            ax.set_xticks(range(len(k_grid)));  ax.set_xticklabels(k_grid)
            ax.set_yticks(range(len(n_grid)));  ax.set_yticklabels(n_grid)
            ax.set_xlabel("$k$");  ax.set_ylabel("$n$")
            for ni in range(len(n_grid)):
                for ki in range(len(k_grid)):
                    v = mat[ni, ki]
                    if not np.isnan(v):
                        ax.text(ki, ni, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="black" if v < 0.5 * vmax + 0.5 * vmin else "white")
            sig = f"$\\sigma={noise}$"
            ax.set_title(f"{_torus_label(m)}, {sig}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Torus shortcut fraction ($\\tau={tau}$, standard ORC)", fontsize=11)
    out = _fig_out("orc_torus_frac_shortcuts_heatmap.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


def fig_f1_vs_k_clean(df_swiss, k_grid):
    """F1 (at κ=0 threshold) vs k for the clean Swiss roll, ORC vs ORC-ManL.

    Two-column layout: left = standard ORC, right = ORC-ManL. F1 averaged
    across n; individual n curves are overlaid in lighter colors.
    """
    tau = PRIMARY_TAU
    noise = 0.0
    sub = df_swiss[(df_swiss["noise_std"] == noise) & (df_swiss["tau"] == tau)]
    if len(sub) == 0 or "f1_0" not in sub.columns:
        print("  [skip] No f1_0 column or no clean Swiss data — skipping F1 vs k clean")
        return

    n_vals = sorted(sub["n"].unique().tolist())
    fig, axes = plt.subplots(1, len(ORC_VARIANTS), figsize=(9, 3.5),
                             constrained_layout=True, sharey=True)

    n_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(n_vals), 1)))

    for j, variant in enumerate(ORC_VARIANTS):
        ax = axes[j]
        var_sub = sub[sub["variant"] == variant]
        for ci, n_val in enumerate(n_vals):
            row = var_sub[var_sub["n"] == n_val].sort_values("k")
            if len(row) == 0:
                continue
            ax.plot(row["k"], row["f1_0"], color=n_colors[ci],
                    marker="o", markersize=4, linewidth=1.0, alpha=0.65,
                    label=f"$n={n_val}$")
        mean_curve = (var_sub.groupby("k")["f1_0"].mean().reset_index()
                                .sort_values("k"))
        if len(mean_curve) > 0:
            ax.plot(mean_curve["k"], mean_curve["f1_0"],
                    color=VAR_COLORS[variant], linestyle="-",
                    marker="s", markersize=6, linewidth=2.0,
                    label=f"Mean over $n$")
        ax.set_xlabel("$k$");  ax.set_ylabel("F1 at $\\kappa=0$")
        ax.set_xlim(k_grid[0] - 1, k_grid[-1] + 1)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{VAR_LABELS[variant]}")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(r"F1 at $\kappa=0$ vs $k$ — clean Swiss roll ($\sigma=0$, $\tau=" +
                 f"{tau}$)", fontsize=11)
    out = _fig_out("orc_f1_vs_k_clean.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


def fig_torus_kappa_separation(df_torus, k_grid):
    """Mean κ for shortcut vs non-shortcut edges on the torus, per ORC variant.

    Two-column layout: standard ORC (left) and ORC-ManL (right). When multiple
    torus aspect ratios are present, each ratio is overlaid as its own pair of
    curves; otherwise we degrade gracefully to a single (R/r=2) view.
    """
    tau = PRIMARY_TAU
    noise = 0.0
    torus_keys = _torus_keys_present(df_torus)
    if not torus_keys:
        print("  [skip] No torus data — skipping torus kappa separation")
        return

    fig, axes = plt.subplots(1, len(ORC_VARIANTS), figsize=(9, 3.5),
                             constrained_layout=True, sharey=True)

    if len(torus_keys) > 1:
        ratio_colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(torus_keys)))
    else:
        ratio_colors = [None]

    for j, variant in enumerate(ORC_VARIANTS):
        ax = axes[j]
        for ri, m in enumerate(torus_keys):
            sub = df_torus[(df_torus["manifold"] == m) &
                           (df_torus["variant"] == variant) &
                           (df_torus["noise_std"] == noise) & (df_torus["tau"] == tau)]
            k_vals, sc_means, nsc_means = [], [], []
            for k in k_grid:
                ks = sub[sub["k"] == k]
                if len(ks) > 0:
                    k_vals.append(k)
                    sc_means.append(ks["mean_kappa_shortcuts"].mean())
                    nsc_means.append(ks["mean_kappa_non_shortcuts"].mean())
            if not k_vals:
                continue
            base_color = (ratio_colors[ri] if len(torus_keys) > 1
                          else VAR_COLORS[variant])
            label_suffix = f" ({_torus_label(m)})" if len(torus_keys) > 1 else ""
            ax.plot(k_vals, nsc_means, color=base_color, linestyle="-",
                    marker="s", linewidth=1.5, markersize=5,
                    label=f"Non-shortcut{label_suffix}")
            ax.plot(k_vals, sc_means, color=base_color, linestyle="--",
                    marker="o", linewidth=1.5, markersize=5,
                    label=f"Shortcut{label_suffix}")
            if len(torus_keys) == 1:
                ax.fill_between(k_vals, sc_means, nsc_means,
                                alpha=0.12, color=base_color)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("$k$");  ax.set_ylabel(r"Mean $\kappa$")
        ax.set_xlim(k_grid[0] - 1, k_grid[-1] + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{VAR_LABELS[variant]}")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(r"Torus: mean $\kappa$ for shortcut vs non-shortcut edges ($\sigma=0$, $\tau=" +
                 f"{tau}$)", fontsize=11)
    out = _fig_out("orc_torus_kappa_separation.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ==============================================================================
# Console report
# ==============================================================================

def print_auroc_report(auroc_df):
    """Print AUROC summary table."""
    tau = PRIMARY_TAU
    noise = 0.0
    sub = auroc_df[(auroc_df["tau"] == tau) & (auroc_df["noise_std"] == noise)]

    auroc_cols = [
        ("orc_auroc", "ORC"),
        ("tangent_auroc", "Tang"),
        ("jaccard_auroc", "Jacc"),
        ("gabriel_auroc", "Gabr"),
        ("kappa_zscore_auroc", "k-z"),
        ("angle_zscore_auroc", "a-z"),
    ]

    print(f"\n## AUROC Summary (tau={tau}, noise=0)")
    hdr = f"{'manifold':8s} {'variant':10s} {'n':6s} {'k':4s}"
    for col, lbl in auroc_cols:
        hdr += f" {lbl:>6s}"
    hdr += f" {'frac':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in sub.sort_values(["manifold", "orc_variant", "n", "k"]).iterrows():
        line = f"{row['manifold']:8s} {row['orc_variant']:10s} {row['n']:6.0f} {row['k']:4.0f}"
        for col, _ in auroc_cols:
            v = row.get(col, np.nan)
            line += f" {v:6.3f}" if not pd.isna(v) else "    NA"
        line += f" {row['frac_pos']:6.3f}"
        print(line)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ORC experiment analysis (v2)")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--swiss-dir", type=str, help="Swiss roll run directory")
    parser.add_argument("--torus-dir", type=str, action="append", default=None,
                        help="Torus run directory; pass multiple times to combine "
                             "torus runs at different R/r aspect ratios")
    parser.add_argument("--figures-out-dir", type=str, default=None,
                        help="Override the directory where PDFs/PNGs are written "
                             "(default: docs/thesis/figures). Useful for comparing "
                             "regenerated figures without overwriting the originals.")
    args = parser.parse_args()

    if args.figures_out_dir:
        _OUTPUT_OVERRIDES["figures_dir"] = os.path.abspath(args.figures_out_dir)
        print(f"Figures output overridden -> {_OUTPUT_OVERRIDES['figures_dir']}")

    print("=" * 80)
    print("ORC EXPERIMENT — UNIFIED ANALYSIS (v2)")
    print("=" * 80)

    # ── Find run directories ──────────────────────────────────────────────────
    swiss_dir = args.swiss_dir or find_latest_run_dir("swiss_roll")
    if args.torus_dir:
        torus_dirs = list(args.torus_dir)
    else:
        latest_torus = find_latest_run_dir("torus")
        torus_dirs = [latest_torus] if latest_torus is not None else []

    df_raw_list = []
    df_edges_list = []
    df_oracle_list = []
    df_rank_list = []
    df_random_list = []

    n_grids = {}
    k_grids = {}

    sources = [("Swiss roll", swiss_dir, "swiss_roll")]
    for td in torus_dirs:
        sources.append((f"Torus ({os.path.basename(td)})", td, "torus"))

    for label, run_dir, prefix in sources:
        if run_dir is None:
            print(f"\n[warn] No {label} run directory found — skipping")
            continue

        print(f"\n{label} directory: {run_dir}")

        # Raw data
        try:
            df = load_raw(run_dir, label)
            df_raw_list.append(df)
            manifold_key = df["manifold"].iloc[0]
            n_grids[manifold_key] = sorted(df["n"].unique().tolist())
            k_grids[manifold_key] = sorted(df["k"].unique().tolist())
        except FileNotFoundError as e:
            print(f"  [warn] {e}")

        # Edges
        e = load_edges(run_dir)
        if e is not None:
            df_edges_list.append(e)

        # Pruning results
        for fname, target_list in [("oracle.csv", df_oracle_list),
                                   ("rank_pruning.csv", df_rank_list),
                                   ("random_pruning.csv", df_random_list)]:
            d = load_csv_if_exists(run_dir, fname)
            if d is not None:
                target_list.append(d)

    if not df_raw_list:
        print("\nNo data found. Nothing to do.")
        sys.exit(1)

    df_all_raw = pd.concat(df_raw_list, ignore_index=True)
    df_all_edges = pd.concat(df_edges_list, ignore_index=True) if df_edges_list else None
    df_all_oracle = pd.concat(df_oracle_list, ignore_index=True) if df_oracle_list else None
    df_all_rank = pd.concat(df_rank_list, ignore_index=True) if df_rank_list else None
    df_all_random = pd.concat(df_random_list, ignore_index=True) if df_random_list else None

    # Combined grids
    all_n = sorted(df_all_raw["n"].unique().tolist())
    all_k = sorted(df_all_raw["k"].unique().tolist())

    # ── AUROC computation ────────────────────────────────────────────────────
    auroc_df = None
    if df_all_edges is not None:
        print("\n--- Computing AUROC ---")
        auroc_df = compute_auroc_table(df_all_edges)
        if len(auroc_df) > 0:
            fpath = os.path.join(RESULTS_DIR, "thesis_auroc_summary.csv")
            auroc_df.to_csv(fpath, index=False, float_format="%.6f")
            print(f"  Wrote {fpath}")
            print_auroc_report(auroc_df)
        else:
            print("  No configs with shortcuts found — skipping AUROC")
            auroc_df = None

    # ── Write thesis CSVs ─────────────────────────────────────────────────────
    print("\n--- Writing thesis CSVs ---")

    if df_all_raw is not None:
        fpath = os.path.join(RESULTS_DIR, "thesis_detection_summary.csv")
        df_all_raw.to_csv(fpath, index=False, float_format="%.6f")
        print(f"  Wrote {fpath}")

    if df_all_oracle is not None:
        fpath = os.path.join(RESULTS_DIR, "thesis_pruning_oracle.csv")
        df_all_oracle.to_csv(fpath, index=False, float_format="%.6f")
        print(f"  Wrote {fpath}")

    if df_all_rank is not None:
        fpath = os.path.join(RESULTS_DIR, "thesis_pruning_rank.csv")
        df_all_rank.to_csv(fpath, index=False, float_format="%.6f")
        print(f"  Wrote {fpath}")

    if df_all_random is not None:
        fpath = os.path.join(RESULTS_DIR, "thesis_pruning_random.csv")
        df_all_random.to_csv(fpath, index=False, float_format="%.6f")
        print(f"  Wrote {fpath}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.skip_figures:
        print("\n--- Generating figures ---")

        # Per-manifold n selection: use largest n that still has shortcuts
        manifold_n = {}
        if df_all_edges is not None:
            for m in df_all_edges["manifold"].unique():
                manifold_n[m] = best_n_for_manifold(df_all_edges, m, all_n)
            print(f"  Per-manifold n for detection/pruning: {manifold_n}")

        # Detection figures
        # Use Swiss roll data for heatmap and kappa separation (if available)
        swiss_raw = df_all_raw[df_all_raw["manifold"] == "swiss"] if "swiss" in df_all_raw["manifold"].values else None
        if swiss_raw is not None and len(swiss_raw) > 0:
            sn = n_grids.get("swiss", all_n)
            sk = k_grids.get("swiss", all_k)
            fig_frac_shortcuts(swiss_raw, "swiss", sn, sk)
            fig_kappa_separation(swiss_raw, sk)
            # Clean Swiss roll F1 vs k (Chapter 5, Fig 5.5)
            fig_f1_vs_k_clean(swiss_raw, sk)

        # Torus-only figures (Chapter 5, Fig 5.2 and Fig 5.7)
        torus_raw = df_all_raw[df_all_raw["manifold"].astype(str).str.startswith("R")]
        if torus_raw is not None and len(torus_raw) > 0:
            tn = sorted(torus_raw["n"].unique().tolist())
            tk = sorted(torus_raw["k"].unique().tolist())
            fig_torus_frac_shortcuts_heatmap(torus_raw, tn, tk)
            fig_torus_kappa_separation(torus_raw, tk)

        if auroc_df is not None and len(auroc_df) > 0:
            fig_auroc_vs_k(auroc_df, all_n, all_k, manifold_n=manifold_n)

        if df_all_edges is not None:
            fig_roc_curves(df_all_edges, all_n, all_k, manifold_n=manifold_n)

        fig_runtime(df_all_raw, all_n, all_k)

        # Pruning figures
        if df_all_oracle is not None and len(df_all_oracle) > 0:
            fig_pruning_oracle(df_all_oracle, all_n, all_k, manifold_n=manifold_n)

        if df_all_rank is not None and len(df_all_rank) > 0:
            fig_pruning_rank_comparison(df_all_rank, df_all_random, all_n, all_k, manifold_n=manifold_n)
            fig_pruning_mre_vs_fraction(df_all_rank, df_all_random, all_n, all_k, manifold_n=manifold_n)

    print("\nDone. All outputs written.")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  CSVs:    {RESULTS_DIR}")


if __name__ == "__main__":
    main()
