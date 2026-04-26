#!/usr/bin/env python3
"""
Visualise ORC Sampling Diagnostic Results

Reads the per-vertex CSV produced by experiment_orc_sampling_diagnostic.jl
and generates two figures:

Figure 1 — Side-by-side heatmap comparison (for a chosen configuration):
  Left:  Swiss roll coloured by local sampling density
  Right: Swiss roll coloured by per-vertex mean ORC (κ̄)
  Shows whether low-density regions (sparse zone) correspond to depressed κ̄.

Figure 2 — Correlation summary:
  Scatter of per-vertex (local density, κ̄) coloured by zone membership,
  one panel per sparsity level (for a fixed k and ORC variant).

Figure 3 — Δκ̄ vs sparsity:
  Mean κ̄ in the sparse zone vs. dense zone as a function of sparsity fraction,
  one line per ORC variant.  Shows whether the gap grows as the zone empties.

Output PDFs go to docs/thesis/figures/ (matching the existing figure naming).

Usage:
    python scripts/plot_sampling_diagnostic.py PATH_TO_VERTEX_CSV PATH_TO_SUMMARY_CSV

If no paths are given, the script looks for the most recent
orc_sampling_diagnostic_*_vertex.csv and _summary.csv in
docs/thesis/results/orc_results/.
"""

import sys
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # non-interactive backend for PDF output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
PACKAGE_DIR  = SCRIPT_DIR.parent
RESULTS_DIR  = PACKAGE_DIR.parent.parent / "docs" / "thesis" / "results" / "orc_results"
FIGURES_DIR  = PACKAGE_DIR.parent.parent / "docs" / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Swiss roll parametrization (must match the Julia experiment constants)
T_MIN    = 1.5 * np.pi
T_RANGE  = 3.0 * np.pi
H_SCALE  = 10.0


def find_latest(pattern):
    """Return the most recently modified file matching the glob pattern."""
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    return Path(files[-1])


def swiss_roll_3d(t, h):
    """Embed (t, h) parameters into 3D ambient space."""
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if len(sys.argv) == 3:
    vertex_path  = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
else:
    # Try new directory layout first (sampling_diag_*/), fall back to legacy flat files
    diag_dirs = sorted(glob.glob(str(RESULTS_DIR / "sampling_diag_*")),
                       key=os.path.getmtime)
    diag_dirs = [d for d in diag_dirs if os.path.isdir(d)]
    if diag_dirs and (Path(diag_dirs[-1]) / "vertex.csv").exists():
        latest_dir = Path(diag_dirs[-1])
        vertex_path  = latest_dir / "vertex.csv"
        summary_path = latest_dir / "summary.csv"
    else:
        vertex_path  = find_latest(RESULTS_DIR / "orc_sampling_diagnostic_*_vertex.csv")
        summary_path = find_latest(RESULTS_DIR / "orc_sampling_diagnostic_*_summary.csv")

print(f"Vertex file  : {vertex_path}")
print(f"Summary file : {summary_path}")

vertex_df  = pd.read_csv(vertex_path,  na_values=["NA"])
summary_df = pd.read_csv(summary_path, na_values=["NA"])

print(f"Loaded {len(vertex_df)} vertex rows, {len(summary_df)} summary rows")
print(f"Sparsity levels : {sorted(vertex_df['sparsity'].unique())}")
print(f"k values        : {sorted(vertex_df['k'].unique())}")
print(f"ORC variants    : {sorted(vertex_df['variant'].unique())}")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

CMAP_DENSITY = "YlOrRd"    # yellow → red for density (high = warm)
CMAP_KAPPA   = "RdBu_r"    # red (negative) → blue (positive) for ORC
ZONE_COLORS  = {"sparse": "#d62728", "dense": "#1f77b4"}


# ---------------------------------------------------------------------------
# Figure 1 — Side-by-side heatmap for a chosen configuration
# ---------------------------------------------------------------------------

def make_heatmap_fig(vertex_df, sparsity=0.1, k=10, variant="standard"):
    """3D scatter of Swiss roll coloured by density (left) and ORC (right)."""
    sub = vertex_df[
        (vertex_df["sparsity"] == sparsity) &
        (vertex_df["k"]        == k)        &
        (vertex_df["variant"]  == variant)
    ].dropna(subset=["mean_kappa", "local_density"])

    if sub.empty:
        print(f"  [heatmap] no data for sparsity={sparsity}, k={k}, variant={variant}")
        return None

    t = sub["t_param"].values
    h = sub["h_param"].values
    xyz = swiss_roll_3d(t, h)

    density  = sub["local_density"].values
    kappa    = sub["mean_kappa"].values
    in_zone  = sub["in_sparse_zone"].astype(bool).values

    fig = plt.figure(figsize=(10, 4.5))
    fig.suptitle(
        f"Sampling diagnostic — Swiss roll  "
        f"(sparsity={sparsity:.2f}, k={k}, {variant} ORC)",
        y=1.01
    )

    for col, (values, cmap, label, sym) in enumerate([
        (density, CMAP_DENSITY, "Local density\n(pts per unit²)", False),
        (kappa,   CMAP_KAPPA,   "Mean vertex ORC  κ̄",            True),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        vmin = np.nanpercentile(values, 2)
        vmax = np.nanpercentile(values, 98)
        if sym:
            absmax = max(abs(vmin), abs(vmax))
            vmin, vmax = -absmax, absmax

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

        sc = ax.scatter(
            xyz[:, 0], xyz[:, 2], xyz[:, 1],   # x, z, y → nice view angle
            c=values, cmap=cmap_obj, norm=norm,
            s=8, alpha=0.7, linewidths=0,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, aspect=18)
        cb.set_label(label)

        # Highlight sparse zone outline
        zone_xyz = xyz[in_zone]
        if len(zone_xyz) > 0:
            ax.scatter(
                zone_xyz[:, 0], zone_xyz[:, 2], zone_xyz[:, 1],
                s=20, facecolors="none", edgecolors="black", linewidths=0.5,
                alpha=0.5, label="sparse zone"
            )
            ax.legend(loc="upper left", markerscale=1.5, handletextpad=0.4)

        ax.set_xlabel("x", labelpad=2)
        ax.set_ylabel("z", labelpad=2)
        ax.set_zlabel("y", labelpad=2)
        ax.tick_params(labelsize=7)
        ax.set_title(["(a) sampling density", "(b) mean ORC per vertex"][col])
        ax.view_init(elev=20, azim=-70)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Scatter: density vs. ORC coloured by zone
# ---------------------------------------------------------------------------

def make_scatter_fig(vertex_df, k=10, variant="standard"):
    """One panel per sparsity level: scatter of (density, κ̄) by zone."""
    sparsities = sorted(vertex_df["sparsity"].unique())
    n_cols = min(len(sparsities), 4)
    n_rows = (len(sparsities) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.0 * n_rows),
                              squeeze=False)
    fig.suptitle(f"Density vs. mean ORC  (k={k}, {variant})", y=1.02)

    for idx, sparsity in enumerate(sparsities):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = vertex_df[
            (vertex_df["sparsity"] == sparsity) &
            (vertex_df["k"]        == k)        &
            (vertex_df["variant"]  == variant)
        ].dropna(subset=["mean_kappa", "local_density"])

        if sub.empty:
            ax.set_visible(False)
            continue

        for zone, color, label in [
            (True,  ZONE_COLORS["sparse"], "sparse zone"),
            (False, ZONE_COLORS["dense"],  "dense zone"),
        ]:
            mask = sub["in_sparse_zone"].astype(bool) == zone
            ax.scatter(
                sub.loc[mask, "local_density"],
                sub.loc[mask, "mean_kappa"],
                c=color, s=6, alpha=0.5, linewidths=0, label=label,
            )

        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("local density")
        ax.set_ylabel("mean κ̄")
        ax.set_title(f"sparsity = {sparsity:.2f}")
        ax.legend(markerscale=2, handletextpad=0.3)

        # Annotate Pearson r
        valid = sub.dropna(subset=["mean_kappa", "local_density"])
        if len(valid) >= 3:
            r = np.corrcoef(valid["local_density"], valid["mean_kappa"])[0, 1]
            ax.text(0.97, 0.04, f"r = {r:.3f}",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=8, color="dimgray")

    # Hide unused subplots
    for idx in range(len(sparsities), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Δκ̄ vs sparsity
# ---------------------------------------------------------------------------

def make_delta_kappa_fig(summary_df, k=10):
    """Mean κ̄ in sparse vs. dense zone as a function of sparsity."""
    variants = sorted(summary_df["variant"].unique())
    colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=False)
    fig.suptitle(f"ORC mean by zone vs. sparsity  (k={k})", y=1.02)

    ax_kappa = axes[0]
    ax_delta = axes[1]

    for i, variant in enumerate(variants):
        sub = summary_df[
            (summary_df["variant"] == variant) &
            (summary_df["k"]       == k)
        ].sort_values("sparsity")

        color = colors[i % len(colors)]

        ax_kappa.plot(sub["sparsity"], sub["mean_kappa_zone"],
                      "o-", color=color, label=f"{variant} (sparse zone)")
        ax_kappa.plot(sub["sparsity"], sub["mean_kappa_dense"],
                      "s--", color=color, alpha=0.5, label=f"{variant} (dense zone)")

        delta = sub["mean_kappa_dense"] - sub["mean_kappa_zone"]
        ax_delta.plot(sub["sparsity"], delta,
                      "o-", color=color, label=variant)

    ax_kappa.axhline(0, color="gray", lw=0.7, ls="--")
    ax_kappa.set_xlabel("sparsity fraction")
    ax_kappa.set_ylabel("mean vertex ORC  κ̄")
    ax_kappa.set_title("(a) κ̄ by zone")
    ax_kappa.legend(fontsize=7)
    ax_kappa.invert_xaxis()   # 1.0 = uniform on left, more sparse to the right

    ax_delta.axhline(0, color="gray", lw=0.7, ls="--")
    ax_delta.set_xlabel("sparsity fraction")
    ax_delta.set_ylabel("Δκ̄  (dense − sparse zone)")
    ax_delta.set_title("(b) ORC gap between zones")
    ax_delta.legend()
    ax_delta.invert_xaxis()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generate and save figures
# ---------------------------------------------------------------------------

# Determine default k and variant for single-panel figures
k_default       = int(vertex_df["k"].mode()[0])
variant_default = str(vertex_df["variant"].mode()[0])

sparsities = sorted(vertex_df["sparsity"].unique())
# For Fig 1 choose the most extreme sparsity (best visual contrast)
sparsity_heatmap = min(s for s in sparsities if s < 1.0) if any(s < 1.0 for s in sparsities) else sparsities[0]

print(f"\nGenerating figures (k={k_default}, variant={variant_default})...")

fig1 = make_heatmap_fig(vertex_df, sparsity=sparsity_heatmap,
                        k=k_default, variant=variant_default)
if fig1 is not None:
    out1 = FIGURES_DIR / "orc_sampling_diagnostic_heatmap.pdf"
    fig1.savefig(out1)
    print(f"  Saved: {out1}")
    plt.close(fig1)

fig2 = make_scatter_fig(vertex_df, k=k_default, variant=variant_default)
out2 = FIGURES_DIR / "orc_sampling_diagnostic_scatter.pdf"
fig2.savefig(out2)
print(f"  Saved: {out2}")
plt.close(fig2)

fig3 = make_delta_kappa_fig(summary_df, k=k_default)
out3 = FIGURES_DIR / "orc_sampling_diagnostic_delta_kappa.pdf"
fig3.savefig(out3)
print(f"  Saved: {out3}")
plt.close(fig3)

print("\nDone.")
