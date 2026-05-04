"""
Produce three plots comparing the ManifoldANN.jl ORC-ManL implementation
against the reference Python orcml implementation. Reads:

  benchmark_results/manl_validation_pairs.csv

with columns (julia_curvature, python_curvature, abs_diff).

Outputs three PNG files into benchmark_results/:
  manl_validation_scatter.png      - scatter with y=x line
  manl_validation_bland_altman.png - Bland-Altman plot
  manl_validation_residual_hist.png - histogram of |julia - python|
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = HERE.parent / "benchmark_results" / "manl_validation_pairs.csv"
OUT = PAIRS.parent

julia, python = [], []
with open(PAIRS) as f:
    r = csv.DictReader(f)
    for row in r:
        julia.append(float(row["julia_curvature"]))
        python.append(float(row["python_curvature"]))

j = np.array(julia)
p = np.array(python)
diff = j - p
mean = (j + p) / 2.0

corr = float(np.corrcoef(j, p)[0, 1])
mae = float(np.mean(np.abs(diff)))
maxabs = float(np.max(np.abs(diff)))
print(f"n={len(j)}  Pearson r={corr:.4f}  MAE={mae:.4f}  max|Δ|={maxabs:.4f}")

# ---- Plot 1: scatter (Julia vs Python) -----------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(p, j, s=6, alpha=0.35, color="#1f77b4", edgecolor="none")
lim = [min(j.min(), p.min()) - 0.05, max(j.max(), p.max()) + 0.05]
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"orcml (Python) $\kappa$")
ax.set_ylabel(r"ManifoldANN.jl $\kappa$")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig(OUT / "manl_validation_scatter.png", dpi=180)
fig.savefig(OUT / "manl_validation_scatter.pdf")
plt.close(fig)
print(f"  saved {OUT / 'manl_validation_scatter.png'} (+ .pdf)")

# ---- Plot 2: Bland-Altman (bias only) ------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(mean, diff, s=6, alpha=0.35, color="#1f77b4", edgecolor="none")
bias = float(np.mean(diff))
ax.axhline(bias, color="k", lw=1, label=f"bias = {bias:+.4f}")
ax.set_xlabel(r"mean $\kappa$ across implementations")
ax.set_ylabel(r"$\kappa_\mathrm{Julia} - \kappa_\mathrm{Python}$")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "manl_validation_bland_altman.png", dpi=180)
fig.savefig(OUT / "manl_validation_bland_altman.pdf")
plt.close(fig)
print(f"  saved {OUT / 'manl_validation_bland_altman.png'} (+ .pdf)")

# ---- Plot 3: residual histogram ------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.abs(diff), bins=60, color="#1f77b4", edgecolor="white")
ax.axvline(mae, color="k", lw=1, label=f"MAE = {mae:.4f}")
ax.axvline(maxabs, color="r", ls="--", lw=1,
           label=fr"max $|\Delta|$ = {maxabs:.4f}")
ax.set_xlabel(r"$|\kappa_\mathrm{Julia} - \kappa_\mathrm{Python}|$")
ax.set_ylabel("number of edges")
ax.set_title(f"Distribution of per-edge absolute differences ({len(j)} edges)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "manl_validation_residual_hist.png", dpi=180)
plt.close(fig)
print(f"  saved {OUT / 'manl_validation_residual_hist.png'}")
