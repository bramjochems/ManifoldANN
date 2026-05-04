"""
Plot the §6.4 end-to-end geodesic-pipeline experiment.

Reads `summary.csv` (and `results.csv` for slope fits) from a
`benchmark_results/geodesic_e2e/{timestamp}/` directory and produces:

  mre_vs_n.{png,pdf}            Headline log-log MRE vs n
  decomposition_vs_n.{png,pdf}  1x3 panel: edge-weight, path-selection, path-deviation
  slopes.csv                    Fitted log-log slopes per scheme

Usage:
  python scripts/plot_geodesic_e2e.py [path/to/run_dir]

If no directory is given, picks the most recent under
benchmark_results/geodesic_e2e/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE.parent / "benchmark_results" / "geodesic_e2e"

SCHEMES = ["d_E", "d_T_sym", "d_g_hat_sym", "d_analytic"]
SCHEME_COLORS = {
    "d_E": "#1f77b4",
    "d_T_sym": "#ff7f0e",
    "d_g_hat_sym": "#2ca02c",
    "d_analytic": "#d62728",
}
SCHEME_LABELS = {
    "d_E": r"$d_E$ (Euclidean chord)",
    "d_T_sym": r"$d_T^{\mathrm{sym}}$ (tangent-projected mean)",
    "d_g_hat_sym": r"$\hat d_g^{\mathrm{sym}}$ (curvature-free)",
    "d_analytic": r"$d_\mathcal{M}$ (analytic baseline)",
}


def resolve_run_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_dir():
            sys.exit(f"error: not a directory: {p}")
        return p
    if not DEFAULT_BASE.is_dir():
        sys.exit(f"error: no default base dir {DEFAULT_BASE}")
    candidates = sorted(p for p in DEFAULT_BASE.iterdir() if p.is_dir())
    if not candidates:
        sys.exit(f"error: no run dirs under {DEFAULT_BASE}")
    return candidates[-1]


def read_summary(run_dir: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Returns scheme -> n -> {col: value}."""
    out: dict[str, dict[int, dict[str, float]]] = {s: {} for s in SCHEMES}
    fp = run_dir / "summary.csv"
    if not fp.exists():
        sys.exit(f"error: missing {fp}")
    with open(fp) as f:
        for row in csv.DictReader(f):
            s = row["scheme"]
            if s not in out:
                continue
            n = int(row["n"])

            def fnum(x: str) -> float:
                return float("nan") if x in ("NaN", "") else float(x)

            out[s][n] = {k: fnum(v) for k, v in row.items() if k not in ("n", "scheme")}
            out[s][n]["n_pairs_used"] = int(float(row["n_pairs_used"]))
    return out


def fit_slope(ns: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(ys) & (ys > 0) & np.isfinite(ns) & (ns > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(ns[mask])
    y = np.log(ys[mask])
    slope, intercept = np.polyfit(x, y, 1)
    # crude SE via residuals
    if mask.sum() >= 3:
        yhat = slope * x + intercept
        resid = y - yhat
        sxx = np.sum((x - x.mean()) ** 2)
        sigma2 = np.sum(resid**2) / (mask.sum() - 2)
        se = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("nan")
    else:
        se = float("nan")
    return float(slope), se


def plot_mre_vs_n(summary, out_dir: Path) -> list[tuple[str, float, float]]:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    slopes_out: list[tuple[str, float, float]] = []

    for scheme in SCHEMES:
        rows = summary[scheme]
        if not rows:
            continue
        ns = np.array(sorted(rows.keys()), dtype=float)
        means = np.array([rows[int(n)]["mre_mean"] for n in ns])
        lo = np.array([rows[int(n)]["mre_ci_lo"] for n in ns])
        hi = np.array([rows[int(n)]["mre_ci_hi"] for n in ns])

        slope, se = fit_slope(ns, means)
        slopes_out.append((scheme, slope, se))
        slope_str = f"slope={slope:+.2f}" + (f"±{se:.2f}" if np.isfinite(se) else "")
        label = f"{SCHEME_LABELS[scheme]}  ({slope_str})"

        linestyle = "--" if scheme == "d_analytic" else "-"
        ax.plot(ns, means, marker="o", linestyle=linestyle,
                color=SCHEME_COLORS[scheme], label=label, lw=1.6, ms=5)
        # fill where CI is finite + positive
        m = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
        if m.any():
            ax.fill_between(ns[m], lo[m], hi[m], color=SCHEME_COLORS[scheme], alpha=0.12)

    # Reference slope guides anchored at smallest-n point of d_T_sym.
    ref_scheme = "d_T_sym" if summary.get("d_T_sym") else SCHEMES[0]
    rows = summary.get(ref_scheme, {})
    if rows:
        ns_sorted = sorted(rows.keys())
        n0 = float(ns_sorted[0])
        y0 = rows[int(n0)]["mre_mean"]
        if np.isfinite(y0) and y0 > 0:
            ns_grid = np.array(ns_sorted, dtype=float)
            for slope, ls, lbl in [(-2.0, "--", "ref slope $-2$"),
                                    (-3.0, ":", "ref slope $-3$")]:
                y_ref = y0 * (ns_grid / n0) ** slope
                ax.plot(ns_grid, y_ref, ls, color="gray", lw=1.0, label=lbl)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel(r"end-to-end MRE")
    ax.set_title(r"End-to-end MRE vs sample size, Swiss roll, $k=15$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mre_vs_n.png", dpi=300)
    fig.savefig(out_dir / "mre_vs_n.pdf")
    plt.close(fig)
    print(f"  saved {out_dir / 'mre_vs_n.png'} (+ .pdf)")
    return slopes_out


def plot_decomposition(summary, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    titles = [
        ("edge_weight_mre_mean", r"Edge-weight component"),
        ("path_selection_mre_mean", r"Path-selection component"),
        ("path_deviation_mre_mean", r"Path-deviation component"),
    ]
    for ax, (col, title) in zip(axes, titles):
        if col == "path_deviation_mre_mean":
            # Single curve (same for all schemes per n); take from analytic.
            scheme = "d_analytic" if summary.get("d_analytic") else SCHEMES[0]
            rows = summary[scheme]
            if rows:
                ns = np.array(sorted(rows.keys()), dtype=float)
                ys = np.array([rows[int(n)][col] for n in ns])
                m = np.isfinite(ys) & (ys > 0)
                if m.any():
                    ax.plot(ns[m], ys[m], marker="o", color="#444444", lw=1.6, ms=5,
                            label="path-deviation (shared)")
        else:
            for scheme in SCHEMES:
                rows = summary[scheme]
                if not rows:
                    continue
                ns = np.array(sorted(rows.keys()), dtype=float)
                ys = np.array([rows[int(n)][col] for n in ns])
                m = np.isfinite(ys) & (ys > 0)
                if not m.any():
                    continue
                ls = "--" if scheme == "d_analytic" else "-"
                ax.plot(ns[m], ys[m], marker="o", linestyle=ls,
                        color=SCHEME_COLORS[scheme], label=SCHEME_LABELS[scheme],
                        lw=1.5, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"sample size $n$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("mean of |error| (relative)")
    fig.suptitle(r"Three-component error decomposition, Swiss roll, $k=15$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition_vs_n.png", dpi=300)
    fig.savefig(out_dir / "decomposition_vs_n.pdf")
    plt.close(fig)
    print(f"  saved {out_dir / 'decomposition_vs_n.png'} (+ .pdf)")


def write_slopes_csv(slopes: list[tuple[str, float, float]], out_dir: Path) -> None:
    fp = out_dir / "slopes.csv"
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "slope", "slope_se"])
        for s, slope, se in slopes:
            w.writerow([s, f"{slope:.6f}", f"{se:.6f}"])
    print(f"  saved {fp}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Path to benchmark_results/geodesic_e2e/{timestamp}/")
    args = ap.parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    print(f"Reading: {run_dir}")
    summary = read_summary(run_dir)
    slopes = plot_mre_vs_n(summary, run_dir)
    plot_decomposition(summary, run_dir)
    write_slopes_csv(slopes, run_dir)


if __name__ == "__main__":
    main()
