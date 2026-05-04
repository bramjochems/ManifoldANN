"""All-pairs MRE on Swiss roll, per pruning method, to check whether the
500-pair sampled MRE in rank_pruning.csv is representative.

For each Swiss roll (n, k, noise) cell with k>=10 (where pruning is meaningful):
  - regenerate the point cloud and the analytic geodesic for every pair
  - score every undirected edge with each detection signal
  - for each pruning fraction p in {0.01, 0.02, 0.05, 0.10, 0.20}:
      - drop top-p by signal, recompute Dijkstra all-pairs distances on the pruned graph
      - compute MRE on ALL n(n-1)/2 pairs
  - also run oracle (drop composite-label shortcuts) and a random baseline
  - emit one CSV row per (cell, signal, fraction)

Output: benchmark_results/composite_shortcut_extended/all_pairs_mre_swiss.csv
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "thesis" / "composite"))
from composite_shortcut_full_eval import RUNS, regen_points_via_julia  # noqa: E402

OUT = ROOT / "benchmark_results" / "composite_shortcut_extended"
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / "all_pairs_mre_swiss.csv"

FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
RNG = np.random.default_rng(0)


def swiss_geodesic_matrix(t: np.ndarray, h: np.ndarray, radial_scale: float) -> np.ndarray:
    """Closed-form geodesic for swiss roll. (n, n) array."""
    arc = lambda x: 0.5 * (x * np.sqrt(1 + x * x) + np.arcsinh(x))
    s = radial_scale * arc(t)  # (n,)
    ds = s[:, None] - s[None, :]
    dh = h[:, None] - h[None, :]
    return np.sqrt(ds * ds + dh * dh)


def all_pairs_mre(graph_sparse, dM_full: np.ndarray) -> float:
    """Mean relative error over all i<j pairs."""
    n = dM_full.shape[0]
    dG = dijkstra(graph_sparse, directed=False)
    mask = np.triu(np.ones((n, n), bool), k=1) & np.isfinite(dG) & (dM_full > 0)
    rel = np.abs(dG[mask] - dM_full[mask]) / dM_full[mask]
    return float(rel.mean())


def build_sparse(edges_uv: np.ndarray, weights: np.ndarray, n: int):
    a = edges_uv[:, 0]
    b = edges_uv[:, 1]
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([weights, weights])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def process_cell(run, df_cell: pd.DataFrame, n: int, k: int, noise_std: float, per_edge_path: Path):
    """Use the existing per-edge dump for signals, but compute all-pairs MRE here."""
    if not per_edge_path.exists():
        return []
    pe = pd.read_csv(per_edge_path) if per_edge_path.suffix == ".csv" else pd.read_parquet(per_edge_path)

    pts = regen_points_via_julia(run, n, noise_std)
    coords = pts[["x", "y", "z"]].to_numpy()
    t_arr = pts["p1"].to_numpy()
    h_arr = pts["p2"].to_numpy()
    radial = run.get("radial_scale", 1.0)
    dM = swiss_geodesic_matrix(t_arr, h_arr, radial)

    # per-edge dump uses 1-indexed Julia node ids
    a_arr = pe.a.to_numpy() - 1
    b_arr = pe.b.to_numpy() - 1
    chord = pe.chord.to_numpy()
    edges_uv = np.column_stack([a_arr, b_arr])

    # Composite label (rebuild from per-edge: ratio<0.67 AND s_ij>2 AND chord >= eps)
    composite = ((pe.ratio < 0.67) & (pe.s_ij > 2) & (pe.chord >= pe.eps_local)).to_numpy()

    # Unpruned baseline
    base_graph = build_sparse(edges_uv, chord, n)
    base_mre = all_pairs_mre(base_graph, dM)

    # Signal scores (higher = more shortcut-like)
    signals = {}
    if "tangent_angle" in pe.columns:
        signals["tangent_angle"] = pe.tangent_angle.to_numpy()
    if "kappa" in pe.columns:
        signals["neg_kappa_std"] = -pe["kappa"].to_numpy()
    if "kappa_orcml" in pe.columns:
        ko = pe["kappa_orcml"].fillna(pe["kappa"]).to_numpy()
        signals["neg_kappa_orcml"] = -ko
    if "jaccard" in pe.columns:
        signals["one_minus_jaccard"] = 1.0 - pe.jaccard.to_numpy()
    if "kappa_zscore" in pe.columns:
        signals["neg_kappa_zscore"] = -pe.kappa_zscore.to_numpy()
    if "angle_zscore" in pe.columns:
        signals["angle_zscore"] = pe.angle_zscore.to_numpy()

    rows = []
    for sig_name, score in signals.items():
        score = np.asarray(score, dtype=float)
        # NaNs sort to bottom (lowest score)
        score_sort = np.where(np.isfinite(score), score, -np.inf)
        order = np.argsort(score_sort)[::-1]  # high score first = drop first
        m = len(order)
        for p in FRACTIONS:
            n_drop = int(round(p * m))
            keep = np.ones(m, bool)
            keep[order[:n_drop]] = False
            g = build_sparse(edges_uv[keep], chord[keep], n)
            mre = all_pairs_mre(g, dM)
            rows.append({
                "manifold": run["label"], "n": n, "k": k, "noise": noise_std,
                "method": f"rank_{sig_name}", "fraction": p,
                "mre_all_pairs": mre, "mre_baseline": base_mre,
                "n_dropped": int(n_drop),
            })

    # Oracle: drop composite shortcuts only (ignores fraction)
    keep = ~composite
    g = build_sparse(edges_uv[keep], chord[keep], n)
    mre = all_pairs_mre(g, dM)
    rows.append({
        "manifold": run["label"], "n": n, "k": k, "noise": noise_std,
        "method": "oracle_composite", "fraction": float(composite.sum() / m),
        "mre_all_pairs": mre, "mre_baseline": base_mre,
        "n_dropped": int(composite.sum()),
    })

    # Random baseline averaged over 5 replicates, at each p
    for p in FRACTIONS:
        n_drop = int(round(p * m))
        mres = []
        rrng = np.random.default_rng(seed=hash((n, k, noise_std, p)) & 0xFFFFFFFF)
        for _ in range(5):
            idx = rrng.choice(m, n_drop, replace=False)
            keep = np.ones(m, bool)
            keep[idx] = False
            g = build_sparse(edges_uv[keep], chord[keep], n)
            mres.append(all_pairs_mre(g, dM))
        rows.append({
            "manifold": run["label"], "n": n, "k": k, "noise": noise_std,
            "method": "random", "fraction": p,
            "mre_all_pairs": float(np.mean(mres)), "mre_baseline": base_mre,
            "n_dropped": int(n_drop),
        })

    return rows


def main():
    swiss_runs = [r for r in RUNS if r["kind"] == "swiss"]
    t0 = time.time()
    all_rows = []
    PER_EDGE = OUT / "per_edge"

    for run in swiss_runs:
        edges_path = run["edges"]
        df_all = pd.read_csv(edges_path)
        cells = sorted(df_all.groupby(["n", "k", "noise_std"]).groups.keys())
        for (n, k, noise_std) in cells:
            if k < 10:
                continue
            t1 = time.time()
            df_cell = df_all[(df_all.n == n) & (df_all.k == k) & (df_all.noise_std == noise_std)]
            stem = f"{run['label']}_n{int(n)}_k{int(k)}_noise{float(noise_std):.2f}"
            per_edge_path = PER_EDGE / f"{stem}.parquet"
            if not per_edge_path.exists():
                per_edge_path = PER_EDGE / f"{stem}.csv"
            try:
                rows = process_cell(run, df_cell, int(n), int(k), float(noise_std), per_edge_path)
            except Exception as e:
                print(f"  cell n={n} k={k} noise={noise_std} FAILED: {e}")
                continue
            all_rows.extend(rows)
            print(f"  {run['label']} n={n:5d} k={k:3d} noise={noise_std:.2f}  rows={len(rows):3d}  ({time.time()-t1:.1f}s)")
            if len(all_rows) % 200 < 30:
                pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\n{len(all_rows)} rows in {time.time()-t0:.1f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
