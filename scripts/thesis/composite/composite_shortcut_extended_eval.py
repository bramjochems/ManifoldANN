r"""Extended composite shortcut evaluation.

Builds on composite_shortcut_full_eval.py and adds three pieces flagged by
TODO_chapter_shortcuts.md:

  3. Min-chord guard at c in {0.5, 1.0}: edges with chord < c * eps(i,j) where
     eps is the local kNN scale (mean of the median kNN distances at endpoints)
     are EXCLUDED from the AUROC denominator. This filters discretisation
     false positives on the torus.

  + Spearman correlation between each detection signal and the continuous
     graph-effect score s_ij. Threshold-free, label-free; complements AUROC.

  + (Torus only) load orcml's algorithmic shortcut flag from per-cell CSVs
     produced by orcml_torus_flag.py for direct comparison.

For each (manifold, n, k, noise) cell, output one row per signal with:
  - AUROC under labels: chord-only, composite, composite_guarded(c=1), composite_guarded(c=0.5)
  - Spearman rho against s_ij (full edge set; guarded edge set)
  - For torus cells, AUROC under orcml's shortcut flag and Jaccard agreement
    of orcml's flag with the chord/composite/guarded labels.

Outputs:
  benchmark_results/composite_shortcut_extended/extended_summary.csv
  benchmark_results/composite_shortcut_extended/per_edge/<manifold>_<n>_<k>_<noise>.parquet
  benchmark_results/composite_shortcut_extended/REPORT.md
"""

from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark_results" / "composite_shortcut_extended_v2"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "per_edge").mkdir(exist_ok=True)

R_THRESH = 0.67
S_THRESH = 2.0
S_CAP = 100.0
GUARD_CS = (1.0, 0.5)

# Reuse RUNS, helpers, JULIA helper from full_eval module
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from composite_shortcut_full_eval import (  # noqa: E402
    RUNS, regen_points_via_julia, build_adj, dijkstra_skip_edge,
)

ORCML_FLAG_DIR = ROOT / "benchmark_results" / "composite_shortcut_extended" / "orcml_flags"


def local_eps(coords: np.ndarray, k: int) -> np.ndarray:
    """For each point, return distance to its k-th nearest neighbour
    (in ambient space). Cheap O(N^2 log k) — fine at N<=2000."""
    n = coords.shape[0]
    eps = np.empty(n)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        # k-th smallest
        eps[i] = np.partition(d, k)[k]
    return eps


def jaccard_index(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    inter = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    return float(inter / union) if union > 0 else float("nan")


def safe_auroc(label: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return float("nan")
    lab = label[mask]
    sc = score[mask]
    if lab.sum() in (0, mask.sum()):
        return float("nan")
    return float(roc_auc_score(lab, sc))


def safe_spearman(score: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return float("nan")
    sc = score[mask]
    tg = target[mask]
    if np.allclose(sc.std(), 0) or np.allclose(tg.std(), 0):
        return float("nan")
    rho, _ = spearmanr(sc, tg)
    return float(rho) if np.isfinite(rho) else float("nan")


def process_cell(run, df_cell: pd.DataFrame, n: int, k: int, noise_std: float):
    df_std = df_cell[df_cell.orc_variant == "standard"].copy()
    df_orc = df_cell[df_cell.orc_variant == "orcml"].copy()
    if len(df_std) == 0:
        return [], None

    orc_map = dict(zip(zip(df_orc.edge_i, df_orc.edge_j), df_orc.kappa))
    df_std["kappa_orcml"] = [orc_map.get((int(i), int(j)), np.nan)
                              for i, j in zip(df_std.edge_i, df_std.edge_j)]

    df_std["a"] = np.minimum(df_std.edge_i, df_std.edge_j).astype(int)
    df_std["b"] = np.maximum(df_std.edge_i, df_std.edge_j).astype(int)
    agg_dict = {
        "kappa": "mean",
        "kappa_orcml": "mean",
        "ratio": "mean",
        "tangent_angle": "mean",
        "jaccard": "mean",
        "gabriel": "mean",
        "kappa_zscore": "mean",
        "angle_zscore": "mean",
    }
    agg_dict = {k_: v for k_, v in agg_dict.items() if k_ in df_std.columns}
    agg = df_std.groupby(["a", "b"]).agg(**{c: (c, fn) for c, fn in agg_dict.items()}).reset_index()
    agg = agg[agg.a != agg.b].reset_index(drop=True)

    pts = regen_points_via_julia(run, n, noise_std)
    coords = pts[["x", "y", "z"]].to_numpy()

    a_arr = agg.a.to_numpy() - 1
    b_arr = agg.b.to_numpy() - 1
    chord = np.linalg.norm(coords[a_arr] - coords[b_arr], axis=1)
    agg["chord"] = chord

    # Local sampling scale: mean of endpoint k-th-NN distances.
    eps_per_pt = local_eps(coords, k)
    agg["eps_local"] = 0.5 * (eps_per_pt[a_arr] + eps_per_pt[b_arr])

    # Per-edge Dijkstra for s_ij
    edges_uv = np.column_stack([a_arr, b_arr])
    adj = build_adj(edges_uv, chord, n)
    s_vals = np.empty(len(agg))
    for idx in range(len(agg)):
        a = int(a_arr[idx]); b = int(b_arr[idx])
        w_ab = float(chord[idx])
        d_skip = dijkstra_skip_edge(adj, a, b, a, b, cap=S_CAP * w_ab)
        s_vals[idx] = S_CAP if not math.isfinite(d_skip) else min(d_skip / w_ab, S_CAP)
    agg["s_ij"] = s_vals

    # Labels
    chord_label = (agg.ratio < R_THRESH).to_numpy()
    composite_label = chord_label & (agg.s_ij > S_THRESH).to_numpy()

    # Guard masks: keep edge in evaluation iff chord >= c * eps_local
    masks = {
        "all": np.ones(len(agg), bool),
    }
    for c in GUARD_CS:
        masks[f"guard_c{c}"] = agg.chord.to_numpy() >= c * agg.eps_local.to_numpy()

    # Optional: orcml algorithmic flag
    orcml_flag = None
    flag_path = ORCML_FLAG_DIR / f"{run['label']}_n{n}_k{k}_noise{noise_std:.2f}.csv"
    if flag_path.exists():
        flag_df = pd.read_csv(flag_path)
        # flag_df columns: a, b, shortcut (0/1)  with a<b, 0-indexed
        flag_map = {(int(r.a), int(r.b)): int(r.shortcut) for r in flag_df.itertuples()}
        orcml_flag = np.array([flag_map.get((int(a_arr[i]), int(b_arr[i])), 0)
                               for i in range(len(agg))], dtype=bool)
        agg["orcml_shortcut"] = orcml_flag

    # Signals (higher = more shortcut-like)
    signals = {}
    if "tangent_angle" in agg.columns:
        signals["tangent_angle"] = agg.tangent_angle.to_numpy()
    if "kappa" in agg.columns:
        signals["neg_kappa_std"] = -agg["kappa"].to_numpy()
    if "kappa_orcml" in agg.columns:
        ko = agg["kappa_orcml"].fillna(agg["kappa"]).to_numpy()
        signals["neg_kappa_orcml"] = -ko
    if "jaccard" in agg.columns:
        signals["one_minus_jaccard"] = 1.0 - agg.jaccard.to_numpy()
    if "gabriel" in agg.columns:
        signals["one_minus_gabriel"] = 1.0 - agg.gabriel.to_numpy()
    if "kappa_zscore" in agg.columns:
        signals["neg_kappa_zscore"] = -agg.kappa_zscore.to_numpy()
    if "angle_zscore" in agg.columns:
        signals["angle_zscore"] = agg.angle_zscore.to_numpy()

    rows = []
    for sig_name, score in signals.items():
        score = np.asarray(score, dtype=float)
        finite = np.isfinite(score)
        row = {
            "manifold": run["label"], "n": n, "k": k, "noise": noise_std,
            "signal": sig_name, "n_edges_undirected": len(agg),
            "n_chord": int(chord_label.sum()),
            "n_composite": int(composite_label.sum()),
        }
        for guard_name, gmask in masks.items():
            mm = finite & gmask
            row[f"auroc_chord_{guard_name}"] = safe_auroc(chord_label, score, mm)
            row[f"auroc_composite_{guard_name}"] = safe_auroc(composite_label, score, mm)
            row[f"spearman_s_{guard_name}"] = safe_spearman(score, agg.s_ij.to_numpy(), mm)
            row[f"n_kept_{guard_name}"] = int(mm.sum())
            row[f"n_chord_{guard_name}"] = int((chord_label & gmask).sum())
            row[f"n_composite_{guard_name}"] = int((composite_label & gmask).sum())
        if orcml_flag is not None:
            row["n_orcml_shortcut"] = int(orcml_flag.sum())
            row["auroc_orcml"] = safe_auroc(orcml_flag, score, finite)
            row["jaccard_orcml_chord"] = jaccard_index(orcml_flag, chord_label)
            row["jaccard_orcml_composite"] = jaccard_index(orcml_flag, composite_label)
            for c in GUARD_CS:
                gm = masks[f"guard_c{c}"]
                row[f"jaccard_orcml_composite_c{c}"] = jaccard_index(
                    orcml_flag & gm, composite_label & gm
                )
        rows.append(row)

    # Persist per-edge frame for later forensics
    per_edge_path = OUT / "per_edge" / f"{run['label']}_n{n}_k{k}_noise{noise_std:.2f}.parquet"
    try:
        agg.to_parquet(per_edge_path)
    except Exception:
        agg.to_csv(per_edge_path.with_suffix(".csv"), index=False)
    return rows, agg


def main():
    t0 = time.time()
    all_rows = []
    for run in RUNS:
        edges_path = run["edges"]
        if not edges_path.exists():
            print(f"!! missing {edges_path}, skip {run['label']}")
            continue
        print(f"\n=== {run['label']} ===")
        df_all = pd.read_csv(edges_path)
        cells = sorted(df_all.groupby(["n", "k", "noise_std"]).groups.keys())
        for (n, k, noise_std) in cells:
            t1 = time.time()
            df_cell = df_all[(df_all.n == n) & (df_all.k == k) & (df_all.noise_std == noise_std)]
            try:
                rows, _ = process_cell(run, df_cell, int(n), int(k), float(noise_std))
            except Exception as e:
                print(f"  cell n={n} k={k} noise={noise_std} FAILED: {e}")
                continue
            all_rows.extend(rows)
            if rows:
                r0 = rows[0]
                kept_c1 = r0.get("n_kept_guard_c1.0", -1)
                print(f"  n={n:5d} k={k:3d} noise={noise_std:.2f}  "
                      f"edges={r0['n_edges_undirected']:6d}  "
                      f"chord={r0['n_chord']:5d}  comp={r0['n_composite']:5d}  "
                      f"kept(c=1)={kept_c1:6d}  ({time.time()-t1:.1f}s)")
            if len(all_rows) % 200 < 10:
                pd.DataFrame(all_rows).to_csv(OUT / "extended_summary.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "extended_summary.csv", index=False)
    print(f"\n{len(df)} rows total in {time.time()-t0:.1f}s -> {OUT/'extended_summary.csv'}")


if __name__ == "__main__":
    main()
