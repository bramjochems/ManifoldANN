r"""Composite shortcut label full evaluation across all (n, k, noise) cells.

For each pre-existing run directory and each (n, k, noise) cell:
  - regenerate point coordinates (matching experiment_orc.jl seed=42 procedure)
  - reconstruct undirected kNN graph from edges.csv
  - compute s_ij = d_{G\(i,j)} / d_G(i,j) per undirected edge via per-edge Dijkstra
  - form composite label (r_ij < 0.67 AND s_ij > 2) and chord-only label (r_ij < 0.67)
  - compute AUROC of detection signals against each label

Detection signals scored such that higher = more shortcut-like:
  - tangent_angle (raw, higher = sharper kink)
  - -kappa (standard or orcml)
  - 1 - jaccard
  - 1 - gabriel (Gabriel test 1=passes; shortcut means failure of Gabriel)
  - kappa_zscore_neg = -kappa_zscore
  - angle_zscore (positive z = unusually-large angle)

Outputs:
  - benchmark_results/composite_shortcut_full_eval/composite_auroc_summary.csv
  - benchmark_results/composite_shortcut_full_eval/REPORT.md
"""

from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark_results" / "composite_shortcut_full_eval"
OUT.mkdir(parents=True, exist_ok=True)

R_THRESH = 0.67
S_THRESH = 2.0
S_CAP = 100.0

# Each run is described by (label, edges_csv, kind, params). We regenerate
# points using the *same* MersenneTwister(42) procedure used in experiment_orc.jl
# / generate_swiss_roll / generate_torus.
RUNS = [
    {
        "label": "swiss_roll",
        "edges": ROOT / "benchmark_results/orc_results_regen/swiss_roll_20260222_012105/edges.csv",
        "kind": "swiss",
        "h_scale": 10.0,
        "t_min": 1.5 * math.pi,
        "t_range": 3.0 * math.pi,
        "radial_scale": 1.0,
    },
    {
        "label": "swiss_roll_tight_s005",
        "edges": ROOT / "benchmark_results/orc_results_tight_swiss/swiss_roll_s005_20260427_012627/edges.csv",
        "kind": "swiss",
        "h_scale": 10.0,
        "t_min": 1.5 * math.pi,
        "t_range": 3.0 * math.pi,
        "radial_scale": 0.05,
    },
    {
        "label": "torus_R2_r1",
        "edges": ROOT / "benchmark_results/orc_results_regen/torus_20260222_015917/edges.csv",
        "kind": "torus",
        "R": 2.0,
        "r": 1.0,
    },
    {
        "label": "torus_R1_5_r1",
        "edges": ROOT / "benchmark_results/orc_results_regen/torus_20260426_232917/edges.csv",
        "kind": "torus",
        "R": 1.5,
        "r": 1.0,
    },
    {
        "label": "torus_R4_r1",
        "edges": ROOT / "benchmark_results/orc_results_regen/torus_20260427_000521/edges.csv",
        "kind": "torus",
        "R": 4.0,
        "r": 1.0,
    },
]

SEED = 42


# ---------------------------------------------------------------------------
# MersenneTwister-compatible RNG via Julia subprocess (one-off per cell)
# ---------------------------------------------------------------------------
# Replicating Julia's MersenneTwister stream in Python is fiddly, so we shell
# out to a tiny Julia helper that writes points to a temp CSV. This guarantees
# bit-exact equivalence with experiment_orc.jl's data generation.
import subprocess
import tempfile

JULIA_HELPER = ROOT / "scripts" / "_composite_full_eval_gen_points.jl"

JULIA_HELPER_SRC = r"""
# Helper: regenerate (x,y,z) for a given (manifold-kind, params, n, noise_std)
# matching the procedure in experiment_orc.jl exactly.
using Random
using DelimitedFiles

include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "torus_utils.jl"))

function gen_swiss(n; t_min, t_range, h_scale, radial_scale)
    rng = MersenneTwister(42)
    if radial_scale == 1.0
        data, params = generate_swiss_roll(n; rng=rng, t_min=t_min, t_range=t_range, h_scale=h_scale)
    else
        t = t_min .+ t_range .* rand(rng, n)
        h = h_scale .* rand(rng, n)
        data = vcat((radial_scale .* t .* cos.(t))', h', (radial_scale .* t .* sin.(t))')
        params = (t=t, h=h, radial_scale=radial_scale)
    end
    return data, params, rng
end

function gen_torus(n; R, r)
    rng = MersenneTwister(42)
    data, params = generate_torus(n; rng=rng, R=R, r=r)
    return data, params, rng
end

function add_noise!(data, rng, noise_std)
    if noise_std > 0
        data .+= noise_std .* randn(rng, size(data))
    end
    return data
end

function main(args)
    out_path = args[1]
    kind = args[2]
    n = parse(Int, args[3])
    noise_std = parse(Float64, args[4])
    if kind == "swiss"
        t_min = parse(Float64, args[5])
        t_range = parse(Float64, args[6])
        h_scale = parse(Float64, args[7])
        radial_scale = parse(Float64, args[8])
        data, params, rng = gen_swiss(n; t_min=t_min, t_range=t_range, h_scale=h_scale, radial_scale=radial_scale)
        add_noise!(data, rng, noise_std)
        # Output: x,y,z,t,h
        out = hcat(data', params.t, params.h)
        open(out_path, "w") do io
            write(io, "x,y,z,p1,p2\n")
            writedlm(io, out, ',')
        end
    elseif kind == "torus"
        R = parse(Float64, args[5])
        r = parse(Float64, args[6])
        data, params, rng = gen_torus(n; R=R, r=r)
        add_noise!(data, rng, noise_std)
        out = hcat(data', params.u, params.v)
        open(out_path, "w") do io
            write(io, "x,y,z,p1,p2\n")
            writedlm(io, out, ',')
        end
    else
        error("unknown kind: $kind")
    end
    return nothing
end

main(ARGS)
"""

if not JULIA_HELPER.exists():
    JULIA_HELPER.write_text(JULIA_HELPER_SRC)


def regen_points_via_julia(run, n: int, noise_std: float) -> pd.DataFrame:
    """Call the Julia helper to regenerate points for (run, n, noise_std)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
        tmp_path = tf.name
    try:
        if run["kind"] == "swiss":
            args = [
                tmp_path, "swiss", str(n), str(noise_std),
                f"{run['t_min']:.17g}", f"{run['t_range']:.17g}",
                f"{run['h_scale']:.17g}", f"{run['radial_scale']:.17g}",
            ]
        else:
            args = [
                tmp_path, "torus", str(n), str(noise_std),
                f"{run['R']:.17g}", f"{run['r']:.17g}",
            ]
        subprocess.run(
            ["julia", "--project=" + str(ROOT), str(JULIA_HELPER), *args],
            check=True, capture_output=True, text=True,
        )
        df = pd.read_csv(tmp_path)
        return df
    finally:
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Per-edge Dijkstra (pure Python heapq; fast enough at n<=2000)
# ---------------------------------------------------------------------------
def build_adj(edges_uv: np.ndarray, weights: np.ndarray, n: int):
    """edges_uv: (m, 2) int array of (a,b) with a<b, 0-indexed."""
    adj = [[] for _ in range(n)]
    for k in range(len(edges_uv)):
        a, b = int(edges_uv[k, 0]), int(edges_uv[k, 1])
        w = float(weights[k])
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def dijkstra_skip_edge(adj, src: int, dst: int, skip_a: int, skip_b: int, cap: float) -> float:
    """Shortest path from src to dst with the undirected edge (skip_a, skip_b)
    removed. Early-terminate at dst. Returns +inf if unreachable or > cap.

    Distances exceeding `cap` are pruned (we only care about s_ij up to S_CAP).
    """
    dist = {src: 0.0}
    heap = [(0.0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        if node == dst:
            return d
        for nb, w in adj[node]:
            if (node == skip_a and nb == skip_b) or (node == skip_b and nb == skip_a):
                continue
            nd = d + w
            if nd > cap:
                continue
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return math.inf


# ---------------------------------------------------------------------------
# Process one cell
# ---------------------------------------------------------------------------
def process_cell(run, df_cell: pd.DataFrame, n: int, k: int, noise_std: float):
    """Compute composite-label-AUROC for one (manifold, n, k, noise) cell.
    Returns a list of dicts (one per signal) for the summary CSV.
    """
    df_std = df_cell[df_cell.orc_variant == "standard"].copy()
    df_orc = df_cell[df_cell.orc_variant == "orcml"].copy()
    if len(df_std) == 0:
        return []

    # Map orcml kappa onto standard rows by directed (i,j) key
    orc_map = dict(zip(zip(df_orc.edge_i, df_orc.edge_j), df_orc.kappa))
    df_std["kappa_orcml"] = [orc_map.get((int(i), int(j)), np.nan)
                              for i, j in zip(df_std.edge_i, df_std.edge_j)]

    # Build undirected edge frame: aggregate (i->j) and (j->i) by mean
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
    agg = agg[agg.a != agg.b]

    # Regenerate points (with noise) and compute chord lengths
    pts = regen_points_via_julia(run, n, noise_std)
    coords = pts[["x", "y", "z"]].to_numpy()

    a_arr = agg.a.to_numpy() - 1
    b_arr = agg.b.to_numpy() - 1
    chord = np.linalg.norm(coords[a_arr] - coords[b_arr], axis=1)
    agg["chord"] = chord

    # Build adjacency with chord weights
    edges_uv = np.column_stack([a_arr, b_arr])
    adj = build_adj(edges_uv, chord, n)

    # Per-edge Dijkstra to compute s_ij
    s_vals = np.empty(len(agg))
    cap = S_CAP
    for idx in range(len(agg)):
        a = int(a_arr[idx]); b = int(b_arr[idx])
        w_ab = float(chord[idx])
        # Cap-based early termination: stop if we exceed cap*w_ab
        d_skip = dijkstra_skip_edge(adj, a, b, a, b, cap=cap * w_ab)
        if not math.isfinite(d_skip):
            s_vals[idx] = cap
        else:
            s_vals[idx] = min(d_skip / w_ab, cap)
    agg["s_ij"] = s_vals

    # Labels
    chord_label = (agg.ratio < R_THRESH).to_numpy()
    composite_label = chord_label & (agg.s_ij > S_THRESH).to_numpy()

    n_chord = int(chord_label.sum())
    n_comp = int(composite_label.sum())

    # Signals (higher = more shortcut-like)
    signals = {}
    if "tangent_angle" in agg.columns:
        signals["tangent_angle"] = agg.tangent_angle.to_numpy()
    if "kappa" in agg.columns:
        signals["neg_kappa_std"] = -agg["kappa"].to_numpy()
    if "kappa_orcml" in agg.columns:
        # fallback to kappa_std if orcml missing
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
        # mask out NaNs (treat as missing rather than tied)
        mask = np.isfinite(score)
        auc_chord = (
            float(roc_auc_score(chord_label[mask], score[mask]))
            if mask.sum() > 0 and chord_label[mask].sum() not in (0, mask.sum())
            else float("nan")
        )
        auc_comp = (
            float(roc_auc_score(composite_label[mask], score[mask]))
            if mask.sum() > 0 and composite_label[mask].sum() not in (0, mask.sum())
            else float("nan")
        )
        rows.append({
            "manifold": run["label"],
            "n": n,
            "k": k,
            "noise": noise_std,
            "signal": sig_name,
            "auroc_chord_only": auc_chord,
            "auroc_composite": auc_comp,
            "n_shortcuts_chord": n_chord,
            "n_shortcuts_composite": n_comp,
            "n_edges_undirected": len(agg),
        })
    return rows


def main():
    t_global = time.time()
    all_rows = []

    for run in RUNS:
        edges_path = run["edges"]
        if not edges_path.exists():
            print(f"!! missing {edges_path}, skipping {run['label']}")
            continue
        print(f"\n=== {run['label']} ({edges_path.parent.name}) ===")
        df_all = pd.read_csv(edges_path)
        cells = sorted(df_all.groupby(["n", "k", "noise_std"]).groups.keys())
        for (n, k, noise_std) in cells:
            t_cell = time.time()
            df_cell = df_all[(df_all.n == n) & (df_all.k == k) & (df_all.noise_std == noise_std)]
            try:
                rows = process_cell(run, df_cell, int(n), int(k), float(noise_std))
            except Exception as e:
                print(f"  cell n={n} k={k} noise={noise_std}  FAILED: {e}")
                continue
            all_rows.extend(rows)
            elapsed = time.time() - t_cell
            n_edges = rows[0]["n_edges_undirected"] if rows else 0
            n_chord = rows[0]["n_shortcuts_chord"] if rows else 0
            n_comp = rows[0]["n_shortcuts_composite"] if rows else 0
            print(f"  n={n:5d} k={k:3d} noise={noise_std:.2f}  edges={n_edges:6d}  "
                  f"chord={n_chord:5d}  composite={n_comp:5d}  ({elapsed:.1f}s)")

            # Periodic save
            if len(all_rows) % 200 < 10:
                pd.DataFrame(all_rows).to_csv(OUT / "composite_auroc_summary.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "composite_auroc_summary.csv", index=False)
    print(f"\nTotal rows: {len(df)}  total elapsed: {time.time()-t_global:.1f}s")
    print(f"Wrote {OUT/'composite_auroc_summary.csv'}")
    write_report(df)


def write_report(df: pd.DataFrame):
    lines = []
    lines.append("# Composite shortcut full evaluation")
    lines.append("")
    lines.append("Per-cell AUROC of detection signals against two shortcut labels:")
    lines.append("")
    lines.append("- **chord-only**: r_ij < 0.67")
    lines.append("- **composite**:  r_ij < 0.67  AND  s_ij > 2  (s_ij = d_{G\\setminus(i,j)} / d_G)")
    lines.append("")
    lines.append(f"Edge weights = ambient chord lengths. Per-edge Dijkstra over all undirected")
    lines.append(f"edges in each (n, k, noise) cell, for all 5 manifold variants. s_ij capped at {S_CAP}.")
    lines.append("")
    lines.append("Per-cell summary CSV: `composite_auroc_summary.csv`")
    lines.append("")

    for manifold in df.manifold.unique():
        sub = df[df.manifold == manifold]
        lines.append(f"## {manifold}")
        lines.append("")
        # Mean AUROC across all cells where the label was non-degenerate
        agg = sub.groupby("signal").agg(
            mean_auroc_chord=("auroc_chord_only", "mean"),
            mean_auroc_composite=("auroc_composite", "mean"),
            n_cells_chord=("auroc_chord_only", lambda s: s.notna().sum()),
            n_cells_composite=("auroc_composite", lambda s: s.notna().sum()),
        ).reset_index().sort_values("mean_auroc_composite", ascending=False)
        lines.append("Mean AUROC across all cells (NaNs = degenerate-label cells excluded):")
        lines.append("")
        lines.append("| signal | mean AUROC (chord-only) | mean AUROC (composite) | cells (chord) | cells (comp) |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in agg.iterrows():
            lines.append(f"| {r.signal} | {r.mean_auroc_chord:.3f} | {r.mean_auroc_composite:.3f} | "
                         f"{int(r.n_cells_chord)} | {int(r.n_cells_composite)} |")
        lines.append("")
        # Headline finding
        if not agg.empty:
            best_comp = agg.iloc[0]
            best_chord_row = agg.sort_values("mean_auroc_chord", ascending=False).iloc[0]
            lines.append(f"**Headline**: under the composite label, **{best_comp.signal}** wins with "
                         f"mean AUROC {best_comp.mean_auroc_composite:.3f} "
                         f"(under chord-only, **{best_chord_row.signal}** wins with "
                         f"{best_chord_row.mean_auroc_chord:.3f}).")
            lines.append("")

        # Per (n, k, noise) breakdown — show only non-degenerate cells
        cell_keys = sub[["n", "k", "noise"]].drop_duplicates().sort_values(["noise", "n", "k"])
        lines.append("<details><summary>Per-cell AUROC table</summary>")
        lines.append("")
        lines.append("| n | k | noise | n_chord | n_comp | signal | AUROC chord | AUROC composite |")
        lines.append("|---:|---:|---:|---:|---:|---|---:|---:|")
        for _, ck in cell_keys.iterrows():
            cell_rows = sub[(sub.n == ck.n) & (sub.k == ck.k) & (sub.noise == ck.noise)]
            for _, r in cell_rows.iterrows():
                ac = "NA" if pd.isna(r.auroc_chord_only) else f"{r.auroc_chord_only:.3f}"
                acm = "NA" if pd.isna(r.auroc_composite) else f"{r.auroc_composite:.3f}"
                lines.append(f"| {int(r.n)} | {int(r.k)} | {r.noise:.2f} | {int(r.n_shortcuts_chord)} | "
                             f"{int(r.n_shortcuts_composite)} | {r.signal} | {ac} | {acm} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Cross-manifold ranking comparison
    lines.append("## Cross-manifold ranking comparison")
    lines.append("")
    lines.append("Best signal (mean AUROC) under each label, per manifold:")
    lines.append("")
    lines.append("| manifold | best (chord-only) | AUROC | best (composite) | AUROC | rank changed? |")
    lines.append("|---|---|---:|---|---:|:---:|")
    for manifold in df.manifold.unique():
        sub = df[df.manifold == manifold]
        ag = sub.groupby("signal").agg(
            mc=("auroc_chord_only", "mean"),
            mp=("auroc_composite", "mean"),
        ).reset_index()
        if ag.mc.notna().any() and ag.mp.notna().any():
            bc = ag.loc[ag.mc.idxmax()]
            bp = ag.loc[ag.mp.idxmax()]
            changed = "yes" if bc.signal != bp.signal else ""
            lines.append(f"| {manifold} | {bc.signal} | {bc.mc:.3f} | {bp.signal} | {bp.mp:.3f} | {changed} |")
        else:
            lines.append(f"| {manifold} | NA | NA | NA | NA | |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Pruning-MRE re-evaluation under composite label was **not** performed in this run "
                 "(the MRE itself is label-independent — it's measured against analytic geodesics — so "
                 "the existing `rank_pruning.csv` plots remain valid; only the AUROC framing changes).")
    lines.append("- Cells where the composite label flags 0 edges (or all edges) yield NaN AUROC and "
                 "are excluded from the mean.")

    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {OUT/'REPORT.md'}")


if __name__ == "__main__":
    main()
