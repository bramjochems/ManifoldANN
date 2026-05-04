"""Composite shortcut label validation (diagnostic, exploratory).

For each of three pre-existing manifold runs at (n=2000, k=15, noise=0):
  - reconstruct the kNN graph from edges.csv
  - compute graph-effect score s_ij via per-edge Dijkstra (sampled)
  - compare composite label (r<0.67 AND s>2) vs chord-only label (r<0.67)
  - sanity-check edges in each label-difference set
  - compute AUROCs for tangent_angle, -kappa(standard), -kappa(orcml) under both labels

NOT a thesis edit. NOT a permanent experiment. Outputs go under
benchmark_results/composite_shortcut_validation/.
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark_results" / "composite_shortcut_validation"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "swiss_roll": {
        "edges": ROOT / "benchmark_results/orc_results_regen/swiss_roll_20260222_012105/edges.csv",
        "points": OUT / "points_swiss_roll.csv",
        "kind": "swiss",
    },
    "swiss_roll_tight": {
        "edges": ROOT / "benchmark_results/orc_results_tight_swiss/swiss_roll_s005_20260427_012627/edges.csv",
        "points": OUT / "points_swiss_roll_tight.csv",
        "kind": "swiss",
        "radial_scale": 0.05,
    },
    "torus_R2r1": {
        "edges": ROOT / "benchmark_results/orc_results_regen/torus_20260222_015917/edges.csv",
        "points": OUT / "points_torus_R2r1.csv",
        "kind": "torus",
        "R": 2.0,
        "r": 1.0,
    },
}

CELL = dict(n=2000, k=15, noise_std=0.0)
R_THRESH = 0.67
S_THRESH = 2.0
S_CAP = 100.0
N_EDGE_SAMPLE = 100000  # effectively all edges (each cell has ~17k undirected)
RNG = np.random.default_rng(0)


def swiss_geodesic(t1, h1, t2, h2, radial_scale=1.0):
    arc = lambda t: 0.5 * (t * np.sqrt(1 + t * t) + np.arcsinh(t))
    ds = radial_scale * (arc(t2) - arc(t1))
    dh = h2 - h1
    return np.sqrt(ds * ds + dh * dh)


def torus_geodesic(u1, v1, u2, v2, R=2.0, r=1.0, n_grid=200):
    """Approximate via small grid Dijkstra around the source angles."""
    # Coarse global grid: build once per call (small, ok for ~1500 lookups
    # if we cache). Implementation uses grid Dijkstra over (u, v) torus.
    # For simplicity reuse a cached grid.
    return _torus_geodesic_cached(u1, v1, u2, v2, R, r, n_grid)


_TORUS_CACHE = {}


def _build_torus_grid(R, r, n_grid):
    key = (R, r, n_grid)
    if key in _TORUS_CACHE:
        return _TORUS_CACHE[key]
    us = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    vs = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    du = 2 * np.pi / n_grid
    # Precompute per-row metric coeffs G_uu = (R + r cos v)^2, G_vv = r^2
    Guu = (R + r * np.cos(vs)) ** 2  # depends on v
    _TORUS_CACHE[key] = (us, vs, du, Guu)
    return _TORUS_CACHE[key]


def _torus_geodesic_cached(u1, v1, u2, v2, R, r, n_grid):
    us, vs, du, Guu = _build_torus_grid(R, r, n_grid)
    iu1 = int(round((u1 % (2 * np.pi)) / du)) % n_grid
    iv1 = int(round((v1 % (2 * np.pi)) / du)) % n_grid
    iu2 = int(round((u2 % (2 * np.pi)) / du)) % n_grid
    iv2 = int(round((v2 % (2 * np.pi)) / du)) % n_grid

    N = n_grid * n_grid
    src = iu1 * n_grid + iv1
    dst = iu2 * n_grid + iv2
    dist = np.full(N, np.inf)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if node == dst:
            break
        iu = node // n_grid
        iv = node % n_grid
        # 8 neighbors
        for diu in (-1, 0, 1):
            for div in (-1, 0, 1):
                if diu == 0 and div == 0:
                    continue
                nu = (iu + diu) % n_grid
                nv = (iv + div) % n_grid
                # Edge length under metric ds^2 = Guu(v)*du^2 + r^2*dv^2
                # Use midpoint v for Guu
                mid_v = vs[iv] + 0.5 * div * du
                guu_mid = (R + r * np.cos(mid_v)) ** 2
                edge_len = np.sqrt(guu_mid * (diu * du) ** 2 + (r * div * du) ** 2)
                nb = nu * n_grid + nv
                nd = d + edge_len
                if nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, nb))
    return dist[dst]


def manifold_geodesic(p1, p2, info):
    if info["kind"] == "swiss":
        rs = info.get("radial_scale", 1.0)
        return swiss_geodesic(p1.t, p1.h, p2.t, p2.h, radial_scale=rs)
    return torus_geodesic(p1.u, p1.v, p2.u, p2.v, R=info["R"], r=info["r"])


def build_adj(df_std, points):
    """Build symmetric adjacency from edges with chord-length weights.

    df_std: standard variant rows (one row per directed kNN edge i->j).
    Returns CSR-like list-of-lists adj[i] = [(j, w), ...]
    """
    n = len(points)
    coords = points[["x", "y", "z"]].to_numpy()
    edge_set = {}
    for i, j in zip(df_std.edge_i.to_numpy() - 1, df_std.edge_j.to_numpy() - 1):
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in edge_set:
            w = float(np.linalg.norm(coords[a] - coords[b]))
            edge_set[(a, b)] = w
    adj = [[] for _ in range(n)]
    for (a, b), w in edge_set.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj, edge_set


def dijkstra_skip_edge(adj, src, dst, skip_a, skip_b):
    """Dijkstra from src to dst with edge (skip_a, skip_b) (undirected) removed."""
    n = len(adj)
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if node == dst:
            return d
        for nb, w in adj[node]:
            # skip the removed undirected edge in both directions
            if (node == skip_a and nb == skip_b) or (node == skip_b and nb == skip_a):
                continue
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist[dst]


def dijkstra_all(adj, src):
    n = len(adj)
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for nb, w in adj[node]:
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist


def process_manifold(name, info):
    print(f"\n=== {name} ===")
    df = pd.read_csv(info["edges"])
    sub = df[(df.n == CELL["n"]) & (df.k == CELL["k"]) & (df.noise_std == CELL["noise_std"])].copy()
    df_std = sub[sub.orc_variant == "standard"].copy()
    df_orc = sub[sub.orc_variant == "orcml"].copy()
    print(f"standard rows: {len(df_std)}  orcml rows: {len(df_orc)}")

    points = pd.read_csv(info["points"])
    coords = points[["x", "y", "z"]].to_numpy()
    n = len(points)

    # Merge orcml kappa onto standard rows by (i,j) key
    df_std["key"] = list(zip(df_std.edge_i, df_std.edge_j))
    df_orc["key"] = list(zip(df_orc.edge_i, df_orc.edge_j))
    orc_map = dict(zip(df_orc.key, df_orc.kappa))
    df_std["kappa_orcml"] = df_std.key.map(orc_map)
    print(f"orcml-kappa merged for {df_std.kappa_orcml.notna().sum()} of {len(df_std)} standard edges")

    # De-duplicate to undirected edges (i<j) for the analysis
    df_std["a"] = np.minimum(df_std.edge_i, df_std.edge_j)
    df_std["b"] = np.maximum(df_std.edge_i, df_std.edge_j)
    # For each undirected (a,b), average kappa values across both directions
    agg = df_std.groupby(["a", "b"]).agg(
        kappa_std=("kappa", "mean"),
        kappa_orcml=("kappa_orcml", "mean"),
        ratio=("ratio", "mean"),
        tangent_angle=("tangent_angle", "mean"),
    ).reset_index()
    print(f"undirected edges: {len(agg)}")

    # Build adjacency
    adj, _ = build_adj(df_std, points)

    # Sample edges for s_ij computation
    n_edges = len(agg)
    if n_edges > N_EDGE_SAMPLE:
        idx = RNG.choice(n_edges, size=N_EDGE_SAMPLE, replace=False)
        agg_sample = agg.iloc[idx].copy().reset_index(drop=True)
    else:
        agg_sample = agg.copy()
    print(f"computing s_ij for {len(agg_sample)} edges...")

    t0 = time.time()
    s_vals = np.empty(len(agg_sample))
    d_g_vals = np.empty(len(agg_sample))
    for k, row in enumerate(agg_sample.itertuples(index=False)):
        a = int(row.a) - 1
        b = int(row.b) - 1
        # d_G(a,b) = direct edge weight (since (a,b) is an edge)
        # but Dijkstra would just give the edge weight unless removed
        # Edge weight:
        w_ab = float(np.linalg.norm(coords[a] - coords[b]))
        d_skip = dijkstra_skip_edge(adj, a, b, a, b)
        d_g_vals[k] = w_ab
        if not np.isfinite(d_skip):
            s_vals[k] = S_CAP
        else:
            s_vals[k] = min(d_skip / w_ab, S_CAP)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(agg_sample)}  elapsed {time.time()-t0:.1f}s")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    agg_sample["s_ij"] = s_vals
    agg_sample["d_G"] = d_g_vals

    # Compute manifold geodesic for sanity-check edges
    def d_M(row):
        return float(manifold_geodesic(points.iloc[int(row.a) - 1], points.iloc[int(row.b) - 1], info))

    # Labels
    agg_sample["chord_label"] = agg_sample.ratio < R_THRESH
    agg_sample["composite_label"] = (agg_sample.ratio < R_THRESH) & (agg_sample.s_ij > S_THRESH)

    n_chord = int(agg_sample.chord_label.sum())
    n_comp = int(agg_sample.composite_label.sum())
    n_overlap = int((agg_sample.chord_label & agg_sample.composite_label).sum())
    n_chord_only = int((agg_sample.chord_label & ~agg_sample.composite_label).sum())
    n_comp_only = int((~agg_sample.chord_label & agg_sample.composite_label).sum())

    print(f"chord-only: {n_chord}, composite: {n_comp}, overlap: {n_overlap}")
    print(f"chord\\composite: {n_chord_only}, composite\\chord: {n_comp_only}")

    # Sanity check examples
    examples = {}
    for label, mask in [
        ("chord_only_not_composite", agg_sample.chord_label & ~agg_sample.composite_label),
        ("composite_only_not_chord", ~agg_sample.chord_label & agg_sample.composite_label),
        ("both", agg_sample.chord_label & agg_sample.composite_label),
    ]:
        sel = agg_sample[mask]
        if len(sel) == 0:
            examples[label] = []
            continue
        pick = sel.head(5).copy()
        pick["chord"] = pick.apply(lambda r: float(np.linalg.norm(coords[int(r.a) - 1] - coords[int(r.b) - 1])), axis=1)
        pick["d_M"] = pick.apply(d_M, axis=1)
        examples[label] = pick[["a", "b", "ratio", "s_ij", "chord", "d_M", "tangent_angle", "kappa_std", "kappa_orcml"]].to_dict(orient="records")

    # AUROCs (use signals where higher = more shortcut-like)
    def auc(y, score):
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        return float(roc_auc_score(y, score))

    aurocs = {}
    for label_name, y in [("chord_only", agg_sample.chord_label.to_numpy()),
                          ("composite", agg_sample.composite_label.to_numpy())]:
        aurocs[label_name] = {
            "tangent_angle": auc(y, agg_sample.tangent_angle.to_numpy()),
            "neg_kappa_std": auc(y, -agg_sample.kappa_std.to_numpy()),
            "neg_kappa_orcml": auc(y, -agg_sample.kappa_orcml.fillna(agg_sample.kappa_std).to_numpy()),
        }

    print("AUROCs:", aurocs)

    # Save per-edge sample CSV
    out_csv = OUT / f"per_edge_{name}.csv"
    agg_sample.to_csv(out_csv, index=False)

    return {
        "name": name,
        "n_edges_undirected": n_edges,
        "n_edges_sampled": len(agg_sample),
        "n_chord": n_chord,
        "n_composite": n_comp,
        "n_overlap": n_overlap,
        "n_chord_only": n_chord_only,
        "n_composite_only": n_comp_only,
        "examples": examples,
        "aurocs": aurocs,
        "elapsed": elapsed,
    }


def write_report(results):
    lines = []
    lines.append("# Composite shortcut label validation")
    lines.append("")
    lines.append("Diagnostic comparison of two shortcut labels on (n=2000, k=15, noise=0) cells.")
    lines.append("")
    lines.append("- Chord-only label: r_ij < 0.67")
    lines.append("- Composite label: r_ij < 0.67 AND s_ij > 2 (s_ij = d_{G\\setminus(i,j)}/d_G)")
    lines.append("")
    lines.append("Edge weights = ambient chord lengths. s_ij computed via per-edge Dijkstra")
    lines.append(f"on a random sample of {N_EDGE_SAMPLE} undirected edges per manifold (capped at {S_CAP}).")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| manifold | edges (undirected) | sampled | runtime (s) |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['name']} | {r['n_edges_undirected']} | {r['n_edges_sampled']} | {r['elapsed']:.1f} |")
    lines.append("")
    lines.append("## Label populations")
    lines.append("")
    lines.append("| manifold | chord-only count | composite count | overlap | chord\\composite | composite\\chord |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['name']} | {r['n_chord']} | {r['n_composite']} | {r['n_overlap']} | {r['n_chord_only']} | {r['n_composite_only']} |")
    lines.append("")
    lines.append("## AUROCs (signal -> label)")
    lines.append("")
    lines.append("| manifold | signal | chord-only label | composite label |")
    lines.append("|---|---|---:|---:|")
    for r in results:
        for sig in ("tangent_angle", "neg_kappa_std", "neg_kappa_orcml"):
            a = r["aurocs"]["chord_only"][sig]
            b = r["aurocs"]["composite"][sig]
            lines.append(f"| {r['name']} | {sig} | {a:.3f} | {b:.3f} |")
    lines.append("")
    lines.append("## Sanity-check examples (5 per cell)")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']}")
        for cat, exs in r["examples"].items():
            lines.append(f"**{cat}** ({len(exs)})")
            if not exs:
                lines.append("  (none)")
                continue
            lines.append("")
            lines.append("| a | b | ratio | s_ij | chord | d_M | tangent_angle | kappa_std | kappa_orcml |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for e in exs:
                ko = e["kappa_orcml"]
                ko_str = f"{ko:.3f}" if ko is not None and not (isinstance(ko, float) and np.isnan(ko)) else "NA"
                lines.append(
                    f"| {int(e['a'])} | {int(e['b'])} | {e['ratio']:.3f} | {e['s_ij']:.2f} | "
                    f"{e['chord']:.3f} | {e['d_M']:.3f} | {e['tangent_angle']:.3f} | "
                    f"{e['kappa_std']:.3f} | {ko_str} |"
                )
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(_interpretation(results))
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"\nWrote {OUT / 'REPORT.md'}")


def _interpretation(results):
    parts = []
    for r in results:
        delta_orcml = r["aurocs"]["composite"]["neg_kappa_orcml"] - r["aurocs"]["chord_only"]["neg_kappa_orcml"]
        delta_std = r["aurocs"]["composite"]["neg_kappa_std"] - r["aurocs"]["chord_only"]["neg_kappa_std"]
        parts.append(
            f"On {r['name']} the chord-only label flags {r['n_chord']} edges and the composite "
            f"label flags {r['n_composite']} (overlap {r['n_overlap']}); the composite label drops "
            f"{r['n_chord_only']} chord-flagged edges that have no graph-effect (small-r but small-s, "
            f"plausibly legitimate high-curvature edges). Switching from chord-only to composite shifts "
            f"the ORC-ManL AUROC by {delta_orcml:+.3f} and the standard-ORC AUROC by {delta_std:+.3f}."
        )
    return " ".join(parts)


def main():
    results = []
    for name, info in RUNS.items():
        results.append(process_manifold(name, info))
    write_report(results)


if __name__ == "__main__":
    main()
