"""
ORC Unified Experiment — Detection + Pruning

Unified script for both shortcut detection (RQ2) and pruning impact (RQ3).
Supports Swiss roll and torus (R2r1) manifolds via MANIFOLD env var.

Detection (Experiment 1):
  - AUROC as primary metric (computed in analysis from edges.csv)
  - F1@κ=0, best-F1, κ class statistics (secondary, for narrative)
  - Per-edge: κ, ratio, tangent_angle, jaccard, gabriel, kappa_zscore, angle_zscore

Pruning (Experiment 2, k≥10 only):
  2a. Oracle pruning — remove ground-truth shortcuts (r_ij < τ)
  2b. ORC rank-based pruning — remove bottom p% by κ
  2c. Connectivity-preserving ORC rank pruning — skip bridges
  2d. Tangent-angle rank-based pruning — remove top p% by angle
  2e. Jaccard rank-based pruning — remove lowest p% by Jaccard similarity
  2f. κ z-score rank-based pruning — remove bottom p% by local z-score
  2g. Tangent-angle z-score rank-based pruning — remove top p% by local z-score
  2h. Gabriel pruning — remove all non-Gabriel edges (single shot)
  2i. Random pruning baseline

Metrics: MRE, Spearman rank correlation, disconnected pairs, fraction removed.

Runtime (Experiment 3):
  - orc_time_s and pca_time_s logged in raw.csv (no separate script)

Outputs (in {manifold}_{ts}/ directory):
  raw.csv          — one row per (n, k, σ, variant, τ)
  edges.csv        — one row per edge with κ, ratio, tangent_angle
  oracle.csv       — oracle pruning results
  rank_pruning.csv — ORC rank, connectivity-preserving, tangent-angle pruning
  random_pruning.csv — random baseline (multiple replicates)
  config.toml      — reproducibility metadata

Environment variables:
  MANIFOLD=swiss|torus    which manifold to run (required)
  SKIP_DETECTION=1        skip detection evaluation
  SKIP_PRUNING=1          skip all pruning sub-experiments
  SKIP_EDGES=1            skip per-edge CSV output
  SMOKE=1                 single minimal config
  N_OVERRIDE=500          single n value
  K_OVERRIDE=10           single k value
  RESUME_DIR=path         resume into a previous run directory

Usage:
  MANIFOLD=swiss julia --project=. -t auto scripts/experiment_orc.jl
"""

using ManifoldANN
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates

include(joinpath(@__DIR__, "orc_helpers.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "torus_utils.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const RESULTS_DIR = joinpath(
    @__DIR__, "..", "..", "..", "..", "..", "docs", "thesis", "results", "orc_results"
)

# Manifold selection
const MANIFOLD = let m = get(ENV, "MANIFOLD", "")
    m in ("swiss", "torus") || error("Set MANIFOLD=swiss or MANIFOLD=torus")
    m
end

# Swiss roll parameters
const T_MIN   = 1.5π
const T_RANGE = 3π
const H_SCALE = 10.0

# Torus parameters (R2r1 only per plan)
const TORUS_R = 2.0
const TORUS_r = 1.0
const TORUS_KEY = "R2r1"
const GEODESIC_N_GRID = 100

# Shortcut ratio thresholds (for labelling — AUROC computed in analysis at multiple τ)
const TAU_GRID = [0.5, 0.67, 0.8]

# Random pruning replicates
const N_RANDOM_REPLICATES = 5

const SEED = 42

# Apply overrides
_smoke          = get(ENV, "SMOKE", "") == "1"
_skip_detection = get(ENV, "SKIP_DETECTION", "") == "1"
_skip_pruning   = get(ENV, "SKIP_PRUNING", "") == "1"
_skip_edges     = get(ENV, "SKIP_EDGES", "") == "1"

_n_grid   = !isempty(get(ENV, "N_OVERRIDE", "")) ? [parse(Int, ENV["N_OVERRIDE"])] :
             [200, 500, 1000, 2000]
_k_grid   = !isempty(get(ENV, "K_OVERRIDE", "")) ? [parse(Int, ENV["K_OVERRIDE"])] :
             [5, 10, 15, 20, 30]
_noise_grid = [0.0, 0.5]
_variants   = ["standard", "orcml"]

if _smoke
    _n_grid     = [200]
    _k_grid     = [10]
    _noise_grid = [0.0]
    _variants   = ["standard"]
end

const N_GRID         = _n_grid
const K_GRID         = _k_grid
const NOISE_GRID     = _noise_grid
const ORC_VARIANTS   = _variants
const TAU_GRID_RUN   = _smoke ? [0.5] : TAU_GRID
const SKIP_DETECTION = _skip_detection
const SKIP_PRUNING   = _skip_pruning
const SKIP_EDGES     = _skip_edges

# ==============================================================================
# Manifold-specific helpers
# ==============================================================================

"""Generate data and params for the configured manifold."""
function generate_manifold_data(n::Int, rng::AbstractRNG)
    if MANIFOLD == "swiss"
        return generate_swiss_roll(n; rng=rng, t_min=T_MIN, t_range=T_RANGE, h_scale=H_SCALE)
    else
        return generate_torus(n; rng=rng, R=TORUS_R, r=TORUS_r)
    end
end

"""Compute geodesic distance between points i and j."""
function manifold_geodesic(params::NamedTuple, i::Int, j::Int)
    if MANIFOLD == "swiss"
        return exact_swiss_roll_geodesic(params, i, j)
    else
        return exact_torus_geodesic(params, i, j; n_grid=GEODESIC_N_GRID)
    end
end

"""Precompute r_ij = d_ambient / d_geodesic for all graph edges."""
function precompute_ratios(graph, data::AbstractMatrix, params::NamedTuple)
    ratios = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:length(graph)
        for j in graph[i]
            d_amb = norm(data[:, i] - data[:, j])
            d_geo = manifold_geodesic(params, i, j)
            ratios[(i, j)] = d_geo > 1e-12 ? d_amb / d_geo : 1.0
        end
    end
    return ratios
end

"""Sample random pairs with true geodesic distances."""
function sample_geodesic_pairs(params::NamedTuple, n::Int, rng::AbstractRNG)
    pairs = Tuple{Int, Int, Float64}[]
    seen = Set{Tuple{Int,Int}}()
    while length(pairs) < min(N_GEO_PAIRS, n * (n - 1) ÷ 2)
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        edge = minmax(i, j)
        edge in seen && continue
        push!(seen, edge)
        d = manifold_geodesic(params, i, j)
        push!(pairs, (i, j, d))
    end
    return pairs
end

"""Manifold label for CSV output."""
manifold_label() = MANIFOLD == "swiss" ? "swiss" : TORUS_KEY

# ==============================================================================
# Output directory setup
# ==============================================================================

mkpath(RESULTS_DIR)
_resume_dir = get(ENV, "RESUME_DIR", "")

run_dir = if !isempty(_resume_dir)
    _resume_dir
else
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    prefix = MANIFOLD == "swiss" ? "swiss_roll" : "torus"
    joinpath(RESULTS_DIR, "$(prefix)_$(ts)")
end
mkpath(run_dir)

# CSV file paths
raw_file           = joinpath(run_dir, "raw.csv")
edges_file         = joinpath(run_dir, "edges.csv")
oracle_file        = joinpath(run_dir, "oracle.csv")
rank_pruning_file  = joinpath(run_dir, "rank_pruning.csv")
random_pruning_file = joinpath(run_dir, "random_pruning.csv")

# Initialize CSV headers
const RAW_HEADER = "manifold,n,k,noise_std,variant,tau," *
    "n_edges,n_shortcuts,frac_shortcuts," *
    "precision_0,recall_0,f1_0,tp_0,fp_0,tn_0,fn_0," *
    "best_threshold,precision_best,recall_best,f1_best," *
    "mean_kappa_shortcuts,mean_kappa_non_shortcuts," *
    "n_shortcut_edges,n_non_shortcut_edges,orc_time_s,pca_time_s"
init_csv(raw_file, RAW_HEADER)

if !SKIP_EDGES
    const EDGES_HEADER = "manifold,n,k,noise_std,orc_variant,edge_i,edge_j,kappa,ratio,tangent_angle,jaccard,gabriel,kappa_zscore,angle_zscore"
    init_csv(edges_file, EDGES_HEADER)
end

if !SKIP_PRUNING
    const ORACLE_HEADER = "manifold,n,k,noise_std,variant,tau," *
        "n_edges_before,n_edges_after,frac_removed,n_shortcuts_removed," *
        "mean_rel_error,median_rel_error,spearman,n_disconnected"
    init_csv(oracle_file, ORACLE_HEADER)

    const RANK_HEADER = "manifold,n,k,noise_std,variant,method,fraction," *
        "n_edges_before,n_edges_after,frac_removed," *
        "mean_rel_error,median_rel_error,spearman,n_disconnected"
    init_csv(rank_pruning_file, RANK_HEADER)

    const RANDOM_HEADER = "manifold,n,k,noise_std,fraction,replicate," *
        "n_edges_before,n_edges_after,frac_removed," *
        "mean_rel_error,median_rel_error,spearman,n_disconnected"
    init_csv(random_pruning_file, RANDOM_HEADER)
end

# Load already-completed configs for resume
completed_configs = load_completed_keys(raw_file, [:manifold, :n, :k, :noise_std, :variant])

println("=" ^ 80)
println("ORC Unified Experiment — $(uppercase(MANIFOLD))")
println("=" ^ 80)
println("Threads         : ", Threads.nthreads())
println("Manifold        : ", MANIFOLD)
println("n grid          : ", N_GRID)
println("k grid          : ", K_GRID)
println("noise grid      : ", NOISE_GRID)
println("ORC variants    : ", ORC_VARIANTS)
println("τ grid          : ", TAU_GRID_RUN)
println("SKIP_DETECTION  : ", SKIP_DETECTION)
println("SKIP_PRUNING    : ", SKIP_PRUNING)
println("SKIP_EDGES      : ", SKIP_EDGES)
println("Output dir      : ", run_dir)
println("Resuming        : ", !isempty(completed_configs), " (", length(completed_configs), " configs done)")
println("=" ^ 80)
println()

# ==============================================================================
# Main loop
# ==============================================================================

total_orc = length(N_GRID) * length(K_GRID) * length(NOISE_GRID) * length(ORC_VARIANTS)
mlabel = manifold_label()

let orc_idx = 0
for n in N_GRID, k in K_GRID, noise_std in NOISE_GRID, variant in ORC_VARIANTS
    orc_idx += 1

    config_key = (mlabel, string(n), string(k), string(noise_std), variant)
    if config_key in completed_configs
        @printf("[%d/%d]  n=%-5d  k=%-3d  noise=%.2f  variant=%s  [SKIP]\n",
            orc_idx, total_orc, n, k, noise_std, variant)
        continue
    end

    @printf("\n[%d/%d]  n=%-5d  k=%-3d  noise=%.2f  variant=%s\n",
        orc_idx, total_orc, n, k, noise_std, variant)

    # ── 1. Generate data ─────────────────────────────────────────────────────
    rng = MersenneTwister(SEED)
    data_clean, params = generate_manifold_data(n, rng)
    data = noise_std > 0 ? data_clean .+ noise_std .* randn(rng, size(data_clean)) :
                           data_clean

    # ── 2. Build kNN graph ───────────────────────────────────────────────────
    directed = (variant == "standard")
    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k, directed=directed)
    n_edges = sum(length(graph[i]) for i in 1:n)

    # ── 3. Compute ORC (expensive, done once) ────────────────────────────────
    orc_variant = variant == "standard" ? StandardORC() : ORCManL()

    t_orc = @elapsed begin
        curvatures = compute_all_curvatures(
            graph, data;
            variant=orc_variant,
            solver=HungarianSolver(),
            fallback_solver=NetworkSimplexSolver(),
            use_threading=true,
            verbose=false
        )
    end
    @printf "  ORC computed in %.2fs  (%d edges)\n" t_orc length(curvatures)

    # ── 4. Precompute ratios + ground-truth labels ───────────────────────────
    ratios = precompute_ratios(graph, data, params)

    # ── 5. Fit PCA at all nodes (for tangent angles) ─────────────────────────
    pca_method = PCAMethod(intrinsic_dim=2)
    t_pca = @elapsed begin
        geometries = Vector{PCAGeometry}(undef, n)
        for i in 1:n
            neighbors = graph[i]
            if !isempty(neighbors)
                geometries[i] = fit_geometry(pca_method, data, i, neighbors)
            else
                # Fallback: use self as center with identity basis
                geometries[i] = fit_geometry(pca_method, data, i, [i])
            end
        end
    end
    @printf "  PCA fitted in %.2fs  (%d nodes)\n" t_pca n

    # ── 6. Compute tangent angles for all edges ──────────────────────────────
    adj, weights = graph_to_adj_weights(graph, data)
    tangent_angles = compute_tangent_angles(adj, data, geometries)

    # ── 6b. Build κ score dict (needed for z-scores and pruning) ─────────────
    kappa_scores = Dict{Tuple{Int,Int}, Float64}()
    for (edge, result) in curvatures
        kappa_scores[edge] = result.curvature
    end

    # ── 6c. Compute Jaccard scores ──────────────────────────────────────────
    jaccard_scores = compute_jaccard_scores(graph)

    # ── 6d. Compute Gabriel mask ────────────────────────────────────────────
    gabriel_mask = compute_gabriel_mask(graph, data)

    # ── 6e. Compute local z-scores ──────────────────────────────────────────
    kappa_zscores = compute_local_zscores(adj, kappa_scores)
    angle_zscores = compute_local_zscores(adj, tangent_angles)

    @printf "  Computed Jaccard (%d), Gabriel (%d), κ-zscore (%d), angle-zscore (%d)\n" length(jaccard_scores) length(gabriel_mask) length(kappa_zscores) length(angle_zscores)

    # ── 7. [Detection] Evaluate for each τ ───────────────────────────────────
    if !SKIP_DETECTION
        for tau in TAU_GRID_RUN
            labels  = label_shortcuts(ratios, tau)
            n_sc    = sum(values(labels))
            frac_sc = n_sc / n_edges

            m0    = evaluate_at_threshold(curvatures, labels, 0.0)
            best  = best_f1_threshold(curvatures, labels)
            kstat = curvature_stats_by_label(curvatures, labels)

            @printf("  τ=%.2f  shortcuts=%d/%d (%.1f%%)  F1@0=%.3f  best_F1=%.3f@κ<%.2f\n",
                tau, n_sc, n_edges, 100frac_sc, m0.f1, best.f1, best.κ)

            # Flush raw row immediately
            raw_row = join([
                mlabel, n, k, noise_std, variant, tau, n_edges, n_sc,
                @sprintf("%.6f", frac_sc),
                @sprintf("%.6f", m0.precision), @sprintf("%.6f", m0.recall), @sprintf("%.6f", m0.f1),
                m0.tp, m0.fp, m0.tn, m0.fn,
                @sprintf("%.6f", best.κ),
                @sprintf("%.6f", best.precision), @sprintf("%.6f", best.recall), @sprintf("%.6f", best.f1),
                isnan(kstat.mean_shortcut) ? "NaN" : @sprintf("%.6f", kstat.mean_shortcut),
                isnan(kstat.mean_non_shortcut) ? "NaN" : @sprintf("%.6f", kstat.mean_non_shortcut),
                kstat.n_shortcut, kstat.n_non_shortcut,
                @sprintf("%.6f", t_orc), @sprintf("%.6f", t_pca),
            ], ",")
            append_csv_rows(raw_file, [raw_row])
        end
    end

    # ── 8. [Detection] Per-edge data for AUROC ───────────────────────────────
    if !SKIP_EDGES && !SKIP_DETECTION
        edge_rows = String[]
        for (edge, result) in curvatures
            haskey(ratios, edge) || continue
            r = ratios[edge]
            ta = get(tangent_angles, edge, NaN)
            jac = get(jaccard_scores, edge, NaN)
            gab = get(gabriel_mask, edge, false) ? 1 : 0
            kz = get(kappa_zscores, edge, NaN)
            az = get(angle_zscores, edge, NaN)
            push!(edge_rows, join([
                mlabel, n, k, noise_std, variant,
                edge[1], edge[2],
                @sprintf("%.8f", result.curvature),
                @sprintf("%.6f", r),
                isnan(ta) ? "NaN" : @sprintf("%.6f", ta),
                isnan(jac) ? "NaN" : @sprintf("%.6f", jac),
                gab,
                isnan(kz) ? "NaN" : @sprintf("%.6f", kz),
                isnan(az) ? "NaN" : @sprintf("%.6f", az),
            ], ","))
        end
        append_csv_rows(edges_file, edge_rows)
        @printf "  Wrote %d edge records\n" length(edge_rows)
    end

    # ── 9. [Pruning] Only for k ≥ 10 ────────────────────────────────────────
    if !SKIP_PRUNING && k >= 10
        @printf "  Computing pruning analysis...\n"

        # Sample geodesic pairs (unpruned baseline)
        rng_pairs = MersenneTwister(SEED + 1)
        pairs = sample_geodesic_pairs(params, n, rng_pairs)

        # Unpruned baseline
        baseline = geodesic_error_at_pairs(adj, weights, pairs, n)
        @printf("    Unpruned: MRE=%.4f  Spearman=%.4f  disconn=%d\n",
            baseline.mean_rel_error, baseline.spearman, baseline.n_disconnected)

        # Find bridges for connectivity-preserving pruning
        bridges = find_bridges(adj)
        @printf "    Found %d bridges\n" length(bridges)

        # ── 9a. Oracle pruning ───────────────────────────────────────────────
        oracle_rows = String[]
        for tau in TAU_GRID_RUN
            adj_oracle, w_oracle = oracle_prune(adj, weights, ratios, tau)
            n_edges_after = sum(length(adj_oracle[i]) for i in 1:n)
            frac_removed = 1.0 - n_edges_after / n_edges

            # Count shortcuts removed
            labels = label_shortcuts(ratios, tau)
            n_sc = sum(values(labels))

            err = geodesic_error_at_pairs(adj_oracle, w_oracle, pairs, n)

            push!(oracle_rows, join([
                mlabel, n, k, noise_std, variant, tau,
                n_edges, n_edges_after,
                @sprintf("%.6f", frac_removed), n_sc,
                @sprintf("%.6f", err.mean_rel_error),
                @sprintf("%.6f", err.median_rel_error),
                @sprintf("%.6f", err.spearman),
                err.n_disconnected,
            ], ","))

            @printf("    Oracle τ=%.2f: removed=%d (%.1f%%)  MRE=%.4f  Spearman=%.4f\n",
                tau, n_edges - n_edges_after, 100frac_removed,
                err.mean_rel_error, err.spearman)
        end
        append_csv_rows(oracle_file, oracle_rows)

        # ── 9b,c,d. Rank-based pruning (ORC, ORC-bridge-safe, tangent-angle) ─
        rank_rows = String[]
        for p in PRUNE_FRACTIONS
            # 9b. ORC rank-based (ascending κ, remove lowest)
            adj_rk, w_rk = rank_prune(adj, weights, kappa_scores, p; ascending=true)
            n_after = sum(length(adj_rk[i]) for i in 1:n)
            err = geodesic_error_at_pairs(adj_rk, w_rk, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "orc_rank", p,
                n_edges, n_after, @sprintf("%.6f", 1.0 - n_after/n_edges),
                @sprintf("%.6f", err.mean_rel_error),
                @sprintf("%.6f", err.median_rel_error),
                @sprintf("%.6f", err.spearman),
                err.n_disconnected,
            ], ","))

            # 9c. Connectivity-preserving ORC rank
            adj_rk_b, w_rk_b = rank_prune(adj, weights, kappa_scores, p;
                                           bridges=bridges, ascending=true)
            n_after_b = sum(length(adj_rk_b[i]) for i in 1:n)
            err_b = geodesic_error_at_pairs(adj_rk_b, w_rk_b, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "orc_rank_bridge_safe", p,
                n_edges, n_after_b, @sprintf("%.6f", 1.0 - n_after_b/n_edges),
                @sprintf("%.6f", err_b.mean_rel_error),
                @sprintf("%.6f", err_b.median_rel_error),
                @sprintf("%.6f", err_b.spearman),
                err_b.n_disconnected,
            ], ","))

            # 9d. Tangent-angle rank-based (descending angle, remove highest)
            adj_ta, w_ta = rank_prune(adj, weights, tangent_angles, p; ascending=false)
            n_after_ta = sum(length(adj_ta[i]) for i in 1:n)
            err_ta = geodesic_error_at_pairs(adj_ta, w_ta, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "tangent_angle_rank", p,
                n_edges, n_after_ta, @sprintf("%.6f", 1.0 - n_after_ta/n_edges),
                @sprintf("%.6f", err_ta.mean_rel_error),
                @sprintf("%.6f", err_ta.median_rel_error),
                @sprintf("%.6f", err_ta.spearman),
                err_ta.n_disconnected,
            ], ","))

            # 9e. Jaccard rank-based (ascending — remove lowest overlap)
            adj_jac, w_jac = rank_prune(adj, weights, jaccard_scores, p; ascending=true)
            n_after_jac = sum(length(adj_jac[i]) for i in 1:n)
            err_jac = geodesic_error_at_pairs(adj_jac, w_jac, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "jaccard_rank", p,
                n_edges, n_after_jac, @sprintf("%.6f", 1.0 - n_after_jac/n_edges),
                @sprintf("%.6f", err_jac.mean_rel_error),
                @sprintf("%.6f", err_jac.median_rel_error),
                @sprintf("%.6f", err_jac.spearman),
                err_jac.n_disconnected,
            ], ","))

            # 9f. κ local z-score rank-based (ascending — remove most negative z-scores)
            adj_kz, w_kz = rank_prune(adj, weights, kappa_zscores, p; ascending=true)
            n_after_kz = sum(length(adj_kz[i]) for i in 1:n)
            err_kz = geodesic_error_at_pairs(adj_kz, w_kz, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "orc_zscore_rank", p,
                n_edges, n_after_kz, @sprintf("%.6f", 1.0 - n_after_kz/n_edges),
                @sprintf("%.6f", err_kz.mean_rel_error),
                @sprintf("%.6f", err_kz.median_rel_error),
                @sprintf("%.6f", err_kz.spearman),
                err_kz.n_disconnected,
            ], ","))

            # 9g. Tangent-angle local z-score rank-based (descending — remove highest z-scores)
            adj_az, w_az = rank_prune(adj, weights, angle_zscores, p; ascending=false)
            n_after_az = sum(length(adj_az[i]) for i in 1:n)
            err_az = geodesic_error_at_pairs(adj_az, w_az, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "tangent_zscore_rank", p,
                n_edges, n_after_az, @sprintf("%.6f", 1.0 - n_after_az/n_edges),
                @sprintf("%.6f", err_az.mean_rel_error),
                @sprintf("%.6f", err_az.median_rel_error),
                @sprintf("%.6f", err_az.spearman),
                err_az.n_disconnected,
            ], ","))

            @printf("    p=%.0f%%: ORC MRE=%.4f  Bridge-safe=%.4f  Tangent=%.4f  Jaccard=%.4f  κ-z=%.4f  angle-z=%.4f\n",
                100p, err.mean_rel_error, err_b.mean_rel_error, err_ta.mean_rel_error,
                err_jac.mean_rel_error, err_kz.mean_rel_error, err_az.mean_rel_error)
        end

        # ── 9h. Gabriel pruning (all-or-nothing: remove all non-Gabriel edges) ─
        gabriel_scores_float = Dict{Tuple{Int,Int}, Float64}(
            e => (v ? 1.0 : 0.0) for (e, v) in gabriel_mask)
        n_non_gabriel = count(!v for v in values(gabriel_mask))
        if n_non_gabriel > 0
            # Remove all non-Gabriel edges by pruning fraction = n_non_gabriel / n_total
            frac_gabriel = n_non_gabriel / length(gabriel_mask)
            adj_gab, w_gab = rank_prune(adj, weights, gabriel_scores_float, frac_gabriel; ascending=true)
            n_after_gab = sum(length(adj_gab[i]) for i in 1:n)
            err_gab = geodesic_error_at_pairs(adj_gab, w_gab, pairs, n)
            push!(rank_rows, join([
                mlabel, n, k, noise_std, variant, "gabriel", @sprintf("%.6f", frac_gabriel),
                n_edges, n_after_gab, @sprintf("%.6f", 1.0 - n_after_gab/n_edges),
                @sprintf("%.6f", err_gab.mean_rel_error),
                @sprintf("%.6f", err_gab.median_rel_error),
                @sprintf("%.6f", err_gab.spearman),
                err_gab.n_disconnected,
            ], ","))
            @printf("    Gabriel: removed %d non-Gabriel edges (%.1f%%)  MRE=%.4f\n",
                n_non_gabriel, 100frac_gabriel, err_gab.mean_rel_error)
        end

        append_csv_rows(rank_pruning_file, rank_rows)

        # ── 9e. Random pruning baseline ──────────────────────────────────────
        random_rows = String[]
        for p in PRUNE_FRACTIONS
            for rep in 1:N_RANDOM_REPLICATES
                rng_rand = MersenneTwister(SEED + 100 + rep)
                adj_rand, w_rand = random_prune(adj, weights, p, rng_rand)
                n_after = sum(length(adj_rand[i]) for i in 1:n)
                err = geodesic_error_at_pairs(adj_rand, w_rand, pairs, n)
                push!(random_rows, join([
                    mlabel, n, k, noise_std, p, rep,
                    n_edges, n_after, @sprintf("%.6f", 1.0 - n_after/n_edges),
                    @sprintf("%.6f", err.mean_rel_error),
                    @sprintf("%.6f", err.median_rel_error),
                    @sprintf("%.6f", err.spearman),
                    err.n_disconnected,
                ], ","))
            end
        end
        append_csv_rows(random_pruning_file, random_rows)
        @printf "    Random baseline: %d rows written\n" length(random_rows)
    end

    @printf "  Config complete.\n"
end
end  # let orc_idx

# ==============================================================================
# Config TOML
# ==============================================================================

config_kwargs = if MANIFOLD == "swiss"
    Dict(
        :manifold   => "swiss_roll",
        :seed       => SEED,
        :t_min      => T_MIN,
        :t_range    => T_RANGE,
        :h_scale    => H_SCALE,
    )
else
    Dict(
        :manifold       => "torus_$(TORUS_KEY)",
        :seed           => SEED,
        :torus_R        => TORUS_R,
        :torus_r        => TORUS_r,
        :geodesic_n_grid => GEODESIC_N_GRID,
    )
end

write_config_toml(joinpath(run_dir, "config.toml");
    config_kwargs...,
    n_grid          = collect(N_GRID),
    k_grid          = collect(K_GRID),
    noise_grid      = NOISE_GRID,
    orc_variants    = ORC_VARIANTS,
    tau_grid        = collect(TAU_GRID_RUN),
    prune_fractions = collect(PRUNE_FRACTIONS),
    skip_detection  = SKIP_DETECTION,
    skip_pruning    = SKIP_PRUNING,
    skip_edges      = SKIP_EDGES,
    smoke           = _smoke,
)

# ==============================================================================
# Console summary
# ==============================================================================

println()
println("=" ^ 80)
println("Output directory: $run_dir")
println("  raw.csv            : $(raw_file)")
!SKIP_EDGES     && println("  edges.csv          : $(edges_file)")
!SKIP_PRUNING   && println("  oracle.csv         : $(oracle_file)")
!SKIP_PRUNING   && println("  rank_pruning.csv   : $(rank_pruning_file)")
!SKIP_PRUNING   && println("  random_pruning.csv : $(random_pruning_file)")
println("  config.toml")
println("=" ^ 80)
