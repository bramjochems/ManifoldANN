"""
End-to-end pipeline evaluation of the curvature-free geodesic estimator
(thesis §6.4). Implements the protocol in
`scripts/experiment_geodesic_e2e_PLAN.md`. Self-contained linear procedural
script: not part of the package proper.

Outputs `benchmark_results/geodesic_e2e/{timestamp}/`:
  results.csv   raw (n, rep, scheme, pair) rows
  summary.csv   one row per (n, scheme), with cluster-bootstrap CIs
  config.toml   reproducibility metadata

Environment variables:
  N_REPS_OVERRIDE=2     override the rep count (quick check)
  OUTPUT_DIR=path       override the timestamped output directory
  SMOKE=1               smallest n, 1 rep, 5 pairs (quick verification)

Usage:
  julia --project=. -t auto scripts/experiment_geodesic_e2e.jl
"""

using ManifoldANN
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates
using Pkg

include(joinpath(@__DIR__, "..", "orc", "orc_helpers.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "..", "benchmark_results", "geodesic_e2e")

# Swiss roll parameters
const T_MIN   = 1.5π
const T_RANGE = 3π
const H_SCALE = 10.0

# Graph
const K = 15
const INTRINSIC_DIM = 2

# Pair selection
const N_PAIRS_DEFAULT = 100
const N_REF = 4000
const K_REF_PAIRS = 50_000
const PAIR_QUANTILE_LO = 0.25
const PAIR_QUANTILE_HI = 0.75
const MAX_CANDIDATE_MULT = 10  # cap candidates at MAX * N_PAIRS

# Reps and bootstrap
const BOOTSTRAP_B = 1000

# Seeds (per plan §2). NOTE: plan literally pins these constants.
const BASE_SEED_DATA  = 0xC0DE_6E0
const BASE_SEED_PAIRS = 0x9A1C_6E0  # plan's "0xPA1R_6E0" stand-in

# Smoke / overrides
const SMOKE = get(ENV, "SMOKE", "") == "1"

const N_GRID = SMOKE ? [200] : [200, 350, 500, 800, 1200, 2000, 4000, 8000]

const N_REPS = let
    if SMOKE
        1
    elseif !isempty(get(ENV, "N_REPS_OVERRIDE", ""))
        parse(Int, ENV["N_REPS_OVERRIDE"])
    else
        10
    end
end

const N_PAIRS = SMOKE ? 5 : N_PAIRS_DEFAULT

# Schemes — names map to API:
#   d_E         -> EuclideanChord()
#   d_T_sym     -> TangentProjectedSymmetricMean()
#   d_g_hat_sym -> CurvatureFreeSymmetric()
#   d_analytic  -> custom edge weights from exact_swiss_roll_geodesic (no API)
# Plan used the (now-stale) trait name `AbstractEdgeGeodesicEstimator` and
# the kwarg `edge_estimator=`; current API is `AbstractEdgeWeight` and
# `edge_weight=`. We use the current names.
const SCHEMES = ["d_E", "d_T_sym", "d_g_hat_sym", "d_analytic"]

# Sanity-check the seed namespacing scheme: rep blocks are 1_000_000 wide,
# largest n is 8000, so adding n never crosses into the next rep block, and
# the two BASE constants are far enough apart that their (n, r) grids do
# not collide.
data_seed(n, r) = BASE_SEED_DATA  + 1_000_000 * r + n
pair_seed(n, r) = BASE_SEED_PAIRS + 1_000_000 * r + n
@assert maximum(N_GRID) < 1_000_000 "rep multiplier collides with n"
@assert abs(BASE_SEED_DATA - BASE_SEED_PAIRS) > 1_000_000 * (N_REPS + 1) "seed namespaces too close"

# ==============================================================================
# Output directory
# ==============================================================================

const ENV_OUT = get(ENV, "OUTPUT_DIR", "")
const RUN_DIR = isempty(ENV_OUT) ?
    joinpath(RESULTS_DIR, Dates.format(now(), "yyyymmdd_HHMMSS")) :
    ENV_OUT
mkpath(RUN_DIR)

const RESULTS_CSV = joinpath(RUN_DIR, "results.csv")
const SUMMARY_CSV = joinpath(RUN_DIR, "summary.csv")

const RESULTS_HEADER = join([
    "n", "rep", "data_seed", "pair_seed", "scheme", "pair_id",
    "src", "tgt", "path_length", "estimated_distance",
    "chosen_path_analytic_length", "ideal_path_analytic_length",
    "true_geodesic", "relative_error", "edge_weight_error",
    "path_selection_error", "path_deviation_error", "disconnected"
], ",")
init_csv(RESULTS_CSV, RESULTS_HEADER)

const SUMMARY_HEADER = join([
    "n", "scheme", "n_pairs_used", "n_disconnected", "n_neg_edges_total",
    "mre_mean", "mre_ci_lo", "mre_ci_hi",
    "edge_weight_mre_mean", "path_selection_mre_mean", "path_deviation_mre_mean"
], ",")
init_csv(SUMMARY_CSV, SUMMARY_HEADER)

# ==============================================================================
# Path-recovering Dijkstra (inline; orc_helpers' dijkstra_from doesn't return
# predecessors).
# ==============================================================================

"""
    dijkstra_with_paths(adj, weights, source, n) -> (dist, prev)

`prev[i]` is the predecessor of node `i` on the shortest path from `source`,
or 0 if unreachable / for the source itself.
"""
function dijkstra_with_paths(adj::Vector{Vector{Int}},
                              weights::Dict{Tuple{Int,Int}, Float64},
                              source::Int, n::Int)
    dist = fill(Inf, n)
    prev = zeros(Int, n)
    visited = falses(n)
    dist[source] = 0.0

    heap = MinHeap()
    heap_push!(heap, 0.0, source)

    while !isempty(heap)
        d, u = heap_pop!(heap)
        visited[u] && continue
        visited[u] = true
        for v in adj[u]
            w = get(weights, (u, v), Inf)
            alt = dist[u] + w
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
                heap_push!(heap, alt, v)
            end
        end
    end
    return dist, prev
end

"""Reconstruct vertex sequence from `source` to `target` given a predecessor
array. Returns an empty vector if `target` is unreachable."""
function reconstruct_path(prev::Vector{Int}, source::Int, target::Int)
    target == source && return [source]
    prev[target] == 0 && return Int[]
    path = Int[target]
    cur = target
    while cur != source
        cur = prev[cur]
        cur == 0 && return Int[]
        push!(path, cur)
    end
    reverse!(path)
    return path
end

"""Sum of analytic (true) geodesic arc lengths along the edges of `path`."""
function path_analytic_length(path::Vector{Int}, params)
    length(path) < 2 && return 0.0
    total = 0.0
    for i in 1:(length(path) - 1)
        total += exact_swiss_roll_geodesic(params, path[i], path[i+1])
    end
    return total
end

# ==============================================================================
# Build adjacency + per-scheme weights
# ==============================================================================

"""
Convert a WeightedKNNGraph (or KNNGraph) into an `(adj, weights)` pair
indexed by (i,j) tuples. `wg.edge_weights[i][j]` is the weight along
the j-th out-edge of node i.
"""
function wg_to_adj_weights(wg::ManifoldANN.WeightedKNNGraph)
    n = length(wg)
    adj = [Int[] for _ in 1:n]
    weights = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        nbrs = wg.graph[i]
        ews  = wg.edge_weights[i]
        for (jpos, j) in enumerate(nbrs)
            push!(adj[i], j)
            weights[(i, j)] = Float64(ews[jpos])
        end
    end
    return adj, weights
end

"""
Build the analytic-baseline weights: for every directed kNN edge (i, j),
assign `exact_swiss_roll_geodesic(params, i, j)` as the weight. Reuses the
adjacency of the existing (Euclidean-weighted) `(adj, _)`.
"""
function build_analytic_weights(adj::Vector{Vector{Int}}, params)
    weights = Dict{Tuple{Int,Int}, Float64}()
    for (i, nbrs) in enumerate(adj)
        for j in nbrs
            weights[(i, j)] = exact_swiss_roll_geodesic(params, i, j)
        end
    end
    return weights
end

# ==============================================================================
# Reference distribution for pair acceptance band
# ==============================================================================

"""
Compute Q25/Q75 of analytic geodesic distances on a reference draw. Done
once at run start; cached.
"""
function compute_pair_acceptance_band()
    @info "Computing pair-distance acceptance band (one-time, n_ref=$(N_REF), K=$(K_REF_PAIRS))"
    rng_ref = MersenneTwister(BASE_SEED_DATA - 1)
    _, params_ref = generate_swiss_roll(N_REF; rng=rng_ref,
                                         t_min=T_MIN, t_range=T_RANGE, h_scale=H_SCALE)
    dists = Vector{Float64}(undef, K_REF_PAIRS)
    rng_pairs = MersenneTwister(BASE_SEED_DATA - 2)
    for k in 1:K_REF_PAIRS
        i = rand(rng_pairs, 1:N_REF)
        j = rand(rng_pairs, 1:N_REF)
        while j == i
            j = rand(rng_pairs, 1:N_REF)
        end
        dists[k] = exact_swiss_roll_geodesic(params_ref, i, j)
    end
    d_lo = quantile(dists, PAIR_QUANTILE_LO)
    d_hi = quantile(dists, PAIR_QUANTILE_HI)
    @info @sprintf("Acceptance band: [%.4f, %.4f] (Q%.0f / Q%.0f of %d ref pairs)",
                   d_lo, d_hi, 100PAIR_QUANTILE_LO, 100PAIR_QUANTILE_HI, K_REF_PAIRS)
    return d_lo, d_hi
end

"""Per-(n, rep) pair selection."""
function select_pairs(params, n::Int, rng::AbstractRNG, d_lo::Float64, d_hi::Float64)
    pairs = Tuple{Int, Int, Float64}[]
    seen = Set{Tuple{Int,Int}}()
    n_candidates = 0
    cap = MAX_CANDIDATE_MULT * N_PAIRS
    while length(pairs) < N_PAIRS && n_candidates < cap
        n_candidates += 1
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        i, j = minmax(i, j)
        (i, j) in seen && continue
        push!(seen, (i, j))
        d = exact_swiss_roll_geodesic(params, i, j)
        if d_lo <= d <= d_hi
            push!(pairs, (i, j, d))
        end
    end
    if length(pairs) < N_PAIRS
        @warn "Only $(length(pairs))/$N_PAIRS pairs accepted at n=$n after $n_candidates draws"
    end
    return pairs
end

# ==============================================================================
# Cluster bootstrap on reps
# ==============================================================================

"""
Cluster bootstrap by reps. `rep_to_errors` maps rep => Vector{Float64} of
per-pair MREs. Returns (lo, hi) percentile band.
"""
function cluster_bootstrap_ci(rep_to_errors::Dict{Int, Vector{Float64}};
                              B::Int=BOOTSTRAP_B,
                              alpha::Float64=0.05,
                              seed::Int=12345)
    reps = collect(keys(rep_to_errors))
    isempty(reps) && return (NaN, NaN)
    R = length(reps)
    rng = MersenneTwister(seed)
    means = Vector{Float64}(undef, B)
    for b in 1:B
        pooled = Float64[]
        for _ in 1:R
            r = reps[rand(rng, 1:R)]
            append!(pooled, rep_to_errors[r])
        end
        means[b] = isempty(pooled) ? NaN : mean(pooled)
    end
    valid = filter(!isnan, means)
    isempty(valid) && return (NaN, NaN)
    return (quantile(valid, alpha/2), quantile(valid, 1 - alpha/2))
end

# ==============================================================================
# Banner
# ==============================================================================

println("=" ^ 80)
println("Geodesic E2E Experiment (thesis §6.4)")
println("=" ^ 80)
println("Threads        : ", Threads.nthreads())
println("Smoke mode     : ", SMOKE)
println("n grid         : ", N_GRID)
println("N reps         : ", N_REPS)
println("N pairs / rep  : ", N_PAIRS)
println("k              : ", K)
println("Schemes        : ", SCHEMES)
println("Output dir     : ", RUN_DIR)
println("=" ^ 80)
println()

# ==============================================================================
# Compute reference acceptance band once
# ==============================================================================

const D_LO, D_HI = compute_pair_acceptance_band()

# ==============================================================================
# In-memory store for summary aggregation
# ==============================================================================

# Keyed by (n, scheme): collect per-row diagnostics for summary.
mutable struct SchemeAgg
    rep_to_rels::Dict{Int, Vector{Float64}}     # rep => [|MRE|, ...]
    edge_weight_abs::Vector{Float64}
    path_select::Vector{Float64}
    path_deviation::Vector{Float64}
    n_disconnected::Int
    n_neg_edges_total::Int
end
SchemeAgg() = SchemeAgg(Dict{Int, Vector{Float64}}(), Float64[], Float64[], Float64[], 0, 0)

const AGG = Dict{Tuple{Int,String}, SchemeAgg}()
for n in N_GRID, s in SCHEMES
    AGG[(n, s)] = SchemeAgg()
end

# ==============================================================================
# Main loop
# ==============================================================================

t_start = time()

for n in N_GRID
    @printf("\n=== n = %d ===\n", n)
    for rep in 0:(N_REPS - 1)
        ds = data_seed(n, rep)
        ps = pair_seed(n, rep)
        @printf("  [n=%d rep=%d/%d]  data_seed=%d pair_seed=%d\n",
                n, rep + 1, N_REPS, ds, ps)

        # ── 1. Generate data ──────────────────────────────────────────────
        rng_data = MersenneTwister(ds)
        data, params = generate_swiss_roll(n; rng=rng_data,
                                           t_min=T_MIN, t_range=T_RANGE, h_scale=H_SCALE)

        # ── 2. ANN index + kNN graph (undirected union symmetrisation) ────
        index = build_index(BruteForceIndex, data)
        # build_geodesic_model builds its graph internally; build a separate
        # one here only so the analytic baseline shares the same adjacency.
        # NOTE: build_geodesic_model uses default `directed`/symmetrisation
        # of build_knn_graph (directed=true at time of writing). We follow
        # the same convention here so the analytic baseline shares topology
        # with the package-built models.
        method = PCAMethod(intrinsic_dim=INTRINSIC_DIM)

        # ── 3. Build models per scheme via the post-refactor API ──────────
        # Scheme weights map (post-refactor):
        weight_traits = Dict(
            "d_E"         => EuclideanChord(),
            "d_T_sym"     => TangentProjectedSymmetricMean(),
            "d_g_hat_sym" => CurvatureFreeSymmetric(),
        )

        # Build all three package models. Each builds its own kNN graph;
        # the topology is deterministic from `data` + `k` so they all
        # share adjacency.
        models = Dict{String, Any}()
        adjs = Dict{String, Vector{Vector{Int}}}()
        wts  = Dict{String, Dict{Tuple{Int,Int}, Float64}}()
        n_neg_log = Dict{String, Int}()
        for (name, w) in weight_traits
            t0 = time()
            mdl = build_geodesic_model(method, index, data; k=K, edge_weight=w)
            adj, weights = wg_to_adj_weights(mdl.weighted_graph)
            models[name] = mdl
            adjs[name] = adj
            wts[name]  = weights
            diag = ManifoldANN.diagnostics(mdl)
            n_neg_log[name] = diag.n_negative_fallbacks
            @printf("    built %-12s in %.2fs (%d edges, n_neg=%d)\n",
                    name, time() - t0, length(weights), diag.n_negative_fallbacks)
        end

        # Analytic baseline: bypass build_geodesic_model. Reuse adjacency
        # from any of the three models (topology identical).
        adj_base = adjs["d_E"]
        wts["d_analytic"]  = build_analytic_weights(adj_base, params)
        adjs["d_analytic"] = adj_base
        n_neg_log["d_analytic"] = 0

        # ── 4. Pair selection ─────────────────────────────────────────────
        rng_pairs = MersenneTwister(ps)
        pairs = select_pairs(params, n, rng_pairs, D_LO, D_HI)
        n_pairs_actual = length(pairs)
        if n_pairs_actual == 0
            @warn "No pairs accepted at n=$n rep=$rep, skipping"
            continue
        end

        # ── 5. Per-scheme: Dijkstra from each unique source, then per-pair ─
        # We need ideal_path_analytic_length first (from d_analytic).
        # Run d_analytic first.
        scheme_dist = Dict{String, Dict{Int, Vector{Float64}}}()  # scheme => src => dist[]
        scheme_prev = Dict{String, Dict{Int, Vector{Int}}}()      # scheme => src => prev[]

        unique_sources = sort!(unique([p[1] for p in pairs]))

        for scheme in SCHEMES
            d_dict = Dict{Int, Vector{Float64}}()
            p_dict = Dict{Int, Vector{Int}}()
            for s in unique_sources
                dist, prev = dijkstra_with_paths(adjs[scheme], wts[scheme], s, n)
                d_dict[s] = dist
                p_dict[s] = prev
            end
            scheme_dist[scheme] = d_dict
            scheme_prev[scheme] = p_dict
        end

        # ── 6. First pass: identify pairs disconnected by the analytic
        #      baseline (these are excluded for ALL schemes per plan §7).
        analytic_disconnected = Set{Tuple{Int,Int}}()
        for (i, j, _) in pairs
            if isinf(scheme_dist["d_analytic"][i][j])
                push!(analytic_disconnected, (i, j))
            end
        end
        if !isempty(analytic_disconnected)
            @warn "  d_analytic disconnects $(length(analytic_disconnected))/$n_pairs_actual pairs at n=$n rep=$rep; excluded for all schemes"
        end

        # ── 7. Per-pair, per-scheme: build rows ───────────────────────────
        rows = String[]
        for (pair_id, (src, tgt, d_true)) in enumerate(pairs)
            # Plan §7: pair_id 0..N-1
            pid = pair_id - 1

            excluded = (src, tgt) in analytic_disconnected

            # Compute ideal first.
            ideal_dist = scheme_dist["d_analytic"][src][tgt]
            ideal_path = excluded ? Int[] :
                         reconstruct_path(scheme_prev["d_analytic"][src], src, tgt)
            # By construction (analytic baseline), ideal_path_analytic_length
            # equals the Dijkstra distance under d_analytic.
            ideal_analytic_len = excluded ? NaN : ideal_dist
            path_dev_err = excluded ? NaN : (ideal_analytic_len - d_true) / d_true

            for scheme in SCHEMES
                est = excluded ? NaN : scheme_dist[scheme][src][tgt]
                disconn_here = excluded || isinf(est)
                if disconn_here
                    push!(rows, join([
                        n, rep, ds, ps, scheme, pid, src, tgt,
                        0, "NaN", "NaN", "NaN",
                        @sprintf("%.6e", d_true),
                        "NaN", "NaN", "NaN", "NaN",
                        1
                    ], ","))
                    AGG[(n, scheme)].n_disconnected += 1
                    continue
                end

                path = reconstruct_path(scheme_prev[scheme][src], src, tgt)
                chosen_analytic_len = path_analytic_length(path, params)

                rel_err = abs(est - d_true) / d_true
                edge_w_err = (est - chosen_analytic_len) / d_true
                path_sel_err = (chosen_analytic_len - ideal_analytic_len) / d_true

                push!(rows, join([
                    n, rep, ds, ps, scheme, pid, src, tgt,
                    length(path) - 1,
                    @sprintf("%.6e", est),
                    @sprintf("%.6e", chosen_analytic_len),
                    @sprintf("%.6e", ideal_analytic_len),
                    @sprintf("%.6e", d_true),
                    @sprintf("%.6e", rel_err),
                    @sprintf("%.6e", edge_w_err),
                    @sprintf("%.6e", path_sel_err),
                    @sprintf("%.6e", path_dev_err),
                    0
                ], ","))

                # Aggregate
                a = AGG[(n, scheme)]
                push!(get!(a.rep_to_rels, rep, Float64[]), rel_err)
                push!(a.edge_weight_abs, abs(edge_w_err))
                push!(a.path_select, path_sel_err)
                push!(a.path_deviation, path_dev_err)
            end
        end

        append_csv_rows(RESULTS_CSV, rows)

        # Add per-rep n_neg counts to aggregates
        for scheme in SCHEMES
            AGG[(n, scheme)].n_neg_edges_total += n_neg_log[scheme]
        end
    end
end

# ==============================================================================
# Summary CSV
# ==============================================================================

@printf("\nWriting summary.csv ...\n")

summary_rows = String[]
for n in N_GRID, scheme in SCHEMES
    a = AGG[(n, scheme)]
    n_used = sum(length(v) for v in values(a.rep_to_rels); init=0)
    if n_used == 0
        push!(summary_rows, join([
            n, scheme, 0, a.n_disconnected, a.n_neg_edges_total,
            "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"
        ], ","))
        continue
    end
    all_rels = Float64[]
    for v in values(a.rep_to_rels); append!(all_rels, v); end
    mre_mean = mean(all_rels)
    ci_lo, ci_hi = cluster_bootstrap_ci(a.rep_to_rels)

    edge_w_mean = isempty(a.edge_weight_abs) ? NaN : mean(a.edge_weight_abs)
    path_sel_mean = isempty(a.path_select) ? NaN : mean(a.path_select)
    path_dev_mean = isempty(a.path_deviation) ? NaN : mean(a.path_deviation)

    push!(summary_rows, join([
        n, scheme, n_used, a.n_disconnected, a.n_neg_edges_total,
        @sprintf("%.6e", mre_mean),
        @sprintf("%.6e", ci_lo),
        @sprintf("%.6e", ci_hi),
        @sprintf("%.6e", edge_w_mean),
        @sprintf("%.6e", path_sel_mean),
        @sprintf("%.6e", path_dev_mean),
    ], ","))
end
append_csv_rows(SUMMARY_CSV, summary_rows)

# ==============================================================================
# Config TOML
# ==============================================================================

# Capture Pkg.status() into a string.
pkg_status_str = let
    io = IOBuffer()
    try
        Pkg.status(io=io)
    catch e
        println(io, "(Pkg.status capture failed: $e)")
    end
    String(take!(io))
end

# Write config.toml. We extend write_config_toml's output by appending a
# `[pkg]` section with the multi-line status text (TOML triple-quoted).
write_config_toml(joinpath(RUN_DIR, "config.toml");
    experiment       = "geodesic_e2e",
    manifold         = "swiss_roll",
    base_seed_data   = BASE_SEED_DATA,
    base_seed_pairs  = BASE_SEED_PAIRS,
    n_grid           = collect(N_GRID),
    n_reps           = N_REPS,
    k                = K,
    n_pairs          = N_PAIRS,
    t_min            = T_MIN,
    t_range          = T_RANGE,
    h_scale          = H_SCALE,
    intrinsic_dim    = INTRINSIC_DIM,
    schemes          = SCHEMES,
    bootstrap_b      = BOOTSTRAP_B,
    symmetrisation   = "directed_default",  # build_geodesic_model uses build_knn_graph default
    n_ref            = N_REF,
    k_ref_pairs      = K_REF_PAIRS,
    pair_q_lo        = PAIR_QUANTILE_LO,
    pair_q_hi        = PAIR_QUANTILE_HI,
    d_lo             = D_LO,
    d_hi             = D_HI,
    smoke            = SMOKE,
    julia_version    = string(VERSION),
)

# Append pkg_status as a triple-quoted block.
open(joinpath(RUN_DIR, "config.toml"), "a") do io
    println(io)
    println(io, "pkg_status = \"\"\"")
    print(io, pkg_status_str)
    endswith(pkg_status_str, "\n") || println(io)
    println(io, "\"\"\"")
end

t_total = time() - t_start

println()
println("=" ^ 80)
@printf("Total wall time: %.1fs\n", t_total)
println("Output directory: $RUN_DIR")
println("  results.csv : $RESULTS_CSV")
println("  summary.csv : $SUMMARY_CSV")
println("  config.toml")
println("=" ^ 80)
