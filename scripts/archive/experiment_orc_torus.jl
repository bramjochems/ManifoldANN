"""
ORC Shortcut Detection on the Torus — Main Experiment

Evaluates Ollivier-Ricci curvature as a shortcut detector on the flat torus,
where ground-truth shortcut edges are defined via numerical geodesic distances.

Unlike the Swiss roll (developable, zero Gaussian curvature), the torus has
nonzero intrinsic curvature that varies with position:
    K(v) = cos(v) / (r · (R + r·cos(v)))
This makes it a richer test bed: shortcuts may be more persistent and ORC
pruning may prove more effective than on the Swiss roll.

Ground-truth definition (matching the Swiss roll experiment):
    r_ij = d_ambient(i,j) / d_geodesic(i,j)
An edge is labelled a shortcut when r_ij < τ.

Experimental factors:
  - torus_variant : (R, r) pair controlling major/minor radius and curvature range
  - n             : number of points (sampling density)
  - k             : neighbourhood size
  - noise_std     : Gaussian noise added to ambient coordinates
  - orc_variant   : "standard" vs "orcml"
  - tau           : shortcut-ratio threshold (ground-truth labelling)

Torus variants explored:
  - (R=4.0, r=1.0) : thin tube, R/r=4 — mild curvature, large hole
  - (R=2.0, r=1.0) : medium tube, R/r=2 — moderate curvature
  - (R=1.5, r=1.0) : fat tube, R/r=1.5 — strong curvature, small hole

Outputs (in timestamped directory results/orc_results/torus_{ts}/):
  raw.csv          — one row per (torus_variant, n, k, noise, variant, τ)
  pivot_f1.csv     — F1 at κ=0 per variant slice
  pivot_best.csv   — best-threshold F1 per variant slice
  edges.csv        — per-edge (κ, ratio) data for AUROC [gated by SKIP_EDGES]
  geoderror.csv    — geodesic error at pruning thresholds [gated by SKIP_GEODERROR]
  config.toml      — git hash, parameters, flags for reproducibility

Results are written incrementally after each ORC computation, so a crash
mid-run loses at most one config.  Restarting skips already-computed configs.

Usage:
    julia --project=. -t auto scripts/experiment_orc_torus.jl

Environment overrides:
    SMOKE=1                  single config (first variant, n=200, k=10, standard, τ=0.5)
    N_OVERRIDE=500            single n value
    K_OVERRIDE=10             single k value
    VARIANT_OVERRIDE=R2r1     single torus variant by key
    SKIP_EDGES=1              skip per-edge CSV
    SKIP_GEODERROR=1          skip geodesic error analysis
    RESUME_DIR=path           resume into a previous run's directory (skips
                              already-computed configs)
"""

using ManifoldANN
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates

include(joinpath(@__DIR__, "orc_helpers.jl"))
include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic", "torus_utils.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const RESULTS_DIR = joinpath(
    @__DIR__, "..", "..", "..", "docs", "thesis", "results", "orc_results"
)

# Torus geometry variants: (key, R, r, description)
const TORUS_VARIANTS_ALL = [
    (key="R15r1", R=1.5, r=1.0, desc="fat tube, R/r=1.5 — strong curvature"),
    (key="R2r1",  R=2.0, r=1.0, desc="medium tube, R/r=2 — moderate curvature"),
    (key="R4r1",  R=4.0, r=1.0, desc="thin tube, R/r=4 — mild curvature"),
]

# τ grid: r_ij = d_amb/d_geo < τ → shortcut
const TAU_GRID = [0.3, 0.5, 0.67, 0.8]

# Grid resolution for torus geodesic approximation
const GEODESIC_N_GRID = 100

const SEED = 42

# Apply overrides
_smoke = get(ENV, "SMOKE", "") == "1"
_skip_edges     = get(ENV, "SKIP_EDGES", "") == "1"
_skip_geoderror = get(ENV, "SKIP_GEODERROR", "") == "1"

_n_grid     = !isempty(get(ENV, "N_OVERRIDE", "")) ? [parse(Int, ENV["N_OVERRIDE"])] :
               [200, 500, 1000, 2000]
_k_grid     = !isempty(get(ENV, "K_OVERRIDE", "")) ? [parse(Int, ENV["K_OVERRIDE"])] :
               [5, 10, 15, 20]
_noise_grid = [0.0, 0.1, 0.5]
_variants   = ["standard", "orcml"]

_variant_override = get(ENV, "VARIANT_OVERRIDE", "")
_torus_variants = isempty(_variant_override) ? TORUS_VARIANTS_ALL :
    filter(v -> v.key == _variant_override, TORUS_VARIANTS_ALL)

if _smoke
    _n_grid           = [200]
    _k_grid           = [10]
    _noise_grid       = [0.0]
    _variants         = ["standard"]
    _torus_variants   = [TORUS_VARIANTS_ALL[1]]
end

const N_GRID          = _n_grid
const K_GRID          = _k_grid
const NOISE_GRID      = _noise_grid
const ORC_VARIANTS    = _variants
const TORUS_VARIANTS  = _torus_variants
const TAU_GRID_RUN    = _smoke ? [0.5] : TAU_GRID
const SKIP_EDGES      = _skip_edges
const SKIP_GEODERROR  = _skip_geoderror

# ==============================================================================
# Manifold-specific helpers
# ==============================================================================

"""
    precompute_ratios_torus(graph, data, params; n_grid)

Return a Dict (i,j) → r_ij = d_ambient / d_geodesic for every graph edge.
Geodesic is computed numerically on the torus parameter grid.
"""
function precompute_ratios_torus(graph, data::AbstractMatrix, params::NamedTuple;
                                 n_grid::Int=GEODESIC_N_GRID)
    ratios = Dict{Tuple{Int,Int}, Float64}()
    n = size(data, 2)
    for i in 1:n
        for j in graph[i]
            d_amb = norm(data[:, i] - data[:, j])
            d_geo = exact_torus_geodesic(params, i, j; n_grid=n_grid)
            ratios[(i, j)] = d_geo > 1e-12 ? d_amb / d_geo : 1.0
        end
    end
    return ratios
end

"""
    torus_pairs(params, n, rng)

Sample random pairs with true geodesic distances for downstream error analysis.
"""
function torus_pairs(params, n, rng)
    pairs = Tuple{Int, Int, Float64}[]
    seen = Set{Tuple{Int,Int}}()
    while length(pairs) < min(N_GEO_PAIRS, n * (n - 1) ÷ 2)
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        edge = minmax(i, j)
        edge in seen && continue
        push!(seen, edge)
        d = exact_torus_geodesic(params, i, j; n_grid=GEODESIC_N_GRID)
        push!(pairs, (i, j, d))
    end
    return pairs
end

# ==============================================================================
# Output directory setup (before the loop, so we can write incrementally)
# ==============================================================================

mkpath(RESULTS_DIR)
_resume_dir = get(ENV, "RESUME_DIR", "")

run_dir = if !isempty(_resume_dir)
    _resume_dir
else
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    joinpath(RESULTS_DIR, "torus_$(ts)")
end
mkpath(run_dir)

# CSV file paths
raw_file      = joinpath(run_dir, "raw.csv")
edges_file    = joinpath(run_dir, "edges.csv")
geoderror_file = joinpath(run_dir, "geoderror.csv")

# Initialize CSV headers (no-op if files already exist)
const RAW_HEADER = "torus_key,torus_R,torus_r,n,k,noise_std,variant,tau," *
    "n_edges,n_shortcuts,frac_shortcuts," *
    "precision_0,recall_0,f1_0,tp_0,fp_0,tn_0,fn_0," *
    "best_threshold,precision_best,recall_best,f1_best," *
    "mean_kappa_shortcuts,mean_kappa_non_shortcuts," *
    "n_shortcut_edges,n_non_shortcut_edges,ratio_time_s,orc_time_s"
init_csv(raw_file, RAW_HEADER)

if !SKIP_EDGES
    const EDGES_HEADER = "manifold,n,k,noise_std,orc_variant,edge_i,edge_j,kappa,ratio,is_sc_05,is_sc_08"
    init_csv(edges_file, EDGES_HEADER)
end

if !SKIP_GEODERROR
    const GEODERROR_HEADER = "manifold,n,k,noise_std,orc_variant,kappa_thresh," *
        "n_edges_before,n_edges_after,frac_removed,mean_rel_error,median_rel_error,n_disconnected"
    init_csv(geoderror_file, GEODERROR_HEADER)
end

# Load already-completed configs for resume
completed_configs = load_completed_keys(raw_file, [:torus_key, :n, :k, :noise_std, :variant])

# ==============================================================================
# Main loop
# ==============================================================================

println("=" ^ 80)
println("ORC Shortcut Detection — Torus Experiment")
println("=" ^ 80)
println("Threads         : ", Threads.nthreads())
println("n grid          : ", N_GRID)
println("k grid          : ", K_GRID)
println("noise grid      : ", NOISE_GRID)
println("ORC variants    : ", ORC_VARIANTS)
println("τ grid          : ", TAU_GRID_RUN)
println("Torus variants  :")
for tv in TORUS_VARIANTS
    @printf "  %-8s  R=%.1f  r=%.1f  (%s)\n" tv.key tv.R tv.r tv.desc
end
println("Geodesic n_grid : ", GEODESIC_N_GRID)
println("SKIP_EDGES      : ", SKIP_EDGES)
println("SKIP_GEODERROR  : ", SKIP_GEODERROR)
println("Output dir      : ", run_dir)
println("Resuming        : ", !isempty(completed_configs), " (", length(completed_configs), " configs done)")
println("=" ^ 80)
println()

results = []
edge_results = []
geoderror_results = []

total_orc = length(TORUS_VARIANTS) * length(N_GRID) * length(K_GRID) *
            length(NOISE_GRID) * length(ORC_VARIANTS)

let orc_idx = 0
for torus_v in TORUS_VARIANTS, n in N_GRID, k in K_GRID,
        noise_std in NOISE_GRID, variant in ORC_VARIANTS

    orc_idx += 1

    # Skip already-computed configs (resume support)
    config_key = (torus_v.key, string(n), string(k), string(noise_std), variant)
    if config_key in completed_configs
        @printf("[%d/%d]  torus=%-8s  n=%-5d  k=%-3d  noise=%.2f  variant=%s  [SKIP — already computed]\n",
            orc_idx, total_orc, torus_v.key, n, k, noise_std, variant)
        continue
    end

    @printf("\n[%d/%d]  torus=%-8s  n=%-5d  k=%-3d  noise=%.2f  variant=%s\n",
        orc_idx, total_orc, torus_v.key, n, k, noise_std, variant)

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = MersenneTwister(SEED)
    data_clean, params = generate_torus(n; rng=rng, R=torus_v.R, r=torus_v.r)
    data = noise_std > 0 ? data_clean .+ noise_std .* randn(rng, size(data_clean)) :
                           data_clean

    # ── kNN graph ─────────────────────────────────────────────────────────────
    directed = (variant == "standard")
    index    = build_index(BruteForceIndex, data)
    graph    = build_knn_graph(index, data; k=k, directed=directed)
    n_edges  = sum(length(graph[i]) for i in 1:n)

    # ── Pre-compute r_ij ratios (τ-independent) ────────────────────────────────
    t_ratio = @elapsed begin
        ratios = precompute_ratios_torus(graph, data, params)
    end
    @printf "  Geodesic ratios computed in %.2fs\n" t_ratio

    # ── Compute ORC (once per (n,k,noise,variant)) ────────────────────────────
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

    # ── Evaluate for each τ ───────────────────────────────────────────────────
    for tau in TAU_GRID_RUN
        labels  = label_shortcuts(ratios, tau)
        n_sc    = sum(values(labels))
        frac_sc = n_sc / n_edges

        m0    = evaluate_at_threshold(curvatures, labels, 0.0)
        best  = best_f1_threshold(curvatures, labels)
        kstat = curvature_stats_by_label(curvatures, labels)

        @printf("  τ=%.2f  shortcuts=%d/%d (%.1f%%)  κ<0: pre=%.3f rec=%.3f F1=%.3f  best_F1=%.3f@κ<%.2f\n",
            tau, n_sc, n_edges, 100frac_sc, m0.precision, m0.recall, m0.f1, best.f1, best.κ)

        push!(results, (
            torus_key                = torus_v.key,
            torus_R                  = torus_v.R,
            torus_r                  = torus_v.r,
            n                        = n,
            k                        = k,
            noise_std                = noise_std,
            variant                  = variant,
            tau                      = tau,
            n_edges                  = n_edges,
            n_shortcuts              = n_sc,
            frac_shortcuts           = frac_sc,
            precision_0              = m0.precision,
            recall_0                 = m0.recall,
            f1_0                     = m0.f1,
            tp_0                     = m0.tp,
            fp_0                     = m0.fp,
            tn_0                     = m0.tn,
            fn_0                     = m0.fn,
            best_threshold           = best.κ,
            precision_best           = best.precision,
            recall_best              = best.recall,
            f1_best                  = best.f1,
            mean_kappa_shortcuts     = kstat.mean_shortcut,
            mean_kappa_non_shortcuts = kstat.mean_non_shortcut,
            n_shortcut_edges         = kstat.n_shortcut,
            n_non_shortcut_edges     = kstat.n_non_shortcut,
            ratio_time_s             = t_ratio,
            orc_time_s               = t_orc,
        ))
    end

    # ── Per-edge data (for AUROC) — τ-independent ─────────────────────────────
    if !SKIP_EDGES
        for (edge, result) in curvatures
            haskey(ratios, edge) || continue
            r = ratios[edge]
            push!(edge_results, (
                manifold    = torus_v.key,
                n           = n,
                k           = k,
                noise_std   = noise_std,
                orc_variant = variant,
                edge_i      = edge[1],
                edge_j      = edge[2],
                kappa       = result.curvature,
                ratio       = r,
                is_sc_05    = r < 0.5,
                is_sc_08    = r < 0.8,
            ))
        end
        @printf "  Saved %d edge records\n" length(curvatures)
    end

    # ── Downstream geodesic error analysis ────────────────────────────────────
    if !SKIP_GEODERROR
        @printf "  Computing geodesic error analysis..."
        t_geo = @elapsed begin
            rng_pairs = MersenneTwister(SEED + 1)
            pairs = torus_pairs(params, n, rng_pairs)
            adj, weights = graph_to_adj_weights(graph, data)

            for κ_thresh in [-Inf; PRUNE_THRESHOLDS]
                if κ_thresh == -Inf
                    adj_use, w_use = adj, weights
                    n_edges_pruned = n_edges
                else
                    adj_use, w_use = prune_graph(adj, weights, curvatures, κ_thresh)
                    n_edges_pruned = sum(length(adj_use[i]) for i in 1:n)
                end

                err = geodesic_error_at_pairs(adj_use, w_use, pairs, n)

                push!(geoderror_results, (
                    manifold       = torus_v.key,
                    n              = n,
                    k              = k,
                    noise_std      = noise_std,
                    orc_variant    = variant,
                    kappa_thresh   = κ_thresh == -Inf ? -999.0 : κ_thresh,
                    n_edges_before = n_edges,
                    n_edges_after  = n_edges_pruned,
                    frac_removed   = 1.0 - n_edges_pruned / n_edges,
                    mean_rel_error = err.mean_rel_error,
                    median_rel_error = err.median_rel_error,
                    n_disconnected = err.n_disconnected,
                ))
            end
        end
        @printf " %.1fs\n" t_geo
    end

    # ── Flush results for this config to CSV ──────────────────────────────────
    # Raw CSV: one row per τ for this (torus_key, n, k, noise_std, variant)
    raw_rows = String[]
    for r in results
        (r.torus_key == torus_v.key && r.n == n && r.k == k &&
         r.noise_std == noise_std && r.variant == variant) || continue
        push!(raw_rows, join([
            r.torus_key, r.torus_R, r.torus_r,
            r.n, r.k, r.noise_std, r.variant, r.tau, r.n_edges, r.n_shortcuts,
            @sprintf("%.6f", r.frac_shortcuts),
            @sprintf("%.6f", r.precision_0), @sprintf("%.6f", r.recall_0), @sprintf("%.6f", r.f1_0),
            r.tp_0, r.fp_0, r.tn_0, r.fn_0,
            @sprintf("%.6f", r.best_threshold),
            @sprintf("%.6f", r.precision_best), @sprintf("%.6f", r.recall_best), @sprintf("%.6f", r.f1_best),
            isnan(r.mean_kappa_shortcuts) ? "NaN" : @sprintf("%.6f", r.mean_kappa_shortcuts),
            isnan(r.mean_kappa_non_shortcuts) ? "NaN" : @sprintf("%.6f", r.mean_kappa_non_shortcuts),
            r.n_shortcut_edges, r.n_non_shortcut_edges,
            @sprintf("%.6f", r.ratio_time_s), @sprintf("%.6f", r.orc_time_s),
        ], ","))
    end
    append_csv_rows(raw_file, raw_rows)

    # Edges CSV (flush all accumulated edge_results, then clear)
    if !SKIP_EDGES
        edge_rows = String[]
        for r in edge_results
            push!(edge_rows, join([
                r.manifold, r.n, r.k, r.noise_std, r.orc_variant,
                r.edge_i, r.edge_j,
                @sprintf("%.8f", r.kappa), @sprintf("%.6f", r.ratio),
                r.is_sc_05 ? 1 : 0, r.is_sc_08 ? 1 : 0,
            ], ","))
        end
        append_csv_rows(edges_file, edge_rows)
        empty!(edge_results)
    end

    # Geoderror CSV (flush all accumulated geoderror_results, then clear)
    if !SKIP_GEODERROR
        geo_rows = String[]
        for r in geoderror_results
            push!(geo_rows, join([
                r.manifold, r.n, r.k, r.noise_std, r.orc_variant,
                @sprintf("%.4f", r.kappa_thresh),
                r.n_edges_before, r.n_edges_after,
                @sprintf("%.6f", r.frac_removed),
                @sprintf("%.6f", r.mean_rel_error),
                @sprintf("%.6f", r.median_rel_error),
                r.n_disconnected,
            ], ","))
        end
        append_csv_rows(geoderror_file, geo_rows)
        empty!(geoderror_results)
    end
end
end  # let orc_idx

# ==============================================================================
# Write pivot CSVs and config (from in-memory results)
# ==============================================================================

# ── Pivoted CSVs — one section per (torus_key, variant, noise, τ) ─────
sections_f1 = []
sections_best = []
for tv in TORUS_VARIANTS, variant in ORC_VARIANTS, noise in NOISE_GRID, tau in TAU_GRID_RUN
    filter_fn = let key=tv.key, v=variant, n=noise, t=tau
        r -> r.torus_key == key && r.variant == v && r.noise_std == n && r.tau == t
    end
    label = "torus=$(tv.key)  variant=$(variant)  noise=$(noise)  tau=$(tau)"
    push!(sections_f1,   (filter_fn, label))
    push!(sections_best, (filter_fn, label))
end

pivot_f1_file   = joinpath(run_dir, "pivot_f1.csv")
pivot_best_file = joinpath(run_dir, "pivot_best.csv")

write_pivot_csv(results, pivot_f1_file,   r -> r.f1_0,    sections_f1,   N_GRID, K_GRID)
write_pivot_csv(results, pivot_best_file, r -> r.f1_best, sections_best, N_GRID, K_GRID)

# ── Config TOML ──────────────────────────────────────────────────────────────
write_config_toml(joinpath(run_dir, "config.toml");
    manifold       = "torus",
    seed           = SEED,
    torus_variants = [tv.key for tv in TORUS_VARIANTS],
    geodesic_n_grid = GEODESIC_N_GRID,
    n_grid         = collect(N_GRID),
    k_grid         = collect(K_GRID),
    noise_grid     = NOISE_GRID,
    orc_variants   = ORC_VARIANTS,
    tau_grid       = collect(TAU_GRID_RUN),
    skip_edges     = SKIP_EDGES,
    skip_geoderror = SKIP_GEODERROR,
    smoke          = _smoke,
)

# ==============================================================================
# Console summary
# ==============================================================================

println()
println("=" ^ 80)
println("Output directory: $run_dir")
println("  raw.csv       : $(raw_file)")
println("  pivot_f1.csv  : $(pivot_f1_file)")
println("  pivot_best.csv: $(pivot_best_file)")
!SKIP_EDGES     && println("  edges.csv     : $(edges_file)")
!SKIP_GEODERROR && println("  geoderror.csv : $(geoderror_file)")
println("  config.toml")
println("=" ^ 80)

println()
println("Summary (κ<0 threshold, no noise, τ=0.5):")
println(rpad("torus", 8), rpad("variant", 10), rpad("n", 7), rpad("k", 5),
        rpad("shortcuts%", 12), rpad("pre", 8), rpad("rec", 8), rpad("F1", 8), "time(s)")
println("─" ^ 76)
for r in results
    (r.noise_std == 0.0 && r.tau == 0.5) || continue
    @printf("%-8s %-10s %-7d %-5d %-12.1f %-8.3f %-8.3f %-8.3f %.2f\n",
        r.torus_key, r.variant, r.n, r.k, (100*r.frac_shortcuts),
        r.precision_0, r.recall_0, r.f1_0, r.orc_time_s)
end
