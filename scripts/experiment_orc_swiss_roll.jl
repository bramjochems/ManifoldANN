"""
ORC Shortcut Detection on the Swiss Roll — Main Experiment

Evaluates Ollivier-Ricci curvature as a shortcut detector on the Swiss roll,
where ground-truth shortcut edges can be identified analytically.

Ground-truth definition (matching thesis Chapter 4):
    r_ij = d_ambient(i,j) / d_geodesic(i,j)
An edge is labelled a shortcut when r_ij < τ, i.e. the ambient distance
understates the geodesic by a factor of at least 1/τ.

τ is swept as an experimental axis (TAU_GRID) because it is a methodological
choice, not a fixed ground truth.  ORC is computed once per (n, k, noise,
variant); τ only affects the labelling step, so sweeping it costs nothing extra.

Experimental factors:
  - n            : number of points (sampling density)
  - k            : neighbourhood size
  - noise_std    : Gaussian noise added to ambient coordinates
  - orc_variant  : "standard" vs "orcml"
  - tau          : shortcut-ratio threshold (ground-truth labelling)

Three CSV outputs (all timestamped, never overwrite):
  _raw.csv          — one row per (n, k, noise, variant, τ); all metrics
  _pivot_f1.csv     — F1 at κ=0; one section per (variant, noise, τ); rows=n cols=k
  _pivot_best.csv   — best-threshold F1; same structure

Usage:
    julia --project=. -t auto scripts/experiment_orc_swiss_roll.jl

Environment overrides:
    SMOKE=1            single config (n=200, k=10, noise=0, standard, τ=0.5)
    N_OVERRIDE=500     single n value
    K_OVERRIDE=10      single k value
"""

using ManifoldANN
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates

include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const RESULTS_DIR = joinpath(
    @__DIR__, "..", "..", "..", "docs", "thesis", "results", "orc_results"
)

const T_MIN   = 1.5π
const T_RANGE = 3π
const H_SCALE = 10.0

# τ grid: r_ij = d_amb/d_geo < τ  →  shortcut
# 0.5 = geodesic at least 2× longer; 0.67 ≈ 1.5×; 0.8 = modest shortcut
const TAU_GRID = [0.3, 0.5, 0.67, 0.8]

# κ sweep for best-F1 search
const THRESHOLD_GRID = -1.0:0.05:1.0

# Full experimental grid
_n_grid     = !isempty(get(ENV, "N_OVERRIDE", "")) ? [parse(Int, ENV["N_OVERRIDE"])] :
               [200, 500, 1000, 2000]
_k_grid     = !isempty(get(ENV, "K_OVERRIDE", "")) ? [parse(Int, ENV["K_OVERRIDE"])] :
               [5, 10, 15, 20]
_noise_grid = [0.0, 0.1, 0.5]
_variants   = ["standard", "orcml"]

if get(ENV, "SMOKE", "") == "1"
    _n_grid     = [200]
    _k_grid     = [10]
    _noise_grid = [0.0]
    _variants   = ["standard"]
    global TAU_GRID_SMOKE = [0.5]   # single τ for smoke test
end

const N_GRID       = _n_grid
const K_GRID       = _k_grid
const NOISE_GRID   = _noise_grid
const ORC_VARIANTS = _variants
const TAU_GRID_RUN = get(ENV, "SMOKE", "") == "1" ? [0.5] : TAU_GRID

const SEED = 42

# ==============================================================================
# Helpers
# ==============================================================================

"""
    precompute_ratios(graph, data, params)

Return a Dict (i,j) → r_ij = d_ambient / d_geodesic for every graph edge.
Computed once per (graph, data) so τ sweeps are free.
"""
function precompute_ratios(graph, data::AbstractMatrix, params::NamedTuple)
    ratios = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:length(graph)
        for j in graph[i]
            d_amb = norm(data[:, i] - data[:, j])
            d_geo = exact_swiss_roll_geodesic(params, i, j)
            ratios[(i, j)] = d_geo > 1e-12 ? d_amb / d_geo : 1.0
        end
    end
    return ratios
end

"""
    label_shortcuts(ratios, tau) → Dict{Tuple{Int,Int}, Bool}

Apply threshold τ to pre-computed ratios to get ground-truth labels.
"""
function label_shortcuts(ratios::Dict, tau::Float64)
    Dict(edge => (r < tau) for (edge, r) in ratios)
end

"""
    evaluate_at_threshold(curvatures, labels, κ_threshold)

Predict shortcut when κ < κ_threshold. Returns (tp, fp, tn, fn, precision, recall, f1).
"""
function evaluate_at_threshold(curvatures::Dict, labels::Dict, κ_threshold::Float64)
    tp = fp = tn = fn = 0
    for (edge, result) in curvatures
        haskey(labels, edge) || continue
        is_sc   = labels[edge]
        pred_sc = result.curvature < κ_threshold
        if pred_sc && is_sc;       tp += 1
        elseif pred_sc && !is_sc;  fp += 1
        elseif !pred_sc && !is_sc; tn += 1
        else;                      fn += 1
        end
    end
    precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    f1        = (precision + recall) > 0 ? 2precision * recall / (precision + recall) : 0.0
    return (tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1)
end

"""
    best_f1_threshold(curvatures, labels) → (κ, precision, recall, f1)

Sweep κ over THRESHOLD_GRID and return the configuration achieving the highest F1.
"""
function best_f1_threshold(curvatures, labels)
    best = (κ=0.0, precision=0.0, recall=0.0, f1=-1.0)
    for κ in THRESHOLD_GRID
        m = evaluate_at_threshold(curvatures, labels, Float64(κ))
        if m.f1 > best.f1
            best = (κ=Float64(κ), precision=m.precision, recall=m.recall, f1=m.f1)
        end
    end
    return best
end

"""
    curvature_stats_by_label(curvatures, labels)

Mean κ split by ground-truth shortcut / non-shortcut label.
"""
function curvature_stats_by_label(curvatures, labels)
    sc_κ  = Float64[]
    nsc_κ = Float64[]
    for (edge, result) in curvatures
        haskey(labels, edge) || continue
        labels[edge] ? push!(sc_κ, result.curvature) :
                       push!(nsc_κ, result.curvature)
    end
    (mean_shortcut     = isempty(sc_κ)  ? NaN : mean(sc_κ),
     mean_non_shortcut = isempty(nsc_κ) ? NaN : mean(nsc_κ),
     n_shortcut        = length(sc_κ),
     n_non_shortcut    = length(nsc_κ))
end

# ==============================================================================
# Main loop
# ==============================================================================

println("=" ^ 80)
println("ORC Shortcut Detection — Swiss Roll Experiment")
println("=" ^ 80)
println("Threads      : ", Threads.nthreads())
println("n grid       : ", N_GRID)
println("k grid       : ", K_GRID)
println("noise grid   : ", NOISE_GRID)
println("ORC variants : ", ORC_VARIANTS)
println("τ grid       : ", TAU_GRID_RUN)
println("=" ^ 80)
println()

results = []

# Outer loop: one ORC computation per (n, k, noise, variant)
# Inner loop: re-label with each τ — free since curvatures already computed
total_orc = length(N_GRID) * length(K_GRID) * length(NOISE_GRID) * length(ORC_VARIANTS)

let orc_idx = 0
for n in N_GRID, k in K_GRID, noise_std in NOISE_GRID, variant in ORC_VARIANTS
    orc_idx += 1
    @printf "\n[%d/%d]  n=%-5d  k=%-3d  noise=%.2f  variant=%s\n" orc_idx total_orc n k noise_std variant

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = MersenneTwister(SEED)
    data_clean, params = generate_swiss_roll(
        n; rng=rng, t_min=T_MIN, t_range=T_RANGE, h_scale=H_SCALE
    )
    data = noise_std > 0 ? data_clean .+ noise_std .* randn(rng, size(data_clean)) :
                           data_clean

    # ── kNN graph ─────────────────────────────────────────────────────────────
    directed = (variant == "standard")
    index    = build_index(BruteForceIndex, data)
    graph    = build_knn_graph(index, data; k=k, directed=directed)
    n_edges  = sum(length(graph[i]) for i in 1:n)

    # ── Pre-compute r_ij ratios (τ-independent) ────────────────────────────────
    ratios = precompute_ratios(graph, data, params)

    # ── Compute ORC (once) ────────────────────────────────────────────────────
    orc_kwargs = variant == "standard" ?
        (exclude_edge_endpoints=false, cost_metric=:euclidean,            denominator_metric=:euclidean) :
        (exclude_edge_endpoints=true,  cost_metric=:geodesic_normalized,  denominator_metric=:normalized)

    t_orc = @elapsed begin
        curvatures = compute_all_curvatures(
            graph, data;
            orc_kwargs...,
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

        @printf "  τ=%.2f  shortcuts=%d/%d (%.1f%%)  κ<0: pre=%.3f rec=%.3f F1=%.3f  best_F1=%.3f@κ<%.2f\n" tau n_sc n_edges 100frac_sc m0.precision m0.recall m0.f1 best.f1 best.κ

        push!(results, (
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
            orc_time_s               = t_orc,
        ))
    end
end
end  # let orc_idx

# ==============================================================================
# Write outputs
# ==============================================================================

mkpath(RESULTS_DIR)
ts = Dates.format(now(), "yyyymmdd_HHMMSS")

# ── 1. Raw tidy CSV ────────────────────────────────────────────────────────────
raw_file = joinpath(RESULTS_DIR, "orc_swiss_roll_$(ts)_raw.csv")
open(raw_file, "w") do io
    println(io, "n,k,noise_std,variant,tau,n_edges,n_shortcuts,frac_shortcuts," *
                "precision_0,recall_0,f1_0,tp_0,fp_0,tn_0,fn_0," *
                "best_threshold,precision_best,recall_best,f1_best," *
                "mean_kappa_shortcuts,mean_kappa_non_shortcuts," *
                "n_shortcut_edges,n_non_shortcut_edges,orc_time_s")
    for r in results
        println(io, join([
            r.n, r.k, r.noise_std, r.variant, r.tau,
            r.n_edges, r.n_shortcuts, @sprintf("%.6f", r.frac_shortcuts),
            @sprintf("%.6f", r.precision_0), @sprintf("%.6f", r.recall_0), @sprintf("%.6f", r.f1_0),
            r.tp_0, r.fp_0, r.tn_0, r.fn_0,
            @sprintf("%.4f", r.best_threshold),
            @sprintf("%.6f", r.precision_best), @sprintf("%.6f", r.recall_best), @sprintf("%.6f", r.f1_best),
            @sprintf("%.6f", r.mean_kappa_shortcuts), @sprintf("%.6f", r.mean_kappa_non_shortcuts),
            r.n_shortcut_edges, r.n_non_shortcut_edges,
            @sprintf("%.3f", r.orc_time_s),
        ], ","))
    end
end

# ── 2 & 3. Pivoted CSVs ───────────────────────────────────────────────────────
#
# One section per (variant, noise, τ) combination; rows = n, columns = k.
# pgfplotstable can read a single section directly when pointed at the right
# line range; sections are separated by a blank line and a comment header.
# Each section has its own column header row so it can be read standalone.
#
# Usage in thesis (example for τ=0.5, standard, no noise):
#   \pgfplotstableread[col sep=comma, skip first n=2]{...pivot_f1.csv}\mytable
# (skip the comment and header of the prior section; or split into per-τ files
#  in post-processing once you know which slice you want for the main text).

function write_pivot(filename, results, value_fn, variants, noises, taus, n_grid, k_grid)
    open(filename, "w") do io
        first_section = true
        for variant in variants, noise in noises, tau in taus
            subset = filter(r -> r.variant == variant &&
                                 r.noise_std == noise &&
                                 r.tau == tau, results)
            isempty(subset) && continue

            first_section || println(io)   # blank line between sections
            first_section = false

            println(io, "# variant=$(variant)  noise=$(noise)  tau=$(tau)")
            println(io, join(["n"; string.(k_grid)], ","))

            for n in n_grid
                row = String[string(n)]
                for k in k_grid
                    m = filter(r -> r.n == n && r.k == k, subset)
                    push!(row, isempty(m) ? "NA" : @sprintf("%.4f", value_fn(m[1])))
                end
                println(io, join(row, ","))
            end
        end
    end
end

pivot_f1_file   = joinpath(RESULTS_DIR, "orc_swiss_roll_$(ts)_pivot_f1.csv")
pivot_best_file = joinpath(RESULTS_DIR, "orc_swiss_roll_$(ts)_pivot_best.csv")

write_pivot(pivot_f1_file,   results, r -> r.f1_0,    ORC_VARIANTS, NOISE_GRID, TAU_GRID_RUN, N_GRID, K_GRID)
write_pivot(pivot_best_file, results, r -> r.f1_best, ORC_VARIANTS, NOISE_GRID, TAU_GRID_RUN, N_GRID, K_GRID)

# ==============================================================================
# Console summary
# ==============================================================================

println()
println("=" ^ 80)
println("Output files:")
@printf "  Raw          : %s\n" raw_file
@printf "  Pivot F1@κ=0 : %s\n" pivot_f1_file
@printf "  Pivot bestF1 : %s\n" pivot_best_file
println("=" ^ 80)

println()
println("Summary (κ<0 threshold, no noise, τ=0.5):")
println(rpad("variant", 10), rpad("n", 7), rpad("k", 5),
        rpad("shortcuts%", 12), rpad("pre", 8), rpad("rec", 8), rpad("F1", 8), "time(s)")
println("─" ^ 68)
for r in results
    (r.noise_std == 0.0 && r.tau == 0.5) || continue
    @printf "%-10s %-7d %-5d %-12.1f %-8.3f %-8.3f %-8.3f %.2f\n" r.variant r.n r.k (100*r.frac_shortcuts) r.precision_0 r.recall_0 r.f1_0 r.orc_time_s
end
