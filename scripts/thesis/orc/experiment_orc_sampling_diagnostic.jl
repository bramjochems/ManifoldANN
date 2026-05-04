"""
ORC as a Sampling-Quality Diagnostic — Experiment

Tests whether regional Ollivier-Ricci curvature statistics can identify
undersampled areas of a manifold, sidestepping the per-edge classification
problem that caused failures in the Swiss roll experiment.

Hypothesis
----------
When a region of the manifold is undersampled, kNN edges from that region
are more likely to be chord-like shortcuts (because denser sampling would
provide intermediate stepping-stones).  This raises the local shortcut
fraction, which in turn depresses the mean ORC in that neighbourhood.
Conversely, oversampled regions produce tightly packed graphs with mostly
manifold-following edges and therefore higher mean ORC.

The experiment deliberately creates non-uniform sampling by partitioning the
Swiss roll's t-parameter into zones and controlling their relative density.
We then check whether the mean per-vertex ORC tracks the local sampling density.

Experimental design
-------------------
1. Generate a Swiss roll with non-uniform sampling: one "sparse zone" and the
   rest densely sampled.  The sparse zone is defined as an angular band
   [t_sparse_min, t_sparse_max].
2. Compute the kNN graph and ORC on all edges.
3. For each vertex, compute its mean κ over incident edges (vertex ORC).
4. Record the true local sampling density for each vertex (number of points
   in a ball of fixed radius divided by the ball's geodesic area).
5. Compare vertex ORC vs. local density — a good diagnostic shows negative
   correlation: undersampled ↔ lower mean ORC.

Three sparsity levels × two kNN sizes × two ORC variants are evaluated.
Output is a CSV with per-vertex data suitable for scatter/heatmap plots.

Results are written incrementally after each ORC computation, so a crash
mid-run loses at most one config.  Restarting skips already-computed configs.

Usage:
    julia --project=. -t auto scripts/experiment_orc_sampling_diagnostic.jl

Environment overrides:
    SMOKE=1         minimal run (n=500, k=10, standard only, one sparsity level)
    RESUME_DIR=path resume into a previous run's directory (skips
                    already-computed configs)
"""

using ManifoldANN
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates

include(joinpath(@__DIR__, "orc_helpers.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const RESULTS_DIR = joinpath(
    @__DIR__, "..", "..", "..", "..", "..", "docs", "thesis", "results", "orc_results"
)

# Swiss roll parameters (match the main Swiss roll experiment)
const T_MIN   = 1.5π
const T_RANGE = 3π
const H_SCALE = 10.0

# Sparse zone: defined as a fraction of the t-range
# The sparse zone covers [T_MIN + T_RANGE*zone_start, T_MIN + T_RANGE*zone_end]
const SPARSE_ZONE = (start=0.35, stop=0.65)   # middle third of the roll

# Total points; the sparse zone will receive only `sparsity_fraction` of its
# "fair share" of points (i.e. if the zone covers 30% of the range and
# sparsity=0.1 then it gets 3% of all points instead of 30%).
const N_TOTAL = 1000

# Sparsity levels: fraction of expected density in the sparse zone
# 1.0 = uniform (baseline), 0.3 = moderately sparse, 0.05 = very sparse
const SPARSITY_GRID = get(ENV, "SMOKE", "") == "1" ? [1.0, 0.1] :
                                                     [1.0, 0.3, 0.1, 0.05]

# kNN and ORC settings
const K_GRID     = get(ENV, "SMOKE", "") == "1" ? [10] : [5, 10, 15]
const ORC_VARIANTS = get(ENV, "SMOKE", "") == "1" ? ["standard"] : ["standard", "orcml"]

# Radius for local density estimation (in geodesic units)
const DENSITY_RADIUS = 1.5

const SEED = 42

# ==============================================================================
# Sampling with non-uniform density
# ==============================================================================

"""
    generate_nonuniform_swiss_roll(n_total, sparsity_fraction;
                                   rng, t_min, t_range, h_scale,
                                   zone_start, zone_end)

Generate a Swiss roll with non-uniform t-density.  The sparse zone
[t_min + t_range*zone_start, t_min + t_range*zone_end] receives
`sparsity_fraction` times the density of the rest.

Returns (data, params, zone_mask) where zone_mask[i]=true iff point i is in
the sparse zone.
"""
function generate_nonuniform_swiss_roll(n_total::Int, sparsity_fraction::Float64;
                                         rng=Random.GLOBAL_RNG,
                                         t_min::Float64=T_MIN,
                                         t_range::Float64=T_RANGE,
                                         h_scale::Float64=H_SCALE,
                                         zone_start::Float64=SPARSE_ZONE.start,
                                         zone_end::Float64=SPARSE_ZONE.stop)
    zone_width  = zone_end - zone_start         # fraction of t_range
    dense_width = 1.0 - zone_width

    # Expected number of points under uniform sampling:
    #   n_zone_uniform = n_total * zone_width
    # Under sparsity:
    #   n_zone_actual  = n_total * zone_width * sparsity_fraction
    # (the remainder goes to the dense part)
    # Solve for allocations:
    #   n_zone + n_dense = n_total
    #   n_zone = zone_width * sparsity_fraction * n_total /
    #            (zone_width*sparsity_fraction + dense_width)
    denom   = zone_width * sparsity_fraction + dense_width
    n_zone  = round(Int, n_total * zone_width  * sparsity_fraction / denom)
    n_dense = n_total - n_zone

    # Dense part: t uniform over the two sub-intervals outside the sparse zone
    t_all = Float64[]
    # sub-interval 1: [0, zone_start) of t_range
    n_dense1 = round(Int, n_dense * (zone_start / dense_width))
    n_dense2 = n_dense - n_dense1
    append!(t_all, t_min .+ (t_range * zone_start)  .* rand(rng, n_dense1))
    append!(t_all, t_min .+ t_range * zone_end .+
                   (t_range * (1 - zone_end)) .* rand(rng, n_dense2))
    # Sparse zone
    append!(t_all, t_min .+ t_range * zone_start .+
                   (t_range * zone_width) .* rand(rng, n_zone))

    n_actual = length(t_all)
    h_all    = h_scale .* rand(rng, n_actual)

    data = vcat(
        (t_all .* cos.(t_all))',
        h_all',
        (t_all .* sin.(t_all))'
    )

    params = (t=t_all, h=h_all)

    # Zone mask: true = point in sparse zone
    t_zone_lo = t_min + t_range * zone_start
    t_zone_hi = t_min + t_range * zone_end
    zone_mask = [t_zone_lo <= t <= t_zone_hi for t in t_all]

    return data, params, zone_mask, n_actual
end

# ==============================================================================
# Local density estimation
# ==============================================================================

"""
    estimate_local_density(params, i, radius)

Estimate local sampling density for point i as the number of points within
geodesic radius `radius` divided by the area of the geodesic disk (π·r²).
"""
function estimate_local_density(params::NamedTuple, i::Int, radius::Float64)
    n = length(params.t)
    count = 0
    for j in 1:n
        j == i && continue
        d = exact_swiss_roll_geodesic(params, i, j)
        count += (d <= radius)
    end
    area = π * radius^2
    return count / area
end

# ==============================================================================
# Per-vertex ORC (mean curvature over incident edges)
# ==============================================================================

"""
    vertex_mean_orc(graph, curvatures, n)

Compute per-vertex mean ORC as the average κ over all edges incident to each
vertex (both as source and target in the directed graph).
"""
function vertex_mean_orc(graph, curvatures::Dict, n::Int)
    sums   = zeros(n)
    counts = zeros(Int, n)
    for (edge, result) in curvatures
        i, j = edge
        κ = result.curvature
        sums[i]   += κ;  counts[i]   += 1
        sums[j]   += κ;  counts[j]   += 1
    end
    [counts[i] > 0 ? sums[i] / counts[i] : NaN for i in 1:n]
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
    joinpath(RESULTS_DIR, "sampling_diag_$(ts)")
end
mkpath(run_dir)

# CSV file paths
summary_file = joinpath(run_dir, "summary.csv")
vertex_file  = joinpath(run_dir, "vertex.csv")

# Initialize CSV headers (no-op if files already exist)
const SUMMARY_HEADER = "sparsity,k,variant,n_actual,n_zone,n_dense," *
    "mean_kappa_zone,mean_kappa_dense,delta_kappa," *
    "pearson_density_orc,n_corr_sample,orc_time_s"
init_csv(summary_file, SUMMARY_HEADER)

const VERTEX_HEADER = "sparsity,k,variant,vertex_idx,t_param,h_param," *
    "in_sparse_zone,mean_kappa,local_density"
init_csv(vertex_file, VERTEX_HEADER)

# Load already-completed configs for resume
completed_configs = load_completed_keys(summary_file, [:sparsity, :k, :variant])

# ==============================================================================
# Main loop
# ==============================================================================

println("=" ^ 80)
println("ORC Sampling Diagnostic — Swiss Roll Non-Uniform Sampling Experiment")
println("=" ^ 80)
println("Threads        : ", Threads.nthreads())
println("n total        : ", N_TOTAL)
println("k grid         : ", K_GRID)
println("ORC variants   : ", ORC_VARIANTS)
println("Sparsity grid  : ", SPARSITY_GRID)
println("Sparse zone    : t-fraction [$(SPARSE_ZONE.start), $(SPARSE_ZONE.stop)]")
println("Density radius : ", DENSITY_RADIUS)
println("Output dir     : ", run_dir)
println("Resuming       : ", !isempty(completed_configs), " (", length(completed_configs), " configs done)")
println("=" ^ 80)
println()

vertex_results   = []   # per-vertex statistics
summary_results  = []   # per-configuration correlation statistics

total_runs = length(SPARSITY_GRID) * length(K_GRID) * length(ORC_VARIANTS)
run_idx = 0

for sparsity in SPARSITY_GRID, k in K_GRID, variant in ORC_VARIANTS
    global run_idx += 1

    # Skip already-computed configs (resume support)
    config_key = (string(sparsity), string(k), variant)
    if config_key in completed_configs
        @printf "[%d/%d]  sparsity=%.2f  k=%-3d  variant=%s  [SKIP — already computed]\n" run_idx total_runs sparsity k variant
        continue
    end

    @printf "\n[%d/%d]  sparsity=%.2f  k=%-3d  variant=%s\n" run_idx total_runs sparsity k variant

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = MersenneTwister(SEED)
    data, params, zone_mask, n_actual = generate_nonuniform_swiss_roll(
        N_TOTAL, sparsity; rng=rng
    )
    @printf "  n_actual=%d  n_zone=%d (%.1f%%)\n" n_actual sum(zone_mask) (100*mean(zone_mask))

    # ── kNN graph ─────────────────────────────────────────────────────────────
    directed = (variant == "standard")
    index    = build_index(BruteForceIndex, data)
    graph    = build_knn_graph(index, data; k=k, directed=directed)
    n_edges  = sum(length(graph[i]) for i in 1:n_actual)

    # ── ORC ───────────────────────────────────────────────────────────────────
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

    # ── Per-vertex statistics ─────────────────────────────────────────────────
    mean_κ_vertex = vertex_mean_orc(graph, curvatures, n_actual)

    # Compute local density (subsampled for speed: use all points for small n)
    density_sample = n_actual <= 300 ? (1:n_actual) :
                                       sort(randperm(rng, n_actual)[1:300])
    densities = fill(NaN, n_actual)
    for i in density_sample
        densities[i] = estimate_local_density(params, i, DENSITY_RADIUS)
    end

    # Correlation between density and vertex ORC (on sampled subset)
    valid = [i for i in density_sample if !isnan(mean_κ_vertex[i]) && !isnan(densities[i])]
    corr_val = NaN
    if length(valid) >= 3
        κ_vals  = mean_κ_vertex[valid]
        d_vals  = densities[valid]
        κ_c  = κ_vals .- mean(κ_vals)
        d_c  = d_vals .- mean(d_vals)
        denom = sqrt(sum(κ_c.^2) * sum(d_c.^2))
        corr_val = denom > 1e-14 ? sum(κ_c .* d_c) / denom : 0.0
    end

    @printf "  Pearson(density, vertex_ORC) = %.4f  (n_sampled=%d)\n" corr_val length(valid)

    # Store per-vertex rows
    for i in 1:n_actual
        push!(vertex_results, (
            sparsity    = sparsity,
            k           = k,
            variant     = variant,
            vertex_idx  = i,
            t_param     = params.t[i],
            h_param     = params.h[i],
            in_sparse_zone = zone_mask[i],
            mean_kappa  = mean_κ_vertex[i],
            local_density = densities[i],
        ))
    end

    # Store per-configuration summary
    zone_κ   = [mean_κ_vertex[i] for i in 1:n_actual if  zone_mask[i] && !isnan(mean_κ_vertex[i])]
    dense_κ  = [mean_κ_vertex[i] for i in 1:n_actual if !zone_mask[i] && !isnan(mean_κ_vertex[i])]

    push!(summary_results, (
        sparsity         = sparsity,
        k                = k,
        variant          = variant,
        n_actual         = n_actual,
        n_zone           = sum(zone_mask),
        n_dense          = sum(.!zone_mask),
        mean_kappa_zone  = isempty(zone_κ)  ? NaN : mean(zone_κ),
        mean_kappa_dense = isempty(dense_κ) ? NaN : mean(dense_κ),
        delta_kappa      = (isempty(zone_κ) || isempty(dense_κ)) ? NaN :
                           mean(dense_κ) - mean(zone_κ),
        pearson_density_orc = corr_val,
        n_corr_sample    = length(valid),
        orc_time_s       = t_orc,
    ))

    # ── Flush results for this config to CSV ──────────────────────────────────
    # Summary CSV: one row per config
    r = summary_results[end]
    append_csv_rows(summary_file, [join([
        r.sparsity, r.k, r.variant,
        r.n_actual, r.n_zone, r.n_dense,
        @sprintf("%.6f", r.mean_kappa_zone),
        @sprintf("%.6f", r.mean_kappa_dense),
        isnan(r.delta_kappa) ? "NA" : @sprintf("%.6f", r.delta_kappa),
        isnan(r.pearson_density_orc) ? "NA" : @sprintf("%.6f", r.pearson_density_orc),
        r.n_corr_sample,
        @sprintf("%.3f", r.orc_time_s),
    ], ",")])

    # Vertex CSV: flush all accumulated vertex_results, then clear
    vtx_rows = String[]
    for r in vertex_results
        push!(vtx_rows, join([
            r.sparsity, r.k, r.variant, r.vertex_idx,
            @sprintf("%.6f", r.t_param),
            @sprintf("%.6f", r.h_param),
            r.in_sparse_zone ? "1" : "0",
            isnan(r.mean_kappa)      ? "NA" : @sprintf("%.6f", r.mean_kappa),
            isnan(r.local_density)   ? "NA" : @sprintf("%.6f", r.local_density),
        ], ","))
    end
    append_csv_rows(vertex_file, vtx_rows)
    empty!(vertex_results)
end

# ==============================================================================
# Console summary
# ==============================================================================

println()
println("=" ^ 80)
println("Output directory: $run_dir")
@printf "  Summary      : %s\n" summary_file
@printf "  Per-vertex   : %s\n" vertex_file
println("=" ^ 80)

println()
println("Summary table (standard ORC, k=$(K_GRID[1])):")
println(rpad("sparsity", 12), rpad("zone κ̄", 12), rpad("dense κ̄", 12),
        rpad("Δκ̄", 10), "Pearson(dens,ORC)")
println("─" ^ 60)
for r in summary_results
    r.variant == "standard" && r.k == K_GRID[1] || continue
    @printf("%-12.2f %-12.4f %-12.4f %-10.4f %.4f\n",
        r.sparsity, r.mean_kappa_zone, r.mean_kappa_dense, r.delta_kappa, r.pearson_density_orc)
end
