#!/usr/bin/env julia
# ORC profiling: where is time + allocations spent in compute_all_curvatures?
#
# Two open ORC perf items in TODO_cleanup.md motivate this:
#   (1) `_create_precomputed_distance_fn` allocates 2 Dicts + a closure per
#       edge, inside the threaded loop. GC pressure scales edges × threads.
#   (2) Pipeline hardcodes Float64 even on Float32 input (~2x memory).
#
# Profiles four cases: {StandardORC, ORCManL} × {Sinkhorn, Hungarian}.
# Outputs go to scripts/profile_results/orc_<case>.txt (one file per case),
# plus a summary on stdout.
#
# Run: julia --project=. -t 8 scripts/orc_profile.jl

using Random, Printf, Profile, LinearAlgebra, Statistics, Logging
using ManifoldANN
const MA = ManifoldANN

# Swiss roll generator (kept in docs/examples/geodesic). Manifold-structured
# data is required: random Gaussian data has uniform neighborhoods that defeat
# Sinkhorn convergence (NaN curvatures) and route Hungarian neighborhoods
# through Tulip LP fallback (250+ GB allocs). Swiss roll gives realistic
# tangent structure with clean OT solutions for both solvers.
include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic",
                 "swiss_roll_utils.jl"))

# ---- Config ----------------------------------------------------------------
const N    = 1_000
const K    = 15
const SEED = 42
# Sinkhorn regularization: package guidance is reg ≈ 5-10% of mean(cost).
# Empirically validated on swiss roll: reg=0.1 → 0 NaN, fast convergence.
const SINKHORN_REG = 0.1
const OUTDIR = joinpath(@__DIR__, "profile_results")
mkpath(OUTDIR)

println("Threads.nthreads() = ", Threads.nthreads(),
        "   BLAS = ", BLAS.get_num_threads())
println("Config: swiss_roll n=$N k=$K  Sinkhorn reg=$SINKHORN_REG")
println("Output: $OUTDIR\n")

# ---- Data + graphs ---------------------------------------------------------
Random.seed!(SEED)
data, _ = generate_swiss_roll(N; rng=MersenneTwister(SEED))
index = MA.build_index(MA.BruteForceIndex, data)
graph_dir   = MA.build_knn_graph(index, data; k=K, directed=true)
graph_undir = MA.build_knn_graph(index, data; k=K, directed=false)
println("Graph (directed)   : $(sum(length(nb) for nb in graph_dir)) edges")
println("Graph (undirected) : $(sum(length(nb) for nb in graph_undir)) edges\n")

struct Case
    name::String
    short::String          # filename-safe
    variant::MA.AbstractORCConfig
    solver::MA.AbstractOTSolver
    graph::Any
end

cases = [
    Case("StandardORC + Sinkhorn",  "standard_sinkhorn",  MA.StandardORC(), MA.SinkhornSolver(reg=SINKHORN_REG),  graph_dir),
    Case("StandardORC + Hungarian", "standard_hungarian", MA.StandardORC(), MA.HungarianSolver(), graph_dir),
    Case("StandardORC + Clp",       "standard_clp",       MA.StandardORC(), MA.ClpSolver(),       graph_dir),
    Case("ORCManL + Sinkhorn",      "orcmanl_sinkhorn",   MA.ORCManL(),     MA.SinkhornSolver(reg=SINKHORN_REG),  graph_undir),
    Case("ORCManL + Hungarian",     "orcmanl_hungarian",  MA.ORCManL(),     MA.HungarianSolver(), graph_undir),
    Case("ORCManL + Clp",           "orcmanl_clp",        MA.ORCManL(),     MA.ClpSolver(),       graph_undir),
]

# ---- Warm-up (compile) -----------------------------------------------------
# Suppress Sinkhorn convergence/regularization warnings — they're advisory
# and not what we're profiling.
const _orig_logger = global_logger(NullLogger())

println("Warming up (compile pass on small graph)...")
let
    small_data, _ = generate_swiss_roll(200; rng=MersenneTwister(SEED+1))
    small_idx = MA.build_index(MA.BruteForceIndex, small_data)
    g_d = MA.build_knn_graph(small_idx, small_data; k=K, directed=true)
    g_u = MA.build_knn_graph(small_idx, small_data; k=K, directed=false)
    for c in cases
        g = c.variant isa MA.StandardORC ? g_d : g_u
        MA.compute_all_curvatures(g, small_data;
            variant=c.variant, solver=c.solver,
            fallback_solver=MA.ClpSolver(),
            use_threading=true, verbose=false)
    end
end
GC.gc()
println("Warmup done.\n")

# ---- Helpers ---------------------------------------------------------------
"""Pick the most informative frame from a stacktrace: prefer ManifoldANN, then
OptimalTransport / Hungarian / SparseArrays, then any non-stdlib package frame.
Skip profiler internals, threading task plumbing, generic Base allocators."""
function pick_site(stack)
    skip_pat = r"(gc-alloc-profiler|gc-stock\.c|gc\.c|/task\.jl|threadingconstructs|/array\.jl|/abstractarray\.jl|/dict\.jl|/loading\.jl|/client\.jl|/boot\.jl|/Base\.jl|/Profile\.jl|/reflection\.jl)"
    candidates = String[]
    for sf in stack
        f = string(sf.file)
        occursin(skip_pat, f) && continue
        push!(candidates, "$(sf.func) @ $(basename(f)):$(sf.line)")
    end
    isempty(candidates) && return "<all-internal>"
    # Prefer ManifoldANN frames; otherwise first non-skipped
    for c in candidates
        occursin("ManifoldANN", c) || occursin("/refinement/", c) && return c
    end
    return candidates[1]
end

function alloc_summary(io, allocs; sample_rate::Float64)
    if isempty(allocs.allocs)
        println(io, "  (no allocation samples captured)")
        return
    end
    by_site = Dict{String, Tuple{Int, Int}}()
    scale = 1.0 / sample_rate
    for a in allocs.allocs
        site = pick_site(a.stacktrace)
        cnt, bts = get(by_site, site, (0, 0))
        by_site[site] = (cnt + 1, bts + a.size)
    end
    sorted = sort(collect(by_site); by=x -> x[2][2], rev=true)
    total_bytes = sum(x[2][2] for x in sorted) * scale
    @printf(io, "  Estimated total alloc bytes (sampled, scaled): %.1f MB\n", total_bytes/1024^2)
    @printf(io, "  %-72s  %14s  %12s\n", "site", "est_bytes", "est_count")
    for (site, (cnt, bts)) in sorted[1:min(25, length(sorted))]
        @printf(io, "  %-72s  %14d  %12d\n", site, Int(round(bts*scale)), Int(round(cnt*scale)))
    end
end

# ---- Per-case run ----------------------------------------------------------
function run_case(c::Case)
    outfile = joinpath(OUTDIR, "orc_$(c.short).txt")
    println("→ $(c.name)  → $outfile")
    open(outfile, "w") do io
        println(io, "="^78)
        println(io, c.name)
        println(io, "swiss_roll n=$N k=$K threads=$(Threads.nthreads()) sinkhorn_reg=$SINKHORN_REG")
        println(io, "="^78)

        # Wall-clock + total alloc baseline (3 reps, take median wall)
        times = Float64[]
        local results
        local alloc_mb
        for rep in 1:3
            GC.gc()
            t0 = time_ns(); a0 = Base.gc_bytes()
            results = MA.compute_all_curvatures(c.graph, data;
                variant=c.variant, solver=c.solver,
                fallback_solver=MA.ClpSolver(),
                use_threading=true, verbose=false)
            t1 = time_ns(); a1 = Base.gc_bytes()
            push!(times, (t1-t0)/1e9)
            alloc_mb = (a1-a0)/(1024^2)
        end
        elapsed = median(times)
        n_results = length(results)
        n_nan = sum(isnan(r.curvature) for r in values(results))
        @printf(io, "\n  wall (median of 3): %.3f s   alloc: %.1f MB   results: %d (NaN: %d)\n",
                elapsed, alloc_mb, n_results, n_nan)
        @printf(io, "  per-edge: %.3f ms   alloc/edge: %.1f KB\n",
                elapsed*1e3/n_results, alloc_mb*1024/n_results)

        # CPU profile (no idle threads — single-threaded so the profile reflects
        # actual work, not 7 threads sitting on `poptask`)
        println(io, "\n--- CPU profile (single-threaded, flat top 30) ---")
        GC.gc()
        Profile.clear()
        Profile.init(n=10^7, delay=0.001)
        @profile MA.compute_all_curvatures(c.graph, data;
            variant=c.variant, solver=c.solver,
            fallback_solver=MA.ClpSolver(),
            use_threading=false, verbose=false)
        Profile.print(io; format=:flat, sortedby=:count, mincount=10, maxdepth=30, C=false)

        # Alloc profile (1% sample rate)
        println(io, "\n--- Alloc profile (single-threaded, 1% sample, top 25 sites) ---")
        GC.gc()
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=0.01 MA.compute_all_curvatures(c.graph, data;
            variant=c.variant, solver=c.solver,
            fallback_solver=MA.ClpSolver(),
            use_threading=false, verbose=false)
        alloc_summary(io, Profile.Allocs.fetch(); sample_rate=0.01)
    end

    # Echo just the headline numbers to stdout
    open(outfile, "r") do io
        for (i, line) in enumerate(eachline(io))
            (i > 8 && !startswith(line, "  wall") && !startswith(line, "  per-edge")) && break
            println("    ", line)
        end
    end
    println()
end

for c in cases
    run_case(c)
end

println("Done. Per-case output in $OUTDIR/orc_*.txt")
