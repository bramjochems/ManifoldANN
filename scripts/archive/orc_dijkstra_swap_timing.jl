#!/usr/bin/env julia
# Focused wall-clock comparison after the FW → per-source Dijkstra swap in
# `compute_shortest_paths`. Same dataset (swiss_roll N=1000, k=15, seed=42)
# as `orc_profile.jl`, so numbers are directly comparable to the recorded
# Clp baselines in `scripts/profile_results/`:
#
#   FW baseline (from profile_results/orc_orcmanl_clp.txt):
#     wall (median of 3): 4.559 s   per-edge: 0.265 ms   alloc: 2295.2 MB
#   FW baseline (orc_standard_clp.txt):
#     wall (median of 3): 1.075 s   per-edge: 0.072 ms   alloc: 1558.0 MB
#
# Run:
#   julia --project=. -t 1 scripts/orc_dijkstra_swap_timing.jl
#   julia --project=. -t 8 scripts/orc_dijkstra_swap_timing.jl

using Random, Printf, LinearAlgebra, Statistics
using ManifoldANN
const MA = ManifoldANN

include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic",
                 "swiss_roll_utils.jl"))

const N    = 1_000
const K    = 15
const SEED = 42

println("Threads.nthreads() = ", Threads.nthreads(),
        "   BLAS = ", BLAS.get_num_threads())
println("Config: swiss_roll n=$N k=$K\n")

Random.seed!(SEED)
data, _ = generate_swiss_roll(N; rng = MersenneTwister(SEED))
index = MA.build_index(MA.BruteForceIndex, data)
graph_dir   = MA.build_knn_graph(index, data; k = K, directed = true)
graph_undir = MA.build_knn_graph(index, data; k = K, directed = false)

# Warmup (compile)
_ = MA.compute_all_curvatures(graph_undir, data;
    variant = MA.ORCManL(), solver = MA.ClpSolver(),
    fallback_solver = MA.ClpSolver(),
    use_threading = false, verbose = false)
_ = MA.compute_all_curvatures(graph_dir, data;
    variant = MA.StandardORC(), solver = MA.ClpSolver(),
    fallback_solver = MA.ClpSolver(),
    use_threading = false, verbose = false)

function bench(label, graph, variant; threaded)
    times = Float64[]
    local results
    local alloc_mb
    for _ in 1:3
        GC.gc()
        t0 = time_ns(); a0 = Base.gc_bytes()
        results = MA.compute_all_curvatures(graph, data;
            variant = variant, solver = MA.ClpSolver(),
            fallback_solver = MA.ClpSolver(),
            use_threading = threaded, verbose = false)
        t1 = time_ns(); a1 = Base.gc_bytes()
        push!(times, (t1 - t0) / 1e9)
        alloc_mb = (a1 - a0) / (1024^2)
    end
    elapsed = median(times)
    n_results = length(results)
    @printf("%-32s  wall (median/3): %6.3f s   per-edge: %5.3f ms   alloc: %7.1f MB\n",
            label, elapsed, elapsed * 1e3 / n_results, alloc_mb)
    return elapsed
end

println("--- single-threaded ---")
bench("StandardORC + Clp",  graph_dir,   MA.StandardORC(); threaded = false)
bench("ORCManL + Clp",      graph_undir, MA.ORCManL();     threaded = false)

println("\n--- multi-threaded ---")
bench("StandardORC + Clp",  graph_dir,   MA.StandardORC(); threaded = true)
bench("ORCManL + Clp",      graph_undir, MA.ORCManL();     threaded = true)
