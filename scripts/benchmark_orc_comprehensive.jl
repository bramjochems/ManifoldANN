"""
Comprehensive ORC benchmarking suite:
1. Compare different Julia solvers on the same graph
2. Compare two ORC configurations (original vs orcml)
3. Benchmark across different graph sizes
4. Export results for comparison with Python implementations

Usage:
    julia --project=. -t 8 scripts/benchmark_orc_comprehensive.jl
"""

using ManifoldANN
using LinearAlgebra
using DelimitedFiles
using Statistics
using Random
using Printf

# Configuration
const SIZES = [100, 500, 1000]  # Graph sizes to test
const K = 15  # Number of neighbors
const DIM = 50  # Data dimensionality
const SEED = 42

# All solvers to benchmark
const SOLVERS = [
    ("Hungarian", ManifoldANN.HungarianSolver(), ManifoldANN.NetworkSimplexSolver()),
    ("NetworkSimplex", ManifoldANN.NetworkSimplexSolver(), ManifoldANN.NetworkSimplexSolver()),
    ("LPReference", ManifoldANN.LPReferenceSolver(), ManifoldANN.NetworkSimplexSolver()),
    ("Sinkhorn", ManifoldANN.SinkhornSolver(), ManifoldANN.NetworkSimplexSolver()),
]

# ORC configurations to test
struct ORCConfig
    name::String
    exclude_edge_endpoints::Bool
    cost_metric::Symbol
    denominator_metric::Symbol
    directed::Bool
end

const CONFIGS = [
    ORCConfig(
        "original",
        false,  # Include edge endpoints
        :euclidean,
        :euclidean,
        true  # Directed graph
    ),
    ORCConfig(
        "orcml",
        true,  # Exclude edge endpoints
        :geodesic_normalized,
        :normalized,
        false  # Undirected graph
    ),
]

println("="^80)
println("ORC Comprehensive Benchmark Suite")
println("="^80)
println("Sizes: ", SIZES)
println("k: ", K)
println("Dimension: ", DIM)
println("Threads: ", Threads.nthreads())
println("Solvers: ", [name for (name, _, _) in SOLVERS])
println("Configs: ", [config.name for config in CONFIGS])
println()

# Results storage
results = []

for n in SIZES
    println("\n" * "="^80)
    println("Graph Size: n=$n")
    println("="^80)

    # Generate random data
    Random.seed!(SEED)
    data = randn(DIM, n)

    # Build index
    index = build_index(BruteForceIndex, data)

    for config in CONFIGS
        println("\n" * "-"^80)
        println("Configuration: $(config.name)")
        println("  directed=$(config.directed), exclude_endpoints=$(config.exclude_edge_endpoints)")
        println("  cost=$(config.cost_metric), denom=$(config.denominator_metric)")
        println("-"^80)

        # Build graph with appropriate direction
        graph = build_knn_graph(index, data; k=K, directed=config.directed)

        n_edges = sum(length(neighbors) for neighbors in graph)
        if !config.directed
            # Undirected: count each edge once (canonical form)
            n_unique_edges = 0
            for i in 1:n
                for j in graph[i]
                    if i < j
                        n_unique_edges += 1
                    end
                end
            end
        else
            n_unique_edges = n_edges
        end

        println("\nGraph stats:")
        println("  Total edges: $n_edges")
        println("  Unique edges: $n_unique_edges")
        println("  Avg degree: $(n_edges / n)")

        for (solver_name, primary_solver, fallback_solver) in SOLVERS
            print("\n  Testing $solver_name solver... ")
            flush(stdout)

            # Time the computation
            start_time = time()

            try
                curvatures = ManifoldANN.compute_all_curvatures(
                    graph, data;
                    exclude_edge_endpoints=config.exclude_edge_endpoints,
                    cost_metric=config.cost_metric,
                    denominator_metric=config.denominator_metric,
                    solver=primary_solver,
                    fallback_solver=fallback_solver,
                    use_threading=true,
                    verbose=false
                )

                elapsed = time() - start_time

                # Compute statistics
                curv_values = [result.curvature for result in values(curvatures)]
                n_computed = length(curv_values)
                n_nan = sum(isnan.(curv_values))

                if n_nan > 0
                    println("⚠ FAILED: $n_nan NaN values")
                    push!(results, (
                        n=n,
                        config=config.name,
                        solver=solver_name,
                        time=elapsed,
                        n_edges=n_computed,
                        n_nan=n_nan,
                        time_per_edge=NaN,
                        mean_curv=NaN,
                        std_curv=NaN
                    ))
                else
                    mean_curv = mean(curv_values)
                    std_curv = std(curv_values)
                    time_per_edge = elapsed / n_computed * 1000  # ms per edge

                    @printf("✓ %.3fs (%.3fms/edge, μ=%.3f, σ=%.3f)\n",
                            elapsed, time_per_edge, mean_curv, std_curv)

                    push!(results, (
                        n=n,
                        config=config.name,
                        solver=solver_name,
                        time=elapsed,
                        n_edges=n_computed,
                        n_nan=0,
                        time_per_edge=time_per_edge,
                        mean_curv=mean_curv,
                        std_curv=std_curv
                    ))
                end

            catch e
                println("✗ ERROR: $e")
                push!(results, (
                    n=n,
                    config=config.name,
                    solver=solver_name,
                    time=NaN,
                    n_edges=0,
                    n_nan=0,
                    time_per_edge=NaN,
                    mean_curv=NaN,
                    std_curv=NaN
                ))
            end
        end
    end
end

# Save results
println("\n" * "="^80)
println("Saving Results")
println("="^80)

open("benchmark_results/orc_benchmark_julia.csv", "w") do io
    println(io, "n,config,solver,time_sec,n_edges,n_nan,ms_per_edge,mean_curv,std_curv")
    for r in results
        println(io, "$(r.n),$(r.config),$(r.solver),$(r.time),$(r.n_edges),$(r.n_nan),$(r.time_per_edge),$(r.mean_curv),$(r.std_curv)")
    end
end

println("Saved to benchmark_results/orc_benchmark_julia.csv")

# Print summary
println("\n" * "="^80)
println("Summary: Time per Edge (ms)")
println("="^80)
println()

for n in SIZES
    println("n=$n:")
    for config in CONFIGS
        println("  $(config.name):")
        for (solver_name, _, _) in SOLVERS
            matching = filter(r -> r.n == n && r.config == config.name && r.solver == solver_name, results)
            if !isempty(matching) && !isnan(matching[1].time_per_edge)
                @printf("    %-15s: %8.3f ms/edge\n", solver_name, matching[1].time_per_edge)
            end
        end
    end
    println()
end

println("="^80)
println("Benchmark Complete!")
println("="^80)
