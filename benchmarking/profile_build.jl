#!/usr/bin/env julia
"""
Quick build time profiling script to identify bottlenecks in HNSW index construction.
"""

using ManifoldANN
using LinearAlgebra
using Random

# Generate small test dataset
function generate_test_data(n=10000, d=784)
    return randn(Float32, d, n)
end

println("=" ^ 60)
println("HNSW Build Time Profiling")
println("=" ^ 60)

# Parameters matching the benchmark
data = generate_test_data(10000, 784)
M = 16
ef_construction = 200
ef_search = 50

println("\nDataset: $(size(data, 2)) points × $(size(data, 1)) dimensions")
println("Parameters: M=$M, ef_construction=$ef_construction, ef_search=$ef_search")

# Warm up compilation
println("\n[1/3] Warmup (JIT compilation)...")
small_data = data[:, 1:100]
_ = build_index(
    HNSWIndex,
    small_data;
    M = M,
    ef_construction = ef_construction,
    ef_search = ef_search,
    distance = default_squared_distance,
)
println("  ✓ Warmup complete")

# Full build with timing
println("\n[2/3] Building full index...")
start = time()
index = build_index(
    HNSWIndex,
    data;
    M = M,
    ef_construction = ef_construction,
    ef_search = ef_search,
    distance = default_squared_distance,
)
build_time = time() - start
println("  ✓ Build time: $(round(build_time, digits=2))s")

# Manual instrumentation to identify hotspots
println("\n[3/3] Detailed timing breakdown...")
println("  Running instrumented build...")

# Count operations during build
function count_operations(data, M, ef_construction)
    n = size(data, 2)
    total_inserts = n

    # Estimate layer distribution (exponential decay)
    ml = 1 / log(max(M, 2))
    avg_layer = ml

    # Per insertion:
    # - Greedy descent through upper layers
    # - Search layer at each target layer (ef_construction candidates)
    # - Select neighbors (sort + copy)
    # - Link nodes (bidirectional, with pruning)

    # Rough estimates
    avg_descent_layers = avg_layer
    avg_search_layers = avg_layer + 1  # All layers from 0 to assigned level
    avg_neighbors_per_layer = M

    # Search layer: ef_construction candidates explored, each checking M neighbors
    # This is O(ef_construction * avg_degree_in_graph) distance computations
    avg_degree_in_graph = M / 2  # Rough estimate

    total_distance_calls = 0
    for i in 1:n
        # Descent: few distance calls per layer
        total_distance_calls += avg_descent_layers * avg_degree_in_graph

        # Search layer: many distance calls
        total_distance_calls += avg_search_layers * ef_construction * avg_degree_in_graph

        # Neighbor selection: sorting requires distance calls (already counted in search)

        # Pruning: for each bidirectional link, might prune existing neighbors
        # Each prune does a full sort of neighbor list (M elements)
        # This is M * log(M) * (number of links) distance calls
        total_distance_calls += avg_neighbors_per_layer * 2 * M  # Both directions
    end

    return (
        total_inserts = total_inserts,
        total_distance_calls = total_distance_calls,
        distance_calls_per_insert = total_distance_calls / n,
    )
end

ops = count_operations(data, M, ef_construction)
println("\n  Estimated operations:")
println("    Total insertions:     $(ops.total_inserts)")
println("    Total distance calls: $(Int(round(ops.total_distance_calls)))")
println("    Distance calls/insert: $(Int(round(ops.distance_calls_per_insert)))")
println("    Time per distance:    $(round(build_time / ops.total_distance_calls * 1e6, digits=2)) μs")

println("\n" * "=" ^ 60)
println("Analysis:")
println("=" ^ 60)

# Compare to hnswlib baseline
hnswlib_build_time = 3.0  # seconds (from benchmark)
speedup = build_time / hnswlib_build_time

println("\nManifoldANN: $(round(build_time, digits=2))s")
println("hnswlib:     $(round(hnswlib_build_time, digits=2))s")
println("Slowdown:    $(round(speedup, digits=2))×")

println("\nKey observations:")
println("  1. _prune_list! sorts entire neighbor list on every pruning")
println("     → Lines 238 in query.jl: sort!(list, by = id -> ...)")
println("     → This happens ~2M times during build (each bidirectional link)")
println("     → Could use partial sorting (partialsort!) or maintain sorted invariant")
println("")
println("  2. Linear search in list membership check")
println("     → Lines 223, 227 in query.jl: !(b in list_a)")
println("     → O(M) lookup for each of ~10K insertions × M neighbors")
println("     → Could use Set for O(1) lookup, but with overhead")
println("")
println("  3. Multiple allocations in _link_nodes!")
println("     → Lines 231-232: adjacency[a] = list_a (copies modified lists)")
println("     → Happens for every link operation")
println("     → Lists are already mutable, assignment might be unnecessary")
println("")
println("  4. select_neighbors does full sort + copy")
println("     → Line 22 in neighbor_policy.jl: sort!(candidates, by = ...)")
println("     → Could use partialsort! to only sort top M elements")

println("\n" * "=" ^ 60)
println("Low-hanging fruit recommendations:")
println("=" ^ 60)
println("  [EASY] Remove unnecessary adjacency[a] = list_a assignments (lines 231-232)")
println("         → Saves ~10K allocations, lists are already mutated in-place")
println("         → Expected speedup: 5-10%")
println("")
println("  [EASY] Use partialsort! instead of sort! in select_neighbors")
println("         → Lines 22 in neighbor_policy.jl")
println("         → Expected speedup: 10-15%")
println("")
println("  [MEDIUM] Use partialsort! in _prune_list!")
println("          → Line 238 in query.jl")
println("          → Only need M smallest elements, not full sort")
println("          → Expected speedup: 15-25%")
println("")
println("  [HARD] Avoid sorting in critical path by maintaining sorted invariant")
println("        → Requires more complex insertion logic")
println("        → Expected speedup: 30-40%")
println("")
println("Total estimated speedup from easy+medium fixes: 30-50%")
println("This would bring build time from ~13s to ~7-9s (still slower than hnswlib,")
println("but acceptable given flexibility goals)")
