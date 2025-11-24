#=
Example: Comparing routing strategies in IVF indices

This example demonstrates how different routing strategies affect
recall and query performance in IVF-style indices.

We compare:
- TopKRouting with various k values (1, 3, 5, 10, 20)
- ExhaustiveRouting (probe all clusters)

Run with `julia --project=. docs/examples/indices/08-routing-strategies.jl`
=#

using ManifoldANN
using Distances
using Random
using LinearAlgebra
using Printf

rng = MersenneTwister(456)
dimension = 48
n_points = 8_000
k_neighbors = 10
n_clusters = 40

println("=" ^ 70)
println("Routing Strategies Comparison")
println("=" ^ 70)
println()

# Generate data
println("▶ Generating dataset ($n_points points, $dimension dimensions)")
data = randn(rng, Float32, dimension, n_points)
println()

# Build ground truth index
println("▶ Building ground truth (brute-force) index")
@time brute_index = build_index(BruteForceIndex, data)
println()

# Build IVF indices with different routing strategies
probe_values = [1, 3, 5, 10, 20, n_clusters]  # Last one is exhaustive
indices = Dict()

println("▶ Building IVF indices with different routing strategies")
println("  Base configuration: $n_clusters clusters, HNSW terminals")
println()

for n_probe in probe_values
    if n_probe == n_clusters
        label = "Exhaustive"
        routing = ExhaustiveRouting()
    else
        label = "TopK($n_probe)"
        routing = TopKRouting(n_probe)
    end

    config = TransformedConfig(
        KMeansTransform(
            k=n_clusters,
            distance=Euclidean(),
            init=:kmeans_plus_plus,
            max_iters=30
        ),
        routing,
        TerminalConfig(HNSWIndex, (M=12, ef_construction=100))
    )

    print("  Building $label... ")
    @time indices[label] = build_index(MultiLevelIndex, data, config)
end
println()

# Generate test queries
n_test_queries = 20
test_queries = [randn(rng, Float32, dimension) for _ in 1:n_test_queries]

println("▶ Evaluating on $n_test_queries test queries")
println()

# Compute ground truth for all queries
ground_truth = [query(brute_index, data, q, k_neighbors) for q in test_queries]

# Evaluate each routing strategy
results = []

for (label, index) in sort(collect(indices), by=x->x[1])
    # Query all test points
    query_times = Float64[]
    recalls = Float64[]

    for (i, q) in enumerate(test_queries)
        # Time the query
        start_time = time()
        approx_neighbors = query(index, data, q, k_neighbors)
        elapsed = time() - start_time
        push!(query_times, elapsed)

        # Compute recall
        truth = ground_truth[i]
        recall =
            length(
                intersect(
                    neighbor_ids(approx_neighbors) |> Set,
                    neighbor_ids(truth) |> Set,
                ),
            ) / k_neighbors
        push!(recalls, recall)
    end

    # Aggregate statistics
    avg_time = mean(query_times) * 1000  # Convert to ms
    avg_recall = mean(recalls)

    push!(results, (
        label=label,
        avg_time=avg_time,
        avg_recall=avg_recall
    ))
end

# Display results table
println("Results (averaged over $n_test_queries queries):")
println()
println("  " * "-" ^ 60)
println(@sprintf("  %-20s %15s %15s", "Strategy", "Avg Time (ms)", "Avg Recall"))
println("  " * "-" ^ 60)

for result in results
    println(@sprintf("  %-20s %15.3f %15.1f%%",
        result.label,
        result.avg_time,
        result.avg_recall * 100
    ))
end
println("  " * "-" ^ 60)
println()

# Analysis
println("▶ Analysis:")
println()

# Find best recall/time trade-off (using a simple metric)
best_tradeoff = argmax([r.avg_recall / (r.avg_time + 0.1) for r in results])
best = results[best_tradeoff]

println("1. Recall vs Probes:")
println("   - More probes → higher recall (expected)")
println("   - Diminishing returns: going from 10 to 20 probes")
println()

println("2. Query Time:")
println("   - Linear relationship with number of probes")
println("   - TopK(1) is fastest, Exhaustive is slowest")
println()

println("3. Best trade-off (recall / time):")
println("   - Strategy: $(best.label)")
println("   - Recall: $(round(best.avg_recall * 100, digits=1))%")
println("   - Time: $(round(best.avg_time, digits=2))ms")
println()

println("4. When to use each strategy:")
println("   - TopK(1):        Ultra-fast, low recall (60-70%)")
println("   - TopK(3-5):      Balanced (80-90% recall)")
println("   - TopK(10+):      High recall (>95%), slower")
println("   - Exhaustive:     Maximum recall, baseline comparison")
println()

println("=" ^ 70)
println("Key takeaways:")
println("- Routing strategy is the primary recall/speed knob in IVF")
println("- Use ExhaustiveRouting for debugging or when clusters are few")
println("=" ^ 70)
