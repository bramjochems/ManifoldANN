using ManifoldANN
using Random
using Printf

println("NN-Descent Performance Benchmark")
println("=" ^ 50)

# Load MNIST-like data
n_points = 10000
n_dims = 784
data = randn(Float32, n_dims, n_points)

println("Dataset: $(n_dims) × $(n_points)")
println()

# Test with different k values
for k in [16, 32, 64]
    println("Building NN-Descent index with k=$k")

    # Warm-up
    _ = build_index(
        NNDescentIndex,
        data[:, 1:100],
        k=min(k, 50),
        max_iterations=2,
        convergence_threshold=0.01,
        sampling_policy=UniformPairSampling(0.5),
        symmetry_policy=:full,
    )

    # Actual benchmark
    GC.gc()
    start_time = time()

    index = build_index(
        NNDescentIndex,
        data,
        k=k,
        max_iterations=10,
        convergence_threshold=0.01,
        sampling_policy=UniformPairSampling(0.5),
        symmetry_policy=:full,
    )

    elapsed = time() - start_time

    # Check index properties
    avg_neighbors = sum(length.(index.neighbors)) / length(index.neighbors)

    @printf("  Build time: %.2f seconds\n", elapsed)
    @printf("  Avg neighbors per node: %.1f (expected >= %d)\n", avg_neighbors, k)
    println()
end

println("Benchmark complete!")
