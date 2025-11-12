# NN-Descent index example
#
# Run with:
#   julia --project=. docs/examples/indices/05-nndescent-index.jl

using ManifoldANN
using Random

Random.seed!(42)

# Toy dataset: 16 dimensions, 2K points
d, n = 16, 2_000
data = randn(Float32, d, n)

# Build an NN-Descent index
index = build_index(
    NNDescentIndex,
    data;
    k = 24,
    max_iterations = 12,
    convergence_threshold = 1e-3,
    sampling_policy = :uniform,
    distance = default_squared_distance,
)

println("Index built: k=$(index.k), dimension=$(index.dimension)")

# Query the index
query_vec = randn(Float32, d)
neighbors = query(index, data, query_vec, 10; ef_search = 48)
println("Approximate neighbors (1-based ids): ", neighbors)

# Materialize the kNN graph if needed for downstream processing
graph = materialize_graph(index)
println("Materialized KNNGraph with $(length(graph)) nodes and k=$(graph.k)")
