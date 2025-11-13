"""
Convenience builder for IVF (KMeans) + HNSW multi-level indices.
"""

using ...ManifoldANN: default_distance
using Distances: Euclidean, SemiMetric

const DEFAULT_IVF_ROUTING = 8

"""
    build_ivf_hnsw_index(
        X::AbstractMatrix;
        nlist::Int,
        routing_k::Int=DEFAULT_IVF_ROUTING,
        kmeans_distance::SemiMetric = Euclidean(),
        kmeans_init::Symbol = :kmeans_plus_plus,
        kmeans_max_iters::Int = 5,
        kmeans_tol::Float64 = 1e-4,
        hnsw_M::Int = 16,
        hnsw_ef_construction::Int = 200,
        hnsw_ef_search::Int = 64,
        hnsw_neighbor_policy::Symbol = :diversified,
        merge_strategy::AbstractMergeStrategy = SimpleMerge(),
        distance::Function = default_distance,
    )::MultiLevelIndex

Build a two-level IVF index where a KMeans transform (coarse quantization) routes
queries to HNSW sub-indices stored per centroid. The helper wires up a `TransformedConfig`
so callers can benchmark IVF-style hierarchies without manually constructing configs.

# Arguments
- `X`: Training data matrix shaped (d × n)
- `nlist`: Number of coarse clusters (L1 centroids)
- `routing_k`: Number of clusters to probe per query (defaults to 8 or `nlist`, whichever is smaller)
- `kmeans_distance`: Metric for the coarse clustering
- `kmeans_init`: Initialization strategy for KMeans (`:kmeans_plus_plus` or `:random`)
- `kmeans_max_iters`: Maximum Lloyd iterations for clustering
- `kmeans_tol`: Convergence tolerance for clustering
- `hnsw_M`, `hnsw_ef_construction`, `hnsw_ef_search`, `hnsw_neighbor_policy`: Parameters forwarded to `HNSWIndex`
- `merge_strategy`: Strategy for merging child results (default `SimpleMerge`)
- `distance`: Distance function used when child indices do not expose their own (fallback for multi-level query)

# Returns
- A `MultiLevelIndex` instance with IVF + HNSW topology
"""
function build_ivf_hnsw_index(
    X::AbstractMatrix;
    nlist::Int,
    routing_k::Int=DEFAULT_IVF_ROUTING,
    kmeans_distance::SemiMetric = Euclidean(),
    kmeans_init::Symbol = :kmeans_plus_plus,
        kmeans_max_iters::Int = 5,
    kmeans_tol::Float64 = 1e-4,
    hnsw_M::Int = 16,
    hnsw_ef_construction::Int = 200,
    hnsw_ef_search::Int = 64,
    hnsw_neighbor_policy::Symbol = :diversified,
    merge_strategy::AbstractMergeStrategy = SimpleMerge(),
    distance::Function = default_distance,
)
    nlist > 0 || throw(ArgumentError("nlist must be positive, got $nlist"))
    routing = TopKRouting(clamp(routing_k, 1, nlist))
    transform = KMeansTransform(
        k = nlist,
        distance = kmeans_distance,
        init = kmeans_init,
        max_iters = kmeans_max_iters,
        tol = kmeans_tol,
    )
    hnsw_params = (
        M = hnsw_M,
        ef_construction = hnsw_ef_construction,
        ef_search = hnsw_ef_search,
        neighbor_policy = hnsw_neighbor_policy,
    )
    config = TransformedConfig(
        transform,
        routing,
        TerminalConfig(HNSWIndex, hnsw_params),
    )
    return build_index(
        MultiLevelIndex,
        X,
        config;
        merge_strategy = merge_strategy,
        distance = distance,
    )
end
