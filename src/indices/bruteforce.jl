using LinearAlgebra

"""
    BruteForceIndex

Reference implementation that scans every point when answering kNN queries.
Useful for correctness checks, small datasets, or as a baseline for recall
metrics. Tracks dataset dimensionality and a point count so it can accept
incremental inserts without rebuilding (the caller still manages the data
matrix or storage).

# Type Parameters
- `T`: Element type (e.g., Float32, Float64)
- `D`: Distance function type (must be thread-safe)
"""
mutable struct BruteForceIndex{T,D} <: AbstractANNIndex
    dimension::Int
    n_points::Int
    distance::D
end

"""
    build_index(BruteForceIndex, data; distance=default_distance)

Create a `BruteForceIndex` for `data`. Only records the dimensionality because
queries always receive the dataset explicitly. The distance function is stored
in the index for type stability and thread safety.
"""
function build_index(::Type{BruteForceIndex}, data::AbstractMatrix{T}; distance::D = default_distance) where {T, D}
    size(data, 1) > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    size(data, 2) > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    return BruteForceIndex{T,D}(size(data, 1), size(data, 2), distance)
end

"""
    query(index::BruteForceIndex, data, q, k)

Compute exact nearest neighbors by scanning the entire dataset with the distance
function stored in the index. Returns up to `k` indices into the columns of
`data` ordered by ascending distance.

Uses multithreading when available (controlled by `Threads.nthreads()`).
"""
function query(
    index::BruteForceIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
) where {T}
    _validate_dimensions(index, data, q)
    n_points = size(data, 2)
    k = min(k, n_points)
    k <= 0 && return Int[]

    # Compute distances with multithreading support
    dists = Vector{Float64}(undef, n_points)
    @inbounds Threads.@threads for j = 1:n_points
        dists[j] = index.distance(@view(data[:, j]), q)
    end

    # Partial sort to find k nearest neighbors
    neighbors = partialsortperm(dists, 1:k)
    return neighbors
end

"""
    query(index::BruteForceIndex, data, queries::Matrix, k)

Batch query interface: compute exact nearest neighbors for multiple queries.
Returns a vector of result vectors, one per query.
"""
function query(
    index::BruteForceIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
) where {T}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))

    n_queries = size(queries, 2)
    results = Vector{Vector{Int}}(undef, n_queries)

    @inbounds for i in 1:n_queries
        q = @view queries[:, i]
        results[i] = query(index, data, q, k)
    end

    return results
end

"""
    query(index::BruteForceIndex, data, queries::Vector{<:Vector}, k)

Convenience batch query interface using a vector of query vectors.
"""
function query(
    index::BruteForceIndex{T},
    data::AbstractMatrix{T},
    queries::Vector{<:AbstractVector{T}},
    k::Integer;
) where {T}
    isempty(queries) && return Vector{Vector{Int}}()
    queries_mat = reduce(hcat, queries)
    return query(index, data, queries_mat, k)
end

"""
    default_distance(x, y)

Default distance function using Euclidean (L2) distance.
Optimized to avoid allocations and use SIMD instructions.
"""
@inline function default_distance(x::AbstractVector{T}, y::AbstractVector{T}) where T
    d = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        diff = x[i] - y[i]
        d += diff * diff
    end
    return sqrt(d)
end

function _validate_dimensions(index::BruteForceIndex, data, q)
    size(data, 1) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    length(q) == index.dimension ||
        throw(DimensionMismatch("Expected query dimension $(index.dimension)"))
    size(data, 2) >= index.n_points || throw(
        ArgumentError(
            "Data contains $(size(data, 2)) points but index tracks $(index.n_points)",
        ),
    )
    return nothing
end

supports_mutation(::BruteForceIndex) = true

"""
    insert!(index::BruteForceIndex, point)
    insert!(index::BruteForceIndex, points)

Register one or multiple points with the index. Callers are responsible for
storing the actual data; this method only keeps the metadata in sync.
"""
function insert!(index::BruteForceIndex{T,D}, point::AbstractVector{T}) where {T,D}
    length(point) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    index.n_points += 1
    return index
end

function insert!(index::BruteForceIndex{T,D}, points::AbstractMatrix{T}) where {T,D}
    size(points, 1) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    size(points, 2) > 0 || return index
    index.n_points += size(points, 2)
    return index
end
