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

index_distance(index::BruteForceIndex) = index.distance

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
    validate_index_dimensions(index, data, q)
    S = float(T)
    n_points = size(data, 2)
    k = min(k, n_points)
    k <= 0 && return Neighbor{S}[]

    # Compute distances with multithreading support
    dists = Vector{S}(undef, n_points)
    @inbounds Threads.@threads for j = 1:n_points
        dists[j] = index.distance(@view(data[:, j]), q)
    end

    # Partial sort to find k nearest neighbors
    ids = partialsortperm(dists, 1:k)
    results = Vector{Neighbor{S}}(undef, length(ids))
    @inbounds for (pos, id) in enumerate(ids)
        results[pos] = Neighbor{S}(id, dists[id])
    end
    return results
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
