using LinearAlgebra

"""
    BruteForceIndex

Reference implementation that scans every point when answering kNN queries.
Useful for correctness checks, small datasets, or as a baseline for recall
metrics. Tracks dataset dimensionality and a point count so it can accept
incremental inserts without rebuilding (the caller still manages the data
matrix or storage).
"""
mutable struct BruteForceIndex{T} <: AbstractANNIndex
    dimension::Int
    n_points::Int
end

"""
    build_index(BruteForceIndex, data; kwargs...)

Create a `BruteForceIndex` for `data`. Only records the dimensionality because
queries always receive the dataset explicitly.
"""
function build_index(::Type{BruteForceIndex}, data::AbstractMatrix{T}; kwargs...) where {T}
    size(data, 1) > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    size(data, 2) > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    return BruteForceIndex{T}(size(data, 1), size(data, 2))
end

"""
    query(index::BruteForceIndex, data, q, k; distance=default_distance)

Compute approximate neighbors by scanning the entire dataset with the supplied
`distance` function (default: Euclidean norm). Returns up to `k` indices into
the columns of `data` ordered by ascending distance.
"""
function query(
    index::BruteForceIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    distance::Function = default_distance,
) where {T}
    _validate_dimensions(index, data, q)
    n_points = size(data, 2)
    k = min(k, n_points)
    k <= 0 && return Int[]

    # Accumulate (distance, index) pairs and partial sort by distance.
    dists = Vector{Float64}(undef, n_points)
    @inbounds for j = 1:n_points
        dists[j] = distance(@view(data[:, j]), q)
    end
    # `partialsortperm` already handles ties deterministically.
    neighbors = partialsortperm(dists, 1:k)
    return neighbors
end

default_distance(x, y) = LinearAlgebra.norm(x .- y)

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
function insert!(index::BruteForceIndex{T}, point::AbstractVector{T}) where {T}
    length(point) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    index.n_points += 1
    return index
end

function insert!(index::BruteForceIndex{T}, points::AbstractMatrix{T}) where {T}
    size(points, 1) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    size(points, 2) > 0 || return index
    index.n_points += size(points, 2)
    return index
end
