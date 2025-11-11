using LinearAlgebra
using Random

const BlasFloat = LinearAlgebra.BlasFloat

"""
    AbstractLSHHash

Shared supertype for all locality-sensitive hash families. Concrete hashes
must implement `hash_point`, `hash_batch`, and `distance_function` so the
`LSHIndex` can operate generically over any similarity metric.
"""
abstract type AbstractLSHHash end

"""
    hash_point(hf::AbstractLSHHash, x) -> UInt64

Hash a single point. Implementations may pack bits, bins, or other signatures
as long as the return value can key into a `Dict{UInt64, Vector{Int}}`.
"""
function hash_point end

"""
    hash_batch(hf::AbstractLSHHash, X) -> Vector{UInt64}

Vectorized hashing of all columns of `X`.
"""
function hash_batch end

"""
    distance_function(hf::AbstractLSHHash)

Return the distance metric that should be used when scoring candidates
originating from the hash family.
"""
function distance_function end

# Helper utilities shared across hash families.
const _MAX_HASH_BITS = 64

pack_bits(bits::AbstractVector{Bool}) = begin
    length(bits) <= _MAX_HASH_BITS ||
        throw(ArgumentError("Hash length $(length(bits)) exceeds supported $_MAX_HASH_BITS bits"))
    value = UInt64(0)
    @inbounds for (i, flag) in enumerate(bits)
        flag || continue
        value |= UInt64(1) << (i - 1)
    end
    return value
end

pack_bins(bins::AbstractVector{Int}) = UInt64(hash(Tuple(bins)))

"""
    euclidean_distance(x, y)

Compute Euclidean (L2) distance between vectors x and y.
Optimized to avoid allocations and use SIMD instructions.
"""
@inline function euclidean_distance(x::AbstractVector{T}, y::AbstractVector{T}) where T
    length(x) == length(y) || throw(DimensionMismatch("Vectors must have same length"))
    d = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        diff = x[i] - y[i]
        d += diff * diff
    end
    return sqrt(d)
end

"""
    cosine_distance(x, y)

Compute cosine distance (1 - cosine_similarity) between vectors x and y.
Optimized to avoid allocations and use SIMD instructions.
Returns Inf if either vector has zero norm.
"""
@inline function cosine_distance(x::AbstractVector{T}, y::AbstractVector{T}) where T
    length(x) == length(y) || throw(DimensionMismatch("Vectors must have same length"))

    dot_prod = zero(T)
    norm_x_sq = zero(T)
    norm_y_sq = zero(T)

    @inbounds @simd for i in eachindex(x, y)
        xi = x[i]
        yi = y[i]
        dot_prod += xi * yi
        norm_x_sq += xi * xi
        norm_y_sq += yi * yi
    end

    norm_x = sqrt(norm_x_sq)
    norm_y = sqrt(norm_y_sq)

    (norm_x == 0 || norm_y == 0) && return T(Inf)

    return 1 - dot_prod / (norm_x * norm_y)
end
