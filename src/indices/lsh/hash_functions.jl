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

cosine_distance(x, y) = begin
    nx = norm(x)
    ny = norm(y)
    (nx == 0 || ny == 0) && return Inf
    return 1 - dot(x, y) / (nx * ny)
end

euclidean_distance(x, y) = norm(x .- y)
