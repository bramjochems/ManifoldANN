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
# Note: MAX_HASH_BITS is defined in utils/constants.jl

pack_bits(bits::AbstractVector{Bool}) = begin
    length(bits) <= MAX_HASH_BITS ||
        throw(ArgumentError("Hash length $(length(bits)) exceeds supported $MAX_HASH_BITS bits"))
    value = UInt64(0)
    @inbounds for (i, flag) in enumerate(bits)
        flag || continue
        value |= UInt64(1) << (i - 1)
    end
    return value
end

# FNV-1a 64-bit folded over the raw bin integers. Avoids the per-call Tuple
# allocation of `hash(Tuple(bins))` on the inner LSH hash path.
const _FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325
const _FNV_PRIME_64        = 0x00000100000001b3

@inline function pack_bins(bins::AbstractVector{Int})
    h = _FNV_OFFSET_BASIS_64
    @inbounds for b in bins
        u = reinterpret(UInt64, Int64(b))
        for shift in (0, 8, 16, 24, 32, 40, 48, 56)
            h = (h ⊻ ((u >> shift) & 0xff)) * _FNV_PRIME_64
        end
    end
    return h
end

# `euclidean_distance` and `cosine_distance` (used by hash families' default
# `distance_function`) are now Distances.jl aliases defined in
# src/utils/distances.jl.
