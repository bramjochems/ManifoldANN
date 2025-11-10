"""
    RandomHyperplaneHash

Classic angular LSH family that hashes by taking the sign of random
projections drawn from `N(0, I)`. Works well for cosine/inner-product
similarity.
"""
struct RandomHyperplaneHash{T<:BlasFloat} <: AbstractLSHHash
    projections::Matrix{T} # hash_length × dimension
end

"""
    make_random_hyperplane_hash(dimension, hash_length; rng=Random.default_rng(), T=Float64)

Construct a random hyperplane hash family with `hash_length` projections.
"""
function make_random_hyperplane_hash(
    dimension::Integer,
    hash_length::Integer;
    rng::AbstractRNG = Random.default_rng(),
    T::Type{<:BlasFloat} = Float64,
)
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    hash_length > 0 || throw(ArgumentError("hash_length must be positive"))
    projections = randn(rng, T, hash_length, dimension)
    return RandomHyperplaneHash{T}(projections)
end

function hash_point(hf::RandomHyperplaneHash{T}, x::AbstractVector{T}) where {T<:BlasFloat}
    length(x) == size(hf.projections, 2) ||
        throw(DimensionMismatch("Expected vector of length $(size(hf.projections, 2))"))
    projection_values = hf.projections * x
    return pack_bits(projection_values .>= 0)
end

function hash_batch(hf::RandomHyperplaneHash{T}, X::AbstractMatrix{T}) where {T<:BlasFloat}
    size(X, 1) == size(hf.projections, 2) ||
        throw(DimensionMismatch("Expected matrix with $(size(hf.projections, 2)) rows"))
    projection_values = hf.projections * X
    n = size(X, 2)
    hashes = Vector{UInt64}(undef, n)
    @inbounds for i in 1:n
        hashes[i] = pack_bits(projection_values[:, i] .>= 0)
    end
    return hashes
end

distance_function(::RandomHyperplaneHash) = cosine_distance
