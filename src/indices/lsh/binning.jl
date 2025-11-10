"""
    BinningHash

LSH family for Euclidean distance using p-stable (Gaussian) projections and uniform binning.
"""
struct BinningHash{T<:BlasFloat} <: AbstractLSHHash
    projections::Matrix{T}
    offsets::Vector{T}
    bin_width::T
end

"""
    make_binning_hash(dimension, hash_length; bin_width, use_offset=false, rng=Random.default_rng(), T=Float64)

Construct a binning hash family. `bin_width` controls how aggressively nearby
points collide. Set `use_offset=true` to sample offsets uniformly in `[0, bin_width)`.
"""
function make_binning_hash(
    dimension::Integer,
    hash_length::Integer;
    bin_width::Real,
    use_offset::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
    T::Type{<:BlasFloat} = Float64,
)
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    hash_length > 0 || throw(ArgumentError("hash_length must be positive"))
    bw = T(bin_width)
    bw > 0 || throw(ArgumentError("bin_width must be positive"))
    projections = randn(rng, T, hash_length, dimension)
    offsets = use_offset ? bw .* rand(rng, T, hash_length) : zeros(T, hash_length)
    return BinningHash{T}(projections, offsets, bw)
end

function hash_point(hf::BinningHash{T}, x::AbstractVector{T}) where {T<:BlasFloat}
    length(x) == size(hf.projections, 2) ||
        throw(DimensionMismatch("Expected vector of length $(size(hf.projections, 2))"))
    values = hf.projections * x .+ hf.offsets
    bins = floor.(Int, values ./ hf.bin_width)
    return pack_bins(bins)
end

function hash_batch(hf::BinningHash{T}, X::AbstractMatrix{T}) where {T<:BlasFloat}
    size(X, 1) == size(hf.projections, 2) ||
        throw(DimensionMismatch("Expected matrix with $(size(hf.projections, 2)) rows"))
    values = hf.projections * X
    values .+= hf.offsets
    n = size(X, 2)
    hashes = Vector{UInt64}(undef, n)
    @inbounds for i in 1:n
        bins = floor.(Int, values[:, i] ./ hf.bin_width)
        hashes[i] = pack_bins(bins)
    end
    return hashes
end

distance_function(::BinningHash) = euclidean_distance
