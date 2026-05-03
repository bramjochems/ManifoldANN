using Distances

# Canonical metric provider: aliases over Distances.jl types.
#
# These names are kept for backwards compatibility with existing callsites
# (`distance = default_distance`, `distance_function(::Hash) = cosine_distance`,
# `idx.distance === default_distance` identity checks). The hand-rolled SIMD
# kernels were removed once a microbenchmark confirmed Distances.jl is at
# parity (and marginally faster at d >= 50). Future Distances.jl improvements
# carry over for free.
#
# Aliases can be removed in a later cleanup if downstream callers migrate to
# the Distances.jl names directly.

const default_distance         = Distances.Euclidean()
const default_squared_distance = Distances.SqEuclidean()
const euclidean_distance       = Distances.Euclidean()
const cosine_distance          = Distances.CosineDist()

"""
    squared_cosine_distance(x, y)

Monotone transform of cosine distance, `2 * (1 - cos_sim(x, y))`, used as a
priority-queue key in angular-metric NN-Descent / HNSW. NOT the same as
`CosineDist()` (which returns `1 - cos_sim`) and NOT the same as squaring
`CosineDist()` (different ordering near the antipode). Equals the squared
Euclidean distance between the L2-normalised vectors. Returns `T(2)` for
zero-norm inputs (the upper bound of the metric) so they sort to the back of
priority queues; `CosineDist()` would return `NaN` in that case.
"""
@inline function squared_cosine_distance(x::AbstractVector{T}, y::AbstractVector{T}) where T
    dot_product = zero(T)
    x_norm_sq = zero(T)
    y_norm_sq = zero(T)

    @inbounds @simd for i in eachindex(x, y)
        dot_product += x[i] * y[i]
        x_norm_sq += x[i] * x[i]
        y_norm_sq += y[i] * y[i]
    end

    if x_norm_sq == 0 || y_norm_sq == 0
        return convert(T, 2)
    end

    cosine_sim = dot_product / sqrt(x_norm_sq * y_norm_sq)
    cosine_sim = clamp(cosine_sim, -one(T), one(T))
    return convert(T, 2) * (one(T) - cosine_sim)
end
