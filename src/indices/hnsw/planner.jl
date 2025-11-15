abstract type AbstractLayerPlanner end

"""
    DefaultLayerPlanner(; lambda = 1 / log(1.0 * M))

Samples HNSW layers using an exponential distribution. Larger `lambda` increases
the probability of placing nodes in higher layers, resulting in denser upper
graphs. The sampler always returns a non-negative integer (layer 0 = base).

The default lambda calculation `1 / log(M)` comes from the original HNSW paper
(Malkov & Yashunin, 2018) and provides optimal scaling properties.
"""
struct DefaultLayerPlanner
    lambda::Float64
end

DefaultLayerPlanner(; lambda::Real = 1.0) = DefaultLayerPlanner(Float64(lambda))

function sample_layer(planner::DefaultLayerPlanner, rng::AbstractRNG)
    # Use inverse transform sampling for exponential distribution with rate lambda
    # Formula: -log(U) / lambda where U ~ Uniform(0, 1)
    uniform_sample = rand(rng)

    # Guard against exact zero to avoid log(0) = -Inf
    uniform_sample = max(uniform_sample, eps(Float64))

    # Sample from exponential distribution and floor to get layer index
    layer = Int(floor(-log(uniform_sample) * planner.lambda))
    return layer
end
