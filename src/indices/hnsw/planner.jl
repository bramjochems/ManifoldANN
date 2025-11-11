abstract type AbstractLayerPlanner end

"""
    DefaultLayerPlanner(; lambda = 1 / log(1.0 * M))

Samples HNSW layers using an exponential distribution. Larger `lambda` increases
the probability of placing nodes in higher layers, resulting in denser upper
graphs. The sampler always returns a non-negative integer (layer 0 = base).
"""
struct DefaultLayerPlanner
    lambda::Float64
end

DefaultLayerPlanner(; lambda::Real = 1.0) = DefaultLayerPlanner(Float64(lambda))

function sample_layer(planner::DefaultLayerPlanner, rng::AbstractRNG)
    # Use inverse transform sampling for exponential distribution with rate lambda.
    u = rand(rng)
    # Guard against exact zero.
    u = max(u, eps(Float64))
    value = Int(floor(-log(u) * planner.lambda))
    return value
end
