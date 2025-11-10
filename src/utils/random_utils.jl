using Random

"""
    spawn_child_rngs(rng, n) -> Vector{AbstractRNG}

Derive `n` independent RNGs from `rng`. Used by randomized indices to ensure
each subcomponent gets a reproducible seed while avoiding shared mutable state.
"""
function spawn_child_rngs(rng::AbstractRNG, n::Integer)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    seeds = rand(rng, UInt, n)
    return [Random.MersenneTwister(seeds[i]) for i in 1:n]
end
