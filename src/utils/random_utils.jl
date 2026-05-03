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

"""
    derive_child_seed(rng) -> UInt64

Draw one `UInt64` from `rng` to use as the parent seed for lazy per-task RNG
derivation via [`query_child_rng`](@ref).
"""
derive_child_seed(rng::AbstractRNG) = rand(rng, UInt64)

"""
    query_child_rng(parent_seed::UInt64, i::Integer) -> Xoshiro

Construct a `Xoshiro` RNG for query index `i` deterministically from
`parent_seed`. Result is independent of thread count and worker assignment, so
batched callers can derive per-query RNGs lazily inside workers without any
upfront vector allocation.
"""
@inline query_child_rng(parent_seed::UInt64, i::Integer) =
    Random.Xoshiro(hash(parent_seed, hash(UInt64(i))))
