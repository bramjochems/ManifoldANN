using LinearAlgebra
using Random

"""
    NNDescentNeighborNode

Working neighbor storage used during NN-Descent construction. Uses dual heaps to
efficiently track "new" neighbors (discovered this iteration) and "old" neighbors
(from previous iterations), which is essential for the NN-Descent local join strategy.
"""
mutable struct NNDescentNeighborNode{T<:AbstractFloat}
    new_neighbors::BoundedMaxHeap{T}
    old_neighbors::BoundedMaxHeap{T}

    function NNDescentNeighborNode{T}(k::Int) where {T<:AbstractFloat}
        return new{T}(BoundedMaxHeap{T}(k), BoundedMaxHeap{T}(k))
    end
end

abstract type AbstractNNDescentSamplingPolicy end

"""
    UniformPairSampling(rate=1.0)

Sampling policy that evaluates each candidate pair with probability `rate`.
Setting `rate=1.0` reproduces the textbook NN-Descent algorithm, while lower
rates trade determinism for fewer distance evaluations.
"""
struct UniformPairSampling <: AbstractNNDescentSamplingPolicy
    sample_rate::Float64
    function UniformPairSampling(rate::Float64 = 1.0)
        (0.0 < rate <= 1.0) ||
            throw(ArgumentError("sample_rate must satisfy 0 < rate ≤ 1, got $rate"))
        return new(rate)
    end
end

"""
    should_consider_pair(policy, rng) -> Bool

Predicate used by the builder to decide whether a candidate pair should be
evaluated. Keeps the decision logic pluggable so we can experiment with
importance sampling or degree-aware throttling later.
"""
@inline function should_consider_pair(
    policy::UniformPairSampling,
    rng::AbstractRNG,
)
    if policy.sample_rate >= 1.0
        return true
    end
    return rand(rng) <= policy.sample_rate
end

abstract type AbstractSymmetryPolicy end

"""
    FullSymmetry()

Symmetry policy that ensures complete graph symmetry: if node `i` has `j` as a
neighbor, then `j` will have `i` as a neighbor. This may result in nodes having
more than `k` neighbors. Provides the best search recall at the cost of higher
memory usage and potentially slower queries.
"""
struct FullSymmetry <: AbstractSymmetryPolicy end

"""
    PrunedSymmetry(degree_multiplier=1.5)

Symmetry policy inspired by PyNNDescent. Adds reverse nearest neighbor edges to
improve search quality, but limits each node to at most `degree_multiplier * k`
edges. This balances search accuracy with memory efficiency.

Setting `degree_multiplier=1.0` gives an asymmetric k-NN graph (strictly k neighbors).
Setting `degree_multiplier=1.5` (default) allows up to 50% more edges for reverse neighbors.
Higher values approach full symmetry but use more memory.
"""
struct PrunedSymmetry <: AbstractSymmetryPolicy
    degree_multiplier::Float64
    function PrunedSymmetry(degree_multiplier::Float64 = 1.5)
        degree_multiplier >= 1.0 ||
            throw(ArgumentError("degree_multiplier must be >= 1.0, got $degree_multiplier"))
        return new(degree_multiplier)
    end
end

"""
    NoSymmetry()

Symmetry policy that maintains an asymmetric (directed) k-NN graph. Each node
has exactly `k` neighbors, but the neighbor relation is not bidirectional. This
minimizes memory usage but may reduce search recall compared to symmetric policies.
"""
struct NoSymmetry <: AbstractSymmetryPolicy end

"""
    NNDescentIndex

Graph-based ANN index produced via NN-Descent. Stores only the final neighbor
lists, so callers must continue passing the dataset when querying. Parameters
such as `k`, iteration limits, and the sampling policy are kept on the index
for introspection.
"""
mutable struct NNDescentIndex{T<:LinearAlgebra.BlasFloat,D,SP,SYM} <: AbstractGraphIndex
    dimension::Int
    n_points::Int
    k::Int
    max_iterations::Int
    distance::D
    sampling_policy::SP
    symmetry_policy::SYM
    neighbors::Vector{Vector{Int}}
end

index_distance(index::NNDescentIndex) = index.distance

configured_k(index::NNDescentIndex) = index.k
supports_layers(::NNDescentIndex) = false
supports_metadata(::NNDescentIndex) = false
supports_mutation(::NNDescentIndex) = false
