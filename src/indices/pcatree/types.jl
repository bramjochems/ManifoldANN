using LinearAlgebra
using Random

"""
    PCASplitter(estimator, direction, stopping, split_value)

Composes the four PCA-tree extension points into a single splitter
object passed to `build_index(PCATreeIndex, ...)`. Each axis is
independent — see [`AbstractSpectrumEstimator`](@ref),
[`AbstractSplitDirectionPolicy`](@ref),
[`AbstractStoppingCriterion`](@ref), and
[`AbstractSplitValuePolicy`](@ref) for the swap surface.

Defaults (no-arg constructor): `ExactSVD`, `TopComponent`,
`AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1, n_floor=256))`,
`MedianSplit`. This is the deterministic single-tree baseline.

For a forest-ready recipe see [`pca_forest_splitter`](@ref).

# How spectrum and direction share the SVD

The splitter calls the spectrum estimator at most ONCE per node. The
result is forwarded to both the stopping check and the direction
policy, so users get one SVD per node regardless of how many traits
read it. If the stopping criteria don't need a spectrum (`MaxLeafSize`
only), the splitter elides the SVD when the size gate fires — no
wasted work on cheap-stop nodes.

If a user wants exact-spectrum-but-randomised-direction (e.g. for a
forest), pair `ExactSVD` with `RandomTopK` / `RandomLinearCombo`. If
they want sketched everything, use `RandomizedSVD` directly.
"""
struct PCASplitter{E<:AbstractSpectrumEstimator,
                   D<:AbstractSplitDirectionPolicy,
                   S<:AbstractStoppingCriterion,
                   V<:AbstractSplitValuePolicy}
    estimator::E
    direction::D
    stopping::S
    split_value::V
end

PCASplitter() = PCASplitter(
    ExactSVD(),
    TopComponent(),
    AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1; n_floor = 256)),
    MedianSplit(),
)

"""
    pca_forest_splitter(; sample_cap=2048, rank=10, n_iter=2, top_k=3,
                          leaf_cap=32, intrinsic_threshold=0.1,
                          n_floor=256)

Forest-ready PCA splitter recipe. Combines:
- `SubsampledSVD(sample_cap, RandomizedSVD(rank, oversample=10, n_iter=n_iter))`
- `RandomTopK(top_k)`
- `AnyOf(MaxLeafSize(leaf_cap), IntrinsicDimRatio(intrinsic_threshold, n_floor=n_floor))`
- `MedianSplit()`

The defaults are documentation-only — tune to dataset.
"""
function pca_forest_splitter(;
    sample_cap::Int = 2048,
    rank::Int = 10,
    n_iter::Int = 2,
    top_k::Int = 3,
    leaf_cap::Int = 32,
    intrinsic_threshold::Real = 0.1,
    n_floor::Int = 256,
)
    return PCASplitter(
        SubsampledSVD(sample_cap, RandomizedSVD(rank; oversample = 10, n_iter = n_iter)),
        RandomTopK(top_k),
        AnyOf(MaxLeafSize(leaf_cap), IntrinsicDimRatio(intrinsic_threshold; n_floor = n_floor)),
        MedianSplit(),
    )
end

"""
    PCANodePayload{T}

Per-internal-node payload stored in the binary-partition-tree node
array. Leaves get a default-constructed sentinel (callers gate on
`bpt_is_leaf`).
"""
struct PCANodePayload{T<:AbstractFloat}
    direction::Vector{T}   # length d, unit-norm
    threshold::T
    center::Vector{T}      # length d, the column mean of this node's points
end

PCANodePayload{T}() where {T<:AbstractFloat} =
    PCANodePayload{T}(Vector{T}(), zero(T), Vector{T}())

"""
    PCATreeIndex{T,D,Sp}

Single PCA tree index. Internal nodes are routers carrying a unit
direction, a scalar threshold, and the per-node center used to
re-center query projections at descent time. Leaves hold buckets of
point ids over the flat `leaf_members` array.

Routes a query to its leaf bucket via inner-product comparison and
brute-force scans the bucket. Recall improves with leaf size; for
forest-recall configurations use multiple trees with `pca_forest_splitter`
(forest wrapper TBD).

# Depth-adaptive note

A depth-adaptive splitter (e.g. PCA up to depth `d_pca`, then RP) can
be implemented as a *meta-splitter* that wraps a `PCASplitter` and an
`AbstractRPSplitter` and dispatches in its own `bpt_split!` overload —
no change to the four trait types here. The trait surface is
deliberately closed over single-rule splitting; depth adaptivity is
composition, not a new trait.
"""
mutable struct PCATreeIndex{T<:AbstractFloat,D,Sp<:PCASplitter} <: AbstractANNIndex
    nodes::Vector{BPTNode{PCANodePayload{T}}}
    leaf_members::Vector{Int}
    root::Int
    dimension::Int
    n_points::Int
    distance::D
    splitter::Sp
end

const PCATREE_DEFAULT_LEAF_CAP = 32

index_distance(index::PCATreeIndex) = index.distance
configured_k(::PCATreeIndex) = nothing
supports_mutation(::PCATreeIndex) = false
