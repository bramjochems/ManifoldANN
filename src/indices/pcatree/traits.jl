using LinearAlgebra
using Random

# ============================================================================
# PCATreeIndex extension points
# ============================================================================
#
# Four orthogonal traits configure the PCA tree's per-node behaviour. They
# follow the same flexibility-point pattern as `AbstractRPSplitter` (rptree),
# `AbstractEdgeWeight` / `AbstractLocalGeometry` (graphs), and `hash_factory`
# (LSH): a small abstract type, concrete strategies that overload a single
# verb, and a composing `PCASplitter` struct.
#
# The four axes are deliberately independent so callers can mix & match:
#   - AbstractSpectrumEstimator     : how to get principal directions/eigvals
#   - AbstractSplitDirectionPolicy  : given a spectrum, pick split direction
#   - AbstractStoppingCriterion     : when to stop recursing
#   - AbstractSplitValuePolicy      : given direction, pick split threshold
#
# A depth-adaptive (e.g. PCA-then-RP) strategy can be implemented as a
# meta-splitter that *wraps* a `PCASplitter` and an `AbstractRPSplitter` and
# routes by depth/n_node — without touching this trait machinery.
# ============================================================================

# ----- Spectrum estimator --------------------------------------------------

"""
    AbstractSpectrumEstimator

Strategy for computing principal directions + eigenvalues at a node. A
concrete estimator implements

    estimate_spectrum(estimator, X_centered, rng) -> NamedTuple{(:U, :S)}

where `X_centered` is `d × n_node` with the column mean already removed,
`U` is `d × k` with `k` orthonormal principal directions, and `S` is the
length-`k` vector of singular values (square roots of eigenvalues of
`X_centered * X_centered'/n`). Estimators are free to return a truncated
spectrum (all `k` they can afford to compute).

Built-in: [`ExactSVD`](@ref), [`RandomizedSVD`](@ref), [`SubsampledSVD`](@ref).
"""
abstract type AbstractSpectrumEstimator end

"""
    ExactSVD()

Exact SVD via `LinearAlgebra.svd`. Default for small nodes; deterministic
and the gold standard for both stopping checks and split-direction
selection. Cost is `O(min(d, n) * d * n)`. Use [`RandomizedSVD`](@ref) or
[`SubsampledSVD`](@ref) when nodes get large.
"""
struct ExactSVD <: AbstractSpectrumEstimator end

function estimate_spectrum(::ExactSVD, X::AbstractMatrix, ::AbstractRNG)
    F = svd(X; full = false)
    return (U = F.U, S = F.S)
end

"""
    RandomizedSVD(rank; oversample=10, n_iter=2)

Halko-Martinsson-Tropp randomized SVD with `rank + oversample` Gaussian
test vectors and `n_iter` power iterations for spectral concentration.
Returns `rank` principal directions. ~30 LOC of pure linear algebra; no
extra dependency.

Forest-friendly: each tree sees a different sketch under per-tree RNG.
"""
struct RandomizedSVD <: AbstractSpectrumEstimator
    rank::Int
    oversample::Int
    n_iter::Int
    function RandomizedSVD(rank::Int = 5; oversample::Int = 10, n_iter::Int = 2)
        rank >= 1 || throw(ArgumentError("rank must be >= 1"))
        oversample >= 0 || throw(ArgumentError("oversample must be >= 0"))
        n_iter >= 0 || throw(ArgumentError("n_iter must be >= 0"))
        return new(rank, oversample, n_iter)
    end
end

function estimate_spectrum(est::RandomizedSVD, X::AbstractMatrix{T}, rng::AbstractRNG) where {T}
    d, n = size(X)
    r = min(est.rank, d, n)
    p = est.oversample
    sketch_dim = min(r + p, n)
    # Random Gaussian test matrix Ω: n × sketch_dim
    Omega = randn(rng, T, n, sketch_dim)
    Y = X * Omega                  # d × sketch_dim
    # Power iterations stabilise the top-r subspace.
    for _ in 1:est.n_iter
        Q1 = Matrix(qr!(Y).Q)      # d × sketch_dim
        Z = X' * Q1                # n × sketch_dim
        Q2 = Matrix(qr!(Z).Q)      # n × sketch_dim
        Y = X * Q2                 # d × sketch_dim
    end
    Q = Matrix(qr!(Y).Q)           # d × sketch_dim
    B = Q' * X                     # sketch_dim × n
    F = svd(B; full = false)
    U_full = Q * F.U               # d × sketch_dim
    rk = min(r, size(U_full, 2))
    return (U = U_full[:, 1:rk], S = F.S[1:rk])
end

"""
    SubsampledSVD(sample_cap, inner)

Decorator: subsample at most `sample_cap` columns of the node matrix and
defer to `inner::AbstractSpectrumEstimator`. Useful when nodes are too
big for exact SVD but the user still wants exact arithmetic on the
sketch.
"""
struct SubsampledSVD{E<:AbstractSpectrumEstimator} <: AbstractSpectrumEstimator
    sample_cap::Int
    inner::E
    function SubsampledSVD(sample_cap::Int, inner::E) where {E<:AbstractSpectrumEstimator}
        sample_cap >= 1 || throw(ArgumentError("sample_cap must be >= 1"))
        return new{E}(sample_cap, inner)
    end
end

function estimate_spectrum(est::SubsampledSVD, X::AbstractMatrix, rng::AbstractRNG)
    n = size(X, 2)
    if n <= est.sample_cap
        return estimate_spectrum(est.inner, X, rng)
    end
    sample_idx = randperm(rng, n)[1:est.sample_cap]
    Xsub = X[:, sample_idx]
    return estimate_spectrum(est.inner, Xsub, rng)
end

# ----- Split direction policy ----------------------------------------------

"""
    AbstractSplitDirectionPolicy

Strategy for picking a split direction from a precomputed spectrum. A
concrete policy implements

    pick_direction(policy, spectrum, rng) -> Vector{T}

returning a unit vector in `R^d`. Decoupled from the spectrum estimator
so callers can, e.g., pair `ExactSVD` with `RandomTopK` for stochastic
forests on a deterministic spectrum.

Built-in: [`TopComponent`](@ref), [`RandomTopK`](@ref), [`RandomLinearCombo`](@ref).
"""
abstract type AbstractSplitDirectionPolicy end

"""
    TopComponent()

Always pick the top principal component. Deterministic; use as the
single-tree baseline.
"""
struct TopComponent <: AbstractSplitDirectionPolicy end

pick_direction(::TopComponent, spectrum, ::AbstractRNG) = collect(spectrum.U[:, 1])

"""
    RandomTopK(k)

Pick uniformly from the top-`k` principal components. Adds randomness
across forest trees; clips to the actual spectrum rank when smaller than
`k`.
"""
struct RandomTopK <: AbstractSplitDirectionPolicy
    k::Int
    function RandomTopK(k::Int = 3)
        k >= 1 || throw(ArgumentError("k must be >= 1"))
        return new(k)
    end
end

function pick_direction(policy::RandomTopK, spectrum, rng::AbstractRNG)
    rk = size(spectrum.U, 2)
    k_eff = min(policy.k, rk)
    j = rand(rng, 1:k_eff)
    return collect(spectrum.U[:, j])
end

"""
    RandomLinearCombo(k)

Random unit vector in the subspace spanned by the top-`k` principal
components. More diverse than [`RandomTopK`](@ref) for forest trees.
"""
struct RandomLinearCombo <: AbstractSplitDirectionPolicy
    k::Int
    function RandomLinearCombo(k::Int = 3)
        k >= 1 || throw(ArgumentError("k must be >= 1"))
        return new(k)
    end
end

function pick_direction(policy::RandomLinearCombo, spectrum, rng::AbstractRNG)
    rk = size(spectrum.U, 2)
    k_eff = min(policy.k, rk)
    coeffs = randn(rng, k_eff)
    nrm = norm(coeffs)
    nrm > 0 || (coeffs[1] = one(eltype(coeffs)); nrm = one(eltype(coeffs)))
    coeffs ./= nrm
    v = spectrum.U[:, 1:k_eff] * coeffs
    return collect(v)
end

# ----- Stopping criterion --------------------------------------------------

"""
    AbstractStoppingCriterion

Strategy for deciding whether a node should stop recursing and become a
leaf. A concrete criterion implements

    should_stop(criterion, n_node, spectrum_or_nothing) -> Bool

`spectrum_or_nothing` is `nothing` for criteria that don't need it (so
the splitter can short-circuit before paying the SVD cost when the size
gate already triggers).

Built-in: [`MaxLeafSize`](@ref), [`IntrinsicDimRatio`](@ref),
[`AnyOf`](@ref), [`AllOf`](@ref).
"""
abstract type AbstractStoppingCriterion end

"""
    MaxLeafSize(cap)

Hard size cap: stop when `n_node <= cap`. Always cheap (no spectrum
needed); ship as a safety net so other criteria can be conservative.
"""
struct MaxLeafSize <: AbstractStoppingCriterion
    cap::Int
    function MaxLeafSize(cap::Int = 32)
        cap >= 1 || throw(ArgumentError("cap must be >= 1"))
        return new(cap)
    end
end

needs_spectrum(::MaxLeafSize) = false
should_stop(c::MaxLeafSize, n_node::Int, _spectrum) = n_node <= c.cap

"""
    IntrinsicDimRatio(threshold; n_floor=256)

Stop when the spectrum is "flat enough" — concretely when
`s2_sum_below / s2_sum_total <= threshold` is FALSE, i.e. when one
principal direction explains most of the variance. The intuition: if
the data in this node lives near a 1-D subspace, one more split won't
help.

Concretely we stop when
    `(S[1]^2) / sum(S .^ 2) >= 1 - threshold`

`n_floor` gates the check on `n_node >= n_floor` so noisy sketches at
small nodes don't trigger spuriously.
"""
struct IntrinsicDimRatio <: AbstractStoppingCriterion
    threshold::Float64
    n_floor::Int
    function IntrinsicDimRatio(threshold::Real = 0.1; n_floor::Int = 256)
        0.0 < threshold < 1.0 || throw(ArgumentError("threshold must be in (0,1)"))
        n_floor >= 1 || throw(ArgumentError("n_floor must be >= 1"))
        return new(Float64(threshold), n_floor)
    end
end

needs_spectrum(::IntrinsicDimRatio) = true
function should_stop(c::IntrinsicDimRatio, n_node::Int, spectrum)
    n_node >= c.n_floor || return false
    spectrum === nothing && return false
    S = spectrum.S
    isempty(S) && return true
    total = sum(abs2, S)
    total > 0 || return true
    top = abs2(S[1])
    return top / total >= 1.0 - c.threshold
end

"""
    AnyOf(criteria...)

Composite criterion: stop if ANY of `criteria` says stop. Supports
short-circuit evaluation; size-only criteria are checked before the
spectrum is computed.
"""
struct AnyOf{Cs<:Tuple} <: AbstractStoppingCriterion
    criteria::Cs
end
AnyOf(cs::AbstractStoppingCriterion...) = AnyOf(cs)

needs_spectrum(c::AnyOf) = any(needs_spectrum, c.criteria)
function should_stop(c::AnyOf, n_node::Int, spectrum)
    for sub in c.criteria
        should_stop(sub, n_node, spectrum) && return true
    end
    return false
end

"""
    AllOf(criteria...)

Composite criterion: stop only if ALL of `criteria` say stop.
"""
struct AllOf{Cs<:Tuple} <: AbstractStoppingCriterion
    criteria::Cs
end
AllOf(cs::AbstractStoppingCriterion...) = AllOf(cs)

needs_spectrum(c::AllOf) = any(needs_spectrum, c.criteria)
function should_stop(c::AllOf, n_node::Int, spectrum)
    isempty(c.criteria) && return false
    for sub in c.criteria
        should_stop(sub, n_node, spectrum) || return false
    end
    return true
end

# ----- Split-value policy --------------------------------------------------

"""
    AbstractSplitValuePolicy

Strategy for picking the split threshold along the chosen direction. A
concrete policy implements

    pick_split_value(policy, projections, rng) -> Real

`projections` is the length-`n_node` vector of inner products with the
split direction.

Built-in: [`MedianSplit`](@ref), [`MeanSplit`](@ref),
[`RandomBetweenQuantiles`](@ref).
"""
abstract type AbstractSplitValuePolicy end

"""
    MedianSplit()

Median of `projections`. Balanced tree; deterministic.
"""
struct MedianSplit <: AbstractSplitValuePolicy end
pick_split_value(::MedianSplit, p::AbstractVector, ::AbstractRNG) =
    _midpoint_median(p)

# Use the average of the lower and upper medians on even-length inputs
# so neither side strictly contains the threshold value, reducing the
# chance of a degenerate (empty-side) split.
function _midpoint_median(p::AbstractVector)
    n = length(p)
    n == 0 && return zero(eltype(p))
    sorted = sort(collect(p))
    if isodd(n)
        return sorted[(n + 1) >>> 1]
    else
        a = sorted[n >>> 1]
        b = sorted[(n >>> 1) + 1]
        return (a + b) / 2
    end
end

"""
    MeanSplit()

Arithmetic mean of `projections`.
"""
struct MeanSplit <: AbstractSplitValuePolicy end
pick_split_value(::MeanSplit, p::AbstractVector, ::AbstractRNG) =
    isempty(p) ? zero(eltype(p)) : sum(p) / length(p)

"""
    RandomBetweenQuantiles(lo, hi)

Pick the threshold uniformly between the `lo` and `hi` quantiles of
`projections`. Default `(0.4, 0.6)` adds split-randomisation while
keeping balance.
"""
struct RandomBetweenQuantiles <: AbstractSplitValuePolicy
    lo::Float64
    hi::Float64
    function RandomBetweenQuantiles(lo::Real = 0.4, hi::Real = 0.6)
        0.0 <= lo < hi <= 1.0 || throw(ArgumentError("require 0 <= lo < hi <= 1"))
        return new(Float64(lo), Float64(hi))
    end
end

function pick_split_value(policy::RandomBetweenQuantiles, p::AbstractVector, rng::AbstractRNG)
    n = length(p)
    n == 0 && return zero(eltype(p))
    sorted = sort(collect(p))
    lo_idx = clamp(round(Int, policy.lo * (n - 1)) + 1, 1, n)
    hi_idx = clamp(round(Int, policy.hi * (n - 1)) + 1, 1, n)
    lo_val = sorted[lo_idx]
    hi_val = sorted[hi_idx]
    lo_val == hi_val && return lo_val
    u = rand(rng)
    return lo_val + u * (hi_val - lo_val)
end
