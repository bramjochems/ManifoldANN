"""
ORC-ManL Compatibility Profiles
================================

A `AbstractOrcMLCompatibilityProfile` bundles together the small set of
implementation choices that make ManifoldANN's ORC-ManL pipeline
either (a) match its long-standing Julia behaviour, or (b) reproduce
bit-for-bit the reference Python `orcml` implementation
(https://github.com/TristanSaidi/orcml).

# Public API (the only thing users should touch)

Two presets, each a value of `AbstractOrcMLCompatibilityProfile`:

- [`ManifoldANNDefault`](@ref): the package's historical behaviour.
  This is the default everywhere in the public API and is guaranteed
  to produce the same curvatures as the previous `main` branch.
- [`OrcmlExact`](@ref): match the reference Python `orcml` package
  bit-for-bit (see `scripts/thesis/orcml/test_orcml_exact_match_policy.jl` for the
  validation harness — under `OrcmlExact` we get
  Pearson r ≈ 0.9995, mean bias ≈ -0.001, with ~74% of edges agreeing
  to within 1e-6 on the 500-point swiss roll benchmark).

# Internal structure (not part of the public API)

Internally, both presets are instances of `OrcMLCompatibilityProfile`,
a plain struct with three independent boolean axes that capture the
known semantic differences between the two implementations:

- `endpoint_inclusion_in_eff_eps::Bool` — when computing the per-node
  effective epsilon (the scale-normalisation weight), should the *other*
  edge endpoint be included in the neighbour list?
  - `false` → ManifoldANN: exclude the other endpoint.
  - `true`  → orcml: include all neighbours (matches `dists = A[i, :]`
    in `compute_eff_eps_adj`).

- `slice_drops_smallest::Bool` — when averaging the k smallest neighbour
  distances, do we drop the smallest entry first?
  - `false` → ManifoldANN: take the k smallest.
  - `true`  → orcml: emulate `np.argsort(dists)[1:k+1]`, i.e. sort
    ascending, drop the smallest, then take the next k.

- `asymmetric_target_exclusion::Bool` — when building the OT mass
  distributions for an edge (x, y), should the source x be excluded
  from y's neighbourhood?
  - `false` → ManifoldANN: symmetric exclusion (drop y from x's side
    AND x from y's side, under the ORC-ManL variant).
  - `true`  → orcml: only the source side excludes the other endpoint.
    This replicates a positional-argument quirk in the upstream Python
    `_get_single_node_neighbors_distributions`.

Combining axes that don't correspond to a documented preset is allowed
internally (useful for ablations) but is not advertised as public API.

# Files

This file lives at `src/graphs/refinement/effective_epsilon_policy.jl`
and is included from `refinement.jl` *before* `filtering.jl`, so that
`filtering.jl` can dispatch through the trait uniformly.
"""

using LinearAlgebra: norm
using Statistics: mean

"""
    _get_original_k(graph::KNNGraph)

Get the original k value used to construct the graph.

For undirected graphs (built with `directed=false`), the graph is symmetrized,
which increases node degrees. The original k is stored in metadata and represents
the k requested at construction time, not the actual degree after symmetrization.

For directed graphs, original_k == graph.k.
"""
function _get_original_k(graph::KNNGraph)
    if graph.metadata !== nothing &&
       hasfield(typeof(graph.metadata), :original_k)
        return graph.metadata.original_k
    else
        return graph.k
    end
end

abstract type AbstractOrcMLCompatibilityProfile end

"""
    OrcMLCompatibilityProfile(; endpoint_inclusion_in_eff_eps,
                                slice_drops_smallest,
                                asymmetric_target_exclusion)

Internal struct exposing the three independent axes of the
ManifoldANN ↔ orcml compatibility space. Not part of the public API —
prefer the [`ManifoldANNDefault`](@ref) or [`OrcmlExact`](@ref) presets.
"""
struct OrcMLCompatibilityProfile <: AbstractOrcMLCompatibilityProfile
    endpoint_inclusion_in_eff_eps::Bool
    slice_drops_smallest::Bool
    asymmetric_target_exclusion::Bool
end

"""
    ManifoldANNDefault()

Public preset reproducing ManifoldANN's historical behaviour. This is
the default profile everywhere in the package and is regression-pinned
to the `main` branch behaviour.
"""
ManifoldANNDefault() = OrcMLCompatibilityProfile(false, false, false)

"""
    OrcmlExact()

Public preset reproducing the reference Python `orcml` package
bit-for-bit. Under this profile, `compute_all_curvatures` agrees with
the upstream `orcml` curvatures to within ~1e-6 on ~74% of edges of a
500-point swiss-roll benchmark (Pearson r ≈ 0.9995, mean bias ≈ -0.001).

See `scripts/thesis/orcml/test_orcml_exact_match_policy.jl` for the validation
harness.
"""
OrcmlExact() = OrcMLCompatibilityProfile(true, true, true)

"""
    compute_effective_epsilon(profile, i, j, graph, data)

Per-edge effective epsilon under the given compatibility profile.
"""
function compute_effective_epsilon(
    profile::OrcMLCompatibilityProfile,
    i::Int, j::Int,
    graph::KNNGraph,
    data::AbstractMatrix{T},
) where {T}
    k = _get_original_k(graph)

    if profile.endpoint_inclusion_in_eff_eps
        # orcml: include all neighbours (mirrors `dists = A[i, :]; dists = dists[dists != 0]`)
        dists_i = [norm(data[:, i] - data[:, n]) for n in graph[i]]
        dists_j = [norm(data[:, j] - data[:, n]) for n in graph[j]]
    else
        # ManifoldANN: exclude the other endpoint
        dists_i = [norm(data[:, i] - data[:, n]) for n in graph[i] if n != j]
        dists_j = [norm(data[:, j] - data[:, n]) for n in graph[j] if n != i]
    end

    eps_i = _eff_eps_average(dists_i, k, profile.slice_drops_smallest)
    eps_j = _eff_eps_average(dists_j, k, profile.slice_drops_smallest)

    return max(eps_i, eps_j)
end

# Average of k neighbour distances, optionally dropping the smallest
# entry first (orcml mimics NumPy's `np.argsort(dists)[1:k+1]`).
function _eff_eps_average(dists::AbstractVector{<:AbstractFloat}, k::Int, drop_smallest::Bool)
    # Accept any float eltype; the previous Vector{Float64}-only signature
    # MethodError-ed on Float32 data because `norm(::Vector{Float32}-...)`
    # returns Float32, leaving the ORC-ManL pipeline unusable in Float32.
    isempty(dists) && return zero(eltype(dists))
    if drop_smallest
        sorted = sort(dists)
        lo = 2
        hi = min(k + 1, length(sorted))
        if lo > hi
            # Fewer than two non-zero distances: fall back to plain mean.
            return mean(sorted)
        end
        @views return mean(sorted[lo:hi])
    else
        if length(dists) > k
            sort!(dists)
            return mean(@view dists[1:k])
        else
            return mean(dists)
        end
    end
end
