#=
Per-Edge Weight Abstraction

Single trait covering both
  * ground-cost weights consumed by Ollivier–Ricci curvature on the
    weighted kNN graph (`build_weighted_graph`), and
  * per-edge geodesic-distance estimates consumed by
    `build_geodesic_model` for shortest-path computation.

In both cases the conceptual quantity is the same: a real-valued weight
derived from the data, the local tangent geometry, and an edge `(x, y)`.
This file defines `AbstractEdgeWeight`, the single function
`compute_edge_weight(weight, x_id, y_id, data, geometries) -> Float64`,
and the five concrete subtypes:

  * `EuclideanChord`                  -- the bare Euclidean chord
                                        d_E = ‖y - x‖.
  * `TangentProjectedSourceOnly`      -- d_T^{(x)}, the displacement
                                        projected onto the source-side
                                        tangent space (asymmetric).
  * `TangentProjectedSymmetricMean`   -- (d_T^{(x)} + d_T^{(y)}) / 2
                                        (symmetric).
  * `TangentProjectedSymmetricMax`    -- max(d_T^{(x)}, d_T^{(y)})
                                        (symmetric, conservative).
  * `CurvatureFreeSymmetric`          -- the curvature-free symmetric
                                        estimator d̂_g^sym
                                        = (8 d_E - d_T^{(x)} - d_T^{(y)}) / 6
                                        from Chapter 6 of the thesis
                                        (§6.2.2–§6.2.3, eq. `eq:geod-sym`).
=#

using LinearAlgebra: norm

# ============================================================================
# Abstract type
# ============================================================================

"""
    AbstractEdgeWeight

Abstract type for per-edge weight computations.

A concrete subtype `W <: AbstractEdgeWeight` provides

    compute_edge_weight(weight::W, x_id::Int, y_id::Int,
                        data::AbstractMatrix, geometries) -> Float64

which returns a real-valued weight for the edge `(x_id, y_id)`.
`geometries` is a vector indexed by vertex id whose entries are local
geometry estimates (typically `PCAGeometry` or `EstimatedGeometry`
wrappers); subtypes that do not consume tangent information may pass
`nothing`.

The same trait is consumed by [`build_weighted_graph`](@ref) (where the
weight is interpreted as a ground cost for Ollivier–Ricci curvature) and
by [`build_geodesic_model`](@ref) (where it is interpreted as a per-edge
geodesic-distance estimate).

See also: [`EuclideanChord`](@ref), [`TangentProjectedSourceOnly`](@ref),
[`TangentProjectedSymmetricMean`](@ref),
[`TangentProjectedSymmetricMax`](@ref), [`CurvatureFreeSymmetric`](@ref).
"""
abstract type AbstractEdgeWeight end

# ============================================================================
# Concrete subtypes
# ============================================================================

"""
    EuclideanChord <: AbstractEdgeWeight

Edge weight equal to the ambient Euclidean chord

    d_E = ‖data[:, y_id] - data[:, x_id]‖.

No tangent-space estimates are required; `geometries` may be `nothing`.
This is the default for [`build_geodesic_model`](@ref) and reproduces the
classical Isomap-style edge length.
"""
struct EuclideanChord <: AbstractEdgeWeight end

"""
    TangentProjectedSourceOnly <: AbstractEdgeWeight

Asymmetric edge weight using only the source node's tangent plane:

    weight(i, j) = d_T^{(i)} = ‖Proj_{T_i M}(x_j - x_i)‖
                 = local_distance(geom_i, x_i, x_j).

This is the fastest tangent-aware mode but produces asymmetric weights
(i→j ≠ j→i) when tangent planes at adjacent nodes differ. It is the
default mode of [`build_weighted_graph`](@ref).
"""
struct TangentProjectedSourceOnly <: AbstractEdgeWeight end

"""
    TangentProjectedSymmetricMean <: AbstractEdgeWeight

Symmetric edge weight averaging both endpoints' tangent projections:

    weight(i, j) = (d_T^{(i)} + d_T^{(j)}) / 2,

where d_T^{(c)} = ‖Proj_{T_c M}(x_j - x_i)‖. Robust when tangent planes
differ significantly at adjacent nodes (e.g. high-curvature regions).
"""
struct TangentProjectedSymmetricMean <: AbstractEdgeWeight end

"""
    TangentProjectedSymmetricMax <: AbstractEdgeWeight

Symmetric edge weight taking the maximum of both endpoints' tangent
projections:

    weight(i, j) = max(d_T^{(i)}, d_T^{(j)}).

Conservative: uses the larger projection, which can be more appropriate
when one tangent plane is a poor fit for the edge.
"""
struct TangentProjectedSymmetricMax <: AbstractEdgeWeight end

"""
    CurvatureFreeSymmetric <: AbstractEdgeWeight

Curvature-free symmetric edge weight from Chapter 6 of the thesis
(§6.2.2–§6.2.3, equation `eq:geod-sym`):

    d̂_g^sym = (8 d_E - d_T^{(x)} - d_T^{(y)}) / 6,

where `d_E = ‖y - x‖` is the ambient Euclidean chord and
`d_T^{(\\cdot)} = ‖Proj_{T_\\cdot M}(y - x)‖`. The Taylor expansions of
`d_E` and `d_T` agree to leading order but disagree at the curvature
correction; the linear combination above cancels that correction, giving
an estimator whose error is one order smaller in the edge length than
either constituent.

Requires fitted local geometries at both endpoints.

# Edge-case handling

For sufficiently long edges, high curvature, or noisy tangent estimates
the Taylor expansion is invalid and the formula can produce a negative
value. When that happens the estimator falls back silently to `d_E` *but*
the fallback is counted; [`build_geodesic_model`](@ref) logs a single
warning per pipeline build reporting the count, so that an experimenter
can detect when this estimator is unreliable on their data.
"""
struct CurvatureFreeSymmetric <: AbstractEdgeWeight end

# ============================================================================
# Dispatch interface
# ============================================================================

"""
    compute_edge_weight(weight, x_id, y_id, data, geometries) -> Float64

Compute the per-edge weight between vertices `x_id` and `y_id`. Dispatches
on the type of `weight`. For [`CurvatureFreeSymmetric`](@ref) a negative
raw value is replaced by `d_E`; the caller is responsible for any
diagnostic bookkeeping (see [`build_geodesic_model`](@ref)).
"""
function compute_edge_weight end

function compute_edge_weight(::EuclideanChord, x_id::Int, y_id::Int,
                             data::AbstractMatrix, geometries)
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    return Float64(norm(y - x))
end

function compute_edge_weight(::TangentProjectedSourceOnly, x_id::Int, y_id::Int,
                             data::AbstractMatrix, geometries)
    geometries === nothing && error(
        "TangentProjectedSourceOnly requires per-vertex tangent estimates; got `geometries=nothing`")
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    return Float64(local_distance(geometries[x_id], x, y))
end

function compute_edge_weight(::TangentProjectedSymmetricMean, x_id::Int, y_id::Int,
                             data::AbstractMatrix, geometries)
    geometries === nothing && error(
        "TangentProjectedSymmetricMean requires per-vertex tangent estimates; got `geometries=nothing`")
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    d_x = Float64(local_distance(geometries[x_id], x, y))
    d_y = Float64(local_distance(geometries[y_id], x, y))
    return (d_x + d_y) / 2
end

function compute_edge_weight(::TangentProjectedSymmetricMax, x_id::Int, y_id::Int,
                             data::AbstractMatrix, geometries)
    geometries === nothing && error(
        "TangentProjectedSymmetricMax requires per-vertex tangent estimates; got `geometries=nothing`")
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    d_x = Float64(local_distance(geometries[x_id], x, y))
    d_y = Float64(local_distance(geometries[y_id], x, y))
    return max(d_x, d_y)
end

function compute_edge_weight(::CurvatureFreeSymmetric, x_id::Int, y_id::Int,
                             data::AbstractMatrix, geometries)
    geometries === nothing && error(
        "CurvatureFreeSymmetric requires per-vertex tangent estimates; got `geometries=nothing`")
    raw, d_E, _ = _curvature_free_symmetric_raw(x_id, y_id, data, geometries)
    return raw < 0 ? Float64(d_E) : Float64(raw)
end

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

"""
Compute the tangent-projected chord d_T^{(c)} = ‖Proj_{T_c M}(y - x)‖ for
the geometry stored at the endpoint whose tangent space is `geom`.

We use the projection interface directly (rather than `local_distance`)
so that the result is the projection of the *displacement* `y - x` onto
the tangent basis, regardless of where `geom.center` happens to sit. For
a PCA geometry centered at exactly one of the endpoints this matches
`local_distance(geom, x, y)`, and §6.2 derives all bounds in terms of
displacement projections, so we mirror that convention.
"""
function _tangent_projected_chord(geom, x::AbstractVector, y::AbstractVector)
    inner = unwrap_geometry(geom)
    if supports_projection(inner)
        px = project(inner, x)
        py = project(inner, y)
        return norm(py - px)
    else
        return local_distance(inner, x, y)
    end
end

"""
Compute the raw curvature-free symmetric estimator (without negative-value
fallback). Returns `(raw, d_E, (d_T_x, d_T_y))`.
"""
function _curvature_free_symmetric_raw(x_id::Int, y_id::Int,
                                       data::AbstractMatrix, geometries)
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    d_E = norm(y - x)
    d_T_x = _tangent_projected_chord(geometries[x_id], x, y)
    d_T_y = _tangent_projected_chord(geometries[y_id], x, y)
    raw = (8 * d_E - d_T_x - d_T_y) / 6
    return raw, d_E, (d_T_x, d_T_y)
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    EstimatorDiagnostics

Lightweight container reporting health information for an edge weight
computation after a pipeline build.

# Fields
- `n_negative_fallbacks::Int`: number of edges on which
  [`CurvatureFreeSymmetric`](@ref) produced a negative raw value and fell
  back to the Euclidean chord. Always `0` for weights that cannot
  produce negative values (e.g. [`EuclideanChord`](@ref),
  [`TangentProjectedSymmetricMean`](@ref)).
- `n_edges::Int`: total number of edges scored.
"""
struct EstimatorDiagnostics
    n_negative_fallbacks::Int
    n_edges::Int
end

EstimatorDiagnostics() = EstimatorDiagnostics(0, 0)
