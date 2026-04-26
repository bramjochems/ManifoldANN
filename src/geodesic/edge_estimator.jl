#=
Per-Edge Geodesic Distance Estimators

This module defines the abstraction for computing per-edge approximations of
the geodesic distance d_g(x, y) between two graph-adjacent vertices.

The estimators here belong to the *geodesic-distance* concern of the package:
they are consumed by `build_geodesic_model` to assign a length to each edge of
the kNN graph before shortest-path search. They are deliberately distinct from
the `AbstractEdgeWeightMode` family in `src/graphs/weighted_knn_graph.jl`,
which lives at the weighted-graph layer and supplies ground costs for
Ollivier–Ricci curvature.

Three estimators are provided:

- `EuclideanChord`: the bare Euclidean chord d_E = ‖y - x‖. No tangent
  estimates required; this is the default and reproduces the behaviour of
  classical Isomap-style pipelines.
- `TangentProjectedSymmetricMean`: the symmetrised tangent-projected length
  d_T = (d_T^{(x)} + d_T^{(y)}) / 2 with d_T^{(x)} = ‖Proj_{T_x M}(y - x)‖.
- `CurvatureFreeSymmetric`: the curvature-free symmetric estimator
  d̂_g^sym = (8 d_E - d_T^{(x)} - d_T^{(y)}) / 6 derived in Chapter 6,
  §6.2.2–§6.2.3 of the thesis (equation `eq:geod-sym`). The leading
  curvature error of d_E and d_T cancels, so this estimator is exact to one
  order higher in the edge length than either constituent.
=#

using LinearAlgebra: norm

# ============================================================================
# Abstract type
# ============================================================================

"""
    AbstractEdgeGeodesicEstimator

Abstract type for per-edge geodesic distance estimators.

A concrete subtype `E <: AbstractEdgeGeodesicEstimator` provides

    compute_edge_distance(estimator::E, x_id::Int, y_id::Int,
                          data::AbstractMatrix, geometries) -> Float64

which returns an approximation of the geodesic distance d_g(x, y) between the
graph-adjacent vertices stored at `data[:, x_id]` and `data[:, y_id]`. The
optional `geometries` argument is a vector indexed by vertex id whose entries
are local-geometry estimates (typically `PCAGeometry` or `EstimatedGeometry`
wrappers). Estimators that do not consume tangent information may ignore it.

Estimators are consumed by [`build_geodesic_model`](@ref) via its
`edge_estimator` keyword argument; they are unrelated to the
[`AbstractEdgeWeightMode`](@ref) family used for ORC ground costs.

See also: [`EuclideanChord`](@ref), [`TangentProjectedSymmetricMean`](@ref),
[`CurvatureFreeSymmetric`](@ref).
"""
abstract type AbstractEdgeGeodesicEstimator end

# ============================================================================
# Concrete estimators
# ============================================================================

"""
    EuclideanChord <: AbstractEdgeGeodesicEstimator

Estimator that returns the ambient Euclidean chord length

    d_E = ‖data[:, y_id] - data[:, x_id]‖.

No tangent-space estimates are required. This is the default estimator used
by [`build_geodesic_model`](@ref) and reproduces the classical Isomap-style
edge length.
"""
struct EuclideanChord <: AbstractEdgeGeodesicEstimator end

"""
    TangentProjectedSymmetricMean <: AbstractEdgeGeodesicEstimator

Estimator that returns the symmetrised tangent-projected chord

    d_T = (d_T^{(x)} + d_T^{(y)}) / 2,

where d_T^{(x)} = ‖Proj_{T_x M}(y - x)‖ is the length of `y - x` projected
onto the tangent space at `x` (and similarly for `y`). Requires
`geometries[x_id]` and `geometries[y_id]` to be fitted local geometries.
"""
struct TangentProjectedSymmetricMean <: AbstractEdgeGeodesicEstimator end

"""
    CurvatureFreeSymmetric <: AbstractEdgeGeodesicEstimator

Curvature-free symmetric edge geodesic estimator from Chapter 6 of the
thesis (§6.2.2–§6.2.3, equation `eq:geod-sym`):

    d̂_g^sym = (8 d_E - d_T^{(x)} - d_T^{(y)}) / 6,

where `d_E = ‖y - x‖` is the ambient Euclidean chord and
`d_T^{(\\cdot)} = ‖Proj_{T_\\cdot M}(y - x)‖` is the chord length projected
onto the tangent plane at the corresponding endpoint. The Taylor expansions
of `d_E` and `d_T` agree to leading order but disagree at the curvature
correction; the linear combination above is constructed to cancel that
correction, giving an estimator whose error is one order smaller in the
edge length than either constituent.

Requires fitted local geometries at both endpoints.

# Edge-case handling

For sufficiently long edges, high curvature, or noisy tangent estimates the
Taylor expansion is invalid and the formula can produce a negative value.
When that happens the estimator falls back silently to `d_E` *but* the
fallback is counted; `build_geodesic_model` logs a single warning per
pipeline build reporting the count, so that an experimenter can detect when
this estimator is unreliable on their data.

See [`build_geodesic_model`](@ref) for how to consume this estimator.
"""
struct CurvatureFreeSymmetric <: AbstractEdgeGeodesicEstimator end

# ============================================================================
# Dispatch interface
# ============================================================================

"""
    compute_edge_distance(estimator, x_id, y_id, data, geometries) -> Float64

Compute the per-edge geodesic distance estimate between vertices `x_id` and
`y_id`. Dispatches on the type of `estimator`.

Returns `Float64`. For [`CurvatureFreeSymmetric`](@ref) a negative raw value
is replaced by `d_E` and the caller is responsible for any diagnostic
bookkeeping; in the standard pipeline that bookkeeping is performed inside
[`build_geodesic_model`](@ref).
"""
function compute_edge_distance end

function compute_edge_distance(::EuclideanChord, x_id::Int, y_id::Int,
                               data::AbstractMatrix, geometries)
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    return Float64(norm(y - x))
end

function compute_edge_distance(::TangentProjectedSymmetricMean, x_id::Int, y_id::Int,
                               data::AbstractMatrix, geometries)
    geometries === nothing && error(
        "TangentProjectedSymmetricMean requires per-vertex tangent estimates; got `geometries=nothing`")
    x = @view data[:, x_id]
    y = @view data[:, y_id]
    geom_x = geometries[x_id]
    geom_y = geometries[y_id]
    d_T_x = _tangent_projected_chord(geom_x, x, y)
    d_T_y = _tangent_projected_chord(geom_y, x, y)
    return Float64((d_T_x + d_T_y) / 2)
end

function compute_edge_distance(::CurvatureFreeSymmetric, x_id::Int, y_id::Int,
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
Compute the tangent-projected chord d_T^{(c)} = ‖Proj_{T_c M}(y - x)‖ for the
geometry stored at the endpoint whose tangent space is `geom`.

We use the projection interface directly (rather than `local_distance`) so
that the result is the projection of the *displacement* `y - x` onto the
tangent basis, regardless of where `geom.center` happens to sit. For a PCA
geometry centered at exactly one of the endpoints this matches
`local_distance(geom, x, y)`, and §6.2 derives all bounds in terms of
displacement projections, so we mirror that convention.
"""
function _tangent_projected_chord(geom, x::AbstractVector, y::AbstractVector)
    inner = unwrap_geometry(geom)
    if supports_projection(inner)
        # basis' * (y - x): projection of the displacement onto T_c M.
        # `project` subtracts `geom.center`, so apply it to both endpoints
        # and take the difference to get the projected displacement.
        px = project(inner, x)
        py = project(inner, y)
        return norm(py - px)
    else
        # Fallback: defer to local_distance with the endpoint as `from`/`to`.
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
    geom_x = geometries[x_id]
    geom_y = geometries[y_id]
    d_T_x = _tangent_projected_chord(geom_x, x, y)
    d_T_y = _tangent_projected_chord(geom_y, x, y)
    raw = (8 * d_E - d_T_x - d_T_y) / 6
    return raw, d_E, (d_T_x, d_T_y)
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    EstimatorDiagnostics

Lightweight container reporting health information for an edge geodesic
estimator after a pipeline build.

# Fields
- `n_negative_fallbacks::Int`: number of edges on which
  [`CurvatureFreeSymmetric`](@ref) produced a negative raw value and fell
  back to the Euclidean chord. Always `0` for estimators that cannot
  produce negative values (e.g. [`EuclideanChord`](@ref),
  [`TangentProjectedSymmetricMean`](@ref)).
- `n_edges::Int`: total number of edges scored.
"""
struct EstimatorDiagnostics
    n_negative_fallbacks::Int
    n_edges::Int
end

EstimatorDiagnostics() = EstimatorDiagnostics(0, 0)
