#=
Local Geometry Estimator

Composes a neighborhood selection strategy with a geometry fitting method.
This separation allows mixing any strategy with any method.
=#

"""
    LocalGeometryEstimator{S, M} <: AbstractLocalGeometryMethod

Combines a neighborhood selection strategy with a geometry fitting method.

This allows orthogonal composition of:
- How neighbors are selected/refined (strategy)
- How geometry is fitted from neighbors (method)

# Type Parameters
- `S <: AbstractNeighborhoodStrategy`: Neighborhood selection strategy
- `M <: AbstractLocalGeometryMethod`: Geometry fitting method

# Fields
- `strategy::S`: Neighborhood selection strategy
- `method::M`: Underlying geometry method

# Example
```julia
# Compose adaptive neighborhood with PCA method
strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=8, max_error=0.1)
method = PCAMethod(intrinsic_dim=2)
estimator = LocalGeometryEstimator(strategy, method)

# Use like any geometry method
geom = fit_geometry(estimator, data, center_idx, candidate_indices)
```

See also: [`AdaptiveNeighborhood`](@ref), [`FixedNeighborhood`](@ref), [`PCAMethod`](@ref)
"""
struct LocalGeometryEstimator{S<:AbstractNeighborhoodStrategy, M<:AbstractLocalGeometryMethod} <: AbstractLocalGeometryMethod
    strategy::S
    method::M
end

"""
    EstimatedGeometry{G} <: AbstractLocalGeometry

Wrapper around a fitted geometry that includes neighborhood selection metadata.

# Fields
- `geometry::G`: The underlying fitted geometry
- `used_neighbors::Vector{Int}`: Neighbor indices actually used
- `selection_iterations::Int`: Number of refinement iterations
- `final_error::Real`: Final maximum fit error
"""
struct EstimatedGeometry{G<:AbstractLocalGeometry, T<:AbstractFloat} <: AbstractLocalGeometry
    geometry::G
    used_neighbors::Vector{Int}
    selection_iterations::Int
    final_error::T
end

# ============================================================================
# fit_geometry implementations
# ============================================================================

function fit_geometry(estimator::LocalGeometryEstimator, data::AbstractMatrix{T},
                      center_idx::Int, candidate_indices::AbstractVector{Int};
                      graph=nothing) where T
    # Select neighbors using strategy
    result = select_neighbors(estimator.strategy, estimator.method, data, center_idx,
                              candidate_indices; graph=graph)

    # Fit geometry using selected neighbors
    geom = fit_geometry(estimator.method, data, center_idx, result.indices)

    EstimatedGeometry{typeof(geom), T}(geom, result.indices, result.iterations, result.final_quality)
end

function fit_geometry(estimator::LocalGeometryEstimator, data::AbstractMatrix{T},
                      query_point::AbstractVector, candidate_indices::AbstractVector{Int};
                      graph=nothing) where T
    # Select neighbors using strategy
    result = select_neighbors(estimator.strategy, estimator.method, data, query_point,
                              candidate_indices; graph=graph)

    # Fit geometry using selected neighbors
    geom = fit_geometry(estimator.method, data, query_point, result.indices)

    EstimatedGeometry{typeof(geom), T}(geom, result.indices, result.iterations, result.final_quality)
end

# ============================================================================
# Delegate AbstractLocalGeometry interface to underlying geometry
# ============================================================================

function local_distance(geom::EstimatedGeometry, from::AbstractVector, to::AbstractVector)
    local_distance(geom.geometry, from, to)
end

supports_projection(geom::EstimatedGeometry) = supports_projection(geom.geometry)

function project(geom::EstimatedGeometry, point::AbstractVector)
    project(geom.geometry, point)
end

function reconstruct(geom::EstimatedGeometry, local_coords::AbstractVector)
    reconstruct(geom.geometry, local_coords)
end

intrinsic_dimension(geom::EstimatedGeometry) = intrinsic_dimension(geom.geometry)
center(geom::EstimatedGeometry) = center(geom.geometry)

function explained_variance_ratio(geom::EstimatedGeometry)
    explained_variance_ratio(geom.geometry)
end

total_variance(geom::EstimatedGeometry) = total_variance(geom.geometry)

function fit_error(geom::EstimatedGeometry, point::AbstractVector)
    fit_error(geom.geometry, point)
end

# ============================================================================
# EstimatedGeometry-specific accessors
# ============================================================================

"""
    used_neighbor_count(geom::EstimatedGeometry) -> Int

Return the number of neighbors actually used after selection.
"""
used_neighbor_count(geom::EstimatedGeometry) = length(geom.used_neighbors)

"""
    refinement_iterations(geom::EstimatedGeometry) -> Int

Return the number of neighborhood refinement iterations.
"""
refinement_iterations(geom::EstimatedGeometry) = geom.selection_iterations

"""
    max_reconstruction_error(geom::EstimatedGeometry) -> Real

Return the maximum fit error in the final neighborhood.
"""
max_reconstruction_error(geom::EstimatedGeometry) = geom.final_error

"""
    unwrap_geometry(geom::EstimatedGeometry)

Return the underlying geometry without the estimation wrapper.
"""
unwrap_geometry(geom::EstimatedGeometry) = geom.geometry

"""
    unwrap_geometry(geom::AbstractLocalGeometry)

For non-wrapped geometries, return as-is.
"""
unwrap_geometry(geom::AbstractLocalGeometry) = geom

"""
    recenter(geom::EstimatedGeometry{G,T}, new_center::AbstractVector) -> EstimatedGeometry{G,T}

Create a new EstimatedGeometry with the underlying geometry re-centered.

This preserves the estimation metadata (used_neighbors, iterations, error)
while changing the center point for correct local distance computation.
"""
function recenter(geom::EstimatedGeometry{G,T}, new_center::AbstractVector) where {G,T}
    recentered_inner = recenter(geom.geometry, new_center)
    EstimatedGeometry{typeof(recentered_inner),T}(
        recentered_inner,
        geom.used_neighbors,
        geom.selection_iterations,
        geom.final_error
    )
end
