#=
Local Geometry Abstraction

This module defines the abstract types and interface for local geometry estimation,
which is used to compute geodesic-aware edge weights in kNN graphs.

Local geometry estimation fits a representation of the manifold's local structure
(e.g., tangent space) at each point, enabling distance computation that respects
the manifold's intrinsic geometry rather than ambient Euclidean distances.
=#

"""
    AbstractLocalGeometryMethod

Abstract type for methods that fit local geometry from neighborhood data.

Implementations must define:
- `fit_geometry(method, data, center_idx, neighbor_indices)` for graph nodes
- `fit_geometry(method, data, query_point, neighbor_indices)` for new points

See also: [`PCAMethod`](@ref), [`AbstractLocalGeometry`](@ref)
"""
abstract type AbstractLocalGeometryMethod end

"""
    AbstractLocalGeometry

Abstract type for fitted local geometry at a point.

Implementations must define:
- `local_distance(geom, from, to)` - compute distance in local coordinates

Optional interface (for path refinement):
- `project(geom, point)` - project point to local coordinates
- `reconstruct(geom, local_coords)` - reconstruct from local coordinates
- `supports_projection(geom)` - capability introspection

See also: [`PCAGeometry`](@ref), [`AbstractLocalGeometryMethod`](@ref)
"""
abstract type AbstractLocalGeometry end

# ============================================================================
# Required Interface
# ============================================================================

"""
    fit_geometry(method::AbstractLocalGeometryMethod, data::AbstractMatrix,
                 center_idx::Int, neighbor_indices::AbstractVector{Int})

Fit local geometry at a graph node using its neighborhood.

# Arguments
- `method`: The geometry fitting method (e.g., `PCAMethod`)
- `data`: Data matrix (dimensions × points)
- `center_idx`: Index of the center point in `data`
- `neighbor_indices`: Indices of neighboring points in `data`

# Returns
An `AbstractLocalGeometry` representing the fitted local structure.
"""
function fit_geometry(method::AbstractLocalGeometryMethod, data::AbstractMatrix,
                      center_idx::Int, neighbor_indices::AbstractVector{Int})
    error("fit_geometry not implemented for $(typeof(method))")
end

"""
    fit_geometry(method::AbstractLocalGeometryMethod, data::AbstractMatrix,
                 query_point::AbstractVector, neighbor_indices::AbstractVector{Int})

Fit local geometry at a new query point (not in the graph) using nearby graph nodes.

# Arguments
- `method`: The geometry fitting method (e.g., `PCAMethod`)
- `data`: Data matrix (dimensions × points)
- `query_point`: The query point coordinates
- `neighbor_indices`: Indices of neighboring points in `data`

# Returns
An `AbstractLocalGeometry` representing the fitted local structure.
"""
function fit_geometry(method::AbstractLocalGeometryMethod, data::AbstractMatrix,
                      query_point::AbstractVector, neighbor_indices::AbstractVector{Int})
    error("fit_geometry not implemented for $(typeof(method))")
end

"""
    local_distance(geom::AbstractLocalGeometry, from::AbstractVector, to::AbstractVector)

Compute the distance between two points using the local geometry.

This typically involves projecting both points onto the local tangent space
and computing the Euclidean distance in that space.

# Arguments
- `geom`: The fitted local geometry
- `from`: Source point coordinates (in ambient space)
- `to`: Target point coordinates (in ambient space)

# Returns
The local distance as a scalar.
"""
function local_distance(geom::AbstractLocalGeometry, from::AbstractVector, to::AbstractVector)
    error("local_distance not implemented for $(typeof(geom))")
end

# ============================================================================
# Optional Interface (for path refinement)
# ============================================================================

"""
    supports_projection(geom::AbstractLocalGeometry) -> Bool

Check if this geometry type supports projection to/from local coordinates.

Returns `true` if `project` and `reconstruct` are implemented.
"""
supports_projection(::AbstractLocalGeometry) = false

"""
    project(geom::AbstractLocalGeometry, point::AbstractVector)

Project a point from ambient space to local coordinates.

Only available if `supports_projection(geom)` returns `true`.

# Arguments
- `geom`: The fitted local geometry
- `point`: Point coordinates in ambient space

# Returns
Local coordinates as a vector (typically lower-dimensional than ambient space).
"""
function project(geom::AbstractLocalGeometry, point::AbstractVector)
    if !supports_projection(geom)
        error("$(typeof(geom)) does not support projection")
    end
    error("project not implemented for $(typeof(geom))")
end

"""
    reconstruct(geom::AbstractLocalGeometry, local_coords::AbstractVector)

Reconstruct a point from local coordinates to ambient space.

Only available if `supports_projection(geom)` returns `true`.

# Arguments
- `geom`: The fitted local geometry
- `local_coords`: Coordinates in the local tangent space

# Returns
Point coordinates in ambient space.
"""
function reconstruct(geom::AbstractLocalGeometry, local_coords::AbstractVector)
    if !supports_projection(geom)
        error("$(typeof(geom)) does not support reconstruction")
    end
    error("reconstruct not implemented for $(typeof(geom))")
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    intrinsic_dimension(geom::AbstractLocalGeometry) -> Int

Return the intrinsic dimension of the local geometry.

For PCA-based geometry, this is the number of principal components retained.
"""
function intrinsic_dimension(geom::AbstractLocalGeometry)
    error("intrinsic_dimension not implemented for $(typeof(geom))")
end

"""
    center(geom::AbstractLocalGeometry)

Return the center point of the local geometry (in ambient coordinates).
"""
function center(geom::AbstractLocalGeometry)
    error("center not implemented for $(typeof(geom))")
end
