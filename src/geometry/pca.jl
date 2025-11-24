#=
PCA-based Local Geometry

This module provides PCA-based tangent space estimation for local geometry.
At each point, PCA is used to find the principal directions of the local
neighborhood, which approximate the tangent space of the manifold.

The local distance between two points is then computed as the Euclidean
distance after projecting both points onto this tangent space.
=#

using LinearAlgebra

"""
    PCAMethod <: AbstractLocalGeometryMethod

Method for fitting local geometry using Principal Component Analysis.

PCA fits a tangent space to the local neighborhood by finding the principal
directions of variation. The number of dimensions to retain can be specified
explicitly or determined automatically based on variance retention.

# Fields
- `intrinsic_dim::Union{Int,Nothing}`: Fixed intrinsic dimension, or `nothing` for auto-detection
- `min_variance_ratio::Float64`: Minimum cumulative variance ratio to retain (for auto-detection)

# Constructor
    PCAMethod(; intrinsic_dim=nothing, min_variance_ratio=0.95)

# Examples
```julia
# Fixed 2D tangent space (for a surface in 3D)
method = PCAMethod(intrinsic_dim=2)

# Auto-detect dimension retaining 95% of variance
method = PCAMethod(min_variance_ratio=0.95)

# Auto-detect dimension retaining 99% of variance
method = PCAMethod(min_variance_ratio=0.99)
```

See also: [`PCAGeometry`](@ref), [`fit_geometry`](@ref)
"""
struct PCAMethod <: AbstractLocalGeometryMethod
    intrinsic_dim::Union{Int,Nothing}
    min_variance_ratio::Float64

    function PCAMethod(; intrinsic_dim::Union{Int,Nothing}=nothing, min_variance_ratio::Float64=0.95)
        if !isnothing(intrinsic_dim) && intrinsic_dim < 1
            throw(ArgumentError("intrinsic_dim must be >= 1, got $intrinsic_dim"))
        end
        if !(0.0 < min_variance_ratio <= 1.0)
            throw(ArgumentError("min_variance_ratio must be in (0, 1], got $min_variance_ratio"))
        end
        new(intrinsic_dim, min_variance_ratio)
    end
end

"""
    PCAGeometry{T} <: AbstractLocalGeometry

Fitted PCA geometry representing a local tangent space.

The tangent space is defined by a center point and an orthonormal basis
of principal directions. Distances are computed by projecting points
onto this tangent space.

# Fields
- `center::Vector{T}`: Center point of the neighborhood (in ambient coordinates)
- `basis::Matrix{T}`: Orthonormal basis matrix (ambient_dim × intrinsic_dim), columns are PCs
- `eigenvalues::Vector{T}`: Eigenvalues (variance along each principal component)

# Properties
- Supports projection: `supports_projection(::PCAGeometry) = true`
- Intrinsic dimension: number of columns in `basis`

See also: [`PCAMethod`](@ref), [`project`](@ref), [`reconstruct`](@ref)
"""
struct PCAGeometry{T<:AbstractFloat} <: AbstractLocalGeometry
    center::Vector{T}
    basis::Matrix{T}
    eigenvalues::Vector{T}
end

# ============================================================================
# fit_geometry implementations
# ============================================================================

"""
    fit_geometry(method::PCAMethod, data::AbstractMatrix{T},
                 center_idx::Int, neighbor_indices::AbstractVector{Int}; graph=nothing) where T

Fit PCA geometry at a graph node.

The center point is taken from `data[:, center_idx]`, and PCA is performed
on the centered neighborhood points.

The `graph` argument is accepted for API compatibility with expanding strategies
but is not used by PCAMethod.
"""
function fit_geometry(method::PCAMethod, data::AbstractMatrix{T},
                      center_idx::Int, neighbor_indices::AbstractVector{Int};
                      graph=nothing) where T
    center_point = Vector{T}(data[:, center_idx])
    neighbor_data = data[:, neighbor_indices]
    _fit_pca_geometry(method, center_point, neighbor_data)
end

"""
    fit_geometry(method::PCAMethod, data::AbstractMatrix{T},
                 query_point::AbstractVector, neighbor_indices::AbstractVector{Int}; graph=nothing) where T

Fit PCA geometry at a query point (not in the graph).

The query point is used as the center, and PCA is performed on the
centered neighborhood points from the graph.

The `graph` argument is accepted for API compatibility with expanding strategies
but is not used by PCAMethod.
"""
function fit_geometry(method::PCAMethod, data::AbstractMatrix{T},
                      query_point::AbstractVector, neighbor_indices::AbstractVector{Int};
                      graph=nothing) where T
    center_point = Vector{T}(query_point)
    neighbor_data = data[:, neighbor_indices]
    _fit_pca_geometry(method, center_point, neighbor_data)
end

"""
Internal function to fit PCA geometry given a center and neighbor data.
"""
function _fit_pca_geometry(method::PCAMethod, center::Vector{T},
                           neighbor_data::AbstractMatrix) where T
    n_neighbors = size(neighbor_data, 2)
    ambient_dim = length(center)

    if n_neighbors < 2
        # Not enough neighbors for meaningful PCA; return trivial geometry
        d = isnothing(method.intrinsic_dim) ? 1 : method.intrinsic_dim
        d = min(d, ambient_dim)
        # Use identity-like basis for the first d dimensions
        basis = zeros(T, ambient_dim, d)
        for i in 1:d
            basis[i, i] = one(T)
        end
        eigenvalues = ones(T, d)
        return PCAGeometry{T}(center, basis, eigenvalues)
    end

    # Center the neighborhood data
    centered = neighbor_data .- center

    # Compute SVD: centered' is (n_neighbors × ambient_dim)
    # We want the right singular vectors (columns of V) which are the PCs
    F = svd(centered')

    # Eigenvalues are S^2 / (n-1) for sample covariance
    # For our purposes, S^2 is proportional and sufficient
    eigenvalues = F.S .^ 2

    # Determine dimension to keep
    d = _determine_dimension(method, eigenvalues, ambient_dim)

    # Extract basis: columns of V corresponding to top d singular values
    # V is ambient_dim × min(n_neighbors, ambient_dim)
    basis = F.V[:, 1:d]

    PCAGeometry{T}(center, Matrix{T}(basis), Vector{T}(eigenvalues[1:d]))
end

"""
Determine the number of dimensions to retain based on method settings.
"""
function _determine_dimension(method::PCAMethod, eigenvalues::AbstractVector,
                              ambient_dim::Int)
    max_d = min(length(eigenvalues), ambient_dim)

    if !isnothing(method.intrinsic_dim)
        # Fixed dimension specified
        return min(method.intrinsic_dim, max_d)
    end

    # Auto-detect based on variance retention
    if isempty(eigenvalues) || all(iszero, eigenvalues)
        return 1
    end

    total_var = sum(eigenvalues)
    if total_var ≈ 0
        return 1
    end

    cumvar = cumsum(eigenvalues) ./ total_var
    d = findfirst(>=(method.min_variance_ratio), cumvar)

    return isnothing(d) ? max_d : d
end

# ============================================================================
# local_distance implementation
# ============================================================================

"""
    local_distance(geom::PCAGeometry, from::AbstractVector, to::AbstractVector)

Compute distance between two points in the local tangent space.

Both points are projected onto the tangent space, and the Euclidean
distance between the projections is returned.
"""
function local_distance(geom::PCAGeometry, from::AbstractVector, to::AbstractVector)
    from_local = project(geom, from)
    to_local = project(geom, to)
    norm(to_local - from_local)
end

# ============================================================================
# Projection interface
# ============================================================================

supports_projection(::PCAGeometry) = true

"""
    project(geom::PCAGeometry, point::AbstractVector)

Project a point from ambient space to local tangent space coordinates.

The projection is: `local_coords = basis' * (point - center)`
"""
function project(geom::PCAGeometry, point::AbstractVector)
    geom.basis' * (point - geom.center)
end

"""
    reconstruct(geom::PCAGeometry, local_coords::AbstractVector)

Reconstruct a point from local tangent space to ambient space.

The reconstruction is: `point = center + basis * local_coords`

Note: This only recovers the component in the tangent space; information
orthogonal to the tangent space is lost.
"""
function reconstruct(geom::PCAGeometry, local_coords::AbstractVector)
    geom.center + geom.basis * local_coords
end

# ============================================================================
# Utility functions
# ============================================================================

"""
    intrinsic_dimension(geom::PCAGeometry) -> Int

Return the intrinsic dimension (number of principal components).
"""
intrinsic_dimension(geom::PCAGeometry) = size(geom.basis, 2)

"""
    center(geom::PCAGeometry)

Return the center point of the local geometry.
"""
center(geom::PCAGeometry) = geom.center

"""
    explained_variance_ratio(geom::PCAGeometry)

Return the proportion of variance explained by each principal component.

Returns a vector where each element is the ratio of that PC's eigenvalue
to the sum of all retained eigenvalues.
"""
function explained_variance_ratio(geom::PCAGeometry)
    total = sum(geom.eigenvalues)
    total ≈ 0 ? fill(1.0 / length(geom.eigenvalues), length(geom.eigenvalues)) : geom.eigenvalues ./ total
end

"""
    total_variance(geom::PCAGeometry)

Return the total variance captured by the retained principal components.
"""
total_variance(geom::PCAGeometry) = sum(geom.eigenvalues)

# ============================================================================
# Fit error (for adaptive neighborhood selection)
# ============================================================================

"""
    fit_error(geom::PCAGeometry, point::AbstractVector) -> Real

Compute the reconstruction error for a point given the PCA geometry.

This is the distance from the point to its projection onto the tangent plane,
i.e., the component orthogonal to the tangent space.

Used by adaptive neighborhood strategies to evaluate fit quality.
"""
function fit_error(geom::PCAGeometry, point::AbstractVector)
    # Vector from center to point
    v = point - geom.center

    # Project onto tangent space and reconstruct
    local_coords = geom.basis' * v
    reconstructed = geom.basis * local_coords

    # Error is the orthogonal component
    norm(v - reconstructed)
end

# ============================================================================
# Re-centering (for tangent plane sharing)
# ============================================================================

"""
    recenter(geom::PCAGeometry{T}, new_center::AbstractVector) -> PCAGeometry{T}

Create a new PCAGeometry with the same basis and eigenvalues but a different center.

This is used for tangent plane sharing: when two nearby nodes have similar tangent
directions, they can share the basis but each needs their own center for correct
local distance computation.

# Arguments
- `geom`: The original geometry to copy the basis from
- `new_center`: The new center point (typically `data[:, node_idx]`)

# Returns
A new `PCAGeometry` with:
- `center = new_center`
- `basis = geom.basis` (same reference, immutable)
- `eigenvalues = geom.eigenvalues` (same reference, immutable)

# Example
```julia
# Share tangent basis between nodes, but with correct centers
shared_basis_geom = recenter(donor_geom, data[:, recipient_idx])
```
"""
function recenter(geom::PCAGeometry{T}, new_center::AbstractVector) where T
    PCAGeometry{T}(Vector{T}(new_center), geom.basis, geom.eigenvalues)
end
