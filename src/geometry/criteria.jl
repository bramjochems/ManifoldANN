#=
Selection Criteria for Neighborhood Strategies

This module provides pluggable criteria for evaluating neighborhood quality.
Criteria can be used with both shrinking (AdaptiveNeighborhood) and expanding
(ExpandingNeighborhood) strategies.

Criteria types:
- FitErrorCriterion: reconstruction error per point
- DistortionCriterion: pairwise distance preservation
- SubspaceAngleCriterion: tangent space rotation
=#

using LinearAlgebra: norm, svd

# ============================================================================
# Abstract type
# ============================================================================

"""
    AbstractSelectionCriterion

Abstract type for neighborhood selection criteria.

Criteria evaluate the quality of a neighborhood for geometry fitting.
They can be used with various neighborhood strategies (shrinking, expanding).

See also: [`FitErrorCriterion`](@ref), [`DistortionCriterion`](@ref),
[`SubspaceAngleCriterion`](@ref)
"""
abstract type AbstractSelectionCriterion end

# ============================================================================
# FitErrorCriterion
# ============================================================================

"""
    FitErrorCriterion <: AbstractSelectionCriterion

Criterion based on per-point reconstruction error.

A point is considered to fit well if its reconstruction error (distance to
the tangent plane) relative to its distance from the center is below threshold.

# Fields
- `max_relative_error::Float64`: Maximum allowed error/distance ratio

# Example
```julia
criterion = FitErrorCriterion(0.1)  # 10% relative error threshold
```
"""
struct FitErrorCriterion <: AbstractSelectionCriterion
    max_relative_error::Float64

    function FitErrorCriterion(max_relative_error::Float64=0.1)
        if !(0.0 < max_relative_error < 1.0)
            throw(ArgumentError("max_relative_error must be in (0, 1)"))
        end
        new(max_relative_error)
    end
end

"""
    evaluate_point(criterion::FitErrorCriterion, geom, point, center) -> Bool

Check if a single point passes the fit error criterion.
Returns true if the point should be kept.
"""
function evaluate_point(criterion::FitErrorCriterion, geom::AbstractLocalGeometry,
                        point::AbstractVector, center::AbstractVector)
    error = fit_error(geom, point)
    distance = norm(point - center)
    relative_error = error / max(distance, eps(eltype(point)))
    return relative_error <= criterion.max_relative_error
end

"""
    evaluate_neighborhood(criterion::FitErrorCriterion, geom, data, center, indices) -> Float64

Compute the maximum relative error across all points in the neighborhood.
Lower is better. Returns value in [0, 1+] range.
"""
function evaluate_neighborhood(criterion::FitErrorCriterion, geom::AbstractLocalGeometry,
                               data::AbstractMatrix{T}, center::AbstractVector,
                               indices::AbstractVector{Int}) where T
    max_error = zero(T)
    for idx in indices
        point = @view data[:, idx]
        error = fit_error(geom, point)
        distance = norm(point - center)
        relative_error = error / max(distance, eps(T))
        max_error = max(max_error, relative_error)
    end
    return max_error
end

# ============================================================================
# DistortionCriterion
# ============================================================================

"""
    DistortionCriterion <: AbstractSelectionCriterion

Criterion based on pairwise distance preservation.

Measures how well the tangent space projection preserves distances between
points. On a curved manifold, projection onto a flat tangent plane shortens
distances, so distortion increases with curvature.

# Fields
- `max_distortion::Float64`: Maximum allowed average relative distortion

# Distortion measure
For points pᵢ, pⱼ:
- Euclidean distance: dₑ = ||pᵢ - pⱼ||
- Tangent space distance: dₜ = ||project(pᵢ) - project(pⱼ)||
- Distortion: |dₑ - dₜ| / dₑ

The criterion computes the mean distortion over all pairs.

# Example
```julia
criterion = DistortionCriterion(0.05)  # 5% average distortion threshold
```
"""
struct DistortionCriterion <: AbstractSelectionCriterion
    max_distortion::Float64

    function DistortionCriterion(max_distortion::Float64=0.05)
        if !(0.0 < max_distortion < 1.0)
            throw(ArgumentError("max_distortion must be in (0, 1)"))
        end
        new(max_distortion)
    end
end

"""
    evaluate_neighborhood(criterion::DistortionCriterion, geom, data, center, indices) -> Float64

Compute the average pairwise distance distortion.
Lower is better. Returns value in [0, 1] range typically.
"""
function evaluate_neighborhood(criterion::DistortionCriterion, geom::AbstractLocalGeometry,
                               data::AbstractMatrix{T}, center::AbstractVector,
                               indices::AbstractVector{Int}) where T
    n = length(indices)
    if n < 2
        return zero(T)
    end

    # Project all points
    projected = [project(geom, @view data[:, idx]) for idx in indices]

    # Compute pairwise distortion
    total_distortion = zero(T)
    n_pairs = 0

    for i in 1:n
        for j in (i+1):n
            # Euclidean distance
            p_i = @view data[:, indices[i]]
            p_j = @view data[:, indices[j]]
            d_euclidean = norm(p_i - p_j)

            # Tangent space distance
            d_tangent = norm(projected[i] - projected[j])

            # Relative distortion
            if d_euclidean > eps(T)
                distortion = abs(d_euclidean - d_tangent) / d_euclidean
                total_distortion += distortion
                n_pairs += 1
            end
        end
    end

    return n_pairs > 0 ? total_distortion / n_pairs : zero(T)
end

"""
    evaluate_point(criterion::DistortionCriterion, geom, point, center) -> Bool

For distortion criterion, we can't evaluate a single point meaningfully.
Always returns true (use evaluate_neighborhood for full assessment).
"""
function evaluate_point(criterion::DistortionCriterion, geom::AbstractLocalGeometry,
                        point::AbstractVector, center::AbstractVector)
    return true  # Distortion needs multiple points
end

# ============================================================================
# SubspaceAngleCriterion
# ============================================================================

"""
    SubspaceAngleCriterion <: AbstractSelectionCriterion

Criterion based on tangent space rotation (max principal angle).

Measures how much the tangent space has rotated from a reference geometry.
Useful for detecting when expansion has gone too far on a curved manifold.

# Fields
- `max_angle::Float64`: Maximum allowed principal angle in radians

# Principal angles
For two subspaces with orthonormal bases U₁, U₂:
- Compute SVD of U₁ᵀU₂
- Principal angles: θᵢ = arccos(σᵢ)
- Max principal angle: max(θᵢ)

# Example
```julia
criterion = SubspaceAngleCriterion(π/6)  # 30 degrees max rotation
```
"""
struct SubspaceAngleCriterion <: AbstractSelectionCriterion
    max_angle::Float64  # radians

    function SubspaceAngleCriterion(max_angle::Float64=π/6)
        if !(0.0 < max_angle <= π/2)
            throw(ArgumentError("max_angle must be in (0, π/2]"))
        end
        new(max_angle)
    end
end

"""
    subspace_angle(geom1::PCAGeometry, geom2::PCAGeometry) -> Float64

Compute the maximum principal angle between two PCA geometries.
Returns angle in radians [0, π/2].
"""
function subspace_angle(geom1::PCAGeometry, geom2::PCAGeometry)
    # Get orthonormal bases
    U1 = geom1.basis
    U2 = geom2.basis

    # Handle dimension mismatch by padding with zeros or truncating
    d1 = size(U1, 2)
    d2 = size(U2, 2)

    if d1 != d2
        # Use the smaller dimension
        d = min(d1, d2)
        U1 = U1[:, 1:d]
        U2 = U2[:, 1:d]
    end

    # Compute principal angles via SVD of U1' * U2
    M = U1' * U2
    F = svd(M)

    # Singular values are cos(θᵢ), clamp for numerical stability
    cos_angles = clamp.(F.S, -1.0, 1.0)
    angles = acos.(cos_angles)

    return maximum(angles)
end

"""
    evaluate_neighborhood(criterion::SubspaceAngleCriterion, geom, data, center, indices) -> Float64

For subspace angle, we need a reference geometry to compare against.
This returns 0 (no angle) when used without reference.
Use `compare_geometries` for actual angle computation.
"""
function evaluate_neighborhood(criterion::SubspaceAngleCriterion, geom::AbstractLocalGeometry,
                               data::AbstractMatrix{T}, center::AbstractVector,
                               indices::AbstractVector{Int}) where T
    # Without a reference geometry, we can't compute angle
    # Return 0 to indicate "no information"
    return zero(T)
end

"""
    compare_geometries(criterion::SubspaceAngleCriterion, geom1, geom2) -> Float64

Compare two geometries using subspace angle.
Returns the maximum principal angle in radians.
"""
function compare_geometries(criterion::SubspaceAngleCriterion,
                            geom1::AbstractLocalGeometry, geom2::AbstractLocalGeometry)
    # Unwrap if needed
    g1 = unwrap_geometry(geom1)
    g2 = unwrap_geometry(geom2)

    if g1 isa PCAGeometry && g2 isa PCAGeometry
        return subspace_angle(g1, g2)
    else
        error("SubspaceAngleCriterion requires PCAGeometry, got $(typeof(g1)) and $(typeof(g2))")
    end
end

"""
    passes_criterion(criterion::SubspaceAngleCriterion, geom1, geom2) -> Bool

Check if the angle between two geometries is within threshold.
"""
function passes_criterion(criterion::SubspaceAngleCriterion,
                          geom1::AbstractLocalGeometry, geom2::AbstractLocalGeometry)
    angle = compare_geometries(criterion, geom1, geom2)
    return angle <= criterion.max_angle
end

"""
    evaluate_point(criterion::SubspaceAngleCriterion, geom, point, center) -> Bool

For subspace angle criterion, we can't evaluate a single point.
Always returns true (use compare_geometries for angle comparison).
"""
function evaluate_point(criterion::SubspaceAngleCriterion, geom::AbstractLocalGeometry,
                        point::AbstractVector, center::AbstractVector)
    return true  # Angle comparison needs two geometries
end

# ============================================================================
# Utility: check if criterion passes
# ============================================================================

"""
    passes_threshold(criterion::AbstractSelectionCriterion, value::Real) -> Bool

Check if a criterion value passes the threshold.
"""
passes_threshold(c::FitErrorCriterion, value::Real) = value <= c.max_relative_error
passes_threshold(c::DistortionCriterion, value::Real) = value <= c.max_distortion
passes_threshold(c::SubspaceAngleCriterion, value::Real) = value <= c.max_angle
