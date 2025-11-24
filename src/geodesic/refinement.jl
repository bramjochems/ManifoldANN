# Geodesic path refinement methods
#
# This module provides methods for refining discrete shortest paths into
# smoother approximations of geodesic curves.
#
# Design Philosophy:
# ------------------
# Refinement methods serve two distinct purposes:
#
# 1. CURVE REFINEMENT: Creating smooth, dense representations of geodesic paths
#    Examples: SubdivisionSmoothing
#    Output: Many points forming a smooth curve
#    Use case: Visualization, dense sampling along manifold
#
# 2. DISTANCE CORRECTION: Improving distance estimates without changing the curve
#    Examples: CurvatureCorrectedDistance
#    Output: Same waypoints but corrected arc length
#    Use case: Accurate distance computation, second-order corrections
#
# The methods can be composed: CurvatureCorrectedDistance can wrap any other
# refinement method via its `base_refinement` parameter. This allows combining
# smooth curves (for visualization) with accurate distances (for metrics).
#
# Example workflow:
#   1. Get discrete shortest path from Dijkstra
#   2. Refine with SubdivisionSmoothing → smooth curve for plotting
#   3. Optionally wrap with CurvatureCorrectedDistance → corrected distances

"""
    AbstractGeodesicRefinement

Abstract base type for geodesic path refinement methods.

All refinement methods should implement:
- `refine_path(method, model, data, path; kwargs...)` -> RefinedPath
"""
abstract type AbstractGeodesicRefinement end

"""
    RefinedPath{T}

Result of refining a discrete path into a smooth geodesic approximation.

# Fields
- `points::Vector{Vector{T}}`: Dense sequence of points along the curve
- `distance::T`: Total arc length of the refined curve
- `original_path::Vector{Int}`: Original discrete path (node indices)
- `segment_lengths::Vector{T}`: Arc length of each segment between consecutive points
"""
struct RefinedPath{T<:AbstractFloat}
    points::Vector{Vector{T}}
    distance::T
    original_path::Vector{Int}
    segment_lengths::Vector{T}
end

"""
    refine_path(method::AbstractGeodesicRefinement,
                model::GeodesicDistanceModel,
                data::AbstractMatrix,
                path::Vector{Int};
                kwargs...) -> RefinedPath

Refine a discrete shortest path into a smooth geodesic approximation.

# Arguments
- `method`: The refinement method to use
- `model`: Geodesic distance model containing the weighted graph
- `data`: Data matrix (D × n)
- `path`: Vector of node indices forming the discrete path

# Returns
- `RefinedPath` containing the refined curve
"""
function refine_path(method::AbstractGeodesicRefinement,
                     model::GeodesicDistanceModel,
                     data::AbstractMatrix,
                     path::Vector{Int};
                     kwargs...)
    error("refine_path not implemented for $(typeof(method))")
end

# ============================================================================
# NoRefinement - Identity refinement (just returns waypoints)
# ============================================================================

"""
    NoRefinement <: AbstractGeodesicRefinement

Identity refinement that returns the original discrete path points with their
graph edge weights preserved.

This ensures that the returned distance matches the shortest path distance from
Dijkstra, even when the graph uses non-Euclidean edge weights (tangent projection,
adaptive methods, etc.).

Useful as a baseline or when refinement is not needed.
"""
struct NoRefinement <: AbstractGeodesicRefinement end

function refine_path(method::NoRefinement,
                     model::GeodesicDistanceModel,
                     data::AbstractMatrix{T},
                     path::Vector{Int};
                     kwargs...) where T
    # Extract waypoints
    points = [data[:, i] for i in path]

    # Use graph edge weights (not Euclidean) to preserve shortest path distance
    segment_lengths = get_graph_edge_weights(model.weighted_graph, path)
    distance = sum(segment_lengths)

    RefinedPath{T}(points, distance, path, segment_lengths)
end

# ============================================================================
# SubdivisionSmoothing - Iterative averaging with tangent projection
# ============================================================================

"""
    SubdivisionSmoothing <: AbstractGeodesicRefinement

Refines a discrete path by subdividing segments and iteratively smoothing
via averaging and tangent plane projection.

# Algorithm
1. Subdivide each segment into `subdivisions` parts
2. For `max_iterations`:
   a. Move each interior point to the average of its neighbors
   b. Project onto the local tangent plane (interpolated from nearby nodes)
   c. Check convergence
3. Return the smoothed dense curve

# Fields
- `subdivisions::Int`: Number of subdivisions per original segment (default: 5)
- `max_iterations::Int`: Maximum smoothing iterations (default: 20)
- `tolerance::Float64`: Convergence tolerance for point movement (default: 1e-4)
- `damping::Float64`: Damping factor for updates (default: 0.5)

# Example
```julia
method = SubdivisionSmoothing(subdivisions=10, max_iterations=50)
refined = refine_path(method, model, data, path)
```
"""
struct SubdivisionSmoothing <: AbstractGeodesicRefinement
    subdivisions::Int
    max_iterations::Int
    tolerance::Float64
    damping::Float64

    function SubdivisionSmoothing(; subdivisions::Int=5,
                                    max_iterations::Int=20,
                                    tolerance::Float64=1e-4,
                                    damping::Float64=0.5)
        subdivisions > 0 || throw(ArgumentError("subdivisions must be > 0"))
        max_iterations > 0 || throw(ArgumentError("max_iterations must be > 0"))
        0 < tolerance || throw(ArgumentError("tolerance must be > 0"))
        0 < damping <= 1 || throw(ArgumentError("damping must be in (0, 1]"))

        new(subdivisions, max_iterations, tolerance, damping)
    end
end

function refine_path(method::SubdivisionSmoothing,
                     model::GeodesicDistanceModel,
                     data::AbstractMatrix{T},
                     path::Vector{Int};
                     kwargs...) where T
    # Step 1: Subdivide the path
    dense_points = subdivide_path(data, path, method.subdivisions)

    # Step 2: Iteratively smooth via averaging + tangent projection
    dense_points = smooth_path_on_manifold(
        method, model, data, dense_points, path;
        max_iterations=method.max_iterations,
        tolerance=method.tolerance,
        damping=method.damping
    )

    # Step 3: Compute arc length
    segment_lengths = [norm(dense_points[i+1] - dense_points[i]) for i in 1:length(dense_points)-1]
    distance = sum(segment_lengths)

    RefinedPath{T}(dense_points, distance, path, segment_lengths)
end

"""
    subdivide_path(data, path, subdivisions)

Subdivide each segment of the discrete path into `subdivisions` linear segments.
"""
function subdivide_path(data::AbstractMatrix{T},
                        path::Vector{Int},
                        subdivisions::Int) where T
    if length(path) < 2
        return [data[:, i] for i in path]
    end

    dense_points = Vector{Vector{T}}()

    for i in 1:length(path)-1
        p_start = data[:, path[i]]
        p_end = data[:, path[i+1]]

        # Add subdivided points (excluding the end to avoid duplicates)
        for j in 0:subdivisions-1
            t = j / subdivisions
            point = (1 - t) * p_start + t * p_end
            push!(dense_points, point)
        end
    end

    # Add the final point
    push!(dense_points, data[:, path[end]])

    return dense_points
end

"""
    smooth_path_on_manifold(method, model, data, points, original_path; kwargs...)

Iteratively smooth a dense path by averaging neighbors and projecting onto
local tangent planes.
"""
function smooth_path_on_manifold(method::SubdivisionSmoothing,
                                  model::GeodesicDistanceModel,
                                  data::AbstractMatrix{T},
                                  points::Vector{Vector{T}},
                                  original_path::Vector{Int};
                                  max_iterations::Int=20,
                                  tolerance::Float64=1e-4,
                                  damping::Float64=0.5) where T
    n = length(points)
    if n < 3
        return points  # Nothing to smooth
    end

    points = deepcopy(points)  # Don't modify input

    for iteration in 1:max_iterations
        max_movement = zero(T)
        new_points = copy(points)

        # Smooth interior points only (keep endpoints fixed)
        for i in 2:n-1
            # Average of neighbors
            avg = (points[i-1] + points[i+1]) / 2

            # Find local tangent plane (interpolate from nearby graph nodes)
            geom = find_local_geometry(model, data, points[i])

            # Project the averaged point onto the tangent plane centered at current point
            # Strategy: project (avg - current) onto tangent space, then add back
            displacement = avg - points[i]

            if supports_projection(geom)
                # Project displacement onto tangent space
                tangent_displacement = project_to_tangent_space(geom, points[i], displacement)

                # Update with damping
                new_points[i] = points[i] + damping * tangent_displacement
            else
                # Fallback: just use averaging without projection
                new_points[i] = points[i] + damping * displacement
            end

            # Track convergence
            movement = norm(new_points[i] - points[i])
            max_movement = max(max_movement, movement)
        end

        points = new_points

        # Check convergence
        if max_movement < tolerance
            break
        end
    end

    return points
end

"""
    find_local_geometry(model, data, query_point)

Find or interpolate local geometry at a query point.

Uses the nearest graph node's geometry as an approximation.
"""
function find_local_geometry(model::GeodesicDistanceModel,
                              data::AbstractMatrix{T},
                              query_point::AbstractVector{T}) where T
    # Find nearest node in the graph
    wg = model.weighted_graph

    # Simple approach: find nearest among all graph nodes
    nearest_idx = 1
    min_dist = norm(query_point - data[:, 1])

    for i in 2:length(wg)
        d = norm(query_point - data[:, i])
        if d < min_dist
            min_dist = d
            nearest_idx = i
        end
    end

    return node_geometry(wg, nearest_idx)
end

"""
    project_to_tangent_space(geom, center, vector)

Project a displacement vector onto the tangent space at a given center point.

Returns the component of `vector` that lies in the tangent space of `geom`,
but with the tangent space translated to be centered at `center` instead of
`geom.center`.

Works with any geometry that supports projection (PCAGeometry, EstimatedGeometry, etc.)
"""
function project_to_tangent_space(geom::AbstractLocalGeometry,
                                   center::AbstractVector{T},
                                   vector::AbstractVector{T}) where T
    # Unwrap if this is an EstimatedGeometry wrapper
    actual_geom = geom isa ManifoldANN.EstimatedGeometry ? unwrap_geometry(geom) : geom

    # Check if geometry supports projection
    if !supports_projection(actual_geom)
        # Fallback: return the vector unchanged (no projection possible)
        return vector
    end

    # Use the public project/reconstruct interface to project the vector
    # Strategy: project(geom, center + v) gives tangent coords of v
    #           reconstruct(geom, coords) gives center + projected_v
    #           Subtract center to get projected_v

    # Project the vector: shift, project, reconstruct, unshift
    tangent_coords = project(actual_geom, center + vector)
    reconstructed = reconstruct(actual_geom, tangent_coords)
    tangent_component = reconstructed - center

    return tangent_component
end

# ============================================================================
# CurvatureCorrectedDistance - Second-order distance correction
# ============================================================================

"""
    CurvatureCorrectedDistance <: AbstractGeodesicRefinement

Improves distance estimates by incorporating second-order curvature information
from PCA eigenvalues.

This method can be applied to any refined path (composable design):
- Applied directly to discrete path: uses graph edge weights + curvature correction
- Applied after another refinement: corrects the refined path's segment lengths

# Algorithm
For each segment in the (possibly refined) path:
1. Estimate local curvature from nearest graph node's PCA geometry
2. Apply second-order correction to the segment distance:
   `d_corrected ≈ d_segment * (1 + (κ * d_segment)² / 24)`
3. Sum corrected distances

# Fields
- `base_refinement::Union{AbstractGeodesicRefinement,Nothing}`: Optional refinement to apply first
- `length_scale::Union{Float64,Nothing}`: Length scale for curvature (default: auto from mean edge weight)

# Understanding length_scale
PCA eigenvalues give a dimensionless curvature indicator (0-1), but the Taylor correction
requires physical curvature κ (units: 1/length). The length_scale converts between them:

    κ = κ_indicator / length_scale

Default uses mean edge weight, which represents the scale at which tangent planes are fit.
For a sphere of radius R, length_scale ≈ R gives the correct curvature κ = 1/R.

Increase length_scale if corrections seem too large (>10%); decrease if too small (<0.1%).

# Examples
```julia
# Apply directly to discrete path (uses graph edge weights)
method = CurvatureCorrectedDistance()
refined = refine_path(method, model, data, path)

# Compose with subdivision smoothing
method = CurvatureCorrectedDistance(base_refinement=SubdivisionSmoothing(subdivisions=10))
refined = refine_path(method, model, data, path)

# Specify length scale explicitly
method = CurvatureCorrectedDistance(length_scale=1.0)
refined = refine_path(method, model, data, path)
```
"""
struct CurvatureCorrectedDistance <: AbstractGeodesicRefinement
    base_refinement::Union{AbstractGeodesicRefinement,Nothing}
    length_scale::Union{Float64,Nothing}

    function CurvatureCorrectedDistance(;
                                        base_refinement::Union{AbstractGeodesicRefinement,Nothing}=nothing,
                                        length_scale::Union{Float64,Nothing}=nothing)
        if !isnothing(length_scale) && length_scale <= 0
            throw(ArgumentError("length_scale must be positive"))
        end
        new(base_refinement, length_scale)
    end
end

function refine_path(method::CurvatureCorrectedDistance,
                     model::GeodesicDistanceModel,
                     data::AbstractMatrix{T},
                     path::Vector{Int};
                     kwargs...) where T
    wg = model.weighted_graph

    # Step 1: Apply base refinement if specified
    if !isnothing(method.base_refinement)
        base_refined = refine_path(method.base_refinement, model, data, path)
        points = base_refined.points
        base_segments = base_refined.segment_lengths
    else
        # Use discrete path with graph edge weights
        points = [data[:, i] for i in path]
        base_segments = get_graph_edge_weights(wg, path)
    end

    # Step 2: Determine length scale for curvature estimation
    L = if !isnothing(method.length_scale)
        method.length_scale
    else
        # Auto: use mean edge weight from graph
        mean_edge_weight(wg)
    end

    # Step 3: Apply curvature correction to each segment
    n_segments = length(base_segments)
    corrected_lengths = Vector{T}(undef, n_segments)

    for i in 1:n_segments
        d_base = base_segments[i]

        # Find nearest graph node to segment midpoint for geometry
        if i < length(points)
            midpoint = (points[i] + points[i+1]) / 2
            geom = find_local_geometry(model, data, midpoint)

            # Estimate dimensionless curvature indicator
            κ_indicator = estimate_local_curvature(geom)

            # Scale to proper curvature: κ ≈ κ_indicator / L
            κ = κ_indicator / L

            # Second-order Taylor correction: d_geodesic ≈ d * (1 + (κ*d)²/24)
            # This is valid for small κ*d << 1
            correction_factor = 1 + (κ * d_base)^2 / 24

            corrected_lengths[i] = d_base * correction_factor
        else
            corrected_lengths[i] = d_base
        end
    end

    distance = sum(corrected_lengths)

    # Return refined path with corrected distances
    # Note: points come from base refinement (or original waypoints)
    RefinedPath{T}(points, distance, path, corrected_lengths)
end

"""
    get_graph_edge_weights(wg, path)

Extract edge weights from the weighted graph for a given path.
Falls back to mean edge weight if edge not found.
"""
function get_graph_edge_weights(wg::WeightedKNNGraph{T}, path::Vector{Int}) where T
    n = length(path)
    if n < 2
        return T[]
    end

    weights = Vector{T}(undef, n-1)

    for i in 1:n-1
        neighbors_i = neighbors(wg, path[i])
        weights_i = neighbor_weights(wg, path[i])
        neighbor_idx = findfirst(==(path[i+1]), neighbors_i)

        if isnothing(neighbor_idx)
            # Edge not in graph - this shouldn't happen for valid shortest paths
            # Use mean edge weight as estimate
            weights[i] = mean_edge_weight(wg)
        else
            weights[i] = weights_i[neighbor_idx]
        end
    end

    return weights
end

"""
    estimate_local_curvature(geom)

Estimate local curvature indicator (dimensionless, 0-1) from geometry.

Returns a value κ_indicator where:
- 0: Tangent plane fits perfectly (flat locally)
- 1: High eigenvalue spread (curved or noisy)

# Implementations
- `PCAGeometry`: Uses eigenvalue spread `sqrt((λ_max - λ_min) / λ_max)`
- `EstimatedGeometry`: Unwraps and delegates to wrapped geometry
- Other types: Returns 0.0 with a warning (safe fallback, no correction applied)

The returned value must be scaled by `1/length_scale` to get physical curvature κ (units: 1/length).
"""
function estimate_local_curvature(geom::PCAGeometry{T}) where T
    if length(geom.eigenvalues) < 2
        return zero(T)
    end

    # Ratio of smallest to largest variance
    λ_min = minimum(geom.eigenvalues)
    λ_max = maximum(geom.eigenvalues)

    if λ_max < eps(T)
        return zero(T)
    end

    # Curvature estimate: inversely related to how "flat" the tangent plane is
    # If all eigenvalues are similar, the tangent plane fits well (low curvature)
    # If there's high variation, the neighborhood is curved

    # Simple estimate: use the geometric mean of curvature indicators
    # κ ~ sqrt((λ_max - λ_min) / λ_max) / typical_length_scale

    # For now, use a normalized measure
    curvature_indicator = sqrt((λ_max - λ_min) / λ_max)

    return curvature_indicator
end

# Handle EstimatedGeometry wrappers by unwrapping
function estimate_local_curvature(geom::ManifoldANN.EstimatedGeometry)
    return estimate_local_curvature(unwrap_geometry(geom))
end

# Fallback for other geometry types
# Returns 0.0 (no correction) for geometries without curvature estimation
function estimate_local_curvature(geom::AbstractLocalGeometry)
    # Warn once per session about unsupported geometry type
    @warn """
        Curvature estimation not implemented for $(typeof(geom)).
        CurvatureCorrectedDistance will return uncorrected distances (κ=0).
        To enable correction, implement: estimate_local_curvature(::$(typeof(geom)))
        """ _id=:curvature_fallback _group=typeof(geom) maxlog=1
    return 0.0
end
