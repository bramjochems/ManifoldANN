#=
Neighborhood Selection Strategies

This module provides abstractions for selecting and refining local neighborhoods
for geometry estimation. The neighborhood strategy is orthogonal to the geometry
method - you can combine any strategy with any method.

Strategies:
- FixedNeighborhood: use all provided neighbors as-is
- AdaptiveNeighborhood: shrink by removing points with poor fit
- ExpandingNeighborhood: grow along graph edges until quality degrades
=#

using LinearAlgebra: norm

# ============================================================================
# Abstract type
# ============================================================================

"""
    AbstractNeighborhoodStrategy

Abstract type for neighborhood selection strategies.

Strategies control how the local neighborhood is selected/refined for
geometry fitting. This is orthogonal to the geometry method itself.

See also: [`FixedNeighborhood`](@ref), [`AdaptiveNeighborhood`](@ref),
[`ExpandingNeighborhood`](@ref)
"""
abstract type AbstractNeighborhoodStrategy end

# ============================================================================
# FixedNeighborhood
# ============================================================================

"""
    FixedNeighborhood <: AbstractNeighborhoodStrategy

Use all provided neighbors without any filtering or adaptation.

This is the simplest strategy - just use the k nearest neighbors as given.

# Example
```julia
strategy = FixedNeighborhood()
```
"""
struct FixedNeighborhood <: AbstractNeighborhoodStrategy end

# ============================================================================
# AdaptiveNeighborhood (Shrinking)
# ============================================================================

"""
    AdaptiveNeighborhood <: AbstractNeighborhoodStrategy

Shrinking strategy: iteratively remove neighbors with poor fit quality.

Starting from a pool of candidate neighbors, this strategy:
1. Fits geometry using the current neighbor set
2. Evaluates fit quality using the criterion
3. Removes neighbors with poor fit
4. Repeats until convergence or minimum size reached

# Fields
- `max_neighbors::Int`: Maximum neighbors to consider (candidate pool size)
- `min_neighbors::Int`: Minimum neighbors to retain
- `criterion::AbstractSelectionCriterion`: Quality criterion for filtering
- `shrink_factor::Float64`: Fraction of worst neighbors to remove per iteration
- `max_iterations::Int`: Maximum refinement iterations

# Example
```julia
# Using fit error criterion (default)
strategy = AdaptiveNeighborhood(
    max_neighbors=30,
    min_neighbors=8,
    criterion=FitErrorCriterion(0.1)
)

# Using distortion criterion
strategy = AdaptiveNeighborhood(
    max_neighbors=30,
    min_neighbors=8,
    criterion=DistortionCriterion(0.05)
)
```

See also: [`FixedNeighborhood`](@ref), [`ExpandingNeighborhood`](@ref),
[`FitErrorCriterion`](@ref), [`DistortionCriterion`](@ref)
"""
struct AdaptiveNeighborhood{C<:AbstractSelectionCriterion} <: AbstractNeighborhoodStrategy
    max_neighbors::Int
    min_neighbors::Int
    criterion::C
    shrink_factor::Float64
    max_iterations::Int

    function AdaptiveNeighborhood(;
        max_neighbors::Int=30,
        min_neighbors::Int=5,
        criterion::AbstractSelectionCriterion=FitErrorCriterion(0.1),
        shrink_factor::Float64=0.2,
        max_iterations::Int=10,
        # Legacy parameter for backwards compatibility
        max_error::Union{Float64,Nothing}=nothing
    )
        # Handle legacy max_error parameter
        if max_error !== nothing
            criterion = FitErrorCriterion(max_error)
        end

        if max_neighbors < 2
            throw(ArgumentError("max_neighbors must be >= 2"))
        end
        if min_neighbors < 2
            throw(ArgumentError("min_neighbors must be >= 2"))
        end
        if min_neighbors > max_neighbors
            throw(ArgumentError("min_neighbors must be <= max_neighbors"))
        end
        if !(0.0 < shrink_factor < 1.0)
            throw(ArgumentError("shrink_factor must be in (0, 1)"))
        end
        if max_iterations < 1
            throw(ArgumentError("max_iterations must be >= 1"))
        end
        new{typeof(criterion)}(max_neighbors, min_neighbors, criterion, shrink_factor, max_iterations)
    end
end

# ============================================================================
# ExpandingNeighborhood
# ============================================================================

"""
    ExpandingNeighborhood <: AbstractNeighborhoodStrategy

Expanding strategy: grow neighborhood along graph edges until quality degrades.

Starting from immediate neighbors, this strategy:
1. Fits geometry on current neighbors
2. Evaluates quality using the criterion
3. Expands to include neighbors-of-neighbors
4. Stops when quality degrades beyond threshold

This strategy requires access to a kNN graph to find neighbors-of-neighbors.

# Fields
- `initial_k::Int`: Starting neighborhood size (immediate neighbors)
- `max_neighbors::Int`: Maximum neighbors to grow to
- `criterion::AbstractSelectionCriterion`: Quality criterion for expansion
- `max_shells::Int`: Maximum expansion depth (shells of neighbors)

# Example
```julia
# Expand until distortion exceeds 5%
strategy = ExpandingNeighborhood(
    initial_k=8,
    max_neighbors=50,
    criterion=DistortionCriterion(0.05),
    max_shells=3
)

# Expand until tangent space rotates more than 30°
strategy = ExpandingNeighborhood(
    initial_k=8,
    max_neighbors=50,
    criterion=SubspaceAngleCriterion(π/6),
    max_shells=3
)
```

See also: [`AdaptiveNeighborhood`](@ref), [`DistortionCriterion`](@ref),
[`SubspaceAngleCriterion`](@ref)
"""
struct ExpandingNeighborhood{C<:AbstractSelectionCriterion} <: AbstractNeighborhoodStrategy
    initial_k::Int
    max_neighbors::Int
    criterion::C
    max_shells::Int

    function ExpandingNeighborhood(;
        initial_k::Int=8,
        max_neighbors::Int=50,
        criterion::AbstractSelectionCriterion=DistortionCriterion(0.05),
        max_shells::Int=3
    )
        if initial_k < 2
            throw(ArgumentError("initial_k must be >= 2"))
        end
        if max_neighbors < initial_k
            throw(ArgumentError("max_neighbors must be >= initial_k"))
        end
        if max_shells < 1
            throw(ArgumentError("max_shells must be >= 1"))
        end
        new{typeof(criterion)}(initial_k, max_neighbors, criterion, max_shells)
    end
end

# ============================================================================
# Neighborhood Selection Result
# ============================================================================

"""
    NeighborhoodResult{T}

Result of neighborhood selection, containing the selected neighbors and metadata.

# Fields
- `indices::Vector{Int}`: Selected neighbor indices
- `iterations::Int`: Number of refinement iterations (or shells for expanding)
- `final_quality::T`: Final quality metric value (interpretation depends on criterion)
"""
struct NeighborhoodResult{T<:AbstractFloat}
    indices::Vector{Int}
    iterations::Int
    final_quality::T
end

# For backwards compatibility
final_max_error(r::NeighborhoodResult) = r.final_quality

# ============================================================================
# select_neighbors interface
# ============================================================================

"""
    select_neighbors(strategy::AbstractNeighborhoodStrategy,
                     method::AbstractLocalGeometryMethod,
                     data::AbstractMatrix, center_idx::Int,
                     candidate_indices::AbstractVector{Int};
                     graph=nothing) -> NeighborhoodResult

Select neighbors for geometry fitting using the given strategy.

# Arguments
- `strategy`: Neighborhood selection strategy
- `method`: Geometry method (used for quality evaluation)
- `data`: Data matrix (dimensions × points)
- `center_idx`: Index of the center point
- `candidate_indices`: Initial candidate neighbor indices
- `graph`: Optional kNN graph (required for ExpandingNeighborhood)

# Returns
A `NeighborhoodResult` containing the selected indices and metadata.
"""
function select_neighbors end

# ============================================================================
# FixedNeighborhood implementation
# ============================================================================

function select_neighbors(strategy::FixedNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          center_idx::Int,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    NeighborhoodResult{T}(collect(candidate_indices), 0, zero(T))
end

function select_neighbors(strategy::FixedNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          query_point::AbstractVector,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    NeighborhoodResult{T}(collect(candidate_indices), 0, zero(T))
end

# ============================================================================
# AdaptiveNeighborhood implementation (Shrinking)
# ============================================================================

function select_neighbors(strategy::AdaptiveNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          center_idx::Int,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    center_point = Vector{T}(data[:, center_idx])
    _select_neighbors_shrinking(strategy, method, data, center_point, candidate_indices)
end

function select_neighbors(strategy::AdaptiveNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          query_point::AbstractVector,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    center_point = Vector{T}(query_point)
    _select_neighbors_shrinking(strategy, method, data, center_point, candidate_indices)
end

"""
Internal shrinking neighborhood selection.
"""
function _select_neighbors_shrinking(strategy::AdaptiveNeighborhood,
                                     method::AbstractLocalGeometryMethod,
                                     data::AbstractMatrix{T},
                                     center::Vector{T},
                                     candidate_indices::AbstractVector{Int}) where T
    # Limit to max_neighbors
    n_candidates = min(length(candidate_indices), strategy.max_neighbors)
    active_indices = collect(candidate_indices[1:n_candidates])

    iteration = 0
    final_quality = zero(T)

    for iter in 1:strategy.max_iterations
        iteration = iter
        n_active = length(active_indices)

        if n_active < strategy.min_neighbors
            break
        end

        # Fit geometry with current neighbors
        geom = fit_geometry(method, data, center, active_indices)

        # Evaluate using criterion
        if strategy.criterion isa FitErrorCriterion
            # For fit error, compute per-point errors and remove worst
            errors, distances = _compute_point_errors(geom, data, center, active_indices)
            relative_errors = errors ./ max.(distances, eps(T))
            final_quality = maximum(relative_errors)

            # Check convergence
            if final_quality <= strategy.criterion.max_relative_error
                break
            end

            # Remove worst points
            n_to_remove = max(1, round(Int, strategy.shrink_factor * n_active))
            n_to_keep = max(strategy.min_neighbors, n_active - n_to_remove)

            if n_to_keep >= n_active
                break
            end

            sorted_order = sortperm(relative_errors)
            active_indices = active_indices[sorted_order[1:n_to_keep]]

        elseif strategy.criterion isa DistortionCriterion
            # For distortion, evaluate neighborhood and remove outliers by fit error
            final_quality = evaluate_neighborhood(strategy.criterion, geom, data, center, active_indices)

            if final_quality <= strategy.criterion.max_distortion
                break
            end

            # Use fit error to identify which points to remove
            errors, distances = _compute_point_errors(geom, data, center, active_indices)
            relative_errors = errors ./ max.(distances, eps(T))

            n_to_remove = max(1, round(Int, strategy.shrink_factor * n_active))
            n_to_keep = max(strategy.min_neighbors, n_active - n_to_remove)

            if n_to_keep >= n_active
                break
            end

            sorted_order = sortperm(relative_errors)
            active_indices = active_indices[sorted_order[1:n_to_keep]]
        else
            # For other criteria, use fit error as fallback for point removal
            errors, distances = _compute_point_errors(geom, data, center, active_indices)
            relative_errors = errors ./ max.(distances, eps(T))
            final_quality = maximum(relative_errors)

            n_to_remove = max(1, round(Int, strategy.shrink_factor * n_active))
            n_to_keep = max(strategy.min_neighbors, n_active - n_to_remove)

            if n_to_keep >= n_active
                break
            end

            sorted_order = sortperm(relative_errors)
            active_indices = active_indices[sorted_order[1:n_to_keep]]
        end
    end

    NeighborhoodResult{T}(active_indices, iteration, final_quality)
end

"""
Compute fit errors and distances for all points.
"""
function _compute_point_errors(geom::AbstractLocalGeometry, data::AbstractMatrix{T},
                               center::AbstractVector, indices::AbstractVector{Int}) where T
    n = length(indices)
    errors = Vector{T}(undef, n)
    distances = Vector{T}(undef, n)

    for (i, idx) in enumerate(indices)
        point = @view data[:, idx]
        errors[i] = fit_error(geom, point)
        distances[i] = norm(point - center)
    end

    return errors, distances
end

# ============================================================================
# ExpandingNeighborhood implementation
# ============================================================================

function select_neighbors(strategy::ExpandingNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          center_idx::Int,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    center_point = Vector{T}(data[:, center_idx])

    if graph === nothing
        # Without graph, fall back to using candidates as single shell
        return _select_neighbors_expanding_no_graph(strategy, method, data, center_point,
                                                    candidate_indices)
    end

    _select_neighbors_expanding(strategy, method, data, center_point, center_idx,
                                candidate_indices, graph)
end

function select_neighbors(strategy::ExpandingNeighborhood,
                          method::AbstractLocalGeometryMethod,
                          data::AbstractMatrix{T},
                          query_point::AbstractVector,
                          candidate_indices::AbstractVector{Int};
                          graph=nothing) where T
    # For query points (not in graph), use candidates directly
    center_point = Vector{T}(query_point)
    _select_neighbors_expanding_no_graph(strategy, method, data, center_point, candidate_indices)
end

"""
Expanding neighborhood selection using graph structure.
"""
function _select_neighbors_expanding(strategy::ExpandingNeighborhood,
                                     method::AbstractLocalGeometryMethod,
                                     data::AbstractMatrix{T},
                                     center::Vector{T},
                                     center_idx::Int,
                                     candidate_indices::AbstractVector{Int},
                                     graph) where T
    # Start with initial_k neighbors
    n_initial = min(strategy.initial_k, length(candidate_indices))
    current_indices = Set(candidate_indices[1:n_initial])
    visited = Set{Int}([center_idx])
    union!(visited, current_indices)

    # Fit initial geometry
    indices_vec = collect(current_indices)
    geom = fit_geometry(method, data, center, indices_vec)
    initial_geom = geom  # Keep reference for SubspaceAngleCriterion

    # Compute initial quality
    current_quality = _evaluate_quality(strategy.criterion, geom, initial_geom,
                                        data, center, indices_vec)

    shells_expanded = 0

    for shell in 1:strategy.max_shells
        if length(current_indices) >= strategy.max_neighbors
            break
        end

        # Find neighbors of current neighbors (next shell)
        frontier = Set{Int}()
        for idx in current_indices
            if idx <= length(graph)
                for neighbor in graph[idx]
                    if neighbor ∉ visited && neighbor != center_idx
                        push!(frontier, neighbor)
                    end
                end
            end
        end

        if isempty(frontier)
            break
        end

        # Sort frontier by distance to center
        frontier_vec = collect(frontier)
        frontier_dists = [norm(data[:, idx] - center) for idx in frontier_vec]
        sorted_frontier = frontier_vec[sortperm(frontier_dists)]

        # Add points from frontier up to max_neighbors
        n_to_add = min(length(sorted_frontier), strategy.max_neighbors - length(current_indices))
        new_indices = sorted_frontier[1:n_to_add]

        # Try adding this shell
        test_indices = union(current_indices, Set(new_indices))
        test_indices_vec = collect(test_indices)
        union!(visited, new_indices)

        # Fit geometry with expanded neighborhood
        test_geom = fit_geometry(method, data, center, test_indices_vec)

        # Evaluate quality
        new_quality = _evaluate_quality(strategy.criterion, test_geom, initial_geom,
                                        data, center, test_indices_vec)

        # Check if we should keep expanding
        if _passes_threshold(strategy.criterion, new_quality)
            current_indices = test_indices
            geom = test_geom
            current_quality = new_quality
            shells_expanded = shell
        else
            # Quality degraded, stop expanding
            break
        end
    end

    NeighborhoodResult{T}(collect(current_indices), shells_expanded, current_quality)
end

"""
Expanding selection without graph - use candidates as a single pool.
"""
function _select_neighbors_expanding_no_graph(strategy::ExpandingNeighborhood,
                                              method::AbstractLocalGeometryMethod,
                                              data::AbstractMatrix{T},
                                              center::Vector{T},
                                              candidate_indices::AbstractVector{Int}) where T
    # Without graph, grow from initial_k up to max by adding closest points
    n_candidates = length(candidate_indices)
    n_initial = min(strategy.initial_k, n_candidates)
    current_indices = collect(candidate_indices[1:n_initial])

    # Fit initial geometry
    geom = fit_geometry(method, data, center, current_indices)
    initial_geom = geom
    current_quality = _evaluate_quality(strategy.criterion, geom, initial_geom,
                                        data, center, current_indices)

    iterations = 0
    # Guard against exhausted candidate list
    remaining = n_initial < n_candidates ? collect(candidate_indices[(n_initial+1):end]) : Int[]

    for iter in 1:strategy.max_shells
        if isempty(remaining) || length(current_indices) >= strategy.max_neighbors
            break
        end

        # Add next batch of candidates
        batch_size = min(strategy.initial_k, length(remaining),
                         strategy.max_neighbors - length(current_indices))
        new_points = remaining[1:batch_size]
        remaining = remaining[(batch_size+1):end]

        test_indices = vcat(current_indices, new_points)
        test_geom = fit_geometry(method, data, center, test_indices)

        new_quality = _evaluate_quality(strategy.criterion, test_geom, initial_geom,
                                        data, center, test_indices)

        if _passes_threshold(strategy.criterion, new_quality)
            current_indices = test_indices
            geom = test_geom
            current_quality = new_quality
            iterations = iter
        else
            break
        end
    end

    NeighborhoodResult{T}(current_indices, iterations, current_quality)
end

"""
Evaluate quality using the appropriate criterion.
"""
function _evaluate_quality(criterion::FitErrorCriterion, geom::AbstractLocalGeometry,
                           initial_geom::AbstractLocalGeometry,
                           data::AbstractMatrix{T}, center::AbstractVector,
                           indices::AbstractVector{Int}) where T
    evaluate_neighborhood(criterion, geom, data, center, indices)
end

function _evaluate_quality(criterion::DistortionCriterion, geom::AbstractLocalGeometry,
                           initial_geom::AbstractLocalGeometry,
                           data::AbstractMatrix{T}, center::AbstractVector,
                           indices::AbstractVector{Int}) where T
    evaluate_neighborhood(criterion, geom, data, center, indices)
end

function _evaluate_quality(criterion::SubspaceAngleCriterion, geom::AbstractLocalGeometry,
                           initial_geom::AbstractLocalGeometry,
                           data::AbstractMatrix{T}, center::AbstractVector,
                           indices::AbstractVector{Int}) where T
    compare_geometries(criterion, initial_geom, geom)
end

"""
Check if quality passes threshold.
"""
_passes_threshold(c::FitErrorCriterion, value::Real) = value <= c.max_relative_error
_passes_threshold(c::DistortionCriterion, value::Real) = value <= c.max_distortion
_passes_threshold(c::SubspaceAngleCriterion, value::Real) = value <= c.max_angle

# ============================================================================
# fit_error interface (for quality evaluation)
# ============================================================================

"""
    fit_error(geom::AbstractLocalGeometry, point::AbstractVector) -> Real

Compute the fit error for a point given the fitted geometry.

This measures how well the point fits the local geometry model.
For PCA, this is the reconstruction error (distance to tangent plane).

This function should be implemented by geometry types to enable
adaptive neighborhood selection.
"""
function fit_error(geom::AbstractLocalGeometry, point::AbstractVector)
    error("fit_error not implemented for $(typeof(geom))")
end
