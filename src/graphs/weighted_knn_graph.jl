#=
Weighted kNN Graph

A kNN graph augmented with local geometry at each node and geodesic-aware
edge weights. This is the core data structure for geodesic distance estimation.

The weighted graph extends KNNGraph by:
1. Storing a fitted local geometry (e.g., PCAGeometry) at each node
2. Computing edge weights using local_distance instead of Euclidean distance

Edge weights approximate local geodesic distances by measuring distances
in the tangent space rather than ambient space.

Edge Weight Modes:
- SourceTangent: use only the source node's tangent plane (asymmetric, fast)
- SymmetricMean: average distance from both tangent planes (symmetric)
- SymmetricMax: max of both (conservative, symmetric)

Tangent Sharing Modes:
- NoSharing: each node gets its own tangent plane (default)
- ShareSimilarTangents: nodes with similar geometry share a tangent plane
=#

# ============================================================================
# Edge Weight Modes
# ============================================================================

"""
    AbstractEdgeWeightMode

Abstract type for edge weight computation modes.

Edge weights can be computed using different strategies for handling
the transition between local charts (tangent planes) at adjacent nodes.

See also: [`SourceTangent`](@ref), [`SymmetricMean`](@ref), [`SymmetricMax`](@ref)
"""
abstract type AbstractEdgeWeightMode end

"""
    SourceTangent <: AbstractEdgeWeightMode

Compute edge weight using only the source node's tangent plane.

For edge i → j: weight = local_distance(geom_i, point_i, point_j)

This is the fastest mode but produces asymmetric weights (i→j ≠ j→i).
Asymmetry occurs when tangent planes at adjacent nodes differ significantly.

# Example
```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SourceTangent())
```
"""
struct SourceTangent <: AbstractEdgeWeightMode end

"""
    SymmetricMean <: AbstractEdgeWeightMode

Compute edge weight as the mean of distances from both tangent planes.

For edge i → j: weight = (local_distance(geom_i, p_i, p_j) +
                          local_distance(geom_j, p_i, p_j)) / 2

This produces symmetric weights and is more robust when tangent planes
differ significantly at adjacent nodes (e.g., high curvature regions).

# Example
```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SymmetricMean())
```
"""
struct SymmetricMean <: AbstractEdgeWeightMode end

"""
    SymmetricMax <: AbstractEdgeWeightMode

Compute edge weight as the maximum of distances from both tangent planes.

For edge i → j: weight = max(local_distance(geom_i, p_i, p_j),
                             local_distance(geom_j, p_i, p_j))

This is a conservative estimate - it uses the larger distance, which may
be more appropriate when one tangent plane is a poor fit for the edge.

# Example
```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SymmetricMax())
```
"""
struct SymmetricMax <: AbstractEdgeWeightMode end

# ============================================================================
# Tangent Sharing Modes
# ============================================================================

"""
    AbstractTangentSharingMode

Abstract type for tangent plane sharing strategies.

Determines whether multiple nodes can share the same tangent plane
when their local geometries are sufficiently similar.

See also: [`NoSharing`](@ref), [`ShareSimilarTangents`](@ref)
"""
abstract type AbstractTangentSharingMode end

"""
    NoSharing <: AbstractTangentSharingMode

Each node gets its own independent tangent plane. This is the default behavior.

# Example
```julia
wg = build_weighted_graph(method, index, data; k=15, tangent_sharing=NoSharing())
```
"""
struct NoSharing <: AbstractTangentSharingMode end

"""
    ShareSimilarTangents{C} <: AbstractTangentSharingMode

Nodes with sufficiently similar tangent planes share the same geometry.

When a node's neighbors include a node with an already-fitted tangent plane,
and the two planes are similar according to the criterion, the new node
reuses the existing tangent plane instead of fitting a new one.

# Fields
- `criterion::C`: Criterion for determining similarity (e.g., `SubspaceAngleCriterion`)
- `max_graph_distance::Int`: Only consider sharing with nodes within this graph distance

# Example
```julia
# Share if tangent planes differ by less than 15 degrees
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12), max_graph_distance=2)
wg = build_weighted_graph(method, index, data; k=15, tangent_sharing=sharing)

# Check how many unique tangent planes were fitted
println("Unique tangent planes: \$(unique_geometry_count(wg))")
```

See also: [`SubspaceAngleCriterion`](@ref), [`unique_geometry_count`](@ref)
"""
struct ShareSimilarTangents{C<:AbstractSelectionCriterion} <: AbstractTangentSharingMode
    criterion::C
    max_graph_distance::Int

    function ShareSimilarTangents(criterion::C; max_graph_distance::Int=1) where C<:AbstractSelectionCriterion
        max_graph_distance < 1 && throw(ArgumentError("max_graph_distance must be >= 1"))
        new{C}(criterion, max_graph_distance)
    end
end

"""
    WeightedKNNGraph{T, G}

A kNN graph with local geometry and geodesic-aware edge weights.

# Type Parameters
- `T`: Element type for edge weights (typically `Float64`)
- `G`: Type of local geometry (e.g., `PCAGeometry{Float64}`)

# Fields
- `graph::KNNGraph`: The underlying kNN graph structure
- `geometries::Vector{G}`: Per-node local geometry
- `edge_weights::Vector{Vector{T}}`: Edge weights, `edge_weights[i][j]` is the
  weight from node `i` to its `j`-th neighbor

# Properties
- Length: number of nodes (same as underlying graph)
- Iteration: iterates over neighbor lists (same as underlying graph)

See also: [`build_weighted_graph`](@ref), [`KNNGraph`](@ref)
"""
struct WeightedKNNGraph{T<:AbstractFloat, G<:AbstractLocalGeometry}
    graph::KNNGraph
    geometries::Vector{G}
    edge_weights::Vector{Vector{T}}
end

# ============================================================================
# Builder
# ============================================================================

"""
    build_weighted_graph(method, graph, data; edge_weight_mode=SourceTangent(), tangent_sharing=NoSharing()) -> WeightedKNNGraph

Build a weighted kNN graph by fitting local geometry at each node and
computing geodesic-aware edge weights.

# Arguments
- `method::AbstractLocalGeometryMethod`: Method for fitting local geometry (e.g., `PCAMethod`)
- `graph::KNNGraph`: The kNN graph to augment with weights
- `data::AbstractMatrix`: Data matrix (dimensions × points)
- `edge_weight_mode::AbstractEdgeWeightMode`: How to compute edge weights (default: `SourceTangent()`)
- `tangent_sharing::AbstractTangentSharingMode`: Whether to share tangent planes (default: `NoSharing()`)

# Edge Weight Modes
- `SourceTangent()`: Use source node's tangent plane only (fast, asymmetric)
- `SymmetricMean()`: Average of both tangent planes (symmetric)
- `SymmetricMax()`: Maximum of both tangent planes (conservative, symmetric)

# Tangent Sharing Modes
- `NoSharing()`: Each node gets its own tangent plane (default)
- `ShareSimilarTangents(criterion)`: Nodes share tangent planes if similar

# Returns
A `WeightedKNNGraph` with:
- Local geometry fitted at each node using its neighbors
- Edge weights computed using the specified mode

# Example
```julia
# Build kNN graph from an index
index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=10)

# Add geodesic-aware weights using PCA geometry
method = PCAMethod(intrinsic_dim=2)

# Asymmetric weights (fast)
wg = build_weighted_graph(method, graph, data)

# Symmetric weights (robust for high curvature)
wg = build_weighted_graph(method, graph, data; edge_weight_mode=SymmetricMean())

# Share similar tangent planes
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12))
wg = build_weighted_graph(method, graph, data; tangent_sharing=sharing)
```

See also: [`WeightedKNNGraph`](@ref), [`SourceTangent`](@ref), [`SymmetricMean`](@ref)
"""
function build_weighted_graph(method::AbstractLocalGeometryMethod,
                               graph::KNNGraph,
                               data::AbstractMatrix{T};
                               edge_weight_mode::AbstractEdgeWeightMode=SourceTangent(),
                               tangent_sharing::AbstractTangentSharingMode=NoSharing()) where T
    n = length(graph)

    # Step 1: Fit geometries (with optional sharing)
    geometries = _fit_geometries(tangent_sharing, method, graph, data)

    # Step 2: Compute edge weights using the specified mode
    edge_weights = _compute_edge_weights(edge_weight_mode, graph, geometries, data)

    G = eltype(geometries)
    WeightedKNNGraph{T, G}(graph, geometries, edge_weights)
end

# ============================================================================
# Geometry Fitting (with optional sharing)
# ============================================================================

"""
    _fit_geometries(sharing_mode, method, graph, data) -> Vector{AbstractLocalGeometry}

Fit local geometry at each node, optionally sharing between similar nodes.
"""
function _fit_geometries(::NoSharing, method::AbstractLocalGeometryMethod,
                          graph::KNNGraph, data::AbstractMatrix{T}) where T
    n = length(graph)
    geometries_any = Vector{Any}(undef, n)

    for i in 1:n
        neighbor_indices = graph[i]
        # Pass graph for ExpandingNeighborhood strategies that need to walk neighbor shells
        geom = fit_geometry(method, data, i, neighbor_indices; graph=graph)
        geometries_any[i] = geom
    end

    # Convert to properly typed vector
    G = typeof(geometries_any[1])
    Vector{G}(geometries_any)
end

function _fit_geometries(sharing::ShareSimilarTangents, method::AbstractLocalGeometryMethod,
                          graph::KNNGraph, data::AbstractMatrix{T}) where T
    n = length(graph)
    geometries_any = Any[nothing for _ in 1:n]
    # Track nodes with assigned geometry (fitted or shared) - all can be donors
    assigned_nodes = Int[]

    for i in 1:n
        neighbor_indices = graph[i]

        # Check if we can reuse a neighbor's tangent plane
        shared_geom = _find_shareable_geometry(sharing, geometries_any, assigned_nodes,
                                                 graph, data, i, method)

        if shared_geom !== nothing
            # Reuse existing geometry basis but with correct center for this node
            node_center = @view data[:, i]
            geometries_any[i] = recenter(shared_geom, node_center)
        else
            # Fit new geometry - pass graph for expanding strategies
            geom = fit_geometry(method, data, i, neighbor_indices; graph=graph)
            geometries_any[i] = geom
        end
        # Node can now act as donor for subsequent nodes (whether fitted or shared)
        push!(assigned_nodes, i)
    end

    G = typeof(geometries_any[1])
    Vector{G}(geometries_any)
end

"""
    _find_shareable_geometry(sharing, geometries, assigned_nodes, graph, data, node_idx, method)

Find an existing geometry that can be shared with the given node, or return nothing.
"""
function _find_shareable_geometry(sharing::ShareSimilarTangents, geometries::Vector{Any},
                                   assigned_nodes::Vector{Int}, graph::KNNGraph,
                                   data::AbstractMatrix, node_idx::Int,
                                   method::AbstractLocalGeometryMethod)
    isempty(assigned_nodes) && return nothing

    # Get neighbors within max_graph_distance
    candidates = _get_nearby_assigned_nodes(graph, assigned_nodes, node_idx, sharing.max_graph_distance)
    isempty(candidates) && return nothing

    # Fit a temporary geometry to compare
    neighbor_indices = graph[node_idx]
    temp_geom = fit_geometry(method, data, node_idx, neighbor_indices; graph=graph)
    temp_geom_unwrapped = unwrap_geometry(temp_geom)

    # Check each candidate for similarity
    for candidate_idx in candidates
        candidate_geom = geometries[candidate_idx]
        candidate_geom === nothing && continue

        candidate_unwrapped = unwrap_geometry(candidate_geom)

        # Use compare_geometries from criteria.jl
        if _geometries_are_similar(sharing.criterion, temp_geom_unwrapped, candidate_unwrapped)
            return candidate_geom
        end
    end

    return nothing
end

"""
    _get_nearby_assigned_nodes(graph, assigned_nodes, node_idx, max_distance) -> Vector{Int}

Get nodes with an assigned geometry within max_distance graph hops from node_idx.
"""
function _get_nearby_assigned_nodes(graph::KNNGraph, assigned_nodes::Vector{Int},
                                     node_idx::Int, max_distance::Int)
    # For distance 1, just check immediate neighbors
    if max_distance == 1
        neighbors = Set(graph[node_idx])
        return filter(n -> n in neighbors, assigned_nodes)
    end

    # For larger distances, do BFS
    visited = Set{Int}([node_idx])
    current_layer = Set{Int}([node_idx])

    for _ in 1:max_distance
        next_layer = Set{Int}()
        for n in current_layer
            for neighbor in graph[n]
                if neighbor ∉ visited
                    push!(visited, neighbor)
                    push!(next_layer, neighbor)
                end
            end
        end
        current_layer = next_layer
    end

    delete!(visited, node_idx)
    return filter(n -> n in visited, assigned_nodes)
end

"""
    _geometries_are_similar(criterion, geom1, geom2) -> Bool

Check if two geometries are similar enough to share, based on the criterion.
"""
function _geometries_are_similar(criterion::SubspaceAngleCriterion, geom1, geom2)
    angle = subspace_angle(geom1, geom2)
    return angle <= criterion.max_angle
end

function _geometries_are_similar(criterion::FitErrorCriterion, geom1, geom2)
    # Compare by checking if centers are well-reconstructed by each other
    c1, c2 = center(geom1), center(geom2)
    err1 = fit_error(geom1, c2)
    err2 = fit_error(geom2, c1)
    max_err = max(err1, err2)
    return max_err <= criterion.max_relative_error
end

function _geometries_are_similar(criterion::DistortionCriterion, geom1, geom2)
    # Compare by checking if both tangent planes give similar distance estimates
    # Use the centers as test points
    c1, c2 = center(geom1), center(geom2)

    # Distance between centers as measured by each tangent plane
    d1 = local_distance(geom1, c1, c2)
    d2 = local_distance(geom2, c1, c2)

    # Avoid division by zero for coincident centers
    if d1 < 1e-10 && d2 < 1e-10
        return true
    end

    # Relative distortion: how much do the distance estimates differ?
    max_d = max(d1, d2)
    min_d = min(d1, d2)
    distortion = (max_d - min_d) / max_d

    return distortion <= criterion.max_distortion
end

# ============================================================================
# Edge Weight Computation (separated for clarity)
# ============================================================================

"""
    _compute_edge_weights(mode, graph, geometries, data) -> Vector{Vector{T}}

Compute edge weights for all edges in the graph using the specified mode.
This is separated from geometry fitting for clarity and to enable different modes.
"""
function _compute_edge_weights(::SourceTangent, graph::KNNGraph,
                                geometries::Vector{G}, data::AbstractMatrix{T}) where {G, T}
    n = length(graph)
    edge_weights = Vector{Vector{T}}(undef, n)

    for i in 1:n
        neighbor_indices = graph[i]
        geom_i = geometries[i]
        center_point = @view data[:, i]

        weights = Vector{T}(undef, length(neighbor_indices))
        for (j, neighbor_idx) in enumerate(neighbor_indices)
            neighbor_point = @view data[:, neighbor_idx]
            # Use only source tangent plane
            weights[j] = T(local_distance(geom_i, center_point, neighbor_point))
        end
        edge_weights[i] = weights
    end

    return edge_weights
end

function _compute_edge_weights(::SymmetricMean, graph::KNNGraph,
                                geometries::Vector{G}, data::AbstractMatrix{T}) where {G, T}
    n = length(graph)
    edge_weights = Vector{Vector{T}}(undef, n)

    for i in 1:n
        neighbor_indices = graph[i]
        geom_i = geometries[i]
        center_point = @view data[:, i]

        weights = Vector{T}(undef, length(neighbor_indices))
        for (j, neighbor_idx) in enumerate(neighbor_indices)
            neighbor_point = @view data[:, neighbor_idx]
            geom_j = geometries[neighbor_idx]

            # Average of both tangent plane distances
            d_from_i = local_distance(geom_i, center_point, neighbor_point)
            d_from_j = local_distance(geom_j, center_point, neighbor_point)
            weights[j] = T((d_from_i + d_from_j) / 2)
        end
        edge_weights[i] = weights
    end

    return edge_weights
end

function _compute_edge_weights(::SymmetricMax, graph::KNNGraph,
                                geometries::Vector{G}, data::AbstractMatrix{T}) where {G, T}
    n = length(graph)
    edge_weights = Vector{Vector{T}}(undef, n)

    for i in 1:n
        neighbor_indices = graph[i]
        geom_i = geometries[i]
        center_point = @view data[:, i]

        weights = Vector{T}(undef, length(neighbor_indices))
        for (j, neighbor_idx) in enumerate(neighbor_indices)
            neighbor_point = @view data[:, neighbor_idx]
            geom_j = geometries[neighbor_idx]

            # Maximum of both tangent plane distances (conservative)
            d_from_i = local_distance(geom_i, center_point, neighbor_point)
            d_from_j = local_distance(geom_j, center_point, neighbor_point)
            weights[j] = T(max(d_from_i, d_from_j))
        end
        edge_weights[i] = weights
    end

    return edge_weights
end

"""
    build_weighted_graph(method, index, data; k, candidate_k=nothing, edge_weight_mode=SourceTangent(), tangent_sharing=NoSharing(), kwargs...) -> WeightedKNNGraph

Convenience method that builds both the kNN graph and weighted graph in one call.

# Arguments
- `method::AbstractLocalGeometryMethod`: Method for fitting local geometry
- `index::AbstractANNIndex`: ANN index for neighbor queries
- `data::AbstractMatrix`: Data matrix
- `k::Int`: Number of neighbors for the final kNN graph edges
- `candidate_k::Union{Int,Nothing}`: Number of candidate neighbors for geometry fitting
  (useful for adaptive methods that filter neighbors). Defaults to `k`.
- `edge_weight_mode::AbstractEdgeWeightMode`: How to compute edge weights (default: `SourceTangent()`)
- `tangent_sharing::AbstractTangentSharingMode`: Whether to share tangent planes (default: `NoSharing()`)
- `kwargs...`: Additional arguments passed to `build_knn_graph`

# Example
```julia
index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
weighted_graph = build_weighted_graph(method, index, data; k=10)

# For adaptive methods, use more candidates:
strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=5)
estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))
weighted_graph = build_weighted_graph(estimator, index, data; k=10, candidate_k=30)

# With symmetric edge weights:
weighted_graph = build_weighted_graph(method, index, data; k=10, edge_weight_mode=SymmetricMean())

# Share similar tangent planes:
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12))
weighted_graph = build_weighted_graph(method, index, data; k=10, tangent_sharing=sharing)
```
"""
function build_weighted_graph(method::AbstractLocalGeometryMethod,
                               index::AbstractANNIndex,
                               data::AbstractMatrix;
                               k::Integer,
                               candidate_k::Union{Integer,Nothing}=nothing,
                               edge_weight_mode::AbstractEdgeWeightMode=SourceTangent(),
                               tangent_sharing::AbstractTangentSharingMode=NoSharing(),
                               kwargs...)
    # Build graph with final k
    graph = build_knn_graph(index, data; k=k, kwargs...)

    if candidate_k === nothing || candidate_k <= k
        # Standard case: use graph neighbors directly
        build_weighted_graph(method, graph, data; edge_weight_mode=edge_weight_mode,
                            tangent_sharing=tangent_sharing)
    else
        # Adaptive case: query for more candidates for geometry fitting
        build_weighted_graph_with_candidates(method, graph, index, data, candidate_k;
                                             edge_weight_mode=edge_weight_mode,
                                             tangent_sharing=tangent_sharing)
    end
end

"""
    build_weighted_graph_with_candidates(method, graph, index, data, candidate_k; edge_weight_mode, tangent_sharing)

Build weighted graph where geometry fitting uses more neighbors than graph edges.
This is useful for adaptive methods that filter outliers.
"""
function build_weighted_graph_with_candidates(method::AbstractLocalGeometryMethod,
                                               graph::KNNGraph,
                                               index::AbstractANNIndex,
                                               data::AbstractMatrix{T},
                                               candidate_k::Integer;
                                               edge_weight_mode::AbstractEdgeWeightMode=SourceTangent(),
                                               tangent_sharing::AbstractTangentSharingMode=NoSharing()) where T
    n = length(graph)

    # Step 1: Fit geometry at all nodes using candidate_k neighbors
    geometries = _fit_geometries_with_candidates(tangent_sharing, method, graph, index, data, candidate_k)

    # Step 2: Compute edge weights using the specified mode
    edge_weights = _compute_edge_weights(edge_weight_mode, graph, geometries, data)

    G = eltype(geometries)
    WeightedKNNGraph{T, G}(graph, geometries, edge_weights)
end

"""
    _fit_geometries_with_candidates(sharing, method, graph, index, data, candidate_k)

Fit geometries using candidate_k neighbors (for adaptive methods).
"""
function _fit_geometries_with_candidates(::NoSharing, method::AbstractLocalGeometryMethod,
                                          graph::KNNGraph, index::AbstractANNIndex,
                                          data::AbstractMatrix{T}, candidate_k::Integer) where T
    n = length(graph)
    geometries_any = Vector{Any}(undef, n)

    for i in 1:n
        center_point = @view data[:, i]
        candidates = query(index, data, center_point, candidate_k + 1)

        candidate_indices = [c.id for c in candidates if c.id != i]
        if length(candidate_indices) > candidate_k
            candidate_indices = candidate_indices[1:candidate_k]
        end

        # Pass graph for expanding strategies
        geom = fit_geometry(method, data, i, candidate_indices; graph=graph)
        geometries_any[i] = geom
    end

    G = typeof(geometries_any[1])
    Vector{G}(geometries_any)
end

function _fit_geometries_with_candidates(sharing::ShareSimilarTangents, method::AbstractLocalGeometryMethod,
                                          graph::KNNGraph, index::AbstractANNIndex,
                                          data::AbstractMatrix{T}, candidate_k::Integer) where T
    n = length(graph)
    geometries_any = Any[nothing for _ in 1:n]
    # Track nodes with assigned geometry (fitted or shared) - all can be donors
    assigned_nodes = Int[]

    for i in 1:n
        # Get candidates
        center_point = @view data[:, i]
        candidates = query(index, data, center_point, candidate_k + 1)
        candidate_indices = [c.id for c in candidates if c.id != i]
        if length(candidate_indices) > candidate_k
            candidate_indices = candidate_indices[1:candidate_k]
        end

        # Check if we can reuse a neighbor's tangent plane
        shared_geom = _find_shareable_geometry_candidates(sharing, geometries_any, assigned_nodes,
                                                           graph, data, i, candidate_indices, method)

        if shared_geom !== nothing
            # Reuse existing geometry basis but with correct center for this node
            node_center = @view data[:, i]
            geometries_any[i] = recenter(shared_geom, node_center)
        else
            # Pass graph for expanding strategies
            geom = fit_geometry(method, data, i, candidate_indices; graph=graph)
            geometries_any[i] = geom
        end
        # Node can now act as donor for subsequent nodes (whether fitted or shared)
        push!(assigned_nodes, i)
    end

    G = typeof(geometries_any[1])
    Vector{G}(geometries_any)
end

function _find_shareable_geometry_candidates(sharing::ShareSimilarTangents, geometries::Vector{Any},
                                              assigned_nodes::Vector{Int}, graph::KNNGraph,
                                              data::AbstractMatrix, node_idx::Int,
                                              candidate_indices::Vector{Int},
                                              method::AbstractLocalGeometryMethod)
    isempty(assigned_nodes) && return nothing

    candidates = _get_nearby_assigned_nodes(graph, assigned_nodes, node_idx, sharing.max_graph_distance)
    isempty(candidates) && return nothing

    # Fit temporary geometry with candidate_indices - pass graph for expanding strategies
    temp_geom = fit_geometry(method, data, node_idx, candidate_indices; graph=graph)
    temp_geom_unwrapped = unwrap_geometry(temp_geom)

    for candidate_idx in candidates
        candidate_geom = geometries[candidate_idx]
        candidate_geom === nothing && continue

        candidate_unwrapped = unwrap_geometry(candidate_geom)

        if _geometries_are_similar(sharing.criterion, temp_geom_unwrapped, candidate_unwrapped)
            return candidate_geom
        end
    end

    return nothing
end

# ============================================================================
# Base interface
# ============================================================================

Base.length(wg::WeightedKNNGraph) = length(wg.graph)
Base.getindex(wg::WeightedKNNGraph, i::Integer) = wg.graph[i]
Base.iterate(wg::WeightedKNNGraph) = iterate(wg.graph)
Base.iterate(wg::WeightedKNNGraph, state) = iterate(wg.graph, state)

# ============================================================================
# Accessors
# ============================================================================

"""
    node_geometry(wg::WeightedKNNGraph, i::Int) -> AbstractLocalGeometry

Get the local geometry fitted at node `i`.
"""
node_geometry(wg::WeightedKNNGraph, i::Int) = wg.geometries[i]

"""
    edge_weight(wg::WeightedKNNGraph, from::Int, neighbor_rank::Int) -> Real

Get the edge weight from node `from` to its `neighbor_rank`-th neighbor.

Note: `neighbor_rank` is the position in the neighbor list (1 to k),
not the actual node index of the neighbor.
"""
edge_weight(wg::WeightedKNNGraph, from::Int, neighbor_rank::Int) = wg.edge_weights[from][neighbor_rank]

"""
    neighbors(wg::WeightedKNNGraph, i::Int) -> Vector{Int}

Get the neighbor indices for node `i`.
"""
neighbors(wg::WeightedKNNGraph, i::Int) = wg.graph[i]

"""
    neighbor_weights(wg::WeightedKNNGraph, i::Int) -> Vector

Get the edge weights for all neighbors of node `i`.
"""
neighbor_weights(wg::WeightedKNNGraph, i::Int) = wg.edge_weights[i]

"""
    neighbors_with_weights(wg::WeightedKNNGraph, i::Int)

Get an iterator over (neighbor_index, weight) pairs for node `i`.

# Example
```julia
for (neighbor, weight) in neighbors_with_weights(wg, 1)
    println("Edge to \$neighbor with weight \$weight")
end
```
"""
neighbors_with_weights(wg::WeightedKNNGraph, i::Int) = zip(wg.graph[i], wg.edge_weights[i])

"""
    configured_k(wg::WeightedKNNGraph) -> Int

Get the k value (number of neighbors per node) from the underlying graph.
"""
configured_k(wg::WeightedKNNGraph) = wg.graph.k

# ============================================================================
# Metadata passthrough
# ============================================================================

"""
    has_metadata(wg::WeightedKNNGraph) -> Bool

Check if the underlying graph has metadata.
"""
has_metadata(wg::WeightedKNNGraph) = has_metadata(wg.graph)

"""
    node_metadata(wg::WeightedKNNGraph, i::Int)

Get metadata for node `i` from the underlying graph.
"""
node_metadata(wg::WeightedKNNGraph, i::Int) = node_metadata(wg.graph, i)

"""
    graph_metadata(wg::WeightedKNNGraph)

Get all metadata from the underlying graph.
"""
graph_metadata(wg::WeightedKNNGraph) = graph_metadata(wg.graph)

# ============================================================================
# Utility functions
# ============================================================================

"""
    total_edge_weight(wg::WeightedKNNGraph) -> Real

Compute the sum of all edge weights in the graph.
"""
function total_edge_weight(wg::WeightedKNNGraph)
    total = zero(eltype(wg.edge_weights[1]))
    for weights in wg.edge_weights
        total += sum(weights)
    end
    total
end

"""
    mean_edge_weight(wg::WeightedKNNGraph) -> Real

Compute the mean edge weight across all edges.
"""
function mean_edge_weight(wg::WeightedKNNGraph)
    n_edges = sum(length, wg.edge_weights)
    n_edges > 0 ? total_edge_weight(wg) / n_edges : zero(eltype(wg.edge_weights[1]))
end

"""
    edge_weight_statistics(wg::WeightedKNNGraph) -> NamedTuple

Compute statistics about edge weights.

Returns a named tuple with fields:
- `min`: minimum edge weight
- `max`: maximum edge weight
- `mean`: mean edge weight
- `total`: sum of all edge weights
- `n_edges`: total number of edges
"""
function edge_weight_statistics(wg::WeightedKNNGraph)
    all_weights = reduce(vcat, wg.edge_weights)
    (
        min = minimum(all_weights),
        max = maximum(all_weights),
        mean = mean_edge_weight(wg),
        total = sum(all_weights),
        n_edges = length(all_weights)
    )
end

"""
    unique_geometry_count(wg::WeightedKNNGraph) -> Int

Count the number of unique tangent plane bases in the weighted graph.

When tangent sharing is used, multiple nodes may share the same basis
(tangent directions) while having different centers. This function counts
how many distinct basis matrices exist by checking object identity of
the underlying basis.

# Example
```julia
# Without sharing: should equal number of nodes
wg1 = build_weighted_graph(method, index, data; k=10)
unique_geometry_count(wg1)  # == 500 (one per node)

# With sharing: may be less
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12))
wg2 = build_weighted_graph(method, index, data; k=10, tangent_sharing=sharing)
unique_geometry_count(wg2)  # < 500 if bases were shared
```

See also: [`ShareSimilarTangents`](@ref)
"""
function unique_geometry_count(wg::WeightedKNNGraph)
    # Count unique basis matrices (shared tangent directions)
    # Use objectid of the basis matrix since recenter reuses the same basis reference
    seen = Set{UInt}()
    for geom in wg.geometries
        inner_geom = unwrap_geometry(geom)
        if hasproperty(inner_geom, :basis)
            push!(seen, objectid(inner_geom.basis))
        else
            # Fallback for non-PCA geometries
            push!(seen, objectid(geom))
        end
    end
    length(seen)
end

"""
    geometry_sharing_ratio(wg::WeightedKNNGraph) -> Float64

Compute the ratio of unique tangent planes to total nodes.

Returns a value between 0 and 1:
- 1.0 means every node has its own tangent plane (no sharing)
- Lower values indicate more sharing

# Example
```julia
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/6))
wg = build_weighted_graph(method, index, data; k=10, tangent_sharing=sharing)
ratio = geometry_sharing_ratio(wg)
println("Sharing ratio: \$(round(ratio * 100, digits=1))% unique planes")
```
"""
function geometry_sharing_ratio(wg::WeightedKNNGraph)
    unique_geometry_count(wg) / length(wg)
end
