#=
Weighted kNN Graph

A kNN graph augmented with local geometry at each node and geodesic-aware
edge weights. This is the core data structure for geodesic distance estimation.

The weighted graph extends KNNGraph by:
1. Storing a fitted local geometry (e.g., PCAGeometry) at each node
2. Computing edge weights using a per-edge weight rule (`AbstractEdgeWeight`)
   instead of Euclidean distance.

Edge weights approximate local geodesic distances by measuring distances
in the tangent space rather than ambient space.

The set of supported edge-weight rules lives in `src/graphs/edge_weight.jl`
and is shared with `build_geodesic_model`. See [`AbstractEdgeWeight`](@ref).

Tangent Sharing Modes:
- NoSharing: each node gets its own tangent plane (default)
- ShareSimilarTangents: nodes with similar geometry share a tangent plane
=#

using Random: AbstractRNG
import Random

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
    build_weighted_graph(method, graph, data; edge_weight=TangentProjectedSourceOnly(), tangent_sharing=NoSharing(), rng=Random.default_rng()) -> WeightedKNNGraph

Build a weighted kNN graph by fitting local geometry at each node and
computing geodesic-aware edge weights.

# Arguments
- `method::AbstractLocalGeometryMethod`: Method for fitting local geometry (e.g., `PCAMethod`)
- `graph::KNNGraph`: The kNN graph to augment with weights
- `data::AbstractMatrix`: Data matrix (dimensions × points)
- `edge_weight::AbstractEdgeWeight`: How to compute edge weights
  (default: `TangentProjectedSourceOnly()`, preserving the historical
  default of this function before the unification of edge-weight and
  edge-geodesic-estimator abstractions).
- `tangent_sharing::AbstractTangentSharingMode`: Whether to share tangent planes (default: `NoSharing()`)
- `rng::AbstractRNG`: RNG threaded through any stochastic step (e.g.
  randomised ANN queries when invoked via the index-overload below).
  Default `Random.default_rng()`.

# Returns
A `WeightedKNNGraph` with:
- Local geometry fitted at each node using its neighbors
- Edge weights computed using the specified rule via [`compute_edge_weight`](@ref).

# Example
```julia
# Build kNN graph from an index
index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=10)

# Add geodesic-aware weights using PCA geometry
method = PCAMethod(intrinsic_dim=2)

# Asymmetric source-only weights (fast)
wg = build_weighted_graph(method, graph, data)

# Symmetric weights (robust for high curvature)
wg = build_weighted_graph(method, graph, data; edge_weight=TangentProjectedSymmetricMean())

# Share similar tangent planes
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12))
wg = build_weighted_graph(method, graph, data; tangent_sharing=sharing)
```

See also: [`WeightedKNNGraph`](@ref), [`AbstractEdgeWeight`](@ref).
"""
function build_weighted_graph(method::AbstractLocalGeometryMethod,
                               graph::KNNGraph,
                               data::AbstractMatrix{T};
                               edge_weight::AbstractEdgeWeight=TangentProjectedSourceOnly(),
                               tangent_sharing::AbstractTangentSharingMode=NoSharing(),
                               rng::AbstractRNG=Random.default_rng()) where T
    # `rng` is accepted here for API uniformity with the index-overload
    # below; the graph-overload itself has no stochastic step (geometry
    # fitting and edge-weight evaluation are both deterministic given a
    # fixed kNN graph).
    _ = rng

    # Step 1: Fit geometries (with optional sharing)
    geometries = _fit_geometries(tangent_sharing, method, graph, data)

    # Step 2: Compute edge weights, with diagnostic bookkeeping
    edge_weights, _ = _compute_edge_weights_and_diagnostics(edge_weight, graph, geometries, data, T)

    G = eltype(geometries)
    WeightedKNNGraph{T, G}(graph, geometries, edge_weights)
end

"""
Internal: compute edge weights for every edge of `graph` using the
unified [`AbstractEdgeWeight`](@ref) rule. Returns
`(edge_weights, diagnostics::EstimatorDiagnostics)` where `diagnostics`
counts negative-fallback events for [`CurvatureFreeSymmetric`](@ref).
"""
function _compute_edge_weights_and_diagnostics(weight::AbstractEdgeWeight,
                                                graph::KNNGraph,
                                                geometries,
                                                data::AbstractMatrix,
                                                ::Type{T}) where {T}
    n = length(graph)
    edge_weights = Vector{Vector{T}}(undef, n)
    n_neg = 0
    n_edges = 0

    for i in 1:n
        neighbor_indices = graph[i]
        weights = Vector{T}(undef, length(neighbor_indices))
        for (j, neighbor_idx) in enumerate(neighbor_indices)
            n_edges += 1
            val, neg_inc = _scored_edge(weight, i, neighbor_idx, data, geometries)
            weights[j] = T(val)
            n_neg += neg_inc
        end
        edge_weights[i] = weights
    end

    return edge_weights, EstimatorDiagnostics(n_neg, n_edges)
end

# Default scoring path: just call `compute_edge_weight`. Negative-fallback
# bookkeeping is only relevant for `CurvatureFreeSymmetric`, so the generic
# path returns `0` for the increment.
function _scored_edge(weight::AbstractEdgeWeight, x_id::Int, y_id::Int,
                      data::AbstractMatrix, geometries)
    val = compute_edge_weight(weight, x_id, y_id, data, geometries)
    return (val, 0)
end

function _scored_edge(::CurvatureFreeSymmetric, x_id::Int, y_id::Int,
                      data::AbstractMatrix, geometries)
    raw, d_E, _ = _curvature_free_symmetric_raw(x_id, y_id, data, geometries)
    if raw < 0
        return (Float64(d_E), 1)
    else
        return (Float64(raw), 0)
    end
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
    n == 0 && return AbstractLocalGeometry[]

    # Fit node 1 first to infer the concrete geometry type, then allocate a
    # type-stable Vector{G}. Avoids n boxing allocations and a final copy.
    g1 = fit_geometry(method, data, 1, graph[1]; graph=graph)
    geometries = Vector{typeof(g1)}(undef, n)
    geometries[1] = g1

    # Each iteration writes a different slot and reads only immutable inputs
    # (data, graph, method); safe to parallelise over nodes.
    Threads.@threads for i in 2:n
        neighbor_indices = graph[i]
        # Pass graph for ExpandingNeighborhood strategies that need to walk neighbor shells
        geometries[i] = fit_geometry(method, data, i, neighbor_indices; graph=graph)
    end

    return geometries
end

function _fit_geometries(sharing::ShareSimilarTangents, method::AbstractLocalGeometryMethod,
                          graph::KNNGraph, data::AbstractMatrix{T}) where T
    n = length(graph)
    n == 0 && return AbstractLocalGeometry[]

    # Node 1 has no donors so it's always fitted; use it to infer the
    # concrete geometry type and switch to a type-stable
    # Vector{Union{Nothing,G}} for the remaining slots.
    g1 = fit_geometry(method, data, 1, graph[1]; graph=graph)
    G = typeof(g1)
    geometries = fill!(Vector{Union{Nothing,G}}(undef, n), nothing)
    geometries[1] = g1
    assigned_nodes = Int[1]

    for i in 2:n
        neighbor_indices = graph[i]

        shared_geom = _find_shareable_geometry(sharing, geometries, assigned_nodes,
                                                 graph, data, i, method)

        if shared_geom !== nothing
            node_center = @view data[:, i]
            geometries[i] = recenter(shared_geom, node_center)
        else
            geometries[i] = fit_geometry(method, data, i, neighbor_indices; graph=graph)
        end
        push!(assigned_nodes, i)
    end

    return Vector{G}(geometries)
end

"""
    _find_shareable_geometry(sharing, geometries, assigned_nodes, graph, data, node_idx, method)

Find an existing geometry that can be shared with the given node, or return nothing.
"""
function _find_shareable_geometry(sharing::ShareSimilarTangents,
                                   geometries::Vector{Union{Nothing,G}},
                                   assigned_nodes::Vector{Int}, graph::KNNGraph,
                                   data::AbstractMatrix, node_idx::Int,
                                   method::AbstractLocalGeometryMethod) where G
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

"""
    build_weighted_graph(method, index, data; k, candidate_k=nothing, edge_weight=TangentProjectedSourceOnly(), tangent_sharing=NoSharing(), rng=Random.default_rng(), kwargs...) -> WeightedKNNGraph

Convenience method that builds both the kNN graph and weighted graph in one call.

# Arguments
- `method::AbstractLocalGeometryMethod`: Method for fitting local geometry
- `index::AbstractANNIndex`: ANN index for neighbor queries
- `data::AbstractMatrix`: Data matrix
- `k::Int`: Number of neighbors for the final kNN graph edges
- `candidate_k::Union{Int,Nothing}`: Number of candidate neighbors for geometry fitting
  (useful for adaptive methods that filter neighbors). Defaults to `k`.
- `edge_weight::AbstractEdgeWeight`: How to compute edge weights
  (default: `TangentProjectedSourceOnly()`).
- `tangent_sharing::AbstractTangentSharingMode`: Whether to share tangent planes (default: `NoSharing()`)
- `rng::AbstractRNG`: RNG threaded through downstream stochastic steps
  (`build_knn_graph` query paths). Default `Random.default_rng()`.
- `kwargs...`: Additional arguments passed to `build_knn_graph`
"""
function build_weighted_graph(method::AbstractLocalGeometryMethod,
                               index::AbstractANNIndex,
                               data::AbstractMatrix;
                               k::Integer,
                               candidate_k::Union{Integer,Nothing}=nothing,
                               edge_weight::AbstractEdgeWeight=TangentProjectedSourceOnly(),
                               tangent_sharing::AbstractTangentSharingMode=NoSharing(),
                               rng::AbstractRNG=Random.default_rng(),
                               kwargs...)
    # Build graph with final k. `build_knn_graph` forwards `kwargs...` to
    # the per-point `query` call; randomised indices (NNDescent, HNSW)
    # consume an `rng` keyword there, while deterministic indices
    # (BruteForce, KDTree) do not accept any kwargs. We only forward
    # `rng` into the query path when the caller explicitly opts in by
    # also placing `rng` in `kwargs`; the bare `rng` argument otherwise
    # exists so that this builder offers a uniform RNG-aware API and so
    # that *its own* code (the geometry-fit + weight-compute steps) is
    # deterministic given a fixed RNG.
    graph = build_knn_graph(index, data; k=k, kwargs...)
    _ = rng

    if candidate_k === nothing || candidate_k <= k
        # Standard case: use graph neighbors directly
        build_weighted_graph(method, graph, data; edge_weight=edge_weight,
                            tangent_sharing=tangent_sharing, rng=rng)
    else
        # Adaptive case: query for more candidates for geometry fitting
        build_weighted_graph_with_candidates(method, graph, index, data, candidate_k;
                                             edge_weight=edge_weight,
                                             tangent_sharing=tangent_sharing,
                                             rng=rng)
    end
end

"""
    build_weighted_graph_with_candidates(method, graph, index, data, candidate_k; edge_weight, tangent_sharing, rng)

Build weighted graph where geometry fitting uses more neighbors than graph edges.
This is useful for adaptive methods that filter outliers.
"""
function build_weighted_graph_with_candidates(method::AbstractLocalGeometryMethod,
                                               graph::KNNGraph,
                                               index::AbstractANNIndex,
                                               data::AbstractMatrix{T},
                                               candidate_k::Integer;
                                               edge_weight::AbstractEdgeWeight=TangentProjectedSourceOnly(),
                                               tangent_sharing::AbstractTangentSharingMode=NoSharing(),
                                               rng::AbstractRNG=Random.default_rng()) where T
    _ = rng  # currently consumed only by `build_knn_graph` upstream

    # Step 1: Fit geometry at all nodes using candidate_k neighbors
    geometries = _fit_geometries_with_candidates(tangent_sharing, method, graph, index, data, candidate_k)

    # Step 2: Compute edge weights via the unified rule
    edge_weights, _ = _compute_edge_weights_and_diagnostics(edge_weight, graph, geometries, data, T)

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
    n == 0 && return AbstractLocalGeometry[]

    function _candidates(i::Int)
        center_point = @view data[:, i]
        candidates = query(index, data, center_point, candidate_k + 1)
        idxs = [c.id for c in candidates if c.id != i]
        if length(idxs) > candidate_k
            idxs = idxs[1:candidate_k]
        end
        return idxs
    end

    g1 = fit_geometry(method, data, 1, _candidates(1); graph=graph)
    geometries = Vector{typeof(g1)}(undef, n)
    geometries[1] = g1

    # Each iteration writes a different slot; index queries inside _candidates
    # are read-only, geometry fits are independent. Safe to parallelise.
    Threads.@threads for i in 2:n
        # Pass graph for expanding strategies
        geometries[i] = fit_geometry(method, data, i, _candidates(i); graph=graph)
    end

    return geometries
end

function _fit_geometries_with_candidates(sharing::ShareSimilarTangents, method::AbstractLocalGeometryMethod,
                                          graph::KNNGraph, index::AbstractANNIndex,
                                          data::AbstractMatrix{T}, candidate_k::Integer) where T
    n = length(graph)
    n == 0 && return AbstractLocalGeometry[]

    # Node 1 has no donors so it's always fitted; use it to infer G and
    # switch to a type-stable Vector{Union{Nothing,G}}.
    center_1 = @view data[:, 1]
    candidates_1 = query(index, data, center_1, candidate_k + 1)
    candidate_indices_1 = [c.id for c in candidates_1 if c.id != 1]
    if length(candidate_indices_1) > candidate_k
        candidate_indices_1 = candidate_indices_1[1:candidate_k]
    end
    g1 = fit_geometry(method, data, 1, candidate_indices_1; graph=graph)
    G = typeof(g1)
    geometries = fill!(Vector{Union{Nothing,G}}(undef, n), nothing)
    geometries[1] = g1
    assigned_nodes = Int[1]

    for i in 2:n
        center_point = @view data[:, i]
        candidates = query(index, data, center_point, candidate_k + 1)
        candidate_indices = [c.id for c in candidates if c.id != i]
        if length(candidate_indices) > candidate_k
            candidate_indices = candidate_indices[1:candidate_k]
        end

        shared_geom = _find_shareable_geometry_candidates(sharing, geometries, assigned_nodes,
                                                           graph, data, i, candidate_indices, method)

        if shared_geom !== nothing
            node_center = @view data[:, i]
            geometries[i] = recenter(shared_geom, node_center)
        else
            geometries[i] = fit_geometry(method, data, i, candidate_indices; graph=graph)
        end
        push!(assigned_nodes, i)
    end

    return Vector{G}(geometries)
end

function _find_shareable_geometry_candidates(sharing::ShareSimilarTangents,
                                              geometries::Vector{Union{Nothing,G}},
                                              assigned_nodes::Vector{Int}, graph::KNNGraph,
                                              data::AbstractMatrix, node_idx::Int,
                                              candidate_indices::Vector{Int},
                                              method::AbstractLocalGeometryMethod) where G
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
