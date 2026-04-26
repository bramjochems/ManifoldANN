#=
Geodesic Distance Model

The main interface for computing geodesic distances on manifold data.
Combines an ANN index (for fast neighbor queries), a weighted kNN graph
(with local geometries and geodesic edge weights), and the geometry method
(for fitting geometry at new query points).

Geodesic distances are computed as shortest paths on the weighted graph,
where edge weights approximate local geodesic distances using tangent
space projections.
=#

using LinearAlgebra
using Random: AbstractRNG
import Random

# ============================================================================
# GeodesicDistanceModel
# ============================================================================

"""
    GeodesicDistanceModel{I, W, M}

Model for computing geodesic distances on manifold data.

# Type Parameters
- `I <: AbstractANNIndex`: Type of the ANN index for neighbor queries
- `W <: WeightedKNNGraph`: Type of the weighted graph
- `M <: AbstractLocalGeometryMethod`: Type of the geometry fitting method

# Fields
- `index::I`: ANN index for fast neighbor queries on new points
- `weighted_graph::W`: Weighted kNN graph with local geometries
- `method::M`: Geometry method for fitting new points

# Usage
The model supports three types of geodesic distance queries:
1. Between two graph nodes: `geodesic_distance(model, data, i, j)`
2. From a new point to a graph node: `geodesic_distance(model, data, point, j)`
3. Between two new points: `geodesic_distance(model, data, point_a, point_b)`

See also: [`build_geodesic_model`](@ref), [`geodesic_distance`](@ref)
"""
struct GeodesicDistanceModel{I<:AbstractANNIndex, W<:WeightedKNNGraph, M<:AbstractLocalGeometryMethod}
    index::I
    weighted_graph::W
    method::M
    diagnostics::EstimatorDiagnostics
end

# Backwards-compatible inner-type-stable constructor (no diagnostics specified).
function GeodesicDistanceModel(index::AbstractANNIndex, wg::WeightedKNNGraph,
                                method::AbstractLocalGeometryMethod)
    GeodesicDistanceModel(index, wg, method, EstimatorDiagnostics())
end

# ============================================================================
# Builder
# ============================================================================

"""
    build_geodesic_model(method, index, data; k=10, edge_weight=EuclideanChord(), rng=Random.default_rng()) -> GeodesicDistanceModel

Build a geodesic distance model from an ANN index and data.

# Arguments
- `method::AbstractLocalGeometryMethod`: Method for fitting local geometry
- `index::AbstractANNIndex`: ANN index for neighbor queries
- `data::AbstractMatrix`: Data matrix (dimensions × points)
- `k::Int=10`: Number of neighbors for the kNN graph
- `edge_weight::AbstractEdgeWeight=EuclideanChord()`: per-edge weight rule
  used to weight every edge of the kNN graph before Dijkstra. Pass
  [`CurvatureFreeSymmetric`](@ref) to use the curvature-free symmetric
  estimator from Chapter 6 of the thesis (§6.2.2–§6.2.3, equation
  `eq:geod-sym`); see also [`TangentProjectedSymmetricMean`](@ref). The
  default [`EuclideanChord`](@ref) reproduces the classical Isomap-style
  edge length.
- `rng::AbstractRNG`: forwarded to [`build_weighted_graph`](@ref) for
  determinism in randomised ANN paths.

# Returns
A `GeodesicDistanceModel` ready for geodesic distance queries.

When `edge_weight` is [`CurvatureFreeSymmetric`](@ref), edges on which
the raw formula returns a negative value (a sign that the Taylor
expansion is invalid for that edge) are silently replaced by the
Euclidean chord; a single warning summarising the count is emitted at
build time, and the count is exposed through [`diagnostics`](@ref).

# Example
```julia
data = randn(3, 1000)
index = build_index(HNSWIndex, data; M=16, ef_construction=100)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=15)

# Use the curvature-free symmetric estimator from Chapter 6
model_cf = build_geodesic_model(method, index, data; k=15,
                                 edge_weight=CurvatureFreeSymmetric())
```

See also: [`GeodesicDistanceModel`](@ref), [`geodesic_distance`](@ref),
[`AbstractEdgeWeight`](@ref).
"""
function build_geodesic_model(method::AbstractLocalGeometryMethod,
                               index::AbstractANNIndex,
                               data::AbstractMatrix;
                               k::Int=10,
                               edge_weight::AbstractEdgeWeight=EuclideanChord(),
                               rng::AbstractRNG=Random.default_rng())
    # Build the weighted graph in one pass, computing edge weights with
    # the requested unified rule directly (no separate rewrite pass).
    weighted_graph, diag = _build_weighted_graph_with_diagnostics(
        method, index, data; k=k, edge_weight=edge_weight, rng=rng)

    if edge_weight isa CurvatureFreeSymmetric && diag.n_negative_fallbacks > 0
        @warn "CurvatureFreeSymmetric: $(diag.n_negative_fallbacks) of $(diag.n_edges) edges produced a negative raw value and were replaced by the Euclidean chord. This may indicate edges that are too long, high curvature, or noisy tangent estimates."
    end

    GeodesicDistanceModel(index, weighted_graph, method, diag)
end

"""
Internal helper that mirrors [`build_weighted_graph`](@ref) on an index
but additionally returns the [`EstimatorDiagnostics`](@ref) emitted by
the unified weight rule. Kept as a separate path so that
`build_weighted_graph` itself need not return diagnostics through its
public API.
"""
function _build_weighted_graph_with_diagnostics(method::AbstractLocalGeometryMethod,
                                                 index::AbstractANNIndex,
                                                 data::AbstractMatrix{T};
                                                 k::Integer,
                                                 edge_weight::AbstractEdgeWeight,
                                                 rng::AbstractRNG) where {T}
    _ = rng
    graph = build_knn_graph(index, data; k=k)

    geometries = _fit_geometries(NoSharing(), method, graph, data)
    edge_weights, diag = _compute_edge_weights_and_diagnostics(
        edge_weight, graph, geometries, data, T)

    G = eltype(geometries)
    wg = WeightedKNNGraph{T, G}(graph, geometries, edge_weights)
    return wg, diag
end

"""
    diagnostics(model::GeodesicDistanceModel) -> EstimatorDiagnostics

Return the [`EstimatorDiagnostics`](@ref) recorded when the model's edge
weights were assigned. Use this to inspect, e.g., how many edges the
[`CurvatureFreeSymmetric`](@ref) estimator had to replace with the
Euclidean chord because its raw value was negative.
"""
diagnostics(model::GeodesicDistanceModel) = model.diagnostics

# ============================================================================
# Dijkstra's Algorithm (internal)
# ============================================================================

"""
Internal priority queue implementation using a binary heap.
Uses (distance, node) pairs sorted by distance.
"""
struct MinHeap{T}
    data::Vector{Tuple{T, Int}}
end

MinHeap{T}() where T = MinHeap{T}(Tuple{T, Int}[])
Base.isempty(h::MinHeap) = isempty(h.data)
Base.length(h::MinHeap) = length(h.data)

function heap_push!(h::MinHeap{T}, dist::T, node::Int) where T
    push!(h.data, (dist, node))
    _sift_up!(h, length(h.data))
end

function heap_pop!(h::MinHeap)
    isempty(h.data) && error("Heap is empty")
    result = h.data[1]
    if length(h.data) > 1
        h.data[1] = pop!(h.data)
        _sift_down!(h, 1)
    else
        pop!(h.data)
    end
    return result
end

function _sift_up!(h::MinHeap, idx::Int)
    while idx > 1
        parent = idx ÷ 2
        if h.data[idx][1] < h.data[parent][1]
            h.data[idx], h.data[parent] = h.data[parent], h.data[idx]
            idx = parent
        else
            break
        end
    end
end

function _sift_down!(h::MinHeap, idx::Int)
    n = length(h.data)
    while true
        smallest = idx
        left = 2 * idx
        right = 2 * idx + 1

        if left <= n && h.data[left][1] < h.data[smallest][1]
            smallest = left
        end
        if right <= n && h.data[right][1] < h.data[smallest][1]
            smallest = right
        end

        if smallest != idx
            h.data[idx], h.data[smallest] = h.data[smallest], h.data[idx]
            idx = smallest
        else
            break
        end
    end
end

"""
Run Dijkstra's algorithm from source node, optionally stopping at target.

Returns (distances, predecessors) where:
- distances[i] = shortest path distance from source to node i
- predecessors[i] = previous node on shortest path (-1 if unreachable or source)
"""
function _dijkstra(wg::WeightedKNNGraph, source::Int; target::Union{Int,Nothing}=nothing)
    n = length(wg)
    T = eltype(wg.edge_weights[1])

    # Use Inf for unreachable nodes (not typemax which gives ~1.8e308)
    dist = fill(T(Inf), n)
    prev = fill(-1, n)
    visited = falses(n)

    dist[source] = zero(T)
    heap = MinHeap{T}()
    heap_push!(heap, zero(T), source)

    while !isempty(heap)
        d, u = heap_pop!(heap)

        # Skip if already visited (we may have duplicate entries)
        visited[u] && continue
        visited[u] = true

        # Early termination if we reached target
        if target !== nothing && u == target
            break
        end

        # Relax edges
        for (v, w) in neighbors_with_weights(wg, u)
            if !visited[v]
                alt = dist[u] + w
                if alt < dist[v]
                    dist[v] = alt
                    prev[v] = u
                    heap_push!(heap, alt, v)
                end
            end
        end
    end

    return dist, prev
end

"""
Reconstruct path from predecessors array.
"""
function _reconstruct_path(prev::Vector{Int}, source::Int, target::Int)
    path = Int[]
    current = target

    # Check if target is reachable
    if prev[target] == -1 && target != source
        return path  # Empty path means unreachable
    end

    while current != -1
        pushfirst!(path, current)
        current = current == source ? -1 : prev[current]
    end

    return path
end

# ============================================================================
# Geodesic Distance: Between Graph Nodes
# ============================================================================

"""
    geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix, i::Int, j::Int)

Compute the geodesic distance between two graph nodes.

Uses Dijkstra's algorithm to find the shortest path on the weighted graph.

# Arguments
- `model`: The geodesic distance model
- `data`: Data matrix (dimensions × points)
- `i`: Source node index
- `j`: Target node index

# Returns
The geodesic distance as a scalar. Returns `Inf` if no path exists.

# Example
```julia
d = geodesic_distance(model, data, 1, 100)
```
"""
function geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix,
                           i::Int, j::Int)
    dist, _ = _dijkstra(model.weighted_graph, i; target=j)
    return dist[j]
end

"""
    shortest_path_with_path(model::GeodesicDistanceModel, data::AbstractMatrix, i::Int, j::Int)

Compute the shortest path between two graph nodes and return both distance and path.

# Arguments
- `model`: The geodesic distance model
- `data`: Data matrix (dimensions × points)
- `i`: Source node index
- `j`: Target node index

# Returns
A named tuple with fields:
- `distance`: The geodesic distance
- `path`: Vector of node indices from source to target

# Example
```julia
result = shortest_path_with_path(model, data, 1, 100)
println("Distance: \$(result.distance)")
println("Path: \$(result.path)")
```
"""
function shortest_path_with_path(model::GeodesicDistanceModel, data::AbstractMatrix,
                                  i::Int, j::Int)
    dist, prev = _dijkstra(model.weighted_graph, i; target=j)
    path = _reconstruct_path(prev, i, j)
    return (distance=dist[j], path=path)
end

# ============================================================================
# Geodesic Distance: From New Point to Graph Node
# ============================================================================

"""
    geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix,
                      point::AbstractVector, j::Int; entry_k::Int=5)

Compute the geodesic distance from a new point (not in the graph) to a graph node.

# Algorithm
1. Find k nearest graph nodes to the query point using the ANN index
2. Fit local geometry at the query point
3. For each entry node, compute: local_distance(query → entry) + graph_distance(entry → target)
4. Return the minimum total distance

# Arguments
- `model`: The geodesic distance model
- `data`: Data matrix (dimensions × points)
- `point`: Query point coordinates
- `j`: Target graph node index
- `entry_k`: Number of entry points to consider (default: 5)

# Returns
The estimated geodesic distance.

# Example
```julia
query = randn(3)
d = geodesic_distance(model, data, query, 100)
```
"""
function geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix,
                           point::AbstractVector, j::Int; entry_k::Int=5)
    wg = model.weighted_graph
    T = eltype(point)

    # Find nearest graph nodes for the query point
    entry_neighbors = query(model.index, data, point, entry_k)

    # Fit local geometry at the query point
    neighbor_indices = [n.id for n in entry_neighbors]
    query_geom = fit_geometry(model.method, data, point, neighbor_indices)

    # Find best path through entry nodes
    # Use Inf for unreachable paths (not typemax which gives ~1.8e308)
    best_dist = T(Inf)

    for entry in entry_neighbors
        entry_idx = entry.id

        # Local distance from query to entry node
        entry_point = @view data[:, entry_idx]
        local_dist = T(local_distance(query_geom, point, entry_point))

        # Graph distance from entry to target
        graph_dist = geodesic_distance(model, data, entry_idx, j)

        total = local_dist + graph_dist
        if total < best_dist
            best_dist = total
        end
    end

    return best_dist
end

# ============================================================================
# Geodesic Distance: Between Two New Points
# ============================================================================

"""
    geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix,
                      point_a::AbstractVector, point_b::AbstractVector; entry_k::Int=5)

Compute the geodesic distance between two new points (not in the graph).

# Algorithm
1. Find k nearest graph nodes to each query point
2. Fit local geometry at each query point
3. For all pairs of entry nodes (entry_a, entry_b):
   - Compute: local_distance(a → entry_a) + graph_distance(entry_a → entry_b) + local_distance(entry_b → b)
4. Return the minimum total distance

# Arguments
- `model`: The geodesic distance model
- `data`: Data matrix (dimensions × points)
- `point_a`: First query point coordinates
- `point_b`: Second query point coordinates
- `entry_k`: Number of entry points to consider for each query (default: 5)

# Returns
The estimated geodesic distance.

# Example
```julia
query_a = randn(3)
query_b = randn(3)
d = geodesic_distance(model, data, query_a, query_b)
```
"""
function geodesic_distance(model::GeodesicDistanceModel, data::AbstractMatrix,
                           point_a::AbstractVector, point_b::AbstractVector;
                           entry_k::Int=5)
    wg = model.weighted_graph
    T = promote_type(eltype(point_a), eltype(point_b))

    # Find entry nodes for both points
    entries_a = query(model.index, data, point_a, entry_k)
    entries_b = query(model.index, data, point_b, entry_k)

    indices_a = [n.id for n in entries_a]
    indices_b = [n.id for n in entries_b]

    # Fit geometries at both query points
    geom_a = fit_geometry(model.method, data, point_a, indices_a)
    geom_b = fit_geometry(model.method, data, point_b, indices_b)

    # Find best path: point_a → entry_a → ... → entry_b → point_b
    # Use Inf for unreachable paths (not typemax which gives ~1.8e308)
    best_dist = T(Inf)

    for entry_a in entries_a
        entry_a_idx = entry_a.id
        entry_a_point = @view data[:, entry_a_idx]

        # Local distance from point_a to entry_a
        dist_a = T(local_distance(geom_a, point_a, entry_a_point))

        for entry_b in entries_b
            entry_b_idx = entry_b.id
            entry_b_point = @view data[:, entry_b_idx]

            # Graph distance between entry nodes
            dist_graph = geodesic_distance(model, data, entry_a_idx, entry_b_idx)

            # Local distance from entry_b to point_b
            dist_b = T(local_distance(geom_b, entry_b_point, point_b))

            total = dist_a + dist_graph + dist_b
            if total < best_dist
                best_dist = total
            end
        end
    end

    return best_dist
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    Base.length(model::GeodesicDistanceModel) -> Int

Return the number of nodes in the geodesic model.
"""
Base.length(model::GeodesicDistanceModel) = length(model.weighted_graph)

"""
    configured_k(model::GeodesicDistanceModel) -> Int

Return the k value (number of neighbors per node) in the model's graph.
"""
configured_k(model::GeodesicDistanceModel) = configured_k(model.weighted_graph)

"""
    all_pairs_geodesic_distances(model::GeodesicDistanceModel, data::AbstractMatrix) -> Matrix

Compute geodesic distances between all pairs of graph nodes.

WARNING: O(n² × complexity of Dijkstra) - only use for small graphs.

# Returns
A symmetric matrix D where D[i,j] is the geodesic distance from node i to j.
"""
function all_pairs_geodesic_distances(model::GeodesicDistanceModel, data::AbstractMatrix)
    n = length(model)
    T = eltype(model.weighted_graph.edge_weights[1])
    D = Matrix{T}(undef, n, n)

    for i in 1:n
        dist, _ = _dijkstra(model.weighted_graph, i)
        D[i, :] = dist
    end

    return D
end
