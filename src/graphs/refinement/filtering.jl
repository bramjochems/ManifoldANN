using LinearAlgebra
using Statistics

# ==============================================================================
# Phase 1: Helper Functions for Flexible Distance Metrics
# ==============================================================================

"""
    effective_epsilon(i, j, graph, data; profile=ManifoldANNDefault())

Compute effective epsilon (scale normalisation) for an edge (i, j) under
the given [`AbstractOrcMLCompatibilityProfile`](@ref). This is a thin
wrapper that dispatches to [`compute_effective_epsilon`](@ref).

# References
- orcml: https://github.com/TristanSaidi/orcml (src/utils/graph_utils.py, lines 34-61)
"""
function effective_epsilon(
    i::Int, j::Int,
    graph::KNNGraph,
    data::AbstractMatrix{T};
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    return compute_effective_epsilon(profile, i, j, graph, data)
end

"""
    compute_shortest_paths(graph::KNNGraph, data::AbstractMatrix{T}, weight_type::Symbol) where T

Compute all-pairs shortest paths using Floyd-Warshall algorithm.

# Arguments
- `graph::KNNGraph`: The k-NN graph
- `data::AbstractMatrix{T}`: Data matrix (d × n)
- `weight_type::Symbol`: Type of edge weights to use
  - `:euclidean`: ||x - y|| (Euclidean distance)
  - `:normalized`: effective_epsilon(x, y) (scale normalization)
  - `:unit`: 1 (hop count / topological distance)

# Returns
- `Matrix{Float64}`: n × n distance matrix where `[i, j]` is shortest path from i to j

# Graph Directedness
Respects `graph.metadata.directed`:
- **Directed graphs**: Only edges i→j that exist in the graph are used
- **Undirected graphs**: Edges are automatically bidirectional (i→j implies j→i)

# Performance
O(n³) time complexity. Pre-compute once and reuse for all edges.
"""
function compute_shortest_paths(
    graph::KNNGraph,
    data::AbstractMatrix{T},
    weight_type::Symbol;
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    n = length(graph)

    # Initialize distance matrix (Inf = no path)
    dist = fill(Inf, n, n)

    # Distance from node to itself is 0
    for i in 1:n
        dist[i, i] = 0.0
    end

    # Set edge weights based on weight_type
    is_undirected = graph.metadata !== nothing &&
                    hasfield(typeof(graph.metadata), :directed) &&
                    !graph.metadata.directed

    for i in 1:n
        for j in graph[i]
            if weight_type == :euclidean
                weight = norm(data[:, i] - data[:, j])
            elseif weight_type == :normalized
                weight = effective_epsilon(i, j, graph, data; profile = profile)
            elseif weight_type == :unit
                weight = 1.0
            else
                error("Unknown weight_type: $weight_type. Use :euclidean, :normalized, or :unit")
            end

            # Set edge weight for i→j
            dist[i, j] = min(dist[i, j], weight)

            # Only add reverse edge for undirected graphs
            if is_undirected
                dist[j, i] = min(dist[j, i], weight)
            end
        end
    end

    # Floyd-Warshall algorithm
    for k in 1:n, i in 1:n, j in 1:n
        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    end

    return dist
end

"""
    validate_geodesic_config(graph::KNNGraph, metric::Symbol)

Validate configuration when using geodesic distance metrics.

Issues a warning if using geodesic distance metrics on a directed graph, as this
produces asymmetric shortest paths where d(i,j) ≠ d(j,i).

Both directed and undirected graphs are valid for ORC computation, but they have
different interpretations:
- **Directed**: Asymmetric shortest paths, measures directional flow
- **Undirected**: Symmetric shortest paths, ensures d(i,j) = d(j,i)

# Arguments
- `graph::KNNGraph`: The k-NN graph
- `metric::Symbol`: Distance metric being used

# Note
The orcml approach uses undirected graphs with geodesic metrics:
```julia
graph = build_knn_graph(index, data; k=15, directed=false)
curvatures = compute_all_curvatures(graph, data; cost_metric=:geodesic_normalized)
```
"""
function validate_geodesic_config(graph::KNNGraph, metric::Symbol)
    # Check if using geodesic metric on directed graph
    is_geodesic = metric in (:geodesic_euclidean, :geodesic_normalized, :geodesic_unit)
    is_directed = graph.metadata !== nothing &&
                  hasfield(typeof(graph.metadata), :directed) &&
                  graph.metadata.directed

    if is_geodesic && is_directed
        @warn """
        Using geodesic metric ($metric) on a directed graph.

        Geodesic distances on directed graphs produce asymmetric shortest paths where
        d(i,j) may differ from d(j,i). This is valid but has different interpretation
        than symmetric (undirected) geodesic distances.

        For symmetric geodesic distances (orcml approach), use:
            graph = build_knn_graph(index, data; k=15, directed=false)

        Both directed and undirected are valid - choose based on your use case.
        """
    end
end

"""
    get_distance_function(metric::Symbol, graph::KNNGraph, data::AbstractMatrix{T}) where T

Get a distance function closure based on the specified metric.

# Arguments
- `metric::Symbol`: Distance metric to use
  - `:euclidean`: Direct Euclidean distance ||i - j||
  - `:geodesic_euclidean`: Shortest path with Euclidean edge weights
  - `:geodesic_normalized`: Shortest path with effective_epsilon weights (orcml)
  - `:geodesic_unit`: Shortest path with unit edge weights (hop count)
  - `:normalized`: effective_epsilon only (for denominator)
- `graph::KNNGraph`: The k-NN graph
- `data::AbstractMatrix{T}`: Data matrix (d × n)

# Returns
- `Function`: (i::Int, j::Int) -> Float64 distance function

# Performance Notes
- `:euclidean` and `:normalized`: O(d) per call, no pre-computation
- `:geodesic_*`: Requires O(n³) pre-computation of shortest paths, but O(1) per call
"""
function get_distance_function(
    metric::Symbol,
    graph::KNNGraph,
    data::AbstractMatrix{T};
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    if metric == :euclidean
        return (i::Int, j::Int) -> norm(data[:, i] - data[:, j])

    elseif metric == :geodesic_euclidean
        sp = compute_shortest_paths(graph, data, :euclidean; profile = profile)
        return (i::Int, j::Int) -> sp[i, j]

    elseif metric == :geodesic_normalized
        sp = compute_shortest_paths(graph, data, :normalized; profile = profile)
        return (i::Int, j::Int) -> sp[i, j]

    elseif metric == :geodesic_unit
        sp = compute_shortest_paths(graph, data, :unit; profile = profile)
        return (i::Int, j::Int) -> sp[i, j]

    elseif metric == :normalized
        return (i::Int, j::Int) -> effective_epsilon(i, j, graph, data; profile = profile)

    else
        error("Unknown metric: $metric. Valid options: :euclidean, :geodesic_euclidean, :geodesic_normalized, :geodesic_unit, :normalized")
    end
end

# ==============================================================================
# Edge Processing Helper Functions
# ==============================================================================

"""
    _collect_edges_to_process(graph::KNNGraph)

Collect unique edges from the graph for curvature computation.

For bidirectional edges (both i→j and j→i exist), only the canonical form (i < j)
is added to avoid duplicate computation. For unidirectional edges, the existing
direction is added. Self-loops (i→i) are always included.

Returns a vector of (x, y) edge tuples.
"""
function _collect_edges_to_process(graph::KNNGraph)
    n_nodes = length(graph)
    edges_to_process = Tuple{Int,Int}[]
    edges_seen = Set{Tuple{Int,Int}}()

    for x in 1:n_nodes
        for y in graph[x]
            # Skip if already processed
            (x, y) ∈ edges_seen && continue

            # Handle self-loops explicitly
            if x == y
                push!(edges_to_process, (x, y))
                push!(edges_seen, (x, y))
                continue
            end

            # Check if edge is bidirectional (both x→y and y→x exist)
            is_bidirectional = y in graph[x] && x in graph[y]

            if is_bidirectional
                # For bidirectional edges, only add canonical form (x < y)
                if x < y
                    push!(edges_to_process, (x, y))
                    push!(edges_seen, (x, y))
                    push!(edges_seen, (y, x))  # Mark reverse as seen
                end
            else
                # For unidirectional edges, add the direction that exists
                push!(edges_to_process, (x, y))
                push!(edges_seen, (x, y))
            end
        end
    end

    return edges_to_process
end

"""
    _setup_distance_functions(cost_metric, denominator_metric, graph, data, distance_fn)

Setup cost and denominator distance functions based on specified metrics.

Handles both new metric-based API and deprecated distance_fn parameter.
"""
function _setup_distance_functions(
    cost_metric::Symbol,
    denominator_metric::Symbol,
    graph::KNNGraph,
    data::AbstractMatrix,
    distance_fn::Union{Nothing,Function};
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
)
    if distance_fn !== nothing
        return distance_fn, distance_fn
    else
        cost_fn = get_distance_function(cost_metric, graph, data; profile = profile)
        denom_fn = get_distance_function(denominator_metric, graph, data; profile = profile)
        return cost_fn, denom_fn
    end
end

"""
    _mirror_bidirectional_results!(results, graph)

Add reverse edges to results for API compatibility.

For bidirectional edges where only the canonical form (x < y) was computed,
this adds the reverse direction (y, x) with the same curvature values.
"""
function _mirror_bidirectional_results!(
    results::Dict{Tuple{Int,Int},CurvatureResult{Float64}},
    graph::KNNGraph
)
    n_nodes = length(graph)
    reverse_edges = Tuple{Int,Int}[]

    # Collect edges that need mirroring
    for x in 1:n_nodes
        for y in graph[x]
            if !haskey(results, (x, y)) && haskey(results, (y, x))
                push!(reverse_edges, (x, y))
            end
        end
    end

    # Mirror the results
    for (x, y) in reverse_edges
        orig = results[(y, x)]
        results[(x, y)] = CurvatureResult{Float64}(
            x, y, orig.curvature, orig.wasserstein_distance,
            orig.edge_distance, orig.solver_type
        )
    end

    return nothing
end

"""
    _process_edge!(results, results_lock, edge, graph, data, cost_fn, denom_fn,
                   exclude_edge_endpoints, solver, fallback_solver)

Process a single edge and store the curvature result.

This helper function is used by compute_all_curvatures for both threaded and sequential processing.
"""
function _process_edge!(
    results::Dict{Tuple{Int,Int},CurvatureResult{Float64}},
    results_lock::ReentrantLock,
    edge::Tuple{Int,Int},
    graph::KNNGraph,
    data::AbstractMatrix{T},
    cost_fn::Function,
    denom_fn::Function,
    exclude_edge_endpoints::Bool,
    solver::AbstractOTSolver,
    fallback_solver::AbstractOTSolver;
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    x, y = edge

    # Build neighborhoods with optional endpoint exclusion.
    # When the profile sets `asymmetric_target_exclusion=true` (orcml),
    # the target node's neighbourhood does NOT exclude the source endpoint
    # (a positional-argument quirk in the upstream Python
    # `_get_single_node_neighbors_distributions`). The source side still
    # excludes the target as usual.
    neighborhood_x = build_neighborhood(x, graph[x], y, exclude_edge_endpoints, Float64)
    exclude_y_side = exclude_edge_endpoints && !profile.asymmetric_target_exclusion
    neighborhood_y = build_neighborhood(y, graph[y], x, exclude_y_side, Float64)

    # Compute edge distance using denominator metric
    edge_dist = denom_fn(x, y)

    # Create edge view
    edge_view = create_edge_view(neighborhood_x, neighborhood_y, Float64(edge_dist))

    # Pre-compute distance matrix for this edge's neighborhood using cost metric
    # This avoids redundant distance calls during OT solving
    precomputed_cost_fn = _create_precomputed_distance_fn(
        edge_view, data, cost_fn
    )

    # Solve OT with precomputed distances
    active_solver = can_handle(solver, edge_view) ? solver : fallback_solver
    result = compute_curvature(
        active_solver, edge_view, precomputed_cost_fn
    )

    # Thread-safe result insertion
    lock(results_lock) do
        results[(x, y)] = result
    end

    return nothing
end

# ==============================================================================
# Graph Filtering Functions
# ==============================================================================

"""
    filter_graph(graph, data; curvature_threshold=0.0, solver=HungarianSolver(),
                 fallback_solver=GenericOTSolver(), distance_fn=nothing,
                 min_neighbors=1, verbose=false)

Filter a kNN graph by removing edges with curvature below threshold.
"""
function filter_graph(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    curvature_threshold::Real=0.0,
    solver::AbstractOTSolver=HungarianSolver(),
    fallback_solver::AbstractOTSolver=SinkhornSolver(),
    distance_fn::Union{Nothing,Function}=nothing,
    min_neighbors::Int=1,
    verbose::Bool=false
) where {T}
    n_nodes = length(graph)
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be >= 1"))

    dist_fn = distance_fn === nothing ? (i, j) -> norm(data[:, i] - data[:, j]) : distance_fn

    verbose && println("Building node neighborhoods...")
    neighborhoods = Dict{Int,NodeNeighborhood{Float64}}(
        i => uniform_neighborhood(i, graph[i], Float64) for i in 1:n_nodes
    )

    verbose && println("Computing edge curvatures...")
    edge_curvatures = Dict{Tuple{Int,Int},CurvatureResult{Float64}}()

    for x in 1:n_nodes
        for y in graph[x]
            haskey(edge_curvatures, (x, y)) && continue

            edge_dist = dist_fn(x, y)
            edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], edge_dist)

            active_solver = can_handle(solver, edge_view) ? solver : fallback_solver
            result = compute_curvature(active_solver, edge_view, dist_fn)
            edge_curvatures[(x, y)] = result

            if y in graph[x] && x in graph[y]
                edge_curvatures[(y, x)] = CurvatureResult{Float64}(
                    y, x, result.curvature, result.wasserstein_distance,
                    result.edge_distance, result.solver_type
                )
            end
        end
        verbose && (x % 100 == 0) && println("  Processed $x / $n_nodes nodes")
    end

    verbose && println("Filtering edges...")
    filtered_neighbors = Vector{Vector{Int}}(undef, n_nodes)
    n_removed = 0

    for i in 1:n_nodes
        neighbor_curvatures = [
            (j, edge_curvatures[(i, j)].curvature)
            for j in graph[i] if haskey(edge_curvatures, (i, j))
        ]

        sort!(neighbor_curvatures, by=x -> x[2], rev=true)

        kept = Int[]
        for (j, curv) in neighbor_curvatures
            if curv >= curvature_threshold || length(kept) < min_neighbors
                push!(kept, j)
            else
                n_removed += 1
            end
        end
        filtered_neighbors[i] = kept
    end

    verbose && println("Removed $n_removed edges total")

    new_k = maximum(length(neighbors) for neighbors in filtered_neighbors)
    KNNGraph(filtered_neighbors, new_k, graph.include_self, graph.metadata)
end

"""
    compute_all_curvatures(graph, data; exclude_edge_endpoints=false,
                          cost_metric=:euclidean, denominator_metric=:euclidean,
                          solver=HungarianSolver(), fallback_solver=SinkhornSolver(),
                          use_threading=true, verbose=false, distance_fn=nothing)

Compute Ollivier-Ricci curvatures for all edges in a k-NN graph.

# Arguments
- `graph::KNNGraph`: The k-NN graph
- `data::AbstractMatrix{T}`: Data matrix (d × n)

# Keyword Arguments
- `exclude_edge_endpoints::Bool=false`: Exclude edge endpoints from neighborhoods (orcml approach)
- `cost_metric::Symbol=:euclidean`: Distance metric for OT cost matrix
  - `:euclidean`: Direct Euclidean distance (default, current approach)
  - `:geodesic_euclidean`: Shortest paths with Euclidean edge weights
  - `:geodesic_normalized`: Shortest paths with effective_epsilon weights (orcml)
  - `:geodesic_unit`: Shortest paths with unit edge weights
  - `:normalized`: effective_epsilon only
- `denominator_metric::Symbol=:euclidean`: Distance metric for curvature denominator
  - Same options as `cost_metric`
- `solver::AbstractOTSolver=HungarianSolver()`: Primary OT solver
- `fallback_solver::AbstractOTSolver=SinkhornSolver()`: Fallback solver
- `use_threading::Bool=true`: Enable multithreaded processing
- `verbose::Bool=false`: Print progress information
- `distance_fn::Union{Nothing,Function}=nothing`: (Deprecated) Custom distance function

# Returns
- `Dict{Tuple{Int,Int}, CurvatureResult{Float64}}`: Curvature results for all edges

# Configuration Examples

## Current ManifoldANN (default):
```julia
compute_all_curvatures(graph, data)
# Same as:
# exclude_edge_endpoints=false, cost_metric=:euclidean, denominator_metric=:euclidean
```

## orcml replication:
```julia
compute_all_curvatures(graph, data;
    exclude_edge_endpoints=true,
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    profile=OrcmlExact(),
)
```

## Hybrid geodesic:
```julia
compute_all_curvatures(graph, data;
    cost_metric=:geodesic_euclidean,
    denominator_metric=:euclidean
)
```

# Performance Notes
- Multithreaded edge processing (scales with Threads.nthreads())
- Pre-computed distance matrices for each edge's neighborhood
- Geodesic metrics require O(n³) shortest path pre-computation

# References
- orcml: https://github.com/TristanSaidi/orcml
- Paper: "Recovering Manifold Structure Using Ollivier-Ricci Curvature" (ICLR 2025)
"""
function compute_all_curvatures(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    exclude_edge_endpoints::Bool=false,
    cost_metric::Symbol=:euclidean,
    denominator_metric::Symbol=:euclidean,
    solver::AbstractOTSolver=HungarianSolver(),
    fallback_solver::AbstractOTSolver=SinkhornSolver(),
    use_threading::Bool=true,
    verbose::Bool=false,
    distance_fn::Union{Nothing,Function}=nothing,  # Deprecated, kept for backward compatibility
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    n_nodes = length(graph)

    # Validate geodesic metric configuration (warn if directed graph)
    validate_geodesic_config(graph, cost_metric)
    validate_geodesic_config(graph, denominator_metric)

    # Setup distance functions (handles both new API and deprecated distance_fn)
    verbose && println("Setting up distance functions...")
    cost_fn, denom_fn = _setup_distance_functions(
        cost_metric, denominator_metric, graph, data, distance_fn;
        profile = profile,
    )

    # Collect edges to process (handles self-loops, bidirectional, and unidirectional edges)
    verbose && println("Collecting edges to process...")
    edges_to_process = _collect_edges_to_process(graph)
    verbose && println("Processing $(length(edges_to_process)) edges (threading: $use_threading)...")

    # Process all edges (threaded or sequential)
    results = Dict{Tuple{Int,Int},CurvatureResult{Float64}}()
    results_lock = ReentrantLock()

    if use_threading
        Threads.@threads for edge_idx in 1:length(edges_to_process)
            _process_edge!(
                results, results_lock, edges_to_process[edge_idx],
                graph, data, cost_fn, denom_fn,
                exclude_edge_endpoints, solver, fallback_solver;
                profile = profile,
            )
        end
    else
        for edge_idx in 1:length(edges_to_process)
            _process_edge!(
                results, results_lock, edges_to_process[edge_idx],
                graph, data, cost_fn, denom_fn,
                exclude_edge_endpoints, solver, fallback_solver;
                profile = profile,
            )
            verbose && (edge_idx % 100 == 0) &&
                println("  Processed $edge_idx / $(length(edges_to_process)) edges")
        end
    end

    # Mirror results for bidirectional edges (API compatibility)
    _mirror_bidirectional_results!(results, graph)

    return results
end

"""
    _create_precomputed_distance_fn(edge_view, data, dist_fn)

Create a distance function with pre-computed distance matrix for the edge's neighborhood.
This avoids redundant norm() computations during optimal transport solving.
"""
function _create_precomputed_distance_fn(
    edge_view::EdgeNeighborhoodView{T},
    data::AbstractMatrix,
    dist_fn::Function
) where {T}
    # Collect all unique nodes in both neighborhoods
    all_nodes_x = vcat(edge_view.shared, edge_view.unique_x)
    all_nodes_y = vcat(edge_view.shared, edge_view.unique_y)

    # Pre-compute all pairwise distances
    n_x, n_y = length(all_nodes_x), length(all_nodes_y)
    dist_matrix = Matrix{Float64}(undef, n_x, n_y)

    for i in 1:n_x, j in 1:n_y
        dist_matrix[i, j] = Float64(dist_fn(all_nodes_x[i], all_nodes_y[j]))
    end

    # Create lookup dictionaries for fast indexing
    x_idx_map = Dict(node => i for (i, node) in enumerate(all_nodes_x))
    y_idx_map = Dict(node => j for (j, node) in enumerate(all_nodes_y))

    # Return closure that looks up pre-computed distances
    return function(node_i::Int, node_j::Int)
        i = get(x_idx_map, node_i, 0)
        j = get(y_idx_map, node_j, 0)

        if i > 0 && j > 0
            return dist_matrix[i, j]
        else
            # Fallback for nodes not in the neighborhood (shouldn't happen)
            return dist_fn(node_i, node_j)
        end
    end
end

"""
    curvature_statistics(curvatures)

Compute summary statistics of curvature distribution.
"""
function curvature_statistics(curvatures::Dict{Tuple{Int,Int}, CurvatureResult{T}}) where {T}
    curv_values = [result.curvature for result in Base.values(curvatures)]

    if isempty(curv_values)
        return (mean=NaN, std=NaN, min=NaN, max=NaN, median=NaN, n_positive=0, n_negative=0)
    end

    sorted_values = sort(curv_values)
    n = length(curv_values)

    (
        mean = sum(curv_values) / n,
        std = sqrt(sum((v - sum(curv_values)/n)^2 for v in curv_values) / n),
        min = minimum(curv_values),
        max = maximum(curv_values),
        median = n % 2 == 1 ? sorted_values[div(n, 2) + 1] : (sorted_values[div(n, 2)] + sorted_values[div(n, 2) + 1]) / 2,
        n_positive = count(v -> v > 0, curv_values),
        n_negative = count(v -> v < 0, curv_values)
    )
end
