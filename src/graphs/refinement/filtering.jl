using LinearAlgebra
using Statistics
using DataStructures: BinaryMinHeap

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

Compute all-pairs shortest paths using per-source Dijkstra (binary-heap)
on the kNN graph's sparse adjacency representation.

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
The kNN graph is sparse with `m ≈ n·k` edges. Per-source Dijkstra runs
in `O(m + n log n)` per source, giving `O(n·(m + n log n))` total —
`~40×` fewer ops than the prior Floyd-Warshall implementation at
`n=1000, k=15`. Mirrors the Python orcml reference, which uses
`networkx.shortest_path_length` (Dijkstra) on the same graph.

Parallel edges (rare, but possible if a node lists the same neighbour
twice) are reduced to their min weight, matching the previous behaviour.
"""
function compute_shortest_paths(
    graph::KNNGraph,
    data::AbstractMatrix{T},
    weight_type::Symbol;
    profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault(),
) where {T}
    n = length(graph)

    is_undirected = graph.metadata !== nothing &&
                    hasfield(typeof(graph.metadata), :directed) &&
                    !graph.metadata.directed

    # Build sparse adjacency: out_neighbors[u] / out_weights[u]. Computing
    # `effective_epsilon` here means each weight is computed exactly once
    # per (i, j) edge — same as the previous FW init loop.
    out_neighbors = [Int[] for _ in 1:n]
    out_weights = [Float64[] for _ in 1:n]

    # `min`-merge parallel edges into a temporary dict per source so the
    # final adjacency carries one weight per (u, v) pair.
    edge_min = Dict{Tuple{Int,Int},Float64}()

    @inline function _add_edge!(u::Int, v::Int, w::Float64)
        key = (u, v)
        prev = get(edge_min, key, Inf)
        if w < prev
            edge_min[key] = w
        end
    end

    for i in 1:n
        for j in graph[i]
            if weight_type == :euclidean
                weight = Float64(norm(data[:, i] - data[:, j]))
            elseif weight_type == :normalized
                weight = Float64(effective_epsilon(i, j, graph, data; profile = profile))
            elseif weight_type == :unit
                weight = 1.0
            else
                error("Unknown weight_type: $weight_type. Use :euclidean, :normalized, or :unit")
            end

            _add_edge!(i, j, weight)
            if is_undirected
                _add_edge!(j, i, weight)
            end
        end
    end

    for ((u, v), w) in edge_min
        push!(out_neighbors[u], v)
        push!(out_weights[u], w)
    end

    # Per-source Dijkstra with a binary min-heap. Settled-node check via
    # `visited[v]` — entries popped after a node is settled (stale entries
    # from earlier `push!`es with worse keys) are skipped.
    dist = fill(Inf, n, n)
    visited = falses(n)

    for src in 1:n
        fill!(visited, false)
        dist[src, src] = 0.0

        heap = BinaryMinHeap{Tuple{Float64,Int}}()
        push!(heap, (0.0, src))

        while !isempty(heap)
            d_u, u = pop!(heap)
            visited[u] && continue
            visited[u] = true

            nbrs = out_neighbors[u]
            wts = out_weights[u]
            @inbounds for idx in eachindex(nbrs)
                v = nbrs[idx]
                visited[v] && continue
                alt = d_u + wts[idx]
                if alt < dist[src, v]
                    dist[src, v] = alt
                    push!(heap, (alt, v))
                end
            end
        end
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
curvatures = compute_all_curvatures(graph, data; variant=ORCManL())
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

            # `y in graph[x]` is tautological inside `for y in graph[x]`,
            # so the bidirectionality test reduces to checking the reverse
            # edge. Skipping the redundant scan saves an O(k) lookup per edge.
            is_bidirectional = x in graph[y]

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
    results::Dict{Tuple{Int,Int},CurvatureResult{T}},
    graph::KNNGraph
) where {T<:AbstractFloat}
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
        results[(x, y)] = CurvatureResult{T}(
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
    results::Dict{Tuple{Int,Int},CurvatureResult{T}},
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
    neighborhood_x = build_neighborhood(x, graph[x], y, exclude_edge_endpoints, T)
    exclude_y_side = exclude_edge_endpoints && !profile.asymmetric_target_exclusion
    neighborhood_y = build_neighborhood(y, graph[y], x, exclude_y_side, T)

    # Compute edge distance using denominator metric
    edge_dist = denom_fn(x, y)

    # Create edge view at input precision T. denom_fn may internally use
    # Float64 (shortest-path matrices, effective_epsilon) — the cast here
    # is the boundary back to the user-facing eltype.
    edge_view = create_edge_view(neighborhood_x, neighborhood_y, T(edge_dist))

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
                 min_neighbors=1, use_threading=true, verbose=false)

Filter a kNN graph by removing edges with curvature below threshold.

This is a thin post-processing wrapper around [`compute_all_curvatures`](@ref):
curvatures are computed once via the shared edge-enumeration path
([`_collect_edges_to_process`](@ref)) and threaded edge processing, then a
per-node sort-and-prune pass keeps the highest-curvature neighbors subject
to `curvature_threshold` and `min_neighbors`.

Uses [`StandardORC`](@ref) as the variant — neighborhoods include the edge
endpoints and the cost/denominator metric defaults to Euclidean. Pass a
custom `distance_fn` to override the metric (the same kwarg is forwarded
to `compute_all_curvatures`).

Folded with `compute_all_curvatures` in 2026-05 to eliminate a divergent
second code path; see commit history.
"""
function filter_graph(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    curvature_threshold::Real=0.0,
    solver::AbstractOTSolver=HungarianSolver(),
    fallback_solver::AbstractOTSolver=SinkhornSolver(),
    distance_fn::Union{Nothing,Function}=nothing,
    min_neighbors::Int=1,
    use_threading::Bool=true,
    verbose::Bool=false
) where {T}
    n_nodes = length(graph)
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be >= 1"))

    verbose && println("Computing edge curvatures (via compute_all_curvatures)...")
    edge_curvatures = compute_all_curvatures(
        graph, data;
        variant = StandardORC(),
        solver = solver,
        fallback_solver = fallback_solver,
        use_threading = use_threading,
        verbose = verbose,
        distance_fn = distance_fn,
    )

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

# ==============================================================================
# ORC Variant Trait (StandardORC vs ORC-ManL)
# ==============================================================================

"""
    AbstractORCConfig

Trait that bundles the algorithmic choices distinguishing **standard
Ollivier-Ricci curvature** from the **ORC-ManL** variant used by
`orcml` (https://github.com/TristanSaidi/orcml).

A single value of `AbstractORCConfig` fixes three internal flags that
must move together to be coherent:

- `exclude_edge_endpoints`
- `cost_metric`
- `denominator_metric`

Two presets are provided:

- [`StandardORC`](@ref) — the package's historical defaults: Euclidean
  cost and denominator, neighbourhoods include the edge endpoints. Not
  affected by [`AbstractOrcMLCompatibilityProfile`](@ref) (which is an
  ORC-ManL-only concept).
- [`ORCManL`](@ref) — the ORC-ManL variant: geodesic-normalised cost,
  effective-epsilon denominator, neighbourhoods exclude the edge
  endpoints. Carries an `AbstractOrcMLCompatibilityProfile` field so
  that the two-variant choice composes cleanly with the orcml-
  compatibility choice.

The previous public API exposed `exclude_edge_endpoints`, `cost_metric`,
`denominator_metric`, and `profile` as four independent kwargs of
[`compute_all_curvatures`](@ref); that API allowed incoherent hybrids
and has been removed in favour of this trait.
"""
abstract type AbstractORCConfig end

"""
    StandardORC()

Public preset for **standard Ollivier-Ricci curvature**:
`exclude_edge_endpoints=false`, `cost_metric=:euclidean`,
`denominator_metric=:euclidean`.

This is the default `variant` of [`compute_all_curvatures`](@ref) and
preserves the package's historical behaviour. The
[`AbstractOrcMLCompatibilityProfile`](@ref) is an ORC-ManL-specific
concept and has no effect under `StandardORC`.
"""
struct StandardORC <: AbstractORCConfig end

"""
    ORCManL(; profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault())

Public preset for the **ORC-ManL** variant:
`exclude_edge_endpoints=true`, `cost_metric=:geodesic_normalized`,
`denominator_metric=:normalized`.

The `profile` field selects between the orcml-compatibility presets
([`ManifoldANNDefault`](@ref) — the package's historical ORC-ManL
behaviour, the default; or [`OrcmlExact`](@ref) — bit-for-bit match
with the reference Python `orcml`).
"""
struct ORCManL{P<:AbstractOrcMLCompatibilityProfile} <: AbstractORCConfig
    profile::P
end
ORCManL(; profile::AbstractOrcMLCompatibilityProfile = ManifoldANNDefault()) =
    ORCManL(profile)

# Internal: unpack a variant into the (exclude_edge_endpoints,
# cost_metric, denominator_metric, profile) 4-tuple consumed by the
# rest of the pipeline. `profile` is irrelevant for `StandardORC`; we
# return `ManifoldANNDefault()` purely to keep types concrete.
function _unpack_variant(::StandardORC)
    return (false, :euclidean, :euclidean, ManifoldANNDefault())
end
function _unpack_variant(v::ORCManL)
    return (true, :geodesic_normalized, :normalized, v.profile)
end

"""
    compute_all_curvatures(graph, data; variant=StandardORC(),
                          solver=HungarianSolver(), fallback_solver=SinkhornSolver(),
                          use_threading=true, verbose=false, distance_fn=nothing)

Compute Ollivier-Ricci curvatures for all edges in a k-NN graph.

# Arguments
- `graph::KNNGraph`: The k-NN graph
- `data::AbstractMatrix{T}`: Data matrix (d × n)

# Keyword Arguments
- `variant::AbstractORCConfig=StandardORC()`: ORC variant preset.
  - [`StandardORC()`](@ref) — Euclidean cost & denominator, endpoints
    included in neighbourhoods (the package default).
  - [`ORCManL()`](@ref) — geodesic-normalised cost, effective-epsilon
    denominator, endpoints excluded. Optionally takes a `profile`
    (defaulting to `ManifoldANNDefault()`; pass `OrcmlExact()` for a
    bit-for-bit match with the reference Python `orcml`).
- `solver::AbstractOTSolver=HungarianSolver()`: Primary OT solver
- `fallback_solver::AbstractOTSolver=SinkhornSolver()`: Fallback solver
- `use_threading::Bool=true`: Enable multithreaded processing
- `verbose::Bool=false`: Print progress information
- `distance_fn::Union{Nothing,Function}=nothing`: (Deprecated) Custom distance function

# Returns
- `Dict{Tuple{Int,Int}, CurvatureResult{T}}`: Curvature results for all edges,
  where `T = eltype(data)` (Float64 input → Float64 output, Float32 → Float32).

# Configuration Examples

## Standard ORC (default):
```julia
compute_all_curvatures(graph, data)
# Equivalent to:
compute_all_curvatures(graph, data; variant=StandardORC())
```

## ORC-ManL (ManifoldANN's historical ORC-ManL behaviour):
```julia
compute_all_curvatures(graph, data; variant=ORCManL())
```

## ORC-ManL matching reference Python `orcml`:
```julia
compute_all_curvatures(graph, data; variant=ORCManL(profile=OrcmlExact()))
```

# Performance Notes
- Multithreaded edge processing (scales with Threads.nthreads())
- Pre-computed distance matrices for each edge's neighborhood
- ORC-ManL requires O(n³) shortest-path pre-computation

# References
- orcml: https://github.com/TristanSaidi/orcml
- Paper: "Recovering Manifold Structure Using Ollivier-Ricci Curvature" (ICLR 2025)
"""
function compute_all_curvatures(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    variant::AbstractORCConfig = StandardORC(),
    solver::AbstractOTSolver=HungarianSolver(),
    fallback_solver::AbstractOTSolver=SinkhornSolver(),
    use_threading::Bool=true,
    verbose::Bool=false,
    distance_fn::Union{Nothing,Function}=nothing,  # Deprecated, kept for backward compatibility
) where {T}
    exclude_edge_endpoints, cost_metric, denominator_metric, profile =
        _unpack_variant(variant)
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

    # Process all edges (threaded or sequential). Container is parameterised
    # on the input data eltype T so Float32 input → CurvatureResult{Float32}
    # without silent promotion through the pipeline.
    results = Dict{Tuple{Int,Int},CurvatureResult{T}}()
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

Create a distance function with pre-computed distance matrix for the edge's
neighborhood. This avoids redundant norm() computations during optimal
transport solving.

The pre-computed matrix is allocated as `Matrix{T}` where `T` is the eltype
of the [`EdgeNeighborhoodView{T}`](@ref). For Float32 graphs this avoids an
unnecessary Float64 promotion + extra allocation.

# Errors
Raises `error(...)` if the OT solver queries a node outside the edge's
neighborhood — by construction this cannot happen, so a query for a
non-neighborhood node is a logic bug elsewhere (silent fallback would
mask it as a numeric drift).

# Performance follow-up (TODO_cleanup)
The closure-allocation-per-edge cost (two `Dict` lookup tables + a
heap-allocated closure for every edge in the graph) dominates wall time on
large graphs. Replacing the closures with a per-edge struct that the OT
solvers consume directly is a design change — flagged for a follow-up
cleanup pass, not part of this mechanical fix.
"""
function _create_precomputed_distance_fn(
    edge_view::EdgeNeighborhoodView{T},
    data::AbstractMatrix,
    dist_fn::Function
) where {T}
    # Collect all unique nodes in both neighborhoods
    all_nodes_x = vcat(edge_view.shared, edge_view.unique_x)
    all_nodes_y = vcat(edge_view.shared, edge_view.unique_y)

    # Pre-compute all pairwise distances at the view's working precision
    n_x, n_y = length(all_nodes_x), length(all_nodes_y)
    dist_matrix = Matrix{T}(undef, n_x, n_y)

    for i in 1:n_x, j in 1:n_y
        dist_matrix[i, j] = T(dist_fn(all_nodes_x[i], all_nodes_y[j]))
    end

    # Create lookup dictionaries for fast indexing
    x_idx_map = Dict(node => i for (i, node) in enumerate(all_nodes_x))
    y_idx_map = Dict(node => j for (j, node) in enumerate(all_nodes_y))

    x_id = edge_view.x_id
    y_id = edge_view.y_id

    # Return closure that looks up pre-computed distances. By construction,
    # the OT solver only queries (node_i, node_j) with node_i in N(x) and
    # node_j in N(y) — both must be in the lookup tables. A miss indicates
    # a logic bug upstream; raise rather than silently falling back to the
    # raw `dist_fn` (which would mask the bug as numeric drift, since the
    # OT cost would be computed from a different distance than the rest of
    # the matrix).
    return function(node_i::Int, node_j::Int)
        i = get(x_idx_map, node_i, 0)
        j = get(y_idx_map, node_j, 0)

        if i > 0 && j > 0
            return dist_matrix[i, j]
        else
            error("_create_precomputed_distance_fn: node ($node_i, $node_j) " *
                  "is outside the neighborhood of edge ($x_id, $y_id). " *
                  "This indicates a logic bug — the OT solver should only " *
                  "query nodes in N(x) × N(y).")
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
