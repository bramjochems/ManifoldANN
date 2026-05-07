"""
    KNNGraph

Lightweight representation of a directed k-nearest-neighbor graph. Stores the
ordered neighbor lists per vertex along with bookkeeping that downstream
algorithms can inspect. Graphs may optionally carry node-level metadata (any
`Vector` whose length matches the vertex count).
"""
struct KNNGraph{I<:Integer,M}
    neighbors::Vector{Vector{I}}
    k::Int
    include_self::Bool
    metadata::M
end

Base.length(graph::KNNGraph) = length(graph.neighbors)
Base.getindex(graph::KNNGraph, i::Integer) = graph.neighbors[i]
Base.eltype(::Type{KNNGraph{I,M}}) where {I,M} = Vector{I}
Base.iterate(graph::KNNGraph) = iterate(graph.neighbors)
Base.iterate(graph::KNNGraph, state) = iterate(graph.neighbors, state)

"""
    build_knn_graph(index, data; k, include_self=false, directed=true, query_kwargs...)

Construct a `KNNGraph` by querying `index` with every column of `data`. The
`k` parameter either comes from the caller or, if omitted, from
`configured_k(index)`. When `include_self=false` (default) the graph omits
self-loops and will raise an error if the index cannot supply enough
non-self neighbors.

# Arguments
- `directed::Bool=true`: If `false`, symmetrize the graph by taking the union of
  i→j and j→i edges.

  **⚠️ Warning**: This changes the semantics of `k`:
  - After symmetrization, `graph.k` becomes the **maximum degree** across all nodes
  - Individual nodes may have varying degrees (typically 1.5-2x the original k)
  - The original k is preserved in `graph.metadata.original_k`
  - This is required for Ollivier-Ricci curvature (orcml configuration)
  - Downstream code should NOT assume `graph.k == original_k` for undirected graphs
"""
function build_knn_graph(
    index::AbstractANNIndex,
    data::AbstractMatrix{T};
    k::Union{Nothing,Integer} = nothing,
    include_self::Bool = false,
    directed::Bool = true,
    metadata::Union{Nothing,AbstractVector} = nothing,
    query_kwargs...,
) where {T}
    n_points = size(data, 2)
    n_points > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    resolved_k = _resolve_graph_k(index, k)
    max_neighbors = n_points - (include_self ? 0 : 1)
    (0 < resolved_k <= max_neighbors) || throw(
        ArgumentError(
            "k must satisfy 0 < k <= $max_neighbors for dataset with $n_points points",
        ),
    )

    adjacency = Vector{Vector{Int}}(undef, n_points)
    request_k = resolved_k + (include_self ? 0 : 1)
    @views for col in 1:n_points
        neighbors = query(index, data, data[:, col], request_k; query_kwargs...)
        ids = neighbor_ids(neighbors)
        adjacency[col] = _prepare_neighbors(ids, col, resolved_k, include_self)
    end

    # Symmetrize if requested
    original_k = resolved_k  # Store original k before potential update
    if !directed
        adjacency = _symmetrize_adjacency(adjacency, n_points)
        # Update k to reflect actual max degree after symmetrization
        resolved_k = maximum(length(neighbors) for neighbors in adjacency)
    end

    # Create graph metadata (structural + optional node-level)
    graph_metadata = _create_graph_metadata(original_k, directed, metadata, n_points)

    return KNNGraph(adjacency, resolved_k, include_self, graph_metadata)
end

function _resolve_graph_k(index::AbstractANNIndex, k::Union{Nothing,Integer})
    configured = configured_k(index)
    if k === nothing
        configured !== nothing ||
            throw(
                ArgumentError(
                    "Index $(typeof(index)) does not expose a default k; pass `k=` explicitly",
                ),
            )
        return configured
    end
    k > 0 || throw(ArgumentError("k must be positive"))
    if configured !== nothing && k > configured
        throw(
            ArgumentError(
                "Requested k=$k exceeds index configuration k=$configured for $(typeof(index))",
            ),
        )
    end
    return Int(k)
end

function _prepare_neighbors(
    ids::Vector{Int},
    self_index::Int,
    k::Int,
    include_self::Bool,
)
    if include_self
        return _trim_or_fail(ids, k)
    end

    filtered = Int[]
    for id in ids
        id == self_index && continue
        push!(filtered, id)
        length(filtered) == k && break
    end

    length(filtered) == k ||
        throw(
            ArgumentError(
                "Index could only supply $(length(filtered)) usable neighbors for point $self_index. " *
                "Consider lowering k or setting include_self=true.",
            ),
        )
    return filtered
end

function _trim_or_fail(ids::Vector{Int}, k::Int)
    length(ids) >= k ||
        throw(
            ArgumentError(
                "Index returned only $(length(ids)) neighbors but graph requires k=$k",
            ),
        )
    if length(ids) > k
        resize!(ids, k)
    end
    return ids
end

function _prepare_metadata(metadata, n_points::Int)
    metadata === nothing && return nothing
    length(metadata) == n_points ||
        throw(
            ArgumentError(
                "Metadata length $(length(metadata)) must equal number of points $n_points",
            ),
        )
    return collect(metadata)
end

"""
    _create_graph_metadata(original_k, directed, node_metadata, n_points)

Create graph metadata structure containing both structural and node-level information.

The metadata is a named tuple with:
- `original_k`: The k value requested at graph construction (before symmetrization)
- `directed`: Whether the graph is directed (false = symmetrized)
- `node_metadata`: Optional per-node metadata (or nothing)

This structure allows downstream code to:
1. Access the original k for algorithms that need it (e.g., effective_epsilon)
2. Determine if graph is directed or undirected (important for geodesic metrics)
3. Attach custom per-node data (labels, coordinates, etc.)
"""
function _create_graph_metadata(
    original_k::Int,
    directed::Bool,
    node_metadata::Union{Nothing,AbstractVector},
    n_points::Int
)
    stored_node_metadata = _prepare_metadata(node_metadata, n_points)

    return (
        original_k = original_k,
        directed = directed,
        node_metadata = stored_node_metadata
    )
end

"""
    _symmetrize_adjacency(adjacency, n_points)

Symmetrize a directed k-NN graph by taking the union of i→j and j→i edges.
If node i has j in its k-NN, and/or j has i in its k-NN, then both i and j
will have each other as neighbors in the symmetrized graph.

This creates an undirected graph where nodes may have varying degrees
(typically more than the original k).
"""
function _symmetrize_adjacency(adjacency::Vector{Vector{Int}}, n_points::Int)
    # Use sets for efficient union
    symmetric = [Set{Int}() for _ in 1:n_points]

    # Add all edges and their reverse
    for i in 1:n_points
        for j in adjacency[i]
            push!(symmetric[i], j)  # i→j
            push!(symmetric[j], i)  # j→i (reverse)
        end
    end

    # Convert back to sorted vectors
    return [sort!(collect(s)) for s in symmetric]
end

has_metadata(graph::KNNGraph) = graph.metadata !== nothing && graph.metadata.node_metadata !== nothing

function node_metadata(graph::KNNGraph, i::Integer)
    graph.metadata !== nothing ||
        throw(ArgumentError("Graph does not store metadata"))
    graph.metadata.node_metadata !== nothing ||
        throw(ArgumentError("Graph does not store node-level metadata"))
    return graph.metadata.node_metadata[i]
end

# Access structural metadata (original_k, directed)
graph_metadata(graph::KNNGraph) = graph.metadata

# Edge-level detection signals (Jaccard, Gabriel, tangent angle) live in
# graphs/detection_signals.jl.
