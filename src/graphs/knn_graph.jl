"""
    KNNGraph

Lightweight representation of a directed k-nearest-neighbor graph. Stores the
ordered neighbor lists per vertex along with bookkeeping that downstream
algorithms can inspect.
"""
struct KNNGraph{I<:Integer}
    neighbors::Vector{Vector{I}}
    k::Int
    include_self::Bool
end

Base.length(graph::KNNGraph) = length(graph.neighbors)
Base.getindex(graph::KNNGraph, i::Integer) = graph.neighbors[i]
Base.eltype(::Type{KNNGraph{I}}) where {I} = Vector{I}
Base.iterate(graph::KNNGraph) = iterate(graph.neighbors)
Base.iterate(graph::KNNGraph, state) = iterate(graph.neighbors, state)

"""
    build_knn_graph(index, data; k, include_self=false, query_kwargs...)

Construct a `KNNGraph` by querying `index` with every column of `data`. The
`k` parameter either comes from the caller or, if omitted, from
`configured_k(index)`. When `include_self=false` (default) the graph omits
self-loops and will raise an error if the index cannot supply enough
non-self neighbors.
"""
function build_knn_graph(
    index::AbstractANNIndex,
    data::AbstractMatrix{T};
    k::Union{Nothing,Integer} = nothing,
    include_self::Bool = false,
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
        result = Vector{Int}(query(index, data, data[:, col], request_k; query_kwargs...))
        adjacency[col] = _prepare_neighbors(result, col, resolved_k, include_self)
    end

    return KNNGraph(adjacency, resolved_k, include_self)
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
