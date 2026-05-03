"""
Query logic for multi-level indices.

The query process:
1. Transform query point at each level
2. Use routing strategy to select child indices to probe
3. Recursively query selected children
4. Merge results using the merge strategy
"""

struct BucketProxy
    count::Int
end

Base.length(proxy::BucketProxy) = proxy.count
Base.eachindex(proxy::BucketProxy) = Base.OneTo(proxy.count)

"""
    query(index::MultiLevelIndex, data, q, k; kwargs...) -> Vector{Neighbor}

Query a multi-level index for k approximate nearest neighbors. Each neighbor
includes both the identifier and the distance produced by the probed child
index, enabling merge strategies to reuse those computations.
"""
function query(
    index::MultiLevelIndex,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    kwargs...
) where {T}
    # Query recursively through the index tree
    all_results = _query_recursive(index.root, data, q, k; kwargs...)

    # Merge results using the merge strategy
    merged_neighbors = merge_results(index.merge_strategy, all_results, k)

    return merged_neighbors
end

"""
    _query_recursive(
        node::TransformedIndex,
        data::AbstractMatrix,
        q::AbstractVector,
        k::Integer
    )::Vector{Vector{Neighbor}}

Recursively query a TransformedIndex node.

This function:
1. Transforms the query point
2. Selects child indices to probe using routing strategy
3. Recursively queries each selected child
4. Returns all result lists (not yet merged)

# Arguments
- `node`: TransformedIndex to query
- `data`: Original data matrix (passed down to terminal indices)
- `q`: Query point (may be transformed at each level)
- `k`: Number of neighbors to find

# Returns
- Vector of result lists, one per probed child index
"""
function _query_recursive(
    node::TransformedIndex,
    data::AbstractMatrix,
    q::AbstractVector,
    k::Integer;
    kwargs...
)
    # Step 1: Transform query point
    result = ManifoldANN.transform(node.transform, q)
    q_transformed = result.data
    assignment = result.assignment

    # Step 2: Select indices to probe based on routing strategy
    route_domain = node.bucket_lookup === nothing ?
        node.indices :
        BucketProxy(length(node.bucket_lookup))
    probe_indices = select_indices(node.routing_strategy, assignment, route_domain)
    child_positions = _resolve_child_positions(node, probe_indices)

    # Step 3: Query each selected child index
    distance_type = _child_distance_type(node, data)
    results = Vector{Vector{Neighbor{distance_type}}}()

    for child_idx in child_positions
        child = node.indices[child_idx]
        child_data = _resolve_child_data(node, child_idx, data)

        child_results = _query_node(
            child,
            child_data,
            q_transformed,
            k;
            kwargs...,
        )

        id_mapping = isnothing(node.id_mappings) ? nothing : node.id_mappings[child_idx]
        _append_child_results!(results, child_results, distance_type, id_mapping)
    end

    return results
end

function _resolve_child_positions(node::TransformedIndex, probe_indices::AbstractVector{Int})
    node.bucket_lookup === nothing && return probe_indices
    lookup = node.bucket_lookup
    selected = Vector{Int}()
    reserve = length(probe_indices)
    reserve > 0 && sizehint!(selected, reserve)
    @inbounds for bucket in probe_indices
        (bucket < 1 || bucket > length(lookup)) && continue
        child_idx = lookup[bucket]
        child_idx == 0 && continue
        push!(selected, child_idx)
    end
    return selected
end

"""
    _query_node(
        node::TransformedIndex,
        data::AbstractMatrix,
        q::AbstractVector,
        k::Integer,
        fallback_distance
    )::Vector{Vector{Neighbor}}

Query a TransformedIndex child (recursive case).

# Returns
- Vector of result lists from this subtree
"""
function _query_node(
    node::TransformedIndex,
    data::AbstractMatrix,
    q::AbstractVector,
    k::Integer;
    kwargs...
)
    return _query_recursive(node, data, q, k; kwargs...)
end

"""
    _query_node(
        index::AbstractANNIndex,
        data::AbstractMatrix,
        q::AbstractVector,
        k::Integer,
        fallback_distance
    )::Vector{Vector{Neighbor}}

Query a terminal index (base case).

Terminal indices return Vector{Int} (point IDs). We need to:
1. Query the terminal index
2. Compute distances for returned IDs
3. Convert to Vector{Neighbor}
4. Wrap in outer vector for consistency

# Returns
- Single-element vector containing the neighbor list from this terminal index
"""
function _query_node(
    index::AbstractANNIndex,
    data::AbstractMatrix,
    q::AbstractVector,
    k::Integer;
    kwargs...
)
    # Query terminal index (returns Vector{Neighbor})
    neighbors = query(index, data, q, k; kwargs...)
    return [neighbors]
end

# Type-stable child-data resolution. Both `child_data` (C) and `id_mappings`
# (M) are encoded as type parameters of TransformedIndex, so each method
# below has a single concrete return type and the per-cluster `_query_node`
# call at the use site can be devirtualised.

# Stored per-child data: return the materialised contiguous matrix.
@inline function _resolve_child_data(
    node::TransformedIndex{<:Any,<:Any,<:AbstractVector{<:AbstractMatrix},<:Any},
    child_idx::Int,
    parent_data::AbstractMatrix,
)
    return node.child_data[child_idx]
end

# No stored data, but bucketed: gather a view over parent ids.
@inline function _resolve_child_data(
    node::TransformedIndex{<:Any,<:Any,Nothing,Vector{Vector{Int}}},
    child_idx::Int,
    parent_data::AbstractMatrix,
)
    ids = node.id_mappings[child_idx]
    return view(parent_data, :, ids)
end

# No stored data, no bucketing: pass parent data through.
@inline function _resolve_child_data(
    node::TransformedIndex{<:Any,<:Any,Nothing,Nothing},
    child_idx::Int,
    parent_data::AbstractMatrix,
)
    return parent_data
end

@inline function _child_distance_type(
    node::TransformedIndex{<:Any,<:Any,<:AbstractVector{<:AbstractMatrix},<:Any},
    parent_data::AbstractMatrix,
)
    return float(eltype(node.child_data[1]))
end

@inline function _child_distance_type(
    node::TransformedIndex{<:Any,<:Any,Nothing,<:Any},
    parent_data::AbstractMatrix,
)
    return float(eltype(parent_data))
end

function _append_child_results!(
    results::Vector{Vector{Neighbor{S}}},
    child_results::Vector{<:AbstractVector{<:Neighbor}},
    ::Type{S},
    id_mapping::Union{Nothing, Vector{Int}},
) where {S<:AbstractFloat}
    for result_list in child_results
        # Fast path: no id mapping and types already match
        if id_mapping === nothing && result_list isa Vector{Neighbor{S}}
            push!(results, result_list)
            continue
        end

        converted = Vector{Neighbor{S}}(undef, length(result_list))
        @inbounds for i in eachindex(result_list)
            neighbor = result_list[i]
            mapped_id = id_mapping === nothing ? neighbor.id : id_mapping[neighbor.id]
            converted[i] = Neighbor{S}(mapped_id, S(neighbor.dist))
        end
        push!(results, converted)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Inverted-loop batch query path for MultiLevelIndex
# ---------------------------------------------------------------------------
#
# The generic batch fallback in ann_index.jl iterates over query columns and
# calls the single-vector method per query, which in turn calls the
# single-vector terminal-index method per probed cluster. For HNSW children
# that bypasses the BatchScratchPool acquired by the matrix-input HNSW
# method.
#
# This batch method instead routes ALL queries first, then *inverts the
# loop*: for each child it gathers the queries that probe it into a single
# sub-batch and issues ONE matrix-input call to that child's `query` method.
# That hits the HNSW matrix path (and its scratch pool) once per child
# instead of once per (query, probed-cluster) pair.

"""
    query(index::MultiLevelIndex, data, Q::AbstractMatrix, k; kwargs...)

Inverted-loop batch query. Routes every column of `Q` to its target children
once, then issues one matrix-input call per child with the gathered
sub-batch. This lets terminal indices that ship a specialised matrix
overload (e.g. HNSW's `BatchScratchPool` path) amortise scratch allocation
across all queries hitting that child.

Equivalent in result to mapping `query(index, data, view(Q, :, i), k)` over
the columns; the loop reorder is purely an allocation/dispatch
optimisation.
"""
function query(
    index::MultiLevelIndex,
    data::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    k::Integer;
    kwargs...,
) where {T}
    size(Q, 1) == size(data, 1) ||
        throw(DimensionMismatch("Expected queries with $(size(data, 1)) rows, got $(size(Q, 1))"))

    n_queries = size(Q, 2)
    S = float(T)
    n_queries == 0 && return Vector{Vector{Neighbor{S}}}()

    per_query_results = _query_recursive_batch(index.root, data, Q, k; kwargs...)

    final = Vector{Vector{Neighbor{S}}}(undef, n_queries)
    @inbounds for i in 1:n_queries
        final[i] = merge_results(index.merge_strategy, per_query_results[i], k)
    end
    return final
end

"""
    _query_recursive_batch(node, data, Q::AbstractMatrix, k; kwargs...)

Batched analogue of `_query_recursive`. Returns a Vector of length
`size(Q, 2)`; entry `i` is the list of per-probed-child result lists for
query column `i` (same shape as `_query_recursive` returns for one query).
"""
function _query_recursive_batch(
    node::TransformedIndex,
    data::AbstractMatrix,
    Q::AbstractMatrix,
    k::Integer;
    kwargs...,
)
    n_queries = size(Q, 2)
    S = _child_distance_type(node, data)

    # Step 1: per-query route. Single-vector transform/select_indices keeps
    # the implementation compatible with every existing transform/routing
    # combo without committing to a batched transform API. The work here is
    # tiny relative to the actual neighbour search, so leaving it
    # column-wise is fine.
    probe_positions = Vector{Vector{Int}}(undef, n_queries)
    # Cache transformed query data for the gather step. Only materialised
    # when the transform does NOT preserve the input representation; if it
    # does, we view the original Q columns directly.
    transform_preserves = preserves_data(node.transform)
    transformed_cols = transform_preserves ? nothing :
        Vector{Any}(undef, n_queries)

    @inbounds for i in 1:n_queries
        result = ManifoldANN.transform(node.transform, view(Q, :, i))
        if !transform_preserves
            transformed_cols[i] = result.data
        end
        route_domain = node.bucket_lookup === nothing ?
            node.indices :
            BucketProxy(length(node.bucket_lookup))
        probe_indices = select_indices(node.routing_strategy, result.assignment, route_domain)
        probe_positions[i] = _resolve_child_positions(node, probe_indices)
    end

    # Step 2: invert the loop — for each child, collect the queries probing
    # it. queries_per_child[c] is the list of (query_index, slot_in_probe_list)
    # pairs so we can scatter results back to the right per-query bucket.
    n_children = length(node.indices)
    queries_per_child = [Tuple{Int,Int}[] for _ in 1:n_children]
    @inbounds for qi in 1:n_queries
        positions = probe_positions[qi]
        for (slot, child_idx) in pairs(positions)
            push!(queries_per_child[child_idx], (qi, slot))
        end
    end

    # Per-query result containers. A child may contribute multiple result
    # lists (e.g. recursive TransformedIndex children), so we push! rather
    # than pre-sizing by probe count.
    per_query_results = Vector{Vector{Vector{Neighbor{S}}}}(undef, n_queries)
    @inbounds for qi in 1:n_queries
        per_query_results[qi] = Vector{Vector{Neighbor{S}}}()
        sizehint!(per_query_results[qi], length(probe_positions[qi]))
    end

    # Step 3: for each child, build a sub-batch matrix and dispatch ONCE.
    @inbounds for child_idx in 1:n_children
        assignments = queries_per_child[child_idx]
        isempty(assignments) && continue

        child = node.indices[child_idx]
        child_data = _resolve_child_data(node, child_idx, data)
        id_mapping = isnothing(node.id_mappings) ? nothing : node.id_mappings[child_idx]

        Q_sub = _gather_sub_batch(Q, transformed_cols, assignments, transform_preserves)

        _query_node_batch_into!(
            per_query_results, child, child_data, Q_sub, k, assignments, S, id_mapping;
            kwargs...,
        )
    end

    return per_query_results
end

# Materialise a contiguous (d × sub_n) Matrix of queries probing one child.
# When the transform preserves data we can copy from the original Q;
# otherwise we copy from the cached transformed column vectors.
function _gather_sub_batch(
    Q::AbstractMatrix{T},
    transformed_cols::Union{Nothing,Vector{Any}},
    assignments::Vector{Tuple{Int,Int}},
    preserves::Bool,
) where {T}
    d = size(Q, 1)
    sub_n = length(assignments)
    if preserves
        Q_sub = Matrix{T}(undef, d, sub_n)
        @inbounds for j in 1:sub_n
            qi, _ = assignments[j]
            for r in 1:d
                Q_sub[r, j] = Q[r, qi]
            end
        end
        return Q_sub
    else
        # transformed_cols[qi] is whatever the transform returned. We assume
        # all entries share a length and elementwise-copyable element type.
        first_qi, _ = assignments[1]
        proto = transformed_cols[first_qi]
        Tt = eltype(proto)
        d_t = length(proto)
        Q_sub = Matrix{Tt}(undef, d_t, sub_n)
        @inbounds for j in 1:sub_n
            qi, _ = assignments[j]
            col = transformed_cols[qi]
            for r in 1:d_t
                Q_sub[r, j] = col[r]
            end
        end
        return Q_sub
    end
end

# Dispatch the batched query at one child and scatter results back into
# `per_query_results`. Two methods: terminal (AbstractANNIndex) and
# recursive (TransformedIndex).

function _query_node_batch_into!(
    per_query_results::Vector{Vector{Vector{Neighbor{S}}}},
    index::AbstractANNIndex,
    child_data::AbstractMatrix,
    Q_sub::AbstractMatrix,
    k::Integer,
    assignments::Vector{Tuple{Int,Int}},
    ::Type{S},
    id_mapping::Union{Nothing,Vector{Int}};
    kwargs...,
) where {S<:AbstractFloat}
    # ONE matrix-input call. For HNSW this acquires from the
    # BatchScratchPool. For other indices it falls back to the generic
    # threaded-over-columns method, but at the granularity of this sub-batch
    # rather than the full IVF query batch.
    sub_results = query(index, child_data, Q_sub, k; kwargs...)

    @inbounds for j in eachindex(assignments)
        qi, _ = assignments[j]
        neighbors = sub_results[j]
        if id_mapping === nothing && neighbors isa Vector{Neighbor{S}}
            push!(per_query_results[qi], neighbors)
        else
            converted = Vector{Neighbor{S}}(undef, length(neighbors))
            for i in eachindex(neighbors)
                nb = neighbors[i]
                mapped_id = id_mapping === nothing ? nb.id : id_mapping[nb.id]
                converted[i] = Neighbor{S}(mapped_id, S(nb.dist))
            end
            push!(per_query_results[qi], converted)
        end
    end
    return nothing
end

function _query_node_batch_into!(
    per_query_results::Vector{Vector{Vector{Neighbor{S}}}},
    node::TransformedIndex,
    child_data::AbstractMatrix,
    Q_sub::AbstractMatrix,
    k::Integer,
    assignments::Vector{Tuple{Int,Int}},
    ::Type{S},
    id_mapping::Union{Nothing,Vector{Int}};
    kwargs...,
) where {S<:AbstractFloat}
    # Recursive case: the child is itself a TransformedIndex. Recurse into
    # the batched path so its grandchildren also get batched dispatch.
    nested = _query_recursive_batch(node, child_data, Q_sub, k; kwargs...)
    # `nested[j]` is a Vector{Vector{Neighbor{S'}}} for sub-batch query j —
    # one inner list per probed grandchild. Push each inner list as its own
    # entry into the parent query's result lists (matching the
    # `_append_child_results!` behaviour on the single-vector path).
    @inbounds for j in eachindex(assignments)
        qi, _ = assignments[j]
        sublists = nested[j]
        for r in sublists
            if id_mapping === nothing && r isa Vector{Neighbor{S}}
                push!(per_query_results[qi], r)
            else
                converted = Vector{Neighbor{S}}(undef, length(r))
                for i in eachindex(r)
                    nb = r[i]
                    mapped_id = id_mapping === nothing ? nb.id : id_mapping[nb.id]
                    converted[i] = Neighbor{S}(mapped_id, S(nb.dist))
                end
                push!(per_query_results[qi], converted)
            end
        end
    end
    return nothing
end
