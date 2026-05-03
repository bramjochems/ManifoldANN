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
        fallback_distance::Function
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
        fallback_distance::Function
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

@inline function _resolve_child_data(
    node::TransformedIndex,
    child_idx::Int,
    parent_data::AbstractMatrix,
)
    if node.child_data === nothing
        if isnothing(node.id_mappings)
            return parent_data
        else
            ids = node.id_mappings[child_idx]
            @views return view(parent_data, :, ids)
        end
    else
        return node.child_data[child_idx]
    end
end

@inline function _child_distance_type(node::TransformedIndex, parent_data::AbstractMatrix)
    if node.child_data === nothing
        return float(eltype(parent_data))
    else
        first_child = node.child_data[1]
        return float(eltype(first_child))
    end
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
