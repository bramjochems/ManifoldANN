"""
Query logic for multi-level indices.

The query process:
1. Transform query point at each level
2. Use routing strategy to select child indices to probe
3. Recursively query selected children
4. Merge results using the merge strategy
"""

"""
    query(
        index::MultiLevelIndex,
        data::AbstractMatrix,
        q::AbstractVector,
        k::Integer;
        kwargs...
    )::Vector{Int}

Query a multi-level index for k approximate nearest neighbors.

# Arguments
- `index`: MultiLevelIndex to query
- `data`: Original data matrix (d × n) - passed to terminal indices
- `q`: Query point (d-dimensional vector)
- `k`: Number of neighbors to return
- `kwargs...`: Additional keyword arguments (currently unused, for future extensions)

# Returns
- Vector of point IDs (up to k neighbors), sorted by distance

# Examples
```julia
# Build IVF index
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16,))
)
index = build_index(MultiLevelIndex, X, config)

# Query for 10 nearest neighbors
q = rand(Float32, size(X, 1))
neighbors = query(index, X, q, 10)
```
"""
function query(
    index::MultiLevelIndex,
    data::AbstractMatrix,
    q::AbstractVector,
    k::Integer;
    kwargs...
)
    # Query recursively through the index tree
    all_results = _query_recursive(index.root, data, q, k, index.distance)

    # Merge results using the merge strategy
    merged_neighbors = merge_results(index.merge_strategy, all_results, k)

    # Extract IDs for API consistency with other indices
    return [neighbor.id for neighbor in merged_neighbors]
end

"""
    query(index::MultiLevelIndex, data::AbstractMatrix, queries::AbstractMatrix, k::Integer)

Batch query variant for multi-level indices. Processes each column independently and
aggregates the neighbor id lists.
"""
function query(
    index::MultiLevelIndex,
    data::AbstractMatrix,
    queries::AbstractMatrix,
    k::Integer;
    kwargs...
)
    size(queries, 1) == size(data, 1) ||
        throw(DimensionMismatch("Expected queries with $(size(data, 1)) rows"))

    n_queries = size(queries, 2)
    results = Vector{Vector{Int}}(undef, n_queries)

    @inbounds for i in 1:n_queries
        q = @view queries[:, i]
        results[i] = query(index, data, q, k; kwargs...)
    end

    return results
end

"""
    query(index::MultiLevelIndex, data::AbstractMatrix, queries::Vector{<:AbstractVector}, k::Integer)

Convenience overload that accepts a vector of query vectors.
"""
function query(
    index::MultiLevelIndex,
    data::AbstractMatrix,
    queries::Vector{<:AbstractVector},
    k::Integer;
    kwargs...
)
    isempty(queries) && return Vector{Vector{Int}}()
    queries_mat = reduce(hcat, queries)
    return query(index, data, queries_mat, k; kwargs...)
end

"""
    _query_recursive(
        node::TransformedIndex,
        data::AbstractMatrix,
        q::AbstractVector,
        k::Integer,
        fallback_distance::Function
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
    k::Integer,
    fallback_distance::Function
)
    # Step 1: Transform query point
    result = ManifoldANN.transform(node.transform, q)
    q_transformed = result.data
    assignment = result.assignment

    # Step 2: Select indices to probe based on routing strategy
    probe_indices = select_indices(node.routing_strategy, assignment, node.indices)

    # Step 3: Query each selected child index
    # Each child may return different result types depending on whether it's
    # a TransformedIndex (returns Vector{Vector{Neighbor}}) or terminal (needs conversion)
    results = Vector{Vector{Neighbor{Float32}}}()

    for (i, child) in enumerate(probe_indices)
        # Find which child index this is
        child_idx = findfirst(==(child), node.indices)

        # Get the appropriate data for this child
        child_data = if !isnothing(node.partition_data)
            node.partition_data[child_idx]
        else
            data  # Use original data if no partitioning
        end

        child_results = _query_node(child, child_data, q_transformed, k, fallback_distance)

        # Map local IDs to global IDs if we have ID mappings
        if !isnothing(node.id_mappings)
            id_mapping = node.id_mappings[child_idx]

            # Remap IDs in all result lists
            for result_list in child_results
                remapped = [Neighbor(id_mapping[n.id], n.dist) for n in result_list]
                push!(results, remapped)
            end
        else
            append!(results, child_results)
        end
    end

    return results
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
    k::Integer,
    fallback_distance::Function
)
    return _query_recursive(node, data, q, k, fallback_distance)
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
    k::Integer,
    fallback_distance::Function
)
    # Query terminal index (returns Vector{Int})
    ids = query(index, data, q, k)

    distance_fn = index_distance(index)
    distance_fn === nothing && (distance_fn = fallback_distance)

    neighbors = Neighbor{Float32}[]
    for id in ids
        point = @view data[:, id]
        dist = Float32(distance_fn(point, q))
        push!(neighbors, Neighbor(id, dist))
    end

    # Return as single-element vector for consistency
    return [neighbors]
end
