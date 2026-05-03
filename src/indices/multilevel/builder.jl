"""
Builder functions for constructing multi-level indices from configurations.

The build process:
1. User provides a declarative config tree (TransformedConfig or TerminalConfig)
2. build_index recursively processes the config:
   - Fits transforms on data
   - Partitions data according to transform bucketing
   - Recursively builds child indices
3. Returns a fully constructed MultiLevelIndex ready for querying
"""

# Import transform utilities
using ...ManifoldANN:
    has_bucketing,
    partition_by_transform,
    apply_transform_batch,
    default_distance,
    preserves_data,
    take_pending_assignments!

"""
    build_index(
        ::Type{MultiLevelIndex},
        X::AbstractMatrix,
        config::TransformedConfig;
        merge_strategy::AbstractMergeStrategy = SimpleMerge(),
        distance = default_distance,
    )::MultiLevelIndex

Build a multi-level index from a declarative configuration tree.

Child indices are built in parallel using Julia threads for improved performance.

# Arguments
- `MultiLevelIndex`: Type marker for dispatch
- `X`: Training data matrix (d × n) where each column is a data point
- `config`: Configuration tree specifying index structure
- `merge_strategy`: Strategy for merging results from multiple probes (default: SimpleMerge)
- `distance`: Fallback distance function used when terminal children do not expose their own

# Returns
- Fully constructed MultiLevelIndex ready for querying

# Examples
```julia
# IVF: KMeans(100) → HNSW per cluster
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean(), init=:kmeans_plus_plus),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

index = build_index(MultiLevelIndex, X, config)
```
"""
function build_index(
    ::Type{MultiLevelIndex},
    X::AbstractMatrix,
    config::TransformedConfig;
    merge_strategy::AbstractMergeStrategy = SimpleMerge(),
    distance::D = default_distance,
) where {D}
    # Build the root TransformedIndex recursively
    root = _build_transformed(X, config)

    # Wrap in MultiLevelIndex with merge strategy and fallback distance
    return MultiLevelIndex(root, merge_strategy, distance)
end

"""
    _build_transformed(X::AbstractMatrix, config::TransformedConfig)::TransformedIndex

Internal function to build a TransformedIndex from a config.

This function:
1. Fits the transform on the data
2. Checks if transform produces bucketing
3. If bucketing: partitions data and builds one child per bucket (in parallel)
4. If no bucketing: transforms data and builds single child
5. Returns TransformedIndex with fitted transform and children

# Arguments
- `X`: Data matrix (d × n)
- `config`: TransformedConfig specifying transform, routing, and child config

# Returns
- TransformedIndex with fitted transform and constructed children
"""
function _build_transformed(X::AbstractMatrix, config::TransformedConfig)
    # Step 1: Fit transform on data
    ManifoldANN.fit!(config.transform, X)

    # Step 2: Check if transform produces bucketing
    # Sample one point to check assignment type
    sample_result = ManifoldANN.transform(config.transform, X[:, 1])

    preserves = ManifoldANN.preserves_data(config.transform)
    precomputed_assignments = preserves ?
        ManifoldANN.take_pending_assignments!(config.transform) :
        nothing
    if precomputed_assignments !== nothing && length(precomputed_assignments) != size(X, 2)
        # Safety: discard mismatched assignments rather than erroring
        precomputed_assignments = nothing
    end

    if has_bucketing(sample_result.assignment)
        # Step 3a: Partition data by bucket assignments
        partitions, id_mappings = partition_by_transform(
            X,
            config.transform;
            capture_data = !preserves,
            precomputed_assignments = preserves ? precomputed_assignments : nothing,
        )

        # Step 3b: Build children only for non-empty partitions
        bucket_lookup = fill(0, length(id_mappings))
        child_inputs = Any[]
        child_id_mappings = Vector{Vector{Int}}()
        child_bucket_ids = Int[]

        for bucket_id in eachindex(id_mappings)
            ids = id_mappings[bucket_id]
            isempty(ids) && continue
            push!(child_bucket_ids, bucket_id)
            push!(child_id_mappings, ids)
            child_input = preserves ? _materialize_partition(X, ids) : partitions[bucket_id]
            push!(child_inputs, child_input)
        end

        n_children = length(child_inputs)
        n_children > 0 ||
            throw(
                ArgumentError(
                    "Transform produced no non-empty buckets; cannot build child indices",
                ),
            )

        # Build all children in parallel into a Vector{Any} buffer, then
        # narrow to Vector{ChildType} once we know the concrete return type.
        # `_per_child_config` returns a fresh copy when needed (nested
        # TransformedConfig — its `transform` field is fitted in-place) and
        # the original otherwise (TerminalConfig — purely immutable).
        children_buf = Vector{Any}(undef, n_children)
        stored_child_data = preserves ? nothing : Vector{typeof(child_inputs[1])}(undef, n_children)

        Threads.@threads for idx in 1:n_children
            children_buf[idx] = _build_from_config(child_inputs[idx], _per_child_config(config.child_config))
            if !preserves
                stored_child_data[idx] = child_inputs[idx]
            end
        end

        ChildType = typeof(children_buf[1])
        children = Vector{ChildType}(undef, n_children)
        @inbounds for idx in 1:n_children
            children[idx] = children_buf[idx]
        end

        # Populate lookup after construction to avoid races
        @inbounds for (pos, bucket_id) in enumerate(child_bucket_ids)
            bucket_lookup[bucket_id] = pos
        end

        return TransformedIndex(
            config.transform,
            config.routing,
            children,
            child_id_mappings,
            stored_child_data,
            bucket_lookup,
        )
    else
        # Step 3a: No bucketing - transform all data
        if preserves
            child_input = X
            stored_child_data = nothing
        else
            child_input = apply_transform_batch(config.transform, X)
            stored_child_data = [child_input]
        end

        # Step 3b: Build single child with transformed data
        children = [_build_from_config(child_input, config.child_config)]

        # Step 4: Create TransformedIndex without ID mappings and with transformed data
        return TransformedIndex(
            config.transform,
            config.routing,
            children,
            nothing,
            stored_child_data,
            nothing,
        )
    end
end

"""
    _build_from_config(X::AbstractMatrix, config::TerminalConfig{I})::I where I

Build a terminal index from a TerminalConfig.

# Arguments
- `X`: Data matrix (d × n)
- `config`: TerminalConfig specifying index type and parameters

# Returns
- Constructed terminal index of type I
"""
function _build_from_config(X::AbstractMatrix, config::TerminalConfig{I}) where I
    # Build terminal index using the specified type and parameters
    # Convert NamedTuple to keyword arguments
    return build_index(config.index_type, X; config.params...)
end

"""
    _build_from_config(X::AbstractMatrix, config::TransformedConfig)::TransformedIndex

Recursive case: build a TransformedIndex from a TransformedConfig.

# Arguments
- `X`: Data matrix (d × n)
- `config`: TransformedConfig specifying another level of transformation

# Returns
- TransformedIndex (recursively built)
"""
function _build_from_config(X::AbstractMatrix, config::TransformedConfig)
    # Recursive case: build another TransformedIndex
    return _build_transformed(X, config)
end

# TerminalConfig is fully immutable (params is a NamedTuple of immutables);
# share across worker tasks. TransformedConfig holds an AbstractTransform
# that is fitted in-place during build, so each task gets its own copy.
@inline _per_child_config(c::TerminalConfig) = c
@inline _per_child_config(c::TransformedConfig) = deepcopy(c)

@inline function _materialize_partition(X::AbstractMatrix, ids::Vector{Int})
    # Materialize a contiguous Matrix rather than a non-strided SubArray.
    # Downstream transforms (e.g. KMeansTransform via mul!/BLAS) require
    # a stride-1 column layout; a `view(X, :, ids)` is a gather pattern
    # that defeats BLAS dispatch and breaks `::Matrix`-typed signatures.
    return X[:, ids]
end
