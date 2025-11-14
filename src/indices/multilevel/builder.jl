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
    preserves_data

"""
    build_index(
        ::Type{MultiLevelIndex},
        X::AbstractMatrix,
        config::TransformedConfig;
        merge_strategy::AbstractMergeStrategy = SimpleMerge(),
        distance::Function = default_distance,
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
    distance::Function = default_distance,
)
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

    if has_bucketing(sample_result.assignment)
        # Step 3a: Partition data by bucket assignments
        partitions, id_mappings = partition_by_transform(
            X,
            config.transform;
            capture_data = !preserves,
        )

        # Step 3b: Build child index for each partition (in parallel)
        # CRITICAL: Must deepcopy child_config for each partition!
        #
        # TransformedConfig is immutable but holds references to mutable transform objects.
        # Without deepcopy, all children would share the same transform instance, and each
        # fit! call would overwrite the previous partition's parameters. This would break
        # multi-level hierarchies completely - all sibling nodes would end up with parameters
        # learned from only the last partition.
        #
        # OPTIMIZATION: Build children in parallel since each partition is independent after deepcopy.
        # This provides near-linear speedup for IVF indices with many clusters.
        n_partitions = length(id_mappings)
        children = Vector{AbstractANNIndex}(undef, n_partitions)
        stored_child_data = preserves ? nothing : partitions

        Threads.@threads for i in 1:n_partitions
            child_input = preserves ?
                _view_partition(X, id_mappings[i]) :
                stored_child_data[i]
            children[i] = _build_from_config(child_input, deepcopy(config.child_config))
        end

        # Step 4: Create TransformedIndex with ID mappings and optional stored data
        return TransformedIndex(
            config.transform,
            config.routing,
            children,
            id_mappings,
            stored_child_data,
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

@inline function _view_partition(X::AbstractMatrix, ids::Vector{Int})
    @views return view(X, :, ids)
end
