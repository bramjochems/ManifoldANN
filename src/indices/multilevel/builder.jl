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

using Random: AbstractRNG
import Random

# Import transform utilities
using ...ManifoldANN:
    has_bucketing,
    partition_by_transform,
    apply_transform_batch,
    default_distance,
    preserves_data,
    take_pending_assignments!,
    spawn_child_rngs

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
    rng::AbstractRNG = Random.default_rng(),
) where {D}
    # Build the root TransformedIndex recursively. The rng is threaded so
    # that KMeansTransform / RandomProjectionTransform fits become
    # reproducible and re-entrant under threaded child builds.
    root = _build_transformed(X, config; rng = rng)

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
function _build_transformed(X::AbstractMatrix, config::TransformedConfig;
                            rng::AbstractRNG = Random.default_rng())
    # Step 1: Treat config.transform as an immutable prototype. Make a fresh
    # copy and fit it; the user's config object is never mutated by build.
    # This is what gives each TransformedIndex its own fitted state and lets
    # us share `config.child_config` across worker tasks without races.
    transform = deepcopy(config.transform)
    # Pass rng to fit! so KMeans / RandomProjection are reproducible. fit!
    # methods that don't accept the kwarg ignore it via the generic
    # AbstractTransform fallback path; transforms that recently grew an
    # `rng` kwarg (KMeansTransform, RandomProjectionTransform) consume it.
    _fit_transform!(transform, X, rng)

    # Step 2: Check if transform produces bucketing
    # Sample one point to check assignment type
    sample_result = ManifoldANN.transform(transform, X[:, 1])

    preserves = ManifoldANN.preserves_data(transform)
    precomputed_assignments = preserves ?
        ManifoldANN.take_pending_assignments!(transform) :
        nothing
    if precomputed_assignments !== nothing && length(precomputed_assignments) != size(X, 2)
        # Safety: discard mismatched assignments rather than erroring
        precomputed_assignments = nothing
    end

    if has_bucketing(sample_result.assignment)
        # Step 3a: Partition data by bucket assignments
        partitions, id_mappings = partition_by_transform(
            X,
            transform;
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
        # `config.child_config` is safe to share across tasks: TerminalConfig is
        # purely immutable, and TransformedConfig is now treated as a declarative
        # prototype — _build_transformed deepcopies its transform before fitting.
        children_buf = Vector{Any}(undef, n_children)
        stored_child_data = preserves ? nothing : Vector{typeof(child_inputs[1])}(undef, n_children)

        # Derive per-child RNGs serially, then build children in parallel.
        # Same pattern as RPTreeForestIndex / PCATreeForestIndex: this keeps
        # the build deterministic regardless of thread count or scheduling.
        child_rngs = spawn_child_rngs(rng, n_children)

        Threads.@threads for idx in 1:n_children
            children_buf[idx] = _build_from_config(child_inputs[idx], config.child_config;
                                                   rng = child_rngs[idx])
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
            transform,
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
            child_input = apply_transform_batch(transform, X)
            stored_child_data = [child_input]
        end

        # Step 3b: Build single child with transformed data
        children = [_build_from_config(child_input, config.child_config; rng = rng)]

        # Step 4: Create TransformedIndex without ID mappings and with transformed data
        return TransformedIndex(
            transform,
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
function _build_from_config(X::AbstractMatrix, config::TerminalConfig{I};
                            rng::AbstractRNG = Random.default_rng()) where I
    # Build terminal index using the specified type and parameters.
    # Convert NamedTuple to keyword arguments. The `rng` kwarg is accepted
    # for API uniformity at the recursion site but is not forwarded to
    # terminal builders, which use their own rng plumbing (config.params
    # may already supply one if the caller wants determinism).
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
function _build_from_config(X::AbstractMatrix, config::TransformedConfig;
                            rng::AbstractRNG = Random.default_rng())
    # Recursive case: build another TransformedIndex
    return _build_transformed(X, config; rng = rng)
end

# Dispatch helper: only forward `rng` to `fit!` for transforms whose
# `fit!` actually accepts an `rng` kwarg. Transforms without rng (e.g.
# PCATransform) fall through to the rng-free overload. This keeps the
# multilevel builder backwards-compatible with transforms that haven't
# adopted explicit rng plumbing.
@inline _fit_transform!(t, X::AbstractMatrix, ::AbstractRNG) =
    ManifoldANN.fit!(t, X)
@inline _fit_transform!(t::ManifoldANN.KMeansTransform, X::AbstractMatrix, rng::AbstractRNG) =
    ManifoldANN.fit!(t, X; rng = rng)
@inline _fit_transform!(t::ManifoldANN.RandomProjectionTransform, X::AbstractMatrix, rng::AbstractRNG) =
    ManifoldANN.fit!(t, X; rng = rng)

@inline function _materialize_partition(X::AbstractMatrix, ids::Vector{Int})
    # Materialize a contiguous Matrix rather than a non-strided SubArray.
    # Downstream transforms (e.g. KMeansTransform via mul!/BLAS) require
    # a stride-1 column layout; a `view(X, :, ids)` is a gather pattern
    # that defeats BLAS dispatch and breaks `::Matrix`-typed signatures.
    return X[:, ids]
end
