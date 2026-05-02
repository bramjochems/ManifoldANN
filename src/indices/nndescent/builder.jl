using Random

"""
    build_index(NNDescentIndex, data; kwargs...)

Construct an NN-Descent index over `data`. Keeps only the neighbor graph, so
callers must still pass `data` when querying. Keywords:

- `k`: Number of neighbors per point (default 32)
- `max_iterations`: Maximum NN-Descent iterations (default 10)
- `convergence_threshold`: Relative improvement threshold to stop early
- `sampling_policy`: `UniformPairSampling`, `:uniform`, or `nothing` (defaults to uniform)
- `symmetry_policy`: `FullSymmetry()`, `PrunedSymmetry(1.5)`, `NoSymmetry()`, or symbol (defaults to `:full`)
- `apply_symmetry_continuously`: If true, apply symmetry after each iteration; if false, apply only at end (default false)
- `rng`: RNG used for initialization and sampling
- `distance`: Distance function (defaults to `default_squared_distance`)
- `max_candidate_neighbors`: Maximum number of neighbor IDs per node considered each iteration (default 64)
"""
function build_index(
    ::Type{NNDescentIndex},
    data::AbstractMatrix{T};
    k::Int = NNDESCENT_DEFAULT_K,
    max_iterations::Int = NNDESCENT_DEFAULT_MAX_ITERATIONS,
    convergence_threshold::Float64 = NNDESCENT_DEFAULT_CONVERGENCE_THRESHOLD,
    sampling_policy::Union{AbstractNNDescentSamplingPolicy,Symbol,Nothing} = nothing,
    symmetry_policy::Union{AbstractSymmetryPolicy,Symbol,Nothing} = nothing,
    apply_symmetry_continuously::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
    distance::D = default_squared_distance,
    max_candidate_neighbors::Int = NNDESCENT_DEFAULT_MAX_CANDIDATE_NEIGHBORS,
) where {T<:LinearAlgebra.BlasFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    k > 0 || throw(ArgumentError("k must be positive"))
    n > 1 || throw(ArgumentError("NN-Descent requires at least two points"))
    k < n || throw(ArgumentError("k must be less than the number of points ($n)"))
    max_iterations >= 1 ||
        throw(ArgumentError("max_iterations must be at least 1"))
    convergence_threshold >= 0 ||
        throw(ArgumentError("convergence_threshold must be non-negative"))
    max_candidate_neighbors > 0 ||
        throw(ArgumentError("max_candidate_neighbors must be positive"))

    resolved_sampling_policy = _resolve_sampling_policy(sampling_policy)
    resolved_symmetry_policy = _resolve_symmetry_policy(symmetry_policy)

    # Determine the distance element type for neighbor bookkeeping
    probe = distance(view(data, :, 1), view(data, :, 1))
    probe isa AbstractFloat ||
        throw(
            ArgumentError(
                "distance function must return an AbstractFloat, got $(typeof(probe))",
            ),
        )
    dist_type = typeof(probe)

    working_graph =
        [NNDescentNeighborNode{dist_type}(k) for _ in 1:n]::Vector{
            NNDescentNeighborNode{dist_type},
        }

    _initialize_random_neighbors!(working_graph, data, k, distance, rng)
    _run_nndescent!(
        working_graph,
        data,
        distance,
        k,
        resolved_sampling_policy,
        resolved_symmetry_policy,
        apply_symmetry_continuously,
        max_iterations,
        convergence_threshold,
        rng,
        max_candidate_neighbors,
    )

    adjacency = _finalize_neighbors(working_graph, k)

    return NNDescentIndex{
        T,
        typeof(distance),
        typeof(resolved_sampling_policy),
        typeof(resolved_symmetry_policy),
    }(
        d,
        n,
        k,
        max_iterations,
        distance,
        resolved_sampling_policy,
        resolved_symmetry_policy,
        adjacency,
    )
end

function _resolve_sampling_policy(
    policy::Union{AbstractNNDescentSamplingPolicy,Symbol,Nothing},
)
    if policy === nothing
        return UniformPairSampling()
    elseif policy isa AbstractNNDescentSamplingPolicy
        return policy
    elseif policy isa Symbol
        if policy === :uniform
            return UniformPairSampling()
        else
            throw(ArgumentError("Unknown NN-Descent sampling policy $policy"))
        end
    else
        throw(ArgumentError("Unsupported sampling policy type $(typeof(policy))"))
    end
end

function _resolve_symmetry_policy(
    policy::Union{AbstractSymmetryPolicy,Symbol,Nothing},
)
    if policy === nothing
        return FullSymmetry()
    elseif policy isa AbstractSymmetryPolicy
        return policy
    elseif policy isa Symbol
        if policy === :full
            return FullSymmetry()
        elseif policy === :pruned
            return PrunedSymmetry()
        elseif policy === :none
            return NoSymmetry()
        else
            throw(ArgumentError("Unknown symmetry policy $policy. Use :full, :pruned, or :none"))
        end
    else
        throw(ArgumentError("Unsupported symmetry policy type $(typeof(policy))"))
    end
end

function _initialize_random_neighbors!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    k::Int,
    distance,
    rng::AbstractRNG,
) where {T}
    n = length(graph)
    @inbounds for i in 1:n
        node = graph[i]
        chosen = Set{Int}()
        added = 0
        while added < k
            candidate = rand(rng, 1:n)
            candidate == i && continue
            candidate in chosen && continue
            push!(chosen, candidate)
            dist = distance(view(data, :, i), view(data, :, candidate))
            # All initial neighbors are "new"
            # Add bidirectional edges (like NearestNeighborDescent.jl)
            # This improves initial graph connectivity significantly
            push!(node.new_neighbors, candidate, dist)
            # Add reverse edge using unsafe_push! to allow exceeding capacity
            # Symmetry policy will prune back to appropriate size later
            unsafe_push!(graph[candidate].new_neighbors, i, dist)
            added += 1
        end
    end
    return nothing
end

function _run_nndescent!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    distance,
    k::Int,
    sampling_policy::AbstractNNDescentSamplingPolicy,
    symmetry_policy::AbstractSymmetryPolicy,
    apply_symmetry_continuously::Bool,
    max_iterations::Int,
    convergence_threshold::Float64,
    rng::AbstractRNG,
    max_candidate_neighbors::Int,
) where {T}
    n = length(graph)
    total_edges = n * k  # Total edges in graph (used to compute improvement ratio)
    for iteration in 1:max_iterations
        updates = 0  # Count of successful neighbor updates this iteration
        @inbounds for node_idx in 1:n
            node = graph[node_idx]
            isempty(node.new_neighbors) && continue

            # Sample candidates for pairing (bounded to prevent memory explosion)
            new_candidates =
                _sample_neighbor_ids(node.new_neighbors, max_candidate_neighbors, rng)
            isempty(new_candidates) && continue
            old_candidates =
                _sample_neighbor_ids(node.old_neighbors, max_candidate_neighbors, rng)

            # NN-Descent core: evaluate all pairs of (new, new) and (new, old) candidates
            # This local join operation discovers transitive neighbors efficiently

            # Pair each new neighbor with other new neighbors (avoid self-pairs)
            for new_idx in 1:length(new_candidates)
                src = new_candidates[new_idx]
                for inner_idx in (new_idx + 1):length(new_candidates)
                    dst = new_candidates[inner_idx]
                    should_consider_pair(sampling_policy, rng) || continue
                    updates += _connect_pair!(graph, data, distance, src, dst)
                end
                # Pair new neighbors with old neighbors
                for dst in old_candidates
                    src == dst && continue
                    should_consider_pair(sampling_policy, rng) || continue
                    updates += _connect_pair!(graph, data, distance, src, dst)
                end
            end
        end

        # Apply symmetry policy after each iteration if requested
        # This maintains graph symmetry during construction, improving neighbor discovery
        if apply_symmetry_continuously
            apply_symmetry_policy!(graph, k, symmetry_policy)
        end

        # Move all "new" neighbors to "old" for next iteration
        @inbounds for node_idx in 1:n
            _transition_neighbors!(graph[node_idx])
        end

        # Compute relative improvement as fraction of total possible edges updated
        improvement = total_edges == 0 ? 0.0 : updates / total_edges
        if improvement < convergence_threshold
            break
        end
    end

    # Apply symmetry policy at the end if not applied continuously
    # When applied only once at the end, this is more efficient but may reduce graph quality
    if !apply_symmetry_continuously
        apply_symmetry_policy!(graph, k, symmetry_policy)
    end

    return nothing
end

"""
    _transition_neighbors!(node)

Move all "new" neighbors to the "old" heap for the next iteration. This clears
the new_neighbors heap, preparing it for the next round of discoveries.
"""
function _transition_neighbors!(node::NNDescentNeighborNode{T}) where {T}
    # Transfer all new neighbors to old neighbors
    for neighbor in node.new_neighbors
        push!(node.old_neighbors, neighbor.id, neighbor.dist)
    end
    # Clear the new neighbors heap for next iteration
    empty!(node.new_neighbors.data)
    return nothing
end

"""
    apply_symmetry_policy!(graph, k, policy)

Apply the specified symmetry policy to the graph after NN-Descent iterations complete.
"""
function apply_symmetry_policy!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    policy::AbstractSymmetryPolicy,
) where {T}
    # Dispatch to the appropriate implementation
    _apply_symmetry!(graph, k, policy)
end

"""
    _apply_symmetry!(graph, k, ::NoSymmetry)

No-op: keep the directed k-NN graph as-is.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    ::NoSymmetry,
) where {T}
    # Nothing to do - graph remains asymmetric
    return nothing
end

"""
    _apply_symmetry!(graph, k, ::FullSymmetry)

Add all reverse edges to ensure complete graph symmetry.
Nodes may end up with > k neighbors.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    ::FullSymmetry,
) where {T}
    n = length(graph)

    # Build sets for O(1) membership checking - combine both heaps
    neighbor_sets = [Set{Int}() for _ in 1:n]
    @inbounds for i in 1:n
        for nb in graph[i].old_neighbors
            push!(neighbor_sets[i], nb.id)
        end
        for nb in graph[i].new_neighbors
            push!(neighbor_sets[i], nb.id)
        end
    end

    # Collect reverse edges to add
    to_add = [Vector{Tuple{Int,T}}() for _ in 1:n]

    @inbounds for i in 1:n
        # Check old neighbors
        for nb in graph[i].old_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
        # Check new neighbors
        for nb in graph[i].new_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
    end

    # Add all the reverse edges to old_neighbors (they're not "new" discoveries)
    # Use unsafe_push! to allow exceeding capacity for full symmetry
    @inbounds for i in 1:n
        for (neighbor_id, dist) in to_add[i]
            # Note: This may exceed k capacity, which is expected for full symmetry
            unsafe_push!(graph[i].old_neighbors, neighbor_id, dist)
        end
    end

    return nothing
end

"""
    _apply_symmetry!(graph, k, policy::PrunedSymmetry)

Add reverse edges like FullSymmetry, but prune each node's combined neighbor list
to at most `policy.degree_multiplier * k` edges. This balances search quality
with memory efficiency, following PyNNDescent's approach.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    policy::PrunedSymmetry,
) where {T}
    n = length(graph)
    max_degree = ceil(Int, policy.degree_multiplier * k)

    # Build sets for O(1) membership checking - combine both heaps
    neighbor_sets = [Set{Int}() for _ in 1:n]
    @inbounds for i in 1:n
        for nb in graph[i].old_neighbors
            push!(neighbor_sets[i], nb.id)
        end
        for nb in graph[i].new_neighbors
            push!(neighbor_sets[i], nb.id)
        end
    end

    # Collect reverse edges to add
    to_add = [Vector{Tuple{Int,T}}() for _ in 1:n]

    @inbounds for i in 1:n
        # Check old neighbors
        for nb in graph[i].old_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
        # Check new neighbors
        for nb in graph[i].new_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
    end

    # Add reverse edges and prune to max_degree
    @inbounds for i in 1:n
        # Collect all neighbors from both heaps plus reverse edges
        all_neighbors = Vector{Neighbor{T}}()
        for nb in graph[i].old_neighbors
            push!(all_neighbors, nb)
        end
        for nb in graph[i].new_neighbors
            push!(all_neighbors, nb)
        end
        for (neighbor_id, dist) in to_add[i]
            push!(all_neighbors, Neighbor{T}(neighbor_id, dist))
        end

        # Sort by distance and keep only the closest max_degree neighbors
        sort!(all_neighbors, by = nb -> nb.dist)
        if length(all_neighbors) > max_degree
            resize!(all_neighbors, max_degree)
        end

        # Rebuild the old_neighbors heap with pruned neighbors, clear new_neighbors
        # NOTE: We directly manipulate heap.data to bypass capacity limits.
        # After empty!(), the heap can accept all_neighbors (up to max_degree)
        # even though max_degree > k (the original heap capacity).
        # This is intentional for PrunedSymmetry where degree_multiplier > 1.0.
        empty!(graph[i].old_neighbors.data)
        empty!(graph[i].new_neighbors.data)
        for nb in all_neighbors
            push!(graph[i].old_neighbors, nb.id, nb.dist)
        end
    end

    return nothing
end

function _connect_pair!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    distance,
    a::Int,
    b::Int,
) where {T}
    a == b && return 0
    dist = distance(view(data, :, a), view(data, :, b))
    inserted = 0
    inserted += _insert_neighbor!(graph[a], b, dist)
    inserted += _insert_neighbor!(graph[b], a, dist)
    return inserted
end

function _insert_neighbor!(
    node::NNDescentNeighborNode{T},
    neighbor_id::Int,
    dist::T,
) where {T}
    # Cheap O(k) duplicate check on new_neighbors only. Skips the (existing,
    # better-or-equal) cases that would otherwise burn heap pushes on duplicates
    # and risk evicting good unique entries before _finalize_neighbors. We do
    # not scan old_neighbors: in the local-join path inserts target
    # new_neighbors, and the heap-push itself absorbs any rare cross-heap
    # duplicate at finalize time. Skips the catastrophic O(k log k) update path
    # the previous implementation paid on every duplicate.
    @inbounds for existing in node.new_neighbors
        if existing.id == neighbor_id
            # Already present: never push again. Distance updates for an
            # existing id are intentionally dropped; PyNNDescent does the same,
            # and our quality tests (recall, MRR) confirm this is fine.
            return 0
        end
    end
    accepted = push!(node.new_neighbors, neighbor_id, dist)
    return accepted ? 1 : 0
end

function _finalize_neighbors(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
) where {T}
    adjacency = Vector{Vector{Int}}(undef, length(graph))
    @inbounds for i in eachindex(graph)
        node = graph[i]

        # Collect all neighbors from both heaps
        all_neighbors = Vector{Neighbor{T}}()
        seen_ids = Set{Int}()  # Track seen IDs to prevent duplicates

        # Add neighbors from old_neighbors heap, checking for duplicates
        for nb in node.old_neighbors
            if !(nb.id in seen_ids)
                push!(all_neighbors, nb)
                push!(seen_ids, nb.id)
            end
        end

        # Add neighbors from new_neighbors heap, checking for duplicates
        for nb in node.new_neighbors
            if !(nb.id in seen_ids)
                push!(all_neighbors, nb)
                push!(seen_ids, nb.id)
            end
        end

        # Sort by distance to ensure we keep the closest neighbors
        sort!(all_neighbors, by = nb -> nb.dist)

        # Extract IDs
        ids = Vector{Int}(undef, length(all_neighbors))
        for (j, nb) in enumerate(all_neighbors)
            ids[j] = nb.id
        end

        adjacency[i] = ids
    end
    return adjacency
end

function _sample_neighbor_ids(
    heap::BoundedMaxHeap{T},
    limit::Int,
    rng::AbstractRNG,
) where {T}
    limit <= 0 && return Int[]
    collected = Int[]
    @inbounds for nb in heap
        push!(collected, nb.id)
    end
    len = length(collected)
    len <= limit && return collected
    shuffle!(rng, collected)
    resize!(collected, limit)
    return collected
end
