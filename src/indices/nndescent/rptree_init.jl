# RP-tree primitives (RPTree, RPTreeNode, build_rptree, leaf_members) live in
# src/utils/rptree.jl and are loaded earlier in src/ManifoldANN.jl. This file
# only carries the NN-Descent-specific forest builder and seeding helper.

"""
    build_rptree_forest(data, n_trees, leaf_cap; rng) -> Vector{RPTree}

Build `n_trees` independent random-projection trees over `data`. Each tree
samples its own splits, providing the decorrelation that NN-Descent
initialization needs.
"""
function build_rptree_forest(
    data::AbstractMatrix{T},
    n_trees::Int,
    leaf_cap::Int;
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}
    n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
    forest = Vector{RPTree{T}}(undef, n_trees)
    for t in 1:n_trees
        forest[t] = build_rptree(data, leaf_cap; rng = rng)
    end
    return forest
end

"""
    _initialize_rptree_neighbors!(graph, data, k, distance, rng, n_trees, leaf_cap)

Seed the NN-Descent working graph with RP-tree-derived candidates. For each
point, the union of leaf-members across the forest (excluding self) is
deduplicated; actual distances are computed and only the closest candidates
land in `new_neighbors` via the bounded heap.
"""
function _initialize_rptree_neighbors!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    k::Int,
    distance,
    rng::AbstractRNG,
    n_trees::Int,
    leaf_cap::Int,
) where {T}
    n = length(graph)
    forest = build_rptree_forest(data, n_trees, leaf_cap; rng = rng)

    seen = Vector{Int}()
    sizehint!(seen, n_trees * leaf_cap)
    # Scratch for (id, dist) pairs while picking the closest k from co-leaf
    # candidates. Reused across nodes; pre-sized to the max possible candidate
    # count (n_trees × leaf_cap) so push! doesn't grow the buffer mid-loop.
    scored = Vector{Tuple{Int,T}}()
    sizehint!(scored, n_trees * leaf_cap)

    @inbounds for i in 1:n
        empty!(seen)
        empty!(scored)
        q = view(data, :, i)
        for tree in forest
            members = leaf_members(tree, q)
            append!(seen, members)
        end
        sort!(seen)
        unique!(seen)

        # Score all unique co-leaf candidates, then seed only the closest k
        # bidirectionally. PyNNDescent's seeding pattern: RP-tree-quality top-k
        # at random-init-style density. Pushing all candidates bidirectionally
        # overcommits and dominates the build (5-10× regression measured);
        # pushing all candidates forward-only undercommits relative to random
        # init's bidirectional setup. This middle path is the canonical fix.
        for cand in seen
            cand == i && continue
            dist = distance(q, view(data, :, cand))::T
            push!(scored, (cand, dist))
        end

        if length(scored) > k
            # Partial sort: keep top-k by distance. sort! on a small list (cap
            # ≈ n_trees × leaf_cap, typically ≤ 600) is fast enough that a
            # quickselect isn't worth the dependency.
            sort!(scored; by = p -> p[2])
            resize!(scored, k)
        end

        node = graph[i]
        for (cand, dist) in scored
            push!(node.new_neighbors, cand, dist)
            unsafe_push!(graph[cand].new_neighbors, i, dist)
        end
    end

    # If any node ended up with fewer than k neighbors (e.g., tiny dataset
    # or a leaf with very few unique members), top up with random picks so
    # NN-Descent has a full starting heap.
    @inbounds for i in 1:n
        node = graph[i]
        deficit = k - length(node.new_neighbors)
        deficit <= 0 && continue
        chosen = Set{Int}()
        for nb in node.new_neighbors
            push!(chosen, nb.id)
        end
        push!(chosen, i)
        added = 0
        attempts = 0
        max_attempts = 20 * deficit + 32
        while added < deficit && attempts < max_attempts
            attempts += 1
            cand = rand(rng, 1:n)
            cand in chosen && continue
            push!(chosen, cand)
            dist = distance(view(data, :, i), view(data, :, cand))::T
            push!(node.new_neighbors, cand, dist)
            added += 1
        end
    end

    return nothing
end
