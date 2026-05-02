using LinearAlgebra
using Random

"""
    RPTreeNode{T}

Node in a random-projection tree built from two-point hyperplane splits.
Leaves are encoded by `left == 0` (and `right == 0`); their member point ids
live in `leaf_range`, which indexes into the parent `RPTree.leaf_members`
flat array. Internal nodes have empty `hyperplane` allocated only for the
split direction and a scalar `offset` threshold.
"""
struct RPTreeNode{T<:AbstractFloat}
    hyperplane::Vector{T}        # length 0 for leaves
    offset::T                    # 0 for leaves
    left::Int32                  # 0 for leaves
    right::Int32                 # 0 for leaves
    leaf_range::UnitRange{Int32} # 1:0 (empty) for internal nodes
end

"""
    RPTree{T}

Random-projection tree over a fixed dataset. `nodes` is a flat node array
indexed by `Int`; `leaf_members` concatenates the point ids belonging to
each leaf in the order they were emitted, and `leaf_range` (per leaf) views
into it. `root` is the index of the root node.
"""
struct RPTree{T<:AbstractFloat}
    nodes::Vector{RPTreeNode{T}}
    leaf_members::Vector{Int}
    root::Int
end

@inline _is_leaf(node::RPTreeNode) = node.left == 0

"""
    build_rptree(data, leaf_cap; rng) -> RPTree

Build a random-projection tree using two-point hyperplane splits. At each
node, two distinct points are sampled from the active set, the split
direction is `data[:, p1] - data[:, p2]`, and the threshold is the
projection of the midpoint of the two points. Recursion stops when a node
holds `<= leaf_cap` points (or when the split degenerates and cannot
separate the points).
"""
function build_rptree(
    data::AbstractMatrix{T},
    leaf_cap::Int;
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}
    leaf_cap > 0 || throw(ArgumentError("leaf_cap must be positive"))
    d, n = size(data)
    n > 0 || throw(ArgumentError("data must contain at least one point"))

    nodes = Vector{RPTreeNode{T}}()
    leaf_members = Vector{Int}()
    sizehint!(leaf_members, n)

    indices = collect(1:n)
    root = _build_rptree_recursive!(nodes, leaf_members, data, indices, leaf_cap, rng)
    return RPTree{T}(nodes, leaf_members, root)
end

function _build_rptree_recursive!(
    nodes::Vector{RPTreeNode{T}},
    leaf_members::Vector{Int},
    data::AbstractMatrix{T},
    indices::Vector{Int},
    leaf_cap::Int,
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    m = length(indices)
    if m <= leaf_cap
        return _emit_leaf!(nodes, leaf_members, indices)
    end

    d = size(data, 1)
    # Pick two distinct points
    i1 = rand(rng, 1:m)
    i2 = rand(rng, 1:(m - 1))
    i2 = i2 >= i1 ? i2 + 1 : i2
    p1 = indices[i1]
    p2 = indices[i2]

    hyperplane = Vector{T}(undef, d)
    @inbounds for j in 1:d
        hyperplane[j] = data[j, p1] - data[j, p2]
    end
    # Threshold = dot(hyperplane, midpoint)
    offset = zero(T)
    @inbounds for j in 1:d
        offset += hyperplane[j] * (data[j, p1] + data[j, p2]) / 2
    end

    left_idx = Vector{Int}()
    right_idx = Vector{Int}()
    sizehint!(left_idx, m >>> 1)
    sizehint!(right_idx, m >>> 1)
    @inbounds for k in 1:m
        idx = indices[k]
        proj = zero(T)
        for j in 1:d
            proj += hyperplane[j] * data[j, idx]
        end
        if proj > offset
            push!(right_idx, idx)
        else
            push!(left_idx, idx)
        end
    end

    # Degenerate split: all on one side. Fall back to leaf to avoid infinite
    # recursion (e.g., duplicate points or pathological geometry).
    if isempty(left_idx) || isempty(right_idx)
        return _emit_leaf!(nodes, leaf_members, indices)
    end

    # Reserve this node's slot, recurse, then patch in child indices. We use
    # a placeholder push first so the node id is stable before recursion.
    push!(nodes, RPTreeNode{T}(hyperplane, offset, Int32(0), Int32(0),
                                Int32(1):Int32(0)))
    self_idx = length(nodes)
    left_child = _build_rptree_recursive!(
        nodes, leaf_members, data, left_idx, leaf_cap, rng)
    right_child = _build_rptree_recursive!(
        nodes, leaf_members, data, right_idx, leaf_cap, rng)
    nodes[self_idx] = RPTreeNode{T}(
        hyperplane, offset, Int32(left_child), Int32(right_child),
        Int32(1):Int32(0))
    return self_idx
end

function _emit_leaf!(
    nodes::Vector{RPTreeNode{T}},
    leaf_members::Vector{Int},
    indices::Vector{Int},
) where {T<:AbstractFloat}
    start = Int32(length(leaf_members) + 1)
    append!(leaf_members, indices)
    stop = Int32(length(leaf_members))
    push!(nodes, RPTreeNode{T}(
        T[], zero(T), Int32(0), Int32(0), start:stop))
    return length(nodes)
end

"""
    leaf_members(tree, point) -> view

Route `point` (a vector or column view) through `tree` and return a `view`
into the matching leaf's member id list. Allocation-free in the hot path.
"""
function leaf_members(tree::RPTree{T}, point::AbstractVector) where {T}
    idx = tree.root
    @inbounds while true
        node = tree.nodes[idx]
        if _is_leaf(node)
            return view(tree.leaf_members, Int(first(node.leaf_range)):Int(last(node.leaf_range)))
        end
        proj = zero(T)
        @simd for j in eachindex(node.hyperplane)
            proj += node.hyperplane[j] * point[j]
        end
        idx = proj > node.offset ? Int(node.right) : Int(node.left)
    end
end

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
    default_rptree_n_trees(n) -> Int

PyNNDescent's default forest size: `max(3, min(12, round(2*log10(n))))`.
"""
default_rptree_n_trees(n::Int) = max(3, min(12, round(Int, 2 * log10(max(n, 10)))))

"""
    default_rptree_leaf_cap(k) -> Int

PyNNDescent's default leaf cap: `min(64, 5*k)`.
"""
default_rptree_leaf_cap(k::Int) = min(64, 5 * k)

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
