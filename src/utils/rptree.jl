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
    hyperplane::Vector{T}
    offset::T
    left::Int32
    right::Int32
    leaf_range::UnitRange{Int32}
end

"""
    RPTree{T}

Random-projection tree over a fixed dataset. `nodes` is a flat node array;
`leaf_members` concatenates the point ids belonging to each leaf in the
order they were emitted, and `leaf_range` (per leaf) views into it. `root`
is the index of the root node.
"""
struct RPTree{T<:AbstractFloat}
    nodes::Vector{RPTreeNode{T}}
    leaf_members::Vector{Int}
    root::Int
end

@inline _is_leaf(node::RPTreeNode) = node.left == 0

"""
    AbstractRPSplitter

Strategy for picking a hyperplane split at each internal node of an
`RPTree`. A concrete splitter implements
`split_node(splitter, data, indices, rng) -> (hyperplane, offset, left_idx, right_idx)`
and returns `nothing` to signal a degenerate split (caller falls back to
emitting a leaf).

Same flexibility-point pattern as HNSW's `AbstractNeighborPolicy` /
LSH's `AbstractLSHHash` / multilevel's `AbstractRoutingStrategy`. The
default `TwoPointSplitter` reproduces the original two-point hyperplane
heuristic; alternative splitters (e.g. random-direction, PCA-aligned,
or Mondrian-style) can plug in without touching the build/query loops.

`split_node` must be safe to call concurrently from multiple tasks on
the *same splitter instance* — `RPTreeForestIndex` builds N trees in
parallel and shares one splitter across them. Stateless splitters (like
the default) satisfy this trivially; user-supplied splitters that cache
state across calls must use thread-local storage or document that they
are not forest-safe.
"""
abstract type AbstractRPSplitter end

"""
    TwoPointSplitter <: AbstractRPSplitter

Two-point hyperplane split (Dasgupta-Freund, the default for RP-tree forests
in PyNNDescent and ANNOY). At each internal node: sample two distinct points
`p1, p2`, set `hyperplane = data[:, p1] - data[:, p2]`, threshold at the
projection of the midpoint.
"""
struct TwoPointSplitter <: AbstractRPSplitter end

function split_node(
    ::TwoPointSplitter,
    data::AbstractMatrix{T},
    indices::Vector{Int},
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    m = length(indices)
    d = size(data, 1)
    i1 = rand(rng, 1:m)
    i2 = rand(rng, 1:(m - 1))
    i2 = i2 >= i1 ? i2 + 1 : i2
    p1 = indices[i1]
    p2 = indices[i2]

    hyperplane = Vector{T}(undef, d)
    @inbounds for j in 1:d
        hyperplane[j] = data[j, p1] - data[j, p2]
    end
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

    (isempty(left_idx) || isempty(right_idx)) && return nothing
    return hyperplane, offset, left_idx, right_idx
end

"""
    build_rptree(data, leaf_cap; splitter=TwoPointSplitter(), rng) -> RPTree

Build a random-projection tree. The split strategy at each internal node is
controlled by `splitter::AbstractRPSplitter` (default: two-point hyperplane).
Recursion stops at `<= leaf_cap` points or when the splitter signals a
degenerate split.
"""
function build_rptree(
    data::AbstractMatrix{T},
    leaf_cap::Int;
    splitter::AbstractRPSplitter = TwoPointSplitter(),
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}
    leaf_cap > 0 || throw(ArgumentError("leaf_cap must be positive"))
    d, n = size(data)
    n > 0 || throw(ArgumentError("data must contain at least one point"))

    nodes = Vector{RPTreeNode{T}}()
    leaf_members = Vector{Int}()
    sizehint!(leaf_members, n)

    indices = collect(1:n)
    root = _build_rptree_recursive!(
        nodes, leaf_members, data, indices, leaf_cap, splitter, rng)
    return RPTree{T}(nodes, leaf_members, root)
end

function _build_rptree_recursive!(
    nodes::Vector{RPTreeNode{T}},
    leaf_members::Vector{Int},
    data::AbstractMatrix{T},
    indices::Vector{Int},
    leaf_cap::Int,
    splitter::AbstractRPSplitter,
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    m = length(indices)
    if m <= leaf_cap
        return _emit_leaf!(nodes, leaf_members, indices)
    end

    result = split_node(splitter, data, indices, rng)
    result === nothing && return _emit_leaf!(nodes, leaf_members, indices)
    hyperplane, offset, left_idx, right_idx = result

    push!(nodes, RPTreeNode{T}(hyperplane, offset, Int32(0), Int32(0),
                                Int32(1):Int32(0)))
    self_idx = length(nodes)
    left_child = _build_rptree_recursive!(
        nodes, leaf_members, data, left_idx, leaf_cap, splitter, rng)
    right_child = _build_rptree_recursive!(
        nodes, leaf_members, data, right_idx, leaf_cap, splitter, rng)
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

Route `point` through `tree` and return a `view` into the matching leaf's
member id list. Allocation-free in the hot path.
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

default_rptree_n_trees(n::Int) = max(3, min(12, round(Int, 2 * log10(max(n, 10)))))
default_rptree_leaf_cap(k::Int) = min(64, 5 * k)

"""
    build_rptree_forest(data, n_trees, leaf_cap; splitter, rng) -> Vector{RPTree}

Build `n_trees` independent random-projection trees over `data`. Per-tree
RNGs are derived serially via `spawn_child_rngs` and the trees are built in
parallel via `Threads.@threads`. The forest is deterministic for fixed `rng`
regardless of thread count or scheduling order.

Used by `RPTreeForestIndex` as the build primitive and by NN-Descent's
`init=:rptree` seeding stage. The `splitter` argument follows the
`AbstractRPSplitter` extension point on `build_rptree`.
"""
function build_rptree_forest(
    data::AbstractMatrix{T},
    n_trees::Int,
    leaf_cap::Int;
    splitter::AbstractRPSplitter = TwoPointSplitter(),
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}
    n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
    tree_rngs = spawn_child_rngs(rng, n_trees)
    forest = Vector{RPTree{T}}(undef, n_trees)
    Threads.@threads for t in 1:n_trees
        forest[t] = build_rptree(data, leaf_cap; splitter = splitter, rng = tree_rngs[t])
    end
    return forest
end
