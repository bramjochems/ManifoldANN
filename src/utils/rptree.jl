using LinearAlgebra
using Random

"""
    RPNodePayload{T}

Per-internal-node payload stored in the binary-partition-tree node array
for an `RPTree`. Carries the hyperplane direction and the scalar offset
used for routing: a query `q` descends right when `dot(hyperplane, q) >
offset`, else left.

Leaves get a default-constructed sentinel (callers gate on
`bpt_is_leaf`); BPT requires payload types to provide a no-arg
constructor for the leaf sentinel slot.
"""
struct RPNodePayload{T<:AbstractFloat}
    hyperplane::Vector{T}
    offset::T
end

RPNodePayload{T}() where {T<:AbstractFloat} = RPNodePayload{T}(Vector{T}(), zero(T))

"""
    RPTree{T}

Random-projection tree over a fixed dataset. `nodes` is a flat
`Vector{BPTNode{RPNodePayload{T}}}` produced by the shared
binary-partition-tree helper (`src/utils/binary_partition_tree.jl`);
`leaf_members` concatenates the point ids belonging to each leaf in the
order they were emitted, and each leaf node carries a `(leaf_lo,
leaf_hi)` range view into it. `root` is the index of the root node.
"""
struct RPTree{T<:AbstractFloat}
    nodes::Vector{BPTNode{RPNodePayload{T}}}
    leaf_members::Vector{Int}
    root::Int
end

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

# Splitter contract (degenerate splits)

A splitter MUST partition non-trivially: if a candidate split would put
all points on one side of the hyperplane, the splitter must return
`nothing` so the caller emits a leaf. The BPT recursion does not
post-correct empty-side splits — see [`bpt_split!`](@ref).
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
    RPBuildContext{T}

Opaque context plumbed through the binary-partition-tree recursion for
`RPTree`. Holds the data matrix, the per-tree RNG, and the leaf-cap
recursion-stop threshold (RP trees stop on a fixed leaf cap rather than
a trait — the BPT helper just walks the recursion).
"""
struct RPBuildContext{T<:AbstractFloat}
    data::AbstractMatrix{T}
    rng::AbstractRNG
    leaf_cap::Int
end

# `bpt_split!` adapter for any `AbstractRPSplitter`. Wraps the existing
# `split_node` extension contract — third-party splitters keep working
# unchanged. Concurrency: each tree owns its own ctx + RNG + indices, so
# the BPT recursion is safe to run in parallel across forest trees as
# long as `split_node` is itself concurrent-safe (documented contract).
function bpt_split!(
    splitter::AbstractRPSplitter,
    ctx::RPBuildContext{T},
    indices::Vector{Int},
    depth::Int,
) where {T<:AbstractFloat}
    m = length(indices)
    if m <= ctx.leaf_cap
        return BPTLeaf()
    end
    result = split_node(splitter, ctx.data, indices, ctx.rng)
    result === nothing && return BPTLeaf()
    hyperplane, offset, left_idx, right_idx = result
    payload = RPNodePayload{T}(hyperplane, offset)
    return BPTInternal{RPNodePayload{T}}(left_idx, right_idx, payload)
end

"""
    build_rptree(data, leaf_cap; splitter=TwoPointSplitter(), rng) -> RPTree

Build a random-projection tree. The split strategy at each internal node is
controlled by `splitter::AbstractRPSplitter` (default: two-point hyperplane).
Recursion stops at `<= leaf_cap` points or when the splitter signals a
degenerate split.

Routes via the binary-partition-tree helper at
`src/utils/binary_partition_tree.jl`; the splitter's existing
`split_node` contract is preserved — an internal adapter converts its
output (or `nothing` sentinel) into the BPT outcome type.
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

    ctx = RPBuildContext{T}(data, rng, leaf_cap)
    indices = collect(1:n)
    nodes, leaf_members, root = bpt_build!(
        splitter, ctx, indices;
        payload_type = RPNodePayload{T},
    )
    return RPTree{T}(nodes, leaf_members, root)
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
        if bpt_is_leaf(node)
            return view(tree.leaf_members, Int(node.leaf_lo):Int(node.leaf_hi))
        end
        payload = node.payload
        proj = zero(T)
        @simd for j in eachindex(payload.hyperplane)
            proj += payload.hyperplane[j] * point[j]
        end
        idx = proj > payload.offset ? Int(node.right) : Int(node.left)
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
