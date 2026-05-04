"""
Generic binary-partition-tree (BPT) recursion skeleton.

Tree-style indices (`KDTreeIndex`, `RPTreeIndex`, `PCATreeIndex`) share
a meta-pattern: recursively partition a point set along some splitting
rule, with router-internal-nodes and leaf buckets. This module factors
out *only* the recursion + leaf-bucket emission. It deliberately does
NOT subsume per-tree storage layout (KD packs `axis + threshold` into 4
ints; RP stores a hyperplane vector; PCA stores a direction + threshold
+ spectral metadata) and it does NOT subsume the query-time descent
(KDTree prunes via componentwise-monotone bounds; RP and PCA trees do
not).

The helper has two consumers today: `PCATreeIndex`
(`src/indices/pcatree/builder.jl`) and `RPTree` (`src/utils/rptree.jl`,
which wraps `AbstractRPSplitter` via an internal adapter). KDTreeIndex
keeps its bespoke in-place `(lo, hi)` + `quickselect!` layout — it would
need a Partitioner trait change to fit BPT's fresh-`Vector{Int}` per
split contract, which isn't worth doing speculatively.

# Splitter contract

A splitter is any object the caller passes through unchanged. It must
support:

    bpt_split!(splitter, ctx, indices, depth) -> SplitOutcome{Payload}

where `SplitOutcome` is one of:
- `BPTLeaf()`               — emit a leaf bucket from `indices`
- `BPTInternal(left_idx, right_idx, payload)` — internal node; `payload`
   is whatever per-internal-node data the caller wants stored.

`ctx` is opaque user state (typically holds the data matrix, RNG, etc.).
`indices` is the live `Vector{Int}` view of point ids in this subtree.
The splitter owns split + recursion-stop decisions; the helper just
walks the recursion.

# Output

`bpt_build!` emits a flat `Vector{BPTNode{Payload}}` plus a flat
`leaf_members::Vector{Int}` and returns the root node id. Callers can
either keep the generic `BPTNode` or unpack into a tighter per-index
struct if memory layout matters.
"""

# ----- Outcome tagged-union ------------------------------------------------

"""
    BPTLeaf

Splitter outcome signalling that the helper should emit a leaf bucket
holding the indices passed in.
"""
struct BPTLeaf end

"""
    BPTInternal{P}

Splitter outcome signalling an internal node. `left_indices` and
`right_indices` partition the parent's `indices` (the splitter is free
to allocate fresh vectors or sort/swap the parent vector in place — the
helper does not reuse the parent vector after the call). `payload`
carries the per-internal-node data the splitter wants stored (split
direction, threshold, etc.).
"""
struct BPTInternal{P}
    left_indices::Vector{Int}
    right_indices::Vector{Int}
    payload::P
end

const BPTSplitOutcome{P} = Union{BPTLeaf, BPTInternal{P}}

# ----- Generic node --------------------------------------------------------

"""
    BPTNode{P}

Generic binary-partition-tree node. `is_leaf == true` means the node
owns the leaf bucket `leaf_members[leaf_lo:leaf_hi]` and `payload` is a
default-constructed sentinel (callers must not read it for leaves).
Internal nodes carry the splitter-supplied `payload` and child ids
(`left`, `right`) into the same `nodes` vector; `leaf_lo == 0` for
internal nodes.
"""
struct BPTNode{P}
    is_leaf::Bool
    left::Int32
    right::Int32
    leaf_lo::Int32
    leaf_hi::Int32
    payload::P
end

@inline bpt_is_leaf(node::BPTNode) = node.is_leaf

# ----- Recursive build -----------------------------------------------------

"""
    bpt_build!(splitter, ctx, indices; payload_type=...) -> (nodes, leaf_members, root)

Recursively partition `indices` into a binary tree by repeatedly calling
`bpt_split!(splitter, ctx, indices, depth)`. Allocates and returns the
flat `nodes::Vector{BPTNode{P}}`, the flat `leaf_members::Vector{Int}`,
and the root node id. Mutates `indices` is not assumed — the splitter is
responsible for not retaining references.

`payload_type` is required: the recursion may emit a leaf at the
root (so there is no internal-node probe to infer `P` from), and the
helper needs the concrete `BPTNode{P}` element type up front for type
stability of the `nodes` vector.

# Payload contract

`P` MUST have a no-arg constructor `P()` returning a sentinel value used
to fill the `payload` field of leaf nodes. Only the internal-node case
reads payload contents (callers gate reads on `bpt_is_leaf`). The
sentinel exists purely to satisfy `BPTNode{P}`'s type stability —
nothing inspects it.
"""
function bpt_build!(
    splitter,
    ctx,
    indices::Vector{Int};
    payload_type::Type{P},
) where {P}
    nodes = BPTNode{P}[]
    leaf_members = Int[]
    sizehint!(leaf_members, length(indices))
    root = _bpt_recurse!(nodes, leaf_members, splitter, ctx, indices, 1)
    return nodes, leaf_members, root
end

function _bpt_recurse!(
    nodes::Vector{BPTNode{P}},
    leaf_members::Vector{Int},
    splitter,
    ctx,
    indices::Vector{Int},
    depth::Int,
) where {P}
    isempty(indices) && return 0
    outcome = bpt_split!(splitter, ctx, indices, depth)
    if outcome isa BPTLeaf
        return _bpt_emit_leaf!(nodes, leaf_members, indices, P)
    end
    internal = outcome::BPTInternal{P}

    # Reserve self-slot before recursing so child ids resolve correctly.
    push!(nodes, BPTNode{P}(false, Int32(0), Int32(0), Int32(0), Int32(0),
                            internal.payload))
    self_idx = length(nodes)

    # Splitters MUST partition non-trivially (see `bpt_split!` docstring).
    # An empty `left_indices` or `right_indices` is a contract violation,
    # not something the helper post-corrects: silently coalescing would
    # leak previously-emitted nodes/members on the non-empty side.
    left_id  = _bpt_recurse!(nodes, leaf_members, splitter, ctx,
                             internal.left_indices, depth + 1)
    right_id = _bpt_recurse!(nodes, leaf_members, splitter, ctx,
                             internal.right_indices, depth + 1)

    nodes[self_idx] = BPTNode{P}(false, Int32(left_id), Int32(right_id),
                                 Int32(0), Int32(0), internal.payload)
    return self_idx
end

function _bpt_emit_leaf!(
    nodes::Vector{BPTNode{P}},
    leaf_members::Vector{Int},
    indices::Vector{Int},
    ::Type{P},
) where {P}
    start = Int32(length(leaf_members) + 1)
    append!(leaf_members, indices)
    stop = Int32(length(leaf_members))
    # Payload for leaves is a default-constructed sentinel; callers must
    # gate reads on `bpt_is_leaf`. We require P to have a no-arg
    # constructor when used directly; PCA tree wraps payload in a struct
    # whose default sentinel is fine.
    push!(nodes, BPTNode{P}(true, Int32(0), Int32(0), start, stop, _bpt_zero_payload(P)))
    return length(nodes)
end

# Default sentinel: try to construct from no args, else use `Ref` trick on
# the type's `instance` (works for singletons). PCA tree's payload is a
# concrete struct that supplies a no-arg constructor.
@inline _bpt_zero_payload(::Type{P}) where {P} = P()

"""
    bpt_split!(splitter, ctx, indices, depth) -> BPTSplitOutcome

Splitter callback. Concrete splitters (e.g. `PCASplitter`,
`AbstractRPSplitter` via its adapter) overload this. Default: error —
callers must specialise.

# Non-degeneracy contract

A splitter that returns `BPTInternal{P}(left_indices, right_indices,
payload)` MUST guarantee `!isempty(left_indices)` and
`!isempty(right_indices)`. If a candidate split would collapse to one
side, the splitter must return `BPTLeaf()` instead. The recursion does
not post-correct empty-side splits; passing one violates the contract
and produces undefined node/leaf-member state.
"""
function bpt_split! end
