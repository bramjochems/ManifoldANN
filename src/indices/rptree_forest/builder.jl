"""
    build_index(RPTreeIndex, data; leaf_cap, distance, splitter, rng)

Build a single random-projection tree wrapped as an `RPTreeIndex`.
The split strategy at each internal node is controlled by
`splitter::AbstractRPSplitter` (default: `TwoPointSplitter`).
"""
function build_index(
    ::Type{RPTreeIndex},
    data::AbstractMatrix{T};
    leaf_cap::Int = RPTREE_INDEX_DEFAULT_LEAF_CAP,
    distance::D = default_distance,
    splitter::AbstractRPSplitter = TwoPointSplitter(),
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have positive dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    leaf_cap >= 1 || throw(ArgumentError("leaf_cap must be >= 1"))

    tree = build_rptree(data, leaf_cap; splitter = splitter, rng = rng)
    return RPTreeIndex{T,D}(tree, d, n, leaf_cap, distance)
end
