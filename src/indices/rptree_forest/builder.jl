"""
    build_index(RPTreeIndex, data; leaf_cap=RPTREE_INDEX_DEFAULT_LEAF_CAP, distance=default_distance, rng=Random.default_rng())

Build a single random-projection tree wrapped as an `RPTreeIndex`.
"""
function build_index(
    ::Type{RPTreeIndex},
    data::AbstractMatrix{T};
    leaf_cap::Int = RPTREE_INDEX_DEFAULT_LEAF_CAP,
    distance::D = default_distance,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have positive dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    leaf_cap >= 1 || throw(ArgumentError("leaf_cap must be >= 1"))

    tree = build_rptree(data, leaf_cap; rng = rng)
    return RPTreeIndex{T,D}(tree, d, n, leaf_cap, distance)
end
