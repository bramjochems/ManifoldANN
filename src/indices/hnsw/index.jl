"""
    build_index(HNSWIndex, data; kwargs...)

Construct an HNSW index over `data`. Exposes keywords mirroring the standard
HNSW parameters (`M`, `ef_construction`, `ef_search`) along with pluggable
components (layer planner, neighbor policy, traversal policy). The index
records only the HNSW graph structure, so callers still supply `data` when
querying.
"""
function build_index(
    ::Type{HNSWIndex},
    data::AbstractMatrix{T};
    M::Int = 16,
    ef_construction::Int = 200,
    ef_search::Int = 64,
    planner::Union{AbstractLayerPlanner,Nothing} = nothing,
    neighbor_policy::Union{AbstractNeighborPolicy,Nothing} = nothing,
    traversal_policy::Union{AbstractTraversalPolicy,Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    distance::Function = default_distance,
) where {T<:LinearAlgebra.BlasFloat}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    M > 0 || throw(ArgumentError("M must be positive"))
    ef_construction >= 1 ||
        throw(ArgumentError("ef_construction must be at least 1"))
    ef_search >= 1 || throw(ArgumentError("ef_search must be at least 1"))

    planner === nothing && (planner = DefaultLayerPlanner(1 / log(max(M, 2))))
    neighbor_policy === nothing && (neighbor_policy = HeuristicNeighborPolicy(M))
    traversal_policy === nothing && (traversal_policy = GreedyTraversalPolicy(ef_search))

    index = HNSWIndex{T,typeof(planner),typeof(neighbor_policy),typeof(traversal_policy)}(
        Vector{HNSWLayer}(),
        0,
        -1,
        d,
        0,
        M,
        ef_construction,
        planner,
        neighbor_policy,
        traversal_policy,
    )

    for col in 1:n
        point = @view data[:, col]
        insert!(index, data, point; point_id = col, rng = rng, distance = distance)
    end
    return index
end

function supports_layers(::HNSWIndex)
    return true
end
