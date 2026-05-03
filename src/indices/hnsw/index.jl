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
    M::Int = HNSW_DEFAULT_M,
    ef_construction::Int = HNSW_DEFAULT_EF_CONSTRUCTION,
    ef_search::Int = HNSW_DEFAULT_EF_SEARCH,
    planner::Union{AbstractLayerPlanner,Nothing} = nothing,
    neighbor_policy::Union{AbstractNeighborPolicy,Symbol,Nothing} = nothing,
    traversal_policy::Union{AbstractTraversalPolicy,Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    distance::D = default_distance,
    threaded::Union{Bool,Nothing} = nothing,
) where {T<:LinearAlgebra.BlasFloat, D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    M > 0 || throw(ArgumentError("M must be positive"))
    ef_construction >= 1 ||
        throw(ArgumentError("ef_construction must be at least 1"))
    ef_search >= 1 || throw(ArgumentError("ef_search must be at least 1"))

    planner === nothing && (planner = DefaultLayerPlanner(HNSW_ML_NORMALIZATION_FACTOR / log(max(M, 2))))
    neighbor_policy = _resolve_neighbor_policy(neighbor_policy, M)
    traversal_policy === nothing && (traversal_policy = GreedyTraversalPolicy(ef_search))

    index = HNSWIndex{T,typeof(planner),typeof(neighbor_policy),typeof(traversal_policy),D}(
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
        distance,
        UInt32[],
        UInt32(0),
        ReentrantLock[],
        ReentrantLock(),
    )

    if threaded === nothing
        threaded = Threads.nthreads() > 1
    end

    if threaded
        _build_index_threaded!(index, data, n, rng)
    else
        for col in 1:n
            point = @view data[:, col]
            insert!(index, data, point; point_id = col, rng = rng)
        end
    end
    return index
end

function _resolve_neighbor_policy(
    policy::Union{AbstractNeighborPolicy,Symbol,Nothing},
    M::Int,
)
    if policy === nothing
        return DiversifiedNeighborPolicy(M)
    elseif policy isa AbstractNeighborPolicy
        return policy
    elseif policy isa Symbol
        if policy === :heuristic
            return HeuristicNeighborPolicy(M)
        elseif policy === :diversified
            return DiversifiedNeighborPolicy(M)
        else
            throw(ArgumentError("Unknown neighbor_policy symbol: $policy"))
        end
    else
        throw(ArgumentError("Unsupported neighbor_policy type: $(typeof(policy))"))
    end
end

function supports_layers(::HNSWIndex)
    return true
end
