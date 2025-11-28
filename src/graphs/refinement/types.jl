"""
    NodeNeighborhood{T}

Cached neighborhood information for efficient curvature computation.
"""
struct NodeNeighborhood{T<:AbstractFloat}
    node_id::Int
    neighbors::Vector{Int}
    probabilities::Vector{T}
    geometry::Union{Nothing,Any}
    projected_coords::Union{Nothing,Matrix{T}}
    barycenter::Union{Nothing,Vector{T}}

    function NodeNeighborhood{T}(
        node_id::Int,
        neighbors::Vector{Int},
        probabilities::Vector{T};
        geometry=nothing,
        projected_coords=nothing,
        barycenter=nothing
    ) where {T<:AbstractFloat}
        length(neighbors) == length(probabilities) || throw(ArgumentError("neighbors and probabilities must have same length"))
        !isempty(probabilities) && !(sum(probabilities) ≈ 1.0) && throw(ArgumentError("probabilities must sum to 1.0"))
        new{T}(node_id, neighbors, probabilities, geometry, projected_coords, barycenter)
    end
end

NodeNeighborhood(node_id::Int, neighbors::Vector{Int}, probabilities::Vector{T}; kwargs...) where {T} =
    NodeNeighborhood{T}(node_id, neighbors, probabilities; kwargs...)

"""
    uniform_neighborhood(node_id::Int, neighbors::Vector{Int}, ::Type{T}=Float64)

Create a NodeNeighborhood with uniform probability distribution.
"""
function uniform_neighborhood(node_id::Int, neighbors::Vector{Int}, ::Type{T}=Float64) where {T<:AbstractFloat}
    k = length(neighbors)
    probs = fill(one(T) / T(k), k)
    NodeNeighborhood{T}(node_id, neighbors, probs)
end

"""
    EdgeNeighborhoodView{T}

Decomposed edge neighborhoods (shared/unique sets) for curvature computation.
"""
struct EdgeNeighborhoodView{T<:AbstractFloat}
    x_id::Int
    y_id::Int
    shared::Vector{Int}
    unique_x::Vector{Int}
    unique_y::Vector{Int}
    x_probs::Dict{Int,T}
    y_probs::Dict{Int,T}
    edge_distance::T
end

"""
    create_edge_view(x_neighborhood, y_neighborhood, edge_distance)

Create an EdgeNeighborhoodView from two NodeNeighborhoods.
"""
function create_edge_view(
    x_neighborhood::NodeNeighborhood{T},
    y_neighborhood::NodeNeighborhood{T},
    edge_distance::T
) where {T<:AbstractFloat}
    x_set = Set(x_neighborhood.neighbors)
    y_set = Set(y_neighborhood.neighbors)

    shared = sort(collect(intersect(x_set, y_set)))
    unique_x = sort(collect(setdiff(x_set, y_set)))
    unique_y = sort(collect(setdiff(y_set, x_set)))

    x_probs = Dict{Int,T}(
        x_neighborhood.neighbors[i] => x_neighborhood.probabilities[i]
        for i in 1:length(x_neighborhood.neighbors)
    )
    y_probs = Dict{Int,T}(
        y_neighborhood.neighbors[i] => y_neighborhood.probabilities[i]
        for i in 1:length(y_neighborhood.neighbors)
    )

    EdgeNeighborhoodView{T}(x_neighborhood.node_id, y_neighborhood.node_id, shared, unique_x, unique_y, x_probs, y_probs, edge_distance)
end

"""
    CurvatureResult{T}

Result of computing Ollivier-Ricci curvature on an edge.
"""
struct CurvatureResult{T<:AbstractFloat}
    x_id::Int
    y_id::Int
    curvature::T
    wasserstein_distance::T
    edge_distance::T
    solver_type::Symbol
end

is_positive_curvature(result::CurvatureResult) = result.curvature > 0
passes_threshold(result::CurvatureResult, threshold::Real) = result.curvature >= threshold
