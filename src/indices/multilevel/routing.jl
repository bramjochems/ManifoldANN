"""
Routing strategies for multi-level indices.

Routing strategies determine which child indices to probe during query execution
based on the transform's assignment information.
"""

"""
    AbstractRoutingStrategy

Abstract base type for routing strategies.

Concrete implementations:
- `TopKRouting`: Probe k nearest clusters/buckets
- `ExhaustiveRouting`: Probe all child indices
"""
abstract type AbstractRoutingStrategy end

"""
    TopKRouting <: AbstractRoutingStrategy

Route queries to the k nearest clusters/buckets based on distance information.

# Fields
- `k::Int`: Number of clusters to probe

# Examples
```julia
# Probe 5 nearest clusters in IVF
routing = TopKRouting(5)

# Probe 10 nearest clusters for higher recall
routing = TopKRouting(10)
```
"""
struct TopKRouting <: AbstractRoutingStrategy
    k::Int

    function TopKRouting(k::Int)
        k > 0 || throw(ArgumentError("k must be positive"))
        new(k)
    end
end

"""
    ExhaustiveRouting <: AbstractRoutingStrategy

Route queries to all child indices (no pruning).

This is useful for:
- Identity transforms (where there's typically only one child)
- Ensuring maximum recall
- Debugging and baseline comparisons

# Examples
```julia
routing = ExhaustiveRouting()
```
"""
struct ExhaustiveRouting <: AbstractRoutingStrategy end

"""
    select_indices(
        strategy::AbstractRoutingStrategy,
        assignment,
        indices::Vector{I}
    )::Vector{I} where {I<:AbstractANNIndex}

Select which child indices to probe based on routing strategy and assignment info.

# Arguments
- `strategy`: Routing strategy to use
- `assignment`: Transform assignment information (e.g., `KMeansAssignment`)
- `indices`: Vector of all available child indices

# Returns
- Vector of indices to probe (subset of `indices`)

# Examples
```julia
# Top-K routing with KMeans
assignment = KMeansAssignment([0.5, 1.2, 0.8, 2.1, 0.9])  # distances to 5 centroids
selected = select_indices(TopKRouting(2), assignment, indices)
# Returns indices[1] and indices[3] (centroids 1 and 3 are closest)

# Exhaustive routing
selected = select_indices(ExhaustiveRouting(), nothing, indices)
# Returns all indices
```
"""
function select_indices end

# TopKRouting implementation
function select_indices(
    strategy::TopKRouting,
    assignment::KMeansAssignment,
    indices::Vector{I}
) where {I<:AbstractANNIndex}
    k = min(strategy.k, length(indices))

    # Get indices of k nearest clusters (smallest distances)
    nearest_indices = partialsortperm(assignment.distances, 1:k)

    return indices[nearest_indices]
end

# ExhaustiveRouting implementation
function select_indices(
    ::ExhaustiveRouting,
    assignment,  # Ignored
    indices::Vector{I}
) where {I<:AbstractANNIndex}
    return indices
end
