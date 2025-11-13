"""
Merge strategies for combining results from multiple child indices.

When a multi-level index probes multiple child indices (e.g., top-K clusters in IVF),
the merge strategy determines how to combine their results into a final k-NN list.
"""

"""
    AbstractMergeStrategy

Abstract base type for result merging strategies.

Concrete implementations:
- `SimpleMerge`: Trust sub-index distances, sort globally, take top-k
- (Future) `RecomputeMerge`: Recompute distances with original data for accuracy
"""
abstract type AbstractMergeStrategy end

"""
    SimpleMerge <: AbstractMergeStrategy

Simple merge strategy that trusts distances from sub-indices.

This strategy:
1. Collects all neighbors from all probed indices
2. Sorts by distance (trusting the distances returned by sub-indices)
3. Returns top-k unique neighbors

This is fast but may be less accurate if sub-indices use approximate distances
(e.g., PQ-encoded distances).

# Examples
```julia
merge_strategy = SimpleMerge()
```
"""
struct SimpleMerge <: AbstractMergeStrategy end

"""
    merge_results(
        strategy::AbstractMergeStrategy,
        result_lists::Vector{Vector{Neighbor{T}}},
        k::Int
    )::Vector{Neighbor{T}} where {T}

Merge results from multiple child indices into a single k-NN list.

# Arguments
- `strategy`: Merge strategy to use
- `result_lists`: Results from each probed child index
- `k`: Number of neighbors to return

# Returns
- Vector of up to k neighbors, sorted by distance

# Examples
```julia
# Merge results from 3 clusters
results = [
    [Neighbor(10, 0.5), Neighbor(20, 0.8)],
    [Neighbor(30, 0.3), Neighbor(40, 1.2)],
    [Neighbor(50, 0.6)]
]
merged = merge_results(SimpleMerge(), results, 3)
# Returns [Neighbor(30, 0.3), Neighbor(10, 0.5), Neighbor(50, 0.6)]
```
"""
function merge_results end

"""
    merge_results(
        ::SimpleMerge,
        result_lists::Vector{Vector{Neighbor{T}}},
        k::Int
    )::Vector{Neighbor{T}} where {T}

SimpleMerge implementation: collect all neighbors, sort by distance, take top-k unique.
"""
function merge_results(
    ::SimpleMerge,
    result_lists::Vector{Vector{Neighbor{T}}},
    k::Int
) where {T}
    # Collect all neighbors from all result lists
    all_neighbors = Neighbor{T}[]
    for result_list in result_lists
        append!(all_neighbors, result_list)
    end

    # Sort by distance
    sort!(all_neighbors, by = n -> n.dist)

    # Remove duplicates (keep first occurrence, which has smallest distance)
    seen_ids = Set{Int}()
    unique_neighbors = Neighbor{T}[]

    for neighbor in all_neighbors
        if !(neighbor.id in seen_ids)
            push!(unique_neighbors, neighbor)
            push!(seen_ids, neighbor.id)

            # Stop once we have k unique neighbors
            if length(unique_neighbors) >= k
                break
            end
        end
    end

    return unique_neighbors
end
