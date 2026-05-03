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
    total = 0
    @inbounds for r in result_lists
        total += length(r)
    end
    total == 0 && return Neighbor{T}[]

    all_neighbors = Vector{Neighbor{T}}(undef, total)
    pos = 1
    @inbounds for r in result_lists
        for n in r
            all_neighbors[pos] = n
            pos += 1
        end
    end
    sort!(all_neighbors, by = n -> n.dist)

    # Dedup by linear scan over the kept prefix. Replaces a Set{Int} for the
    # typical case (k ≤ 32, total ≤ a few hundred) where hashset overhead
    # exceeds the O(k²) scan. For IVF with disjoint partitions the dedup is
    # a no-op; the check is kept so this function works for future
    # overlapping-partition strategies.
    cap = min(k, total)
    unique_neighbors = Vector{Neighbor{T}}(undef, cap)
    n_kept = 0
    @inbounds for i in 1:total
        cand = all_neighbors[i]
        already = false
        for j in 1:n_kept
            if unique_neighbors[j].id == cand.id
                already = true
                break
            end
        end
        already && continue
        n_kept += 1
        unique_neighbors[n_kept] = cand
        n_kept >= cap && break
    end
    resize!(unique_neighbors, n_kept)
    return unique_neighbors
end
