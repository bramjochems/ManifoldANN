"""
    recall_at_k(found, ground_truth) -> Float64

Compute recall@k for a single query, where `ground_truth` holds the true
neighbor ids ordered by distance. Returns a value in `[0, 1]`.
"""
function recall_at_k(found::AbstractVector{<:Integer}, ground_truth::AbstractVector{<:Integer})
    k = length(ground_truth)
    k > 0 || return 0.0
    hits = 0
    gt = Set(ground_truth)
    for id in found
        id in gt && (hits += 1)
    end
    return hits / k
end

"""
    recall_at_k(found_batches, ground_truth_batches) -> Float64

Average recall@k across multiple queries.
"""
function recall_at_k(
    found_batches::AbstractVector{<:AbstractVector{<:Integer}},
    ground_truth_batches::AbstractVector{<:AbstractVector{<:Integer}},
)
    length(found_batches) == length(ground_truth_batches) ||
        throw(ArgumentError("found and ground truth batches must match in length"))
    n = length(found_batches)
    n == 0 && return 0.0
    total = 0.0
    @inbounds for i in 1:n
        total += recall_at_k(found_batches[i], ground_truth_batches[i])
    end
    return total / n
end
