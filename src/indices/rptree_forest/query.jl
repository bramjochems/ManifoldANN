"""
    query(index::RPTreeIndex, data, q, k)

Approximate kNN by routing `q` to its leaf bucket and brute-force scanning
the bucket. Returns up to `k` neighbours, sorted by distance.
"""
function query(
    index::RPTreeIndex{T,D},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
) where {T<:AbstractFloat,D}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    heap = BoundedMaxHeap{S}(actual_k)
    members = leaf_members(index.tree, q)
    @inbounds for j in eachindex(members)
        point_id = members[j]
        dist = S(index.distance(@view(data[:, point_id]), q))
        push!(heap, point_id, dist)
    end
    return to_sorted_vector(heap)
end
