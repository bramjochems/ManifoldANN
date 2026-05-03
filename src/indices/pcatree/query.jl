"""
    query(index::PCATreeIndex, data, q, k)

Approximate kNN via PCA-tree routing: at each internal node compare
`direction · (q - center)` against the stored threshold; descend into
the matching child. Brute-force scan the leaf bucket and return the
top-`k` by distance. Concurrent-safe (read-only on `index` and `data`).

The descent logic deliberately mirrors `RPTreeIndex.query`'s
no-pruning pattern: PCA splits use general directions, so the
componentwise-monotone bound that powers KDTree pruning does not apply.
A future tighter descent (multi-probe, beam search) could be a
separate query method.
"""
function query(
    index::PCATreeIndex{T,D,Sp},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
) where {T<:AbstractFloat,D,Sp}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    members = _pcatree_route(index, q)
    heap = BoundedMaxHeap{S}(actual_k)
    @inbounds for j in eachindex(members)
        point_id = members[j]
        dist = S(index.distance(@view(data[:, point_id]), q))
        push!(heap, point_id, dist)
    end
    return to_sorted_vector(heap)
end

@inline function _pcatree_route(
    index::PCATreeIndex{T,D,Sp},
    q::AbstractVector{T},
) where {T<:AbstractFloat,D,Sp}
    idx = index.root
    @inbounds while true
        node = index.nodes[idx]
        if bpt_is_leaf(node)
            lo = Int(node.leaf_lo)
            hi = Int(node.leaf_hi)
            return view(index.leaf_members, lo:hi)
        end
        payload = node.payload
        proj = zero(T)
        d = length(payload.direction)
        @simd for i in 1:d
            proj += payload.direction[i] * (q[i] - payload.center[i])
        end
        idx = proj <= payload.threshold ? Int(node.left) : Int(node.right)
    end
end
