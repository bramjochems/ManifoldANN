using ..ManifoldANN: default_distance, KMeansTransform, fit!, take_pending_assignments!
using Distances

"""
    IVFFlatIndex

Lightweight IVF-Flat index optimized for build/query speed (no PQ, no
graph layers). Stores centroids and inverted lists of point ids; data is always
supplied at query time to honor the shared ANN contract.
"""
struct IVFFlatIndex{T,Dm,Dc} <: AbstractANNIndex
    centroids::Matrix{T}
    centroid_norms::Vector{T}
    lists::Vector{Vector{Int}}
    dimension::Int
    default_nprobe::Int
    distance::Dm          # point-point distance
    centroid_metric::Dc   # centroid assignment metric
end

index_distance(index::IVFFlatIndex) = index.distance

"""
    build_index(IVFFlatIndex, data; nlist, nprobe=10, distance=default_distance, centroid_metric)

Build an IVF-Flat style index. Centroids come from k-means; inverted lists hold
point ids only (no stored data). `centroid_metric` defaults to Euclidean unless
an angular distance is requested.
"""
function build_index(
    ::Type{IVFFlatIndex},
    data::AbstractMatrix{T};
    nlist::Int,
    nprobe::Int = 10,
    distance::Dm = default_distance,
    centroid_metric::Dc = Distances.Euclidean(),
) where {T<:Real,Dm,Dc<:Distances.SemiMetric}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    nlist > 0 || throw(ArgumentError("nlist must be positive"))

    # Fit k-means using existing transform to avoid reinventing distance kernels
    kmeans = KMeansTransform(k = nlist, distance = centroid_metric, init = :kmeans_plus_plus)
    fit!(kmeans, data)
    assignments = take_pending_assignments!(kmeans)
    assignments === nothing && error("KMeansTransform did not return assignments")
    length(assignments) == n || error("Assignment length mismatch")

    lists = [Int[] for _ in 1:nlist]
    @inbounds for (idx, cid) in enumerate(assignments)
        push!(lists[cid], idx)
    end

    centroids = kmeans.centroids
    centroids === nothing && error("KMeans centroids missing after fit!")
    centroid_norms = kmeans.centroid_norms
    centroid_norms === nothing && (centroid_norms = vec(sum(abs2, centroids; dims=1)))

    return IVFFlatIndex{eltype(centroids),Dm,Dc}(
        centroids,
        centroid_norms,
        lists,
        d,
        max(1, min(nprobe, nlist)),
        distance,
        centroid_metric,
    )
end

@inline function _validate_dimensions(index::IVFFlatIndex, data, qlen::Int)
    size(data, 1) == index.dimension ||
        throw(DimensionMismatch("Expected data with $(index.dimension) rows"))
    qlen == index.dimension ||
        throw(DimensionMismatch("Expected query of length $(index.dimension)"))
    return nothing
end

@inline function _centroid_distances(
    q::AbstractVector{T},
    centroids::AbstractMatrix{T},
    metric::Distances.SemiMetric,
    centroid_norms::AbstractVector{T},
) where {T}
    k = size(centroids, 2)
    dists = Vector{T}(undef, k)
    if metric isa Distances.Euclidean
        x_norm_sq = sum(abs2, q)
        dots = centroids' * q
        @inbounds for i in 1:k
            sq = centroid_norms[i] + x_norm_sq - 2 * dots[i]
            dists[i] = sqrt(max(sq, zero(T)))
        end
    elseif metric isa Distances.SqEuclidean
        x_norm_sq = sum(abs2, q)
        dots = centroids' * q
        @inbounds for i in 1:k
            sq = centroid_norms[i] + x_norm_sq - 2 * dots[i]
            dists[i] = max(sq, zero(T))
        end
    else
        @inbounds for i in 1:k
            dists[i] = Distances.evaluate(metric, @view(centroids[:, i]), q)
        end
    end
    return dists
end

"""
    query(index::IVFFlatIndex, data, q, k; nprobe=index.default_nprobe)

Probe `nprobe` closest centroids and brute-force within their inverted lists.
"""
function query(
    index::IVFFlatIndex{T,Dm},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    nprobe::Int = index.default_nprobe,
) where {T,Dm}
    _validate_dimensions(index, data, length(q))
    nprobe = clamp(nprobe, 1, length(index.lists))

    # Route to nearest centroids
    centroid_dists = _centroid_distances(q, index.centroids, index.centroid_metric, index.centroid_norms)
    probe_ids = partialsortperm(centroid_dists, 1:nprobe)

    # Collect candidate neighbors
    candidates = Neighbor{float(T)}[]
    dist_fn = index.distance
    for cid in probe_ids
        ids = index.lists[cid]
        @inbounds for id in ids
            dist = dist_fn(@view(data[:, id]), q)
            push!(candidates, Neighbor{float(T)}(id, dist))
        end
    end

    isempty(candidates) && return Neighbor{float(T)}[]
    sort!(candidates, by = n -> n.dist)
    k_eff = min(k, length(candidates))
    return candidates[1:k_eff]
end

function query(
    index::IVFFlatIndex{T,Dm},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    nprobe::Int = index.default_nprobe,
) where {T,Dm}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))
    n_queries = size(queries, 2)
    results = Vector{Vector{Neighbor{float(T)}}}(undef, n_queries)
    @inbounds for i in 1:n_queries
        q = @view queries[:, i]
        results[i] = query(index, data, q, k; nprobe=nprobe)
    end
    return results
end
