using LinearAlgebra: norm

"""
    compute_jaccard_scores(graph::KNNGraph) → Dict{Tuple{Int,Int}, Float64}

Compute Jaccard similarity for each edge (i,j):
  J(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|

Low Jaccard → endpoints live in different neighborhoods → likely shortcut.
"""
function compute_jaccard_scores(graph::KNNGraph)
    scores = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:length(graph)
        ni = Set(graph[i])
        for j in graph[i]
            nj = Set(graph[j])
            shared = length(ni ∩ nj)
            total  = length(ni ∪ nj)
            scores[(i, j)] = total > 0 ? shared / total : 0.0
        end
    end
    return scores
end

"""
    compute_gabriel_mask(graph::KNNGraph, data::AbstractMatrix) → Dict{Tuple{Int,Int}, Bool}

Gabriel graph test for each edge (i,j): edge is Gabriel if no other point falls
inside the diametral sphere (sphere with (i,j) as diameter).
Non-Gabriel edges are candidates for removal.
"""
function compute_gabriel_mask(graph::KNNGraph, data::AbstractMatrix)
    mask = Dict{Tuple{Int,Int}, Bool}()
    n = length(graph)
    for i in 1:n
        for j in graph[i]
            midpoint = (data[:, i] .+ data[:, j]) ./ 2
            radius_sq = sum(abs2, data[:, i] .- data[:, j]) / 4
            is_gabriel = true
            for k in 1:n
                k == i && continue
                k == j && continue
                if sum(abs2, data[:, k] .- midpoint) < radius_sq
                    is_gabriel = false
                    break
                end
            end
            mask[(i, j)] = is_gabriel
        end
    end
    return mask
end

"""
    compute_tangent_angles(adj, data, geometries) → Dict{Tuple{Int,Int}, Float64}

For each edge (i,j), compute the angle between the edge vector and the local
tangent plane. Uses pre-fitted PCA geometries at each node.

Angle = arcsin(|orthogonal component| / |edge vector|), averaged over both endpoints.
Range: [0, π/2]. 0 = tangent to manifold, π/2 = perpendicular.
Higher angle → more likely a shortcut.
"""
function compute_tangent_angles(adj::Vector{Vector{Int}},
                                data::AbstractMatrix,
                                geometries::Vector)
    angles = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:length(adj)
        for j in adj[i]
            edge_vec = data[:, j] - data[:, i]
            edge_len = norm(edge_vec)
            edge_len < 1e-12 && continue

            err_i = fit_error(geometries[i], data[:, j])
            angle_i = asin(clamp(err_i / edge_len, 0.0, 1.0))

            err_j = fit_error(geometries[j], data[:, i])
            angle_j = asin(clamp(err_j / edge_len, 0.0, 1.0))

            angles[(i, j)] = (angle_i + angle_j) / 2.0
        end
    end
    return angles
end
