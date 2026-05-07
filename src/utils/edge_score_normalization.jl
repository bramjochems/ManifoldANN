using Statistics: median

"""
    compute_local_zscores(adj, scores) → Dict{Tuple{Int,Int}, Float64}

Normalize per-edge scores against the local neighborhood distribution at each
source node, using MAD (median absolute deviation) for robustness.

For each edge (i,j), computes z_i = (s_ij - median_i) / (1.4826 * MAD_i),
where the median and MAD are taken over all edges incident to node i.
The 1.4826 factor scales MAD to be comparable to σ for normal distributions.
"""
function compute_local_zscores(adj::Vector{Vector{Int}},
                                scores::Dict{Tuple{Int,Int}, Float64})
    zscores = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:length(adj)
        local_scores = Float64[]
        for j in adj[i]
            s = get(scores, (i, j), NaN)
            isnan(s) || push!(local_scores, s)
        end
        length(local_scores) < 2 && continue
        med = median(local_scores)
        mad_val = median(abs.(local_scores .- med))
        mad_val < 1e-12 && continue
        for j in adj[i]
            s = get(scores, (i, j), NaN)
            isnan(s) && continue
            zscores[(i, j)] = (s - med) / (1.4826 * mad_val)
        end
    end
    return zscores
end
