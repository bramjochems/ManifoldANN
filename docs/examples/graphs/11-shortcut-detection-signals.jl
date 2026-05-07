#=
Example: Per-Edge Shortcut Detection Signals

This example demonstrates the four non-ORC shortcut detection signals shipped
with ManifoldANN.jl, evaluated on a noisy Swiss roll where ground-truth
"shortcut" edges (those that bridge across rolls) can be identified from
the chord-vs-geodesic ratio.

Signals demonstrated:
  - Jaccard neighbourhood overlap   (compute_jaccard_scores)
  - Gabriel graph test              (compute_gabriel_mask)
  - Tangent-plane angle             (compute_tangent_angles)
  - Local MAD-based z-score         (compute_local_zscores)

For each signal we report AUROC against ground-truth shortcut labels.
=#

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "geodesic", "swiss_roll_utils.jl"))

println("=== Per-Edge Shortcut Detection Signals ===\n")

# ----------------------------------------------------------------------------
# Step 1: Generate a noisy Swiss roll
# ----------------------------------------------------------------------------
Random.seed!(42)
n = 800
data, params = generate_swiss_roll(n; rng=MersenneTwister(42))
data .+= 0.5 .* randn(MersenneTwister(7), size(data))

println("Generated Swiss roll: $(size(data, 2)) points in $(size(data, 1))D")

# ----------------------------------------------------------------------------
# Step 2: Build kNN graph + per-node PCA geometries
# ----------------------------------------------------------------------------
k = 12
index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=k)

graph = model.weighted_graph.graph
geometries = model.weighted_graph.geometries
adj = graph.neighbors

println("Built kNN graph (k=$k) with PCA tangent planes at each node\n")

# ----------------------------------------------------------------------------
# Step 3: Label edges as shortcuts via the chord/geodesic ratio
# ----------------------------------------------------------------------------
# An edge (i,j) is a shortcut if the ambient (chord) distance is much smaller
# than the true manifold geodesic — i.e. the edge cuts across the roll.
shortcut_threshold = 2.0  # geodesic / chord > 2.0 ⇒ shortcut

labels = Dict{Tuple{Int,Int}, Bool}()
for i in 1:length(adj)
    for j in adj[i]
        chord = norm(data[:, i] - data[:, j])
        geo = exact_swiss_roll_geodesic(params.t[i], params.h[i],
                                        params.t[j], params.h[j])
        labels[(i, j)] = (geo / max(chord, 1e-12)) > shortcut_threshold
    end
end

n_pos = count(values(labels))
n_neg = length(labels) - n_pos
println("Edges: $(length(labels)) total, $n_pos shortcuts, $n_neg manifold-following\n")

# ----------------------------------------------------------------------------
# Step 4: Compute the four detection signals
# ----------------------------------------------------------------------------
println("Computing signals...")
jaccard = compute_jaccard_scores(graph)
gabriel = compute_gabriel_mask(graph, data)
angles  = compute_tangent_angles(adj, data, geometries)
angle_z = compute_local_zscores(adj, angles)

# Orient signals so higher score = more shortcut-like.
neg_jaccard = Dict(e => -v for (e, v) in jaccard)         # low Jaccard ⇒ shortcut
non_gabriel = Dict(e => v ? 0.0 : 1.0 for (e, v) in gabriel)  # non-Gabriel ⇒ shortcut

# ----------------------------------------------------------------------------
# Step 5: AUROC
# ----------------------------------------------------------------------------
function auroc(scores::Dict{Tuple{Int,Int}, T}, labels::Dict{Tuple{Int,Int}, Bool}) where {T<:Real}
    edges = collect(keys(scores))
    intersect!(edges, collect(keys(labels)))
    isempty(edges) && return NaN
    s = [Float64(scores[e]) for e in edges]
    y = [labels[e] for e in edges]
    pos = sum(y); neg = length(y) - pos
    (pos == 0 || neg == 0) && return NaN
    # Mann–Whitney U via average ranks
    order = sortperm(s)
    ranks = similar(s)
    i = 1
    while i <= length(s)
        j = i
        while j < length(s) && s[order[j+1]] == s[order[i]]
            j += 1
        end
        avg = (i + j) / 2
        for r in i:j
            ranks[order[r]] = avg
        end
        i = j + 1
    end
    rank_sum_pos = sum(ranks[k] for k in eachindex(y) if y[k])
    U = rank_sum_pos - pos * (pos + 1) / 2
    return U / (pos * neg)
end

println("\nAUROC (higher = better shortcut discrimination):")
@printf "  Jaccard (negated)        %.3f\n" auroc(neg_jaccard, labels)
@printf "  Gabriel (non-Gabriel)    %.3f\n" auroc(non_gabriel, labels)
@printf "  Tangent angle            %.3f\n" auroc(angles, labels)
@printf "  Tangent-angle z-score    %.3f\n" auroc(angle_z, labels)

println("""

Note: AUROC values on a single (n, k, noise) cell are noisy. Chapter 5 of the
thesis aggregates over many cells to compare signals systematically.
""")
