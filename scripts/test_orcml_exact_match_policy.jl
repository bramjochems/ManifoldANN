"""
Validate the `OrcmlExact` compatibility profile against the reference
Python `orcml` curvatures. Mirrors `scripts/test_orcml_exact_match.jl`
but enables the orcml-compatible profile preset.
"""

using ManifoldANN
using LinearAlgebra
using DelimitedFiles
using Statistics

println("="^70)
println("OrcmlExact compatibility profile — validation")
println("="^70)

data = readdlm("benchmark_results/test_data.csv", ',')
data = data'

index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=15, directed=false)

println("\nGraph: nodes=", length(graph),
        " original_k=", graph.metadata.original_k,
        " directed=", graph.metadata.directed)

println("\nComputing ORC with OrcmlExact compatibility profile ...")
curvatures = ManifoldANN.compute_all_curvatures(
    graph, data;
    exclude_edge_endpoints=true,
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    solver=ManifoldANN.HungarianSolver(),
    fallback_solver=ManifoldANN.NetworkSimplexSolver(),
    use_threading=false,
    verbose=false,
    profile=OrcmlExact(),
)

python_results = readdlm("benchmark_results/curvatures_orcml_python.csv", ',', skipstart=1)

julia_curvs = Float64[]
python_curvs = Float64[]
for row in eachrow(python_results)
    i, j = Int(row[1]) + 1, Int(row[2]) + 1
    if haskey(curvatures, (i, j))
        push!(julia_curvs, curvatures[(i, j)].curvature)
        push!(python_curvs, row[3])
    end
end

println("\nMatched edges: ", length(julia_curvs))
println("Pearson r:    ", cor(julia_curvs, python_curvs))
println("Mean |Δ|:     ", mean(abs.(julia_curvs .- python_curvs)))
println("Max |Δ|:      ", maximum(abs.(julia_curvs .- python_curvs)))
println("Mean Δ:       ", mean(julia_curvs .- python_curvs))
println("Matches < 1e-6: ", sum(abs.(julia_curvs .- python_curvs) .< 1e-6),
        " / ", length(julia_curvs))
println("Matches < 1e-3: ", sum(abs.(julia_curvs .- python_curvs) .< 1e-3),
        " / ", length(julia_curvs))

pairs_path = "benchmark_results/manl_validation_pairs_orcml_policy.csv"
open(pairs_path, "w") do io
    println(io, "julia_curvature,python_curvature,abs_diff")
    for (jc, pc) in zip(julia_curvs, python_curvs)
        println(io, "$(jc),$(pc),$(abs(jc - pc))")
    end
end
println("Saved pairs to $(pairs_path)")
