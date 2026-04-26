"""
Test exact match with orcml after implementing:
1. Undirected graph support (directed=false)
2. k-nearest selection in effective_epsilon for symmetric graphs
"""

using ManifoldANN
using LinearAlgebra
using DelimitedFiles
using Statistics

println("="^70)
println("Testing Exact orcml Match")
println("="^70)

# Load the test data
data = readdlm("benchmark_results/test_data.csv", ',')
data = data'  # Transpose to get d x n

# Build graph
index = build_index(BruteForceIndex, data)

# NEW: Build undirected graph (symmetric, like orcml)
graph = build_knn_graph(index, data; k=15, directed=false)

println("\n1. Graph Structure:")
println("   Nodes: ", length(graph))
println("   k (max degree): ", graph.k)
println("   original_k: ", graph.metadata.original_k)
println("   directed: ", graph.metadata.directed)

# Check specific nodes match Python
test_nodes = [1, 12]  # (0, 11) in Python
for node in test_nodes
    neighbors = sort(graph[node] .- 1)  # 0-indexed
    println("\n   Node $(node-1) [0-indexed]:")
    println("     Neighbors: ", neighbors)
    println("     Count: ", length(graph[node]))
end

# Compute curvatures with orcml configuration
# Use NetworkSimplexSolver as fallback instead of Sinkhorn to avoid convergence issues
println("\n2. Computing ORC (orcml config, OrcmlExact profile)...")
curvatures = ManifoldANN.compute_all_curvatures(
    graph, data;
    variant=ManifoldANN.ORCManL(profile=ManifoldANN.OrcmlExact()),
    solver=ManifoldANN.HungarianSolver(),
    fallback_solver=ManifoldANN.NetworkSimplexSolver(),
    use_threading=false,
    verbose=false
)

# Test specific edge
test_edge = (1, 12)
println("\n3. Test Edge ($test_edge) [Python: (0, 11)]:")
if haskey(curvatures, test_edge)
    result = curvatures[test_edge]
    println("   Julia:")
    println("     κ = ", result.curvature)
    println("     W1 = ", result.wasserstein_distance)
    println("     d = ", result.edge_distance)
end

# Load Python results
python_results = readdlm("benchmark_results/curvatures_orcml_python.csv", ',', skipstart=1)
for row in eachrow(python_results)
    if Int(row[1]) == 0 && Int(row[2]) == 11
        python_curv = row[3]
        println("   Python:")
        println("     κ = ", python_curv)

        if haskey(curvatures, test_edge)
            diff = abs(result.curvature - python_curv)
            println("   Difference: ", diff)
            if diff < 1e-6
                println("   ✅ EXACT MATCH (within 1e-6)!")
            elseif diff < 0.01
                println("   ✓ Very close (< 0.01)")
            else
                println("   ⚠ Still differs")
            end
        end
        break
    end
end

# Save results for full comparison
println("\n4. Saving Results...")
open("benchmark_results/curvatures_manl_exact_test.csv", "w") do io
    println(io, "source,target,curvature")
    for ((i, j), result) in curvatures
        println(io, "$(i-1),$(j-1),$(result.curvature)")
    end
end
println("   Saved to benchmark_results/curvatures_manl_exact_test.csv")

# Compute full correlation
println("\n5. Full Comparison:")
julia_curvs = Float64[]
python_curvs = Float64[]

for row in eachrow(python_results)
    i, j = Int(row[1]) + 1, Int(row[2]) + 1  # Convert to 1-indexed
    if haskey(curvatures, (i, j))
        push!(julia_curvs, curvatures[(i, j)].curvature)
        push!(python_curvs, row[3])
    end
end

println("   Matched edges: ", length(julia_curvs))
println("   Correlation: ", cor(julia_curvs, python_curvs))
println("   Mean absolute error: ", mean(abs.(julia_curvs .- python_curvs)))
println("   Max absolute error: ", maximum(abs.(julia_curvs .- python_curvs)))

# Count how many match within different tolerances
exact_matches = sum(abs.(julia_curvs .- python_curvs) .< 1e-6)
close_matches = sum(abs.(julia_curvs .- python_curvs) .< 1e-3)
println("   Matches within 1e-6: $exact_matches / $(length(julia_curvs))")
println("   Matches within 1e-3: $close_matches / $(length(julia_curvs))")

# Persist matched (julia, python) curvature pairs for plotting
pairs_path = "benchmark_results/manl_validation_pairs.csv"
open(pairs_path, "w") do io
    println(io, "julia_curvature,python_curvature,abs_diff")
    for (jc, pc) in zip(julia_curvs, python_curvs)
        println(io, "$(jc),$(pc),$(abs(jc - pc))")
    end
end
println("   Matched pairs saved to $(pairs_path)")

println("\n" * "="^70)
if cor(julia_curvs, python_curvs) > 0.99
    println("✅ EXCELLENT MATCH!")
elseif cor(julia_curvs, python_curvs) > 0.95
    println("✓ Very good match")
else
    println("⚠ Still room for improvement")
end
println("="^70)
