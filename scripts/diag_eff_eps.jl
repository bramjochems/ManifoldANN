using ManifoldANN
using LinearAlgebra
using DelimitedFiles
using Statistics

data = readdlm("benchmark_results/test_data.csv", ',')
data = data'

index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=15, directed=false)

# Pick edge (1, 12) [Python (0,11)]
i, j = 1, 12
println("graph[i] (1-indexed) = ", sort(collect(graph[i])))
println("graph[j] (1-indexed) = ", sort(collect(graph[j])))
println("|N(i)|=", length(graph[i]), "  |N(j)|=", length(graph[j]))

eps_default = ManifoldANN.effective_epsilon(i, j, graph, data)
eps_orcml = ManifoldANN.effective_epsilon(i, j, graph, data; policy=OrcmlEffectiveEps())
println("ManifoldANN (default) effective_eps  = ", eps_default)
println("OrcmlEffectiveEps        effective_eps = ", eps_orcml)

# Expected from Python: print all neighbor dists for i then drop smallest, take next k=15
dists_i = sort([norm(data[:, i] - data[:, n]) for n in graph[i]])
dists_j = sort([norm(data[:, j] - data[:, n]) for n in graph[j]])
println("\nSorted N(i) dists: ", dists_i)
println("Sorted N(j) dists: ", dists_j)
println("orcml mean i (slice [2:16]): ", mean(dists_i[2:min(16,end)]))
println("orcml mean j (slice [2:16]): ", mean(dists_j[2:min(16,end)]))
