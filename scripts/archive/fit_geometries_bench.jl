#!/usr/bin/env julia
# Microbench for _fit_geometries(::ShareSimilarTangents).
# Measures the impact of replacing Vector{Any} with Vector{Union{Nothing,G}}.
#
# Run: julia --project=. -t N scripts/fit_geometries_bench.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
using ManifoldANN: KNNGraph, ShareSimilarTangents, SubspaceAngleCriterion,
                    PCAMethod, _fit_geometries, NoSharing
const MA = ManifoldANN

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())

const N      = parse(Int, get(ENV, "FG_N",   "5000"))
const D      = parse(Int, get(ENV, "FG_D",   "32"))
const K      = parse(Int, get(ENV, "FG_K",   "20"))
const TARGET = parse(Int, get(ENV, "FG_TGT", "8"))   # PCA target dim
const MAXGD  = parse(Int, get(ENV, "FG_MGD", "2"))   # max_graph_distance

println("config: n=$N d=$D k=$K target_dim=$TARGET max_graph_distance=$MAXGD")

Random.seed!(0xC0FFEE)
data = randn(Float32, D, N)

# Build a kNN graph to feed _fit_geometries
println("Building kNN graph...")
idx = MA.build_index(MA.NNDescentIndex, data; k=K, max_iterations=8, threaded=true,
                     rng=MersenneTwister(1))
graph = MA.build_knn_graph(idx, data; k=K, ef_search=60)

method = PCAMethod(intrinsic_dim=TARGET)
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/12); max_graph_distance=MAXGD)

# Warm
let smaller = randn(Float32, D, 500)
    sidx = MA.build_index(MA.NNDescentIndex, smaller; k=10, max_iterations=4,
                          rng=MersenneTwister(2))
    sgraph = MA.build_knn_graph(sidx, smaller; k=10, ef_search=30)
    _fit_geometries(sharing, method, sgraph, smaller)
    _fit_geometries(NoSharing(), method, sgraph, smaller)
end
GC.gc()

# Bench ShareSimilarTangents (the optimised path)
println("\n[_fit_geometries(::ShareSimilarTangents) — optimised path]")
times = Float64[]
for r in 1:5
    GC.gc()
    t0 = time_ns()
    _fit_geometries(sharing, method, graph, data)
    push!(times, (time_ns() - t0) / 1e9)
end
@printf "  best: %.3fs   median: %.3fs   range: [%.3fs, %.3fs]\n" minimum(times) sort(times)[3] minimum(times) maximum(times)

# Bench NoSharing for reference (already type-stable)
println("\n[_fit_geometries(::NoSharing) — reference, unchanged path]")
times_ns = Float64[]
for r in 1:5
    GC.gc()
    t0 = time_ns()
    _fit_geometries(NoSharing(), method, graph, data)
    push!(times_ns, (time_ns() - t0) / 1e9)
end
@printf "  best: %.3fs   median: %.3fs\n" minimum(times_ns) sort(times_ns)[3]

println("\nDone.")
