#!/usr/bin/env julia
# Side-by-side profile: MANN-NNDescent vs NND.jl on identical config.
# Run: julia --project=benchmarking/julia -t 4 scripts/nndescent_jl_profile.jl

using Random, Printf, LinearAlgebra, Profile
using ManifoldANN
using NearestNeighborDescent
const MA = ManifoldANN
const NND = NearestNeighborDescent

const N    = 20000
const D    = 32
const K    = 20
const EFS  = 60
const NQ   = 5000
const SEED = 0xC0FFEE
const MAX_IT = 8

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
println("config: n=$N d=$D k=$K ef_s=$EFS n_queries=$NQ")

Random.seed!(SEED)
data_mat = randn(Float32, D, N)
queries_mat = randn(Float32, D, NQ)
data_cols    = [collect(data_mat[:, i])    for i in 1:N]
queries_cols = [collect(queries_mat[:, i]) for i in 1:NQ]

# Build both
idx_mann = MA.build_index(MA.NNDescentIndex, data_mat;
    k=K, max_iterations=MAX_IT, threaded=true, rng=MersenneTwister(SEED))
g_nnd = NND.nndescent(data_cols, K, NND.Euclidean(); max_iters=MAX_IT)

# Warm
MA.query(idx_mann, data_mat, queries_mat[:, 1:8], K; ef_search=EFS)
NND.search(g_nnd, queries_cols[1:8], K; max_candidates=EFS)
GC.gc()

# ---- MANN profile ----
println("\n=== MANN-NNDescent profile ===")
Profile.clear()
@profile for _ in 1:3
    MA.query(idx_mann, data_mat, queries_mat, K; ef_search=EFS)
end
Profile.print(format=:flat, sortedby=:count, mincount=200, maxdepth=20)

GC.gc()

# ---- NND.jl profile ----
println("\n=== NND.jl profile ===")
Profile.clear()
@profile for _ in 1:3
    NND.search(g_nnd, queries_cols, K; max_candidates=EFS)
end
Profile.print(format=:flat, sortedby=:count, mincount=200, maxdepth=20)

println("\nDone.")
