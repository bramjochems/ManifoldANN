#!/usr/bin/env julia
# Batch-query benchmark for NN-Descent. Times the Matrix-input variant of
# `query` at multiple thread counts on a graph built once.
#
# Run: JULIA_NUM_THREADS=N julia --project=. scripts/nndescent_batch_query_bench.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "ND_N",    "20000"))
const D    = parse(Int, get(ENV, "ND_D",    "32"))
const K    = parse(Int, get(ENV, "ND_K",    "20"))
const EFS  = parse(Int, get(ENV, "ND_EFS",  "60"))
const NQ   = parse(Int, get(ENV, "ND_NQ",   "5000"))
const SEED = 0xC0FFEE
const REPS = 3
const MAX_IT = 8

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
println("config: n=$N d=$D k=$K ef_s=$EFS n_queries=$NQ reps=$REPS")

Random.seed!(SEED)
data    = randn(Float32, D, N)
queries = randn(Float32, D, NQ)

# Warm
let small_data = randn(Float32, D, 1_000), q = randn(Float32, D, 8)
    idx = MA.build_index(MA.NNDescentIndex, small_data; k=10, max_iterations=4,
                         rng=MersenneTwister(1))
    MA.query(idx, small_data, @view(q[:, 1]), 5; ef_search=20)
    MA.query(idx, small_data, q, 5; ef_search=20)
end
GC.gc()

@printf "Building index (n=%d d=%d k=%d)...\n" N D K
build_t = @elapsed idx = MA.build_index(MA.NNDescentIndex, data; k=K,
                                        max_iterations=MAX_IT, threaded=true,
                                        rng=MersenneTwister(SEED))
@printf "  build = %.3fs\n" build_t

MA.query(idx, data, @view(queries[:, 1:8]), K; ef_search=EFS)
GC.gc()

@printf "\nBatch query (Matrix input, %d queries):\n" NQ
times = Float64[]
allocs = Int[]
for r in 1:REPS
    GC.gc(); a0 = Base.gc_bytes()
    t = @elapsed MA.query(idx, data, queries, K; ef_search=EFS)
    a1 = Base.gc_bytes()
    push!(times, t)
    push!(allocs, a1 - a0)
end
let best = minimum(times), median = sort(times)[ceil(Int, length(times)/2)]
    @printf "  best:   %.3fs   qps %6.0f   alloc %.1f MB total = %.2f KB/query\n" best (NQ/best) (minimum(allocs)/(1024^2)) (minimum(allocs)/NQ/1024)
    @printf "  median: %.3fs   qps %6.0f\n" median (NQ/median)
end

println("\nDone.")
