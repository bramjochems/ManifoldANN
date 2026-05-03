#!/usr/bin/env julia
# Batch-query benchmark for HNSW. Times the Matrix-input variant of `query`
# at multiple thread counts, on a graph built once.
#
# Run: JULIA_NUM_THREADS=N julia --project=. scripts/hnsw_batch_query_bench.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "HNSW_N", "50000"))
const D    = parse(Int, get(ENV, "HNSW_D", "32"))
const M    = 16
const EFC  = 200
const EFS  = 80
const K    = 10
const NQ   = parse(Int, get(ENV, "HNSW_NQ", "5000"))
const SEED = 0xC0FFEE
const REPS = 3

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
println("config: n=$N d=$D M=$M ef_c=$EFC ef_s=$EFS k=$K n_queries=$NQ reps=$REPS")

Random.seed!(SEED)
data = randn(Float32, D, N)
queries = randn(Float32, D, NQ)

# Warm
let small = randn(Float32, D, 1_000), q = randn(Float32, D, 8)
    idx = MA.build_index(MA.HNSWIndex, small; M=8, ef_construction=40, ef_search=20)
    MA.query(idx, small, @view(q[:, 1]), 5; ef_search=20)
    MA.query(idx, small, q, 5; ef_search=20)
end
GC.gc()

# Build the real index (timed but not the focus)
@printf "Building index (n=%d d=%d)...\n" N D
build_t = @elapsed idx = MA.build_index(MA.HNSWIndex, data; M=M, ef_construction=EFC, ef_search=EFS)
@printf "  build = %.3fs\n" build_t

# Warm batch query path
MA.query(idx, data, @view(queries[:, 1:8]), K; ef_search=EFS)
GC.gc()

# Timed batch queries
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
