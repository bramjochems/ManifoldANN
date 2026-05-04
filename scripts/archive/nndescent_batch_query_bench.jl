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

function _bench_batch(label::AbstractString, idx, data, qs, k, ef, reps)
    nq = size(qs, 2)
    times = Float64[]
    allocs = Int[]
    for _ in 1:reps
        GC.gc(); a0 = Base.gc_bytes()
        t = @elapsed MA.query(idx, data, qs, k; ef_search=ef)
        a1 = Base.gc_bytes()
        push!(times, t); push!(allocs, a1 - a0)
    end
    best = minimum(times)
    median = sort(times)[ceil(Int, length(times)/2)]
    @printf "%s (n_queries=%d):\n" label nq
    @printf "  best:   %.3fs   qps %6.0f   alloc %.2f KB/query\n" best (nq/best) (minimum(allocs)/nq/1024)
    @printf "  median: %.3fs   qps %6.0f\n" median (nq/median)
end

println()
_bench_batch("Batch query (large)", idx, data, queries, K, EFS, REPS)

# Small-batch sweep — the threaded path's Channel + worker spawn has a fixed
# cost; a small batch can be slower than the serial fallback. Verify the
# BATCH_THREAD_THRESHOLD gate keeps small-batch performance reasonable.
println()
small_qs = @view queries[:, 1:min(100, NQ)]
_bench_batch("Batch query (small)", idx, data, small_qs, K, EFS, REPS)

println("\nDone.")
