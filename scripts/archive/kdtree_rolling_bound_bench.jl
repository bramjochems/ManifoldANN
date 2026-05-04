#!/usr/bin/env julia
# Bench KDTree query throughput under the FBF77 rolling cell-distance
# prune across the safe-metric set. Compares Euclidean (linear bound) vs
# SqEuclidean (squared bound; previously not supported) on identical data.
#
# The rolling bound is expected to be at-parity-or-faster than the legacy
# linear `axis_distance <= worst` check: tighter bound → more far-child
# prunes succeed → fewer leaf scans. SqEuclidean has the additional win of
# saving a sqrt per leaf-scan distance evaluation.
#
# Run: JULIA_NUM_THREADS=N julia --project=. scripts/kdtree_rolling_bound_bench.jl

using Random, Printf, LinearAlgebra
using Distances
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "KDB_N",  "20000"))
const D    = parse(Int, get(ENV, "KDB_D",  "16"))
const K    = parse(Int, get(ENV, "KDB_K",  "10"))
const NQ   = parse(Int, get(ENV, "KDB_NQ", "5000"))
const SEED = 0xCAFE
const REPS = 3

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
@printf "config: n=%d d=%d k=%d n_queries=%d reps=%d\n" N D K NQ REPS

Random.seed!(SEED)
data    = randn(Float32, D, N)
queries = randn(Float32, D, NQ)

# Warm
let small = randn(Float32, D, 1_000), q = randn(Float32, D, 8)
    for m in (Euclidean(), SqEuclidean(), Cityblock())
        idx = MA.build_index(MA.KDTreeIndex, small; distance = m)
        MA.query(idx, small, @view(q[:, 1]), 5)
        MA.query(idx, small, q, 5)
    end
end
GC.gc()

function bench(metric)
    idx = MA.build_index(MA.KDTreeIndex, data; distance = metric)
    best = Inf
    for _ in 1:REPS
        t = @elapsed MA.query(idx, data, queries, K)
        best = min(best, t)
    end
    qps = NQ / best
    return best, qps
end

println()
@printf "%-22s %12s %12s\n" "metric" "best_s" "qps"
for m in (Euclidean(), SqEuclidean(), Cityblock(), Minkowski(3.0))
    t, qps = bench(m)
    @printf "%-22s %12.4f %12.1f\n" string(typeof(m).name.name) t qps
end
