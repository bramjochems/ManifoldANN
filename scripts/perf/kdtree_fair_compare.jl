#!/usr/bin/env julia
# Fair head-to-head: ManifoldANN KDTreeIndex vs NearestNeighbors.jl KDTree
# (and BallTree, since BallTree is the canonical higher-d choice in NN.jl).
#
# Both libraries are Julia, so no juliacall / Python overhead. The timed
# region excludes JIT (warmup runs), data conversion (data is in the right
# layout once), and result-conversion (queried but not unpacked into a
# uniform output type).
#
# Run from the benchmarking/julia environment:
#   julia --project=benchmarking/julia -t auto scripts/kdtree_fair_compare.jl
#
# Env knobs:
#   KDT_N=10000       number of training points
#   KDT_NQ=1000       number of test queries
#   KDT_DS=8,32,128   comma-separated dimensions to sweep
#   KDT_K=10          neighbours per query
#   KDT_REPS=3        timed reps per config (best of)

using Random, Printf, LinearAlgebra
using ManifoldANN
using NearestNeighbors
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "KDT_N", "10000"))
const NQ   = parse(Int, get(ENV, "KDT_NQ", "1000"))
const DS   = [parse(Int, x) for x in split(get(ENV, "KDT_DS", "8,32,128"), ",")]
const K    = parse(Int, get(ENV, "KDT_K", "10"))
const REPS = parse(Int, get(ENV, "KDT_REPS", "3"))
const SEED = 0xC0FFEE

BLAS.set_num_threads(1)
println("nthreads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads())")
println("config: n=$N nq=$NQ k=$K reps=$REPS dims=$DS")

# Warmup: compile every code path on a tiny dataset before any timed run.
let
    Random.seed!(SEED)
    warm = randn(Float32, 4, 200)
    warm_q = randn(Float32, 4, 8)
    # MANN
    let idx = MA.build_index(MA.KDTreeIndex, warm; axis_selector=:variance)
        MA.query(idx, warm, @view(warm_q[:, 1]), 5)
    end
    # NN.jl KDTree
    let tree = NearestNeighbors.KDTree(warm, NearestNeighbors.Euclidean(); leafsize=10)
        NearestNeighbors.knn(tree, @view(warm_q[:, 1]), 5)
        NearestNeighbors.knn(tree, warm_q, 5)
    end
    # NN.jl BallTree
    let tree = NearestNeighbors.BallTree(warm, NearestNeighbors.Euclidean(); leafsize=10)
        NearestNeighbors.knn(tree, @view(warm_q[:, 1]), 5)
        NearestNeighbors.knn(tree, warm_q, 5)
    end
end
GC.gc()
println("warmup done\n")

# --- Single-thread loop over query (fair against both libraries' single-query API) ---
function bench_mann(data, queries, k; axis=:variance)
    t_build = @elapsed idx = MA.build_index(MA.KDTreeIndex, data; axis_selector=axis)
    GC.gc()
    a0 = Base.gc_bytes()
    nq = size(queries, 2)
    t_query = @elapsed for j in 1:nq
        MA.query(idx, data, @view(queries[:, j]), k)
    end
    a1 = Base.gc_bytes()
    return (build=t_build, query=t_query, alloc=(a1-a0), idx=idx)
end

function bench_nn_kdtree(data, queries, k; leafsize=10)
    t_build = @elapsed tree = NearestNeighbors.KDTree(data, NearestNeighbors.Euclidean(); leafsize=leafsize)
    GC.gc()
    a0 = Base.gc_bytes()
    nq = size(queries, 2)
    t_query = @elapsed for j in 1:nq
        NearestNeighbors.knn(tree, @view(queries[:, j]), k)
    end
    a1 = Base.gc_bytes()
    return (build=t_build, query=t_query, alloc=(a1-a0), idx=tree)
end

function bench_nn_balltree(data, queries, k; leafsize=10)
    t_build = @elapsed tree = NearestNeighbors.BallTree(data, NearestNeighbors.Euclidean(); leafsize=leafsize)
    GC.gc()
    a0 = Base.gc_bytes()
    nq = size(queries, 2)
    t_query = @elapsed for j in 1:nq
        NearestNeighbors.knn(tree, @view(queries[:, j]), k)
    end
    a1 = Base.gc_bytes()
    return (build=t_build, query=t_query, alloc=(a1-a0), idx=tree)
end

function best_of(f, reps)
    results = [f() for _ in 1:reps]
    return (build=minimum(r.build for r in results),
            query=minimum(r.query for r in results),
            alloc=minimum(r.alloc for r in results))
end

# Recall vs ground-truth (use BruteForce as oracle, single sample).
function recall_at_k(idx, data, queries, k, query_fn)
    brute = MA.build_index(MA.BruteForceIndex, data)
    hits, total = 0, 0
    nq = size(queries, 2)
    for j in 1:nq
        truth = Set(n.id for n in MA.query(brute, data, @view(queries[:, j]), k))
        got_ids = query_fn(idx, @view(queries[:, j]))
        hits += length(intersect(Set(got_ids), truth))
        total += k
    end
    return hits / total
end

println("="^96)
@printf "%-25s %-6s %-10s %-10s %-12s %-8s\n" "library" "d" "build_ms" "query_ms" "alloc_KB/q" "recall@$K"
println("="^96)

for d in DS
    Random.seed!(SEED)
    data = randn(Float32, d, N)
    queries = randn(Float32, d, NQ)

    GC.gc()
    mann = best_of(() -> bench_mann(data, queries, K), REPS)
    GC.gc()
    nn_kd = best_of(() -> bench_nn_kdtree(data, queries, K), REPS)
    GC.gc()
    nn_ball = best_of(() -> bench_nn_balltree(data, queries, K), REPS)

    # Recall — exact for KD-trees in low-d, may degrade in high-d due to early-stop.
    # MANN
    let idx = MA.build_index(MA.KDTreeIndex, data; axis_selector=:variance)
        rec = recall_at_k(idx, data, queries, K,
            (idx, q) -> [n.id for n in MA.query(idx, data, q, K)])
        @printf "%-25s %-6d %-10.2f %-10.2f %-12.2f %-8.4f\n" "MANN-KDTree (variance)" d (mann.build*1000) (mann.query*1000/NQ) (mann.alloc/NQ/1024) rec
    end
    # NN-KDTree
    let tree = NearestNeighbors.KDTree(data, NearestNeighbors.Euclidean(); leafsize=10)
        rec = recall_at_k(tree, data, queries, K,
            (t, q) -> NearestNeighbors.knn(t, q, K)[1])
        @printf "%-25s %-6d %-10.2f %-10.2f %-12.2f %-8.4f\n" "NN.jl-KDTree" d (nn_kd.build*1000) (nn_kd.query*1000/NQ) (nn_kd.alloc/NQ/1024) rec
    end
    # NN-BallTree
    let tree = NearestNeighbors.BallTree(data, NearestNeighbors.Euclidean(); leafsize=10)
        rec = recall_at_k(tree, data, queries, K,
            (t, q) -> NearestNeighbors.knn(t, q, K)[1])
        @printf "%-25s %-6d %-10.2f %-10.2f %-12.2f %-8.4f\n" "NN.jl-BallTree" d (nn_ball.build*1000) (nn_ball.query*1000/NQ) (nn_ball.alloc/NQ/1024) rec
    end
    println()
end

println("Done.")
