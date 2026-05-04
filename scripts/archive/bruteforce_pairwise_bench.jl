#!/usr/bin/env julia
# Bench: per-pair loop vs Distances.pairwise! (BLAS3) for BruteForceIndex.
#
# Compares the *integrated* `query` entry point (single-query and batch).
# The integrated bench is what matters; microbenches over the inner pairwise
# call have historically misled (cf. NN-Descent batch-query data_cache, where
# a 3.76× microbench win became a 37-50% regression on the integrated path).
#
# Run:
#   JULIA_NUM_THREADS=N julia --project=. scripts/bruteforce_pairwise_bench.jl
#
# Config (env):
#   BF_N    dataset size (default 10000)
#   BF_D    dimension (default 128)
#   BF_NQ   batch query sizes, comma-separated (default "1,100,1000")
#   BF_K    k for kNN (default 10)
#   BF_REPS reps (default 5)

using Random, Printf, LinearAlgebra
using Distances
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "BF_N", "10000"))
const Dd   = parse(Int, get(ENV, "BF_D", "128"))
const NQs  = parse.(Int, split(get(ENV, "BF_NQ", "1,100,1000"), ","))
const K    = parse(Int, get(ENV, "BF_K", "10"))
const REPS = parse(Int, get(ENV, "BF_REPS", "5"))
const SEED = 0xBEEF

println("Threads.nthreads() = ", Threads.nthreads())
@printf "config: n=%d d=%d k=%d reps=%d\n" N Dd K REPS
println("BLAS.get_num_threads() = ", BLAS.get_num_threads())

Random.seed!(SEED)
data    = randn(Float32, Dd, N)
queries_all = randn(Float32, Dd, maximum(NQs))

# ---------------------------------------------------------------
# Reference: current per-pair-loop query implementation (a copy of what's in
# bruteforce.jl now). We re-implement here so the bench is self-contained
# and the comparison is apples-to-apples regardless of any code changes.
# ---------------------------------------------------------------
function query_loop(index, data::AbstractMatrix{T}, q::AbstractVector{T}, k::Integer) where {T}
    S = float(T)
    n_points = size(data, 2)
    k = min(k, n_points)
    k <= 0 && return MA.Neighbor{S}[]
    dists = Vector{S}(undef, n_points)
    @inbounds Threads.@threads for j in 1:n_points
        dists[j] = index.distance(@view(data[:, j]), q)
    end
    ids = partialsortperm(dists, 1:k)
    results = Vector{MA.Neighbor{S}}(undef, length(ids))
    @inbounds for (pos, id) in enumerate(ids)
        results[pos] = MA.Neighbor{S}(id, dists[id])
    end
    return results
end

# ---------------------------------------------------------------
# Candidate: BLAS3-friendly variant via Distances.pairwise!.
# For Euclidean / SqEuclidean / CosineDist, Distances.pairwise! is GEMM-backed.
# For other metrics it loops; we still fall back to that to keep code uniform.
# ---------------------------------------------------------------
function query_pairwise(index, data::AbstractMatrix{T}, q::AbstractVector{T}, k::Integer) where {T}
    S = float(T)
    n_points = size(data, 2)
    k = min(k, n_points)
    k <= 0 && return MA.Neighbor{S}[]
    metric = index.distance
    # Single-column query matrix, dims=2 means columns are observations.
    qmat = reshape(q, :, 1)
    D = Matrix{S}(undef, n_points, 1)
    Distances.pairwise!(metric, D, data, qmat; dims=2)
    dists = vec(D)
    ids = partialsortperm(dists, 1:k)
    results = Vector{MA.Neighbor{S}}(undef, length(ids))
    @inbounds for (pos, id) in enumerate(ids)
        results[pos] = MA.Neighbor{S}(id, dists[id])
    end
    return results
end

# Batch: per-query loop using the per-pair scanner (same as generic batch fallback today).
function batch_loop(index, data::AbstractMatrix{T}, queries::AbstractMatrix{T}, k::Integer) where {T}
    S = float(T)
    nq = size(queries, 2)
    results = Vector{Vector{MA.Neighbor{S}}}(undef, nq)
    if Threads.nthreads() == 1 || nq < 64
        @inbounds for i in 1:nq
            results[i] = query_loop(index, data, view(queries, :, i), k)
        end
    else
        Threads.@threads for i in 1:nq
            results[i] = query_loop(index, data, view(queries, :, i), k)
        end
    end
    return results
end

# Batch: single GEMM call for the whole batch, then partial-sort per query.
function batch_pairwise(index, data::AbstractMatrix{T}, queries::AbstractMatrix{T}, k::Integer) where {T}
    S = float(T)
    n_points = size(data, 2)
    nq = size(queries, 2)
    metric = index.distance
    D = Matrix{S}(undef, n_points, nq)
    Distances.pairwise!(metric, D, data, queries; dims=2)
    results = Vector{Vector{MA.Neighbor{S}}}(undef, nq)
    if Threads.nthreads() == 1 || nq < 64
        @inbounds for i in 1:nq
            col = view(D, :, i)
            ids = partialsortperm(col, 1:min(k, n_points))
            v = Vector{MA.Neighbor{S}}(undef, length(ids))
            for (pos, id) in enumerate(ids)
                v[pos] = MA.Neighbor{S}(id, col[id])
            end
            results[i] = v
        end
    else
        Threads.@threads for i in 1:nq
            col = view(D, :, i)
            ids = partialsortperm(col, 1:min(k, n_points))
            v = Vector{MA.Neighbor{S}}(undef, length(ids))
            for (pos, id) in enumerate(ids)
                v[pos] = MA.Neighbor{S}(id, col[id])
            end
            results[i] = v
        end
    end
    return results
end

# ---------------------------------------------------------------
# Timing helper: run f() REPS times, return best+median seconds.
# ---------------------------------------------------------------
function timed(f; reps=REPS)
    times = Float64[]
    for _ in 1:reps
        GC.gc()
        push!(times, @elapsed f())
    end
    sort!(times)
    return (best=times[1], median=times[(length(times)+1) ÷ 2])
end

function bench_metric(metric, label)
    @printf "\n=== Metric: %s ===\n" label
    index = MA.build_index(MA.BruteForceIndex, data; distance=metric)

    # Correctness sanity: neighbor IDs match between paths
    let q = view(queries_all, :, 1)
        a = sort(MA.neighbor_ids(query_loop(index, data, q, K)))
        b = sort(MA.neighbor_ids(query_pairwise(index, data, q, K)))
        @assert a == b "single-query NN IDs disagree"
    end

    # Warm up
    query_loop(index, data, view(queries_all, :, 1), K)
    query_pairwise(index, data, view(queries_all, :, 1), K)

    # Single-query
    @printf "%-22s  %-14s  %-14s  %s\n" "case" "loop best (s)" "pair best (s)" "speedup"
    let q = view(queries_all, :, 1)
        t1 = timed(() -> query_loop(index, data, q, K))
        t2 = timed(() -> query_pairwise(index, data, q, K))
        @printf "%-22s  %-14.6f  %-14.6f  %.2fx\n" "single (1 query)" t1.best t2.best (t1.best / t2.best)
    end

    # Batch
    for nq in NQs
        nq == 1 && continue
        Q = view(queries_all, :, 1:nq)
        # warm batch path
        batch_loop(index, data, Q, K); batch_pairwise(index, data, Q, K)
        t1 = timed(() -> batch_loop(index, data, Q, K))
        t2 = timed(() -> batch_pairwise(index, data, Q, K))
        @printf "%-22s  %-14.6f  %-14.6f  %.2fx   (loop qps=%.0f, pair qps=%.0f)\n" "batch nq=$(nq)" t1.best t2.best (t1.best / t2.best) (nq/t1.best) (nq/t2.best)
    end
end

bench_metric(Distances.Euclidean(), "Euclidean")
bench_metric(Distances.CosineDist(), "CosineDist")
bench_metric(Distances.SqEuclidean(), "SqEuclidean")

println("\nDone.")
