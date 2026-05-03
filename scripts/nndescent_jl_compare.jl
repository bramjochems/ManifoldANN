#!/usr/bin/env julia
# Fair-compare bench: NND.jl vs MANN-NNDescent on the same config and quiet
# machine. Run from a checkout where benchmarking/julia is the active project
# (it brings in NearestNeighborDescent.jl).
#
# Run: JULIA_NUM_THREADS=N julia --project=benchmarking/julia scripts/nndescent_jl_compare.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
using NearestNeighborDescent
const MA = ManifoldANN
const NND = NearestNeighborDescent

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
data_mat = randn(Float32, D, N)
queries_mat = randn(Float32, D, NQ)

# NND.jl wants Vector{<:AbstractVector}
data_cols    = [collect(data_mat[:, i])    for i in 1:N]
queries_cols = [collect(queries_mat[:, i]) for i in 1:NQ]

# Ground truth: brute-force kNN for recall calculation
@printf "\nComputing ground truth (brute force, k=%d)...\n" K
gt_t = @elapsed begin
    gt_ids = Matrix{Int}(undef, K, NQ)
    Threads.@threads for qi in 1:NQ
        q = view(queries_mat, :, qi)
        dists = Vector{Float32}(undef, N)
        @inbounds for j in 1:N
            d = 0.0f0
            @simd for k in 1:D
                δ = data_mat[k, j] - q[k]
                d += δ * δ
            end
            dists[j] = d
        end
        perm = partialsortperm(dists, 1:K)
        gt_ids[:, qi] = perm
    end
end
@printf "  ground truth: %.2fs\n" gt_t

function recall_at_k(result_ids::AbstractVector{<:AbstractVector{Int}}, gt::AbstractMatrix{Int}, k::Int)
    nq = length(result_ids)
    total = 0
    for qi in 1:nq
        gt_set = Set(view(gt, 1:k, qi))
        for id in result_ids[qi]
            id in gt_set && (total += 1)
        end
    end
    return total / (nq * k)
end

function recall_at_k_matrix(result_ids::AbstractMatrix{Int}, gt::AbstractMatrix{Int}, k::Int)
    nq = size(result_ids, 2)
    total = 0
    for qi in 1:nq
        gt_set = Set(view(gt, 1:k, qi))
        @inbounds for r in 1:k
            result_ids[r, qi] in gt_set && (total += 1)
        end
    end
    return total / (nq * k)
end

# Warm both libraries
let small_data = randn(Float32, D, 1000), small_data_cols = [collect(small_data[:,i]) for i in 1:1000]
    idx = MA.build_index(MA.NNDescentIndex, small_data; k=10, max_iterations=4, rng=MersenneTwister(1))
    MA.query(idx, small_data, @view(small_data[:, 1]), 5; ef_search=20)
    MA.query(idx, small_data, small_data, 5; ef_search=20)

    g = NND.nndescent(small_data_cols, 10, NND.Euclidean(); max_iters=4)
    NND.search(g, small_data_cols[1:8], 5; max_candidates=20)
end
GC.gc()

# ----- MANN -----
@printf "\n[MANN-NNDescent]\n"
build_t_mann = @elapsed idx_mann = MA.build_index(MA.NNDescentIndex, data_mat;
    k=K, max_iterations=MAX_IT, threaded=true, rng=MersenneTwister(SEED))
@printf "  build = %.3fs\n" build_t_mann

MA.query(idx_mann, data_mat, @view(queries_mat[:, 1:8]), K; ef_search=EFS)
GC.gc()

times_m = Float64[]
for _ in 1:REPS
    GC.gc()
    t0 = time_ns()
    MA.query(idx_mann, data_mat, queries_mat, K; ef_search=EFS)
    push!(times_m, (time_ns() - t0) / 1e9)
end
mann_results = MA.query(idx_mann, data_mat, queries_mat, K; ef_search=EFS)
mann_ids = [[n.id for n in res] for res in mann_results]
mann_recall = recall_at_k(mann_ids, gt_ids, K)
let best = minimum(times_m), median = sort(times_m)[ceil(Int, length(times_m)/2)]
    @printf "  query best:   %.3fs   qps %6.0f\n" best (NQ/best)
    @printf "  query median: %.3fs   qps %6.0f\n" median (NQ/median)
    @printf "  recall@%d:     %.4f\n" K mann_recall
end

# ----- NND.jl -----
@printf "\n[NND.jl]\n"
build_t_nnd = @elapsed g = NND.nndescent(data_cols, K, NND.Euclidean();
    max_iters=MAX_IT)
@printf "  build = %.3fs\n" build_t_nnd

NND.search(g, queries_cols[1:8], K; max_candidates=EFS)
GC.gc()

times_n = Float64[]
for _ in 1:REPS
    GC.gc()
    t0 = time_ns()
    NND.search(g, queries_cols, K; max_candidates=EFS)
    push!(times_n, (time_ns() - t0) / 1e9)
end
nnd_ids_mat, _ = NND.search(g, queries_cols, K; max_candidates=EFS)
nnd_recall = recall_at_k_matrix(nnd_ids_mat, gt_ids, K)
let best = minimum(times_n), median = sort(times_n)[ceil(Int, length(times_n)/2)]
    @printf "  query best:   %.3fs   qps %6.0f\n" best (NQ/best)
    @printf "  query median: %.3fs   qps %6.0f\n" median (NQ/median)
    @printf "  recall@%d:     %.4f\n" K nnd_recall
end

# ----- Ratio -----
@printf "\n[Ratio MANN / NND.jl]\n"
@printf "  build: MANN %.3fs / NND.jl %.3fs = %.2f×\n" build_t_mann build_t_nnd (build_t_mann/build_t_nnd)
let m = NQ/minimum(times_m), n = NQ/minimum(times_n)
    @printf "  query: MANN %.0f qps / NND.jl %.0f qps = %.2f×\n" m n (m/n)
end

println("\nDone.")
