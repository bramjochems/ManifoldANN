#!/usr/bin/env julia
# Pareto sweep: MANN-NNDescent vs NND.jl at varying ef_search / max_candidates.
# Reports (recall, qps) pairs for both libraries so the apples-to-apples
# comparison can be made at matched recall.
#
# Run: julia --project=benchmarking/julia -t N scripts/nndescent_jl_pareto.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
using NearestNeighborDescent
const MA = ManifoldANN
const NND = NearestNeighborDescent

const N      = parse(Int, get(ENV, "ND_N",   "20000"))
const D      = parse(Int, get(ENV, "ND_D",   "32"))
const K      = parse(Int, get(ENV, "ND_K",   "20"))
const NQ     = parse(Int, get(ENV, "ND_NQ",  "5000"))
const SEED   = 0xC0FFEE
const REPS   = 3
const MAX_IT = 8

const SWEEP = [20, 30, 40, 60, 100, 150, 200, 300]

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
println("config: n=$N d=$D k=$K n_queries=$NQ reps=$REPS")
println("sweep: ", SWEEP)

Random.seed!(SEED)
data_mat     = randn(Float32, D, N)
queries_mat  = randn(Float32, D, NQ)
data_cols    = [collect(data_mat[:, i])    for i in 1:N]
queries_cols = [collect(queries_mat[:, i]) for i in 1:NQ]

# Ground truth (brute force)
@printf "\nComputing ground truth (k=%d)...\n" K
gt_t = @elapsed begin
    gt_ids = Matrix{Int}(undef, K, NQ)
    Threads.@threads for qi in 1:NQ
        q = view(queries_mat, :, qi)
        dists = Vector{Float32}(undef, N)
        @inbounds for j in 1:N
            d = 0.0f0
            @simd for kk in 1:D
                δ = data_mat[kk, j] - q[kk]
                d += δ * δ
            end
            dists[j] = d
        end
        perm = partialsortperm(dists, 1:K)
        gt_ids[:, qi] = perm
    end
end
@printf "  ground truth: %.2fs\n" gt_t

function recall_vec(ids_vec, gt, k)
    nq = length(ids_vec); total = 0
    for qi in 1:nq
        s = Set(view(gt, 1:k, qi))
        for id in ids_vec[qi]; id in s && (total += 1); end
    end
    return total / (nq*k)
end
function recall_mat(ids_mat, gt, k)
    nq = size(ids_mat, 2); total = 0
    for qi in 1:nq
        s = Set(view(gt, 1:k, qi))
        @inbounds for r in 1:k; ids_mat[r, qi] in s && (total += 1); end
    end
    return total / (nq*k)
end

# Build both
@printf "\nBuilding indexes...\n"
build_t_mann = @elapsed idx_mann = MA.build_index(MA.NNDescentIndex, data_mat;
    k=K, max_iterations=MAX_IT, threaded=true, rng=MersenneTwister(SEED))
@printf "  MANN build:   %.3fs\n" build_t_mann
build_t_nnd = @elapsed g_nnd = NND.nndescent(data_cols, K, NND.Euclidean();
    max_iters=MAX_IT)
@printf "  NND.jl build: %.3fs\n" build_t_nnd

# Warm
MA.query(idx_mann, data_mat, view(queries_mat, :, 1:8), K; ef_search=60)
NND.search(g_nnd, queries_cols[1:8], K; max_candidates=60)
GC.gc()

#==============================================================================
# Sweep MANN ef_search
==============================================================================#
println("\n[MANN sweep on ef_search]")
@printf "%4s  %8s  %8s\n" "ef" "qps" "recall@$K"
mann_pareto = Tuple{Int,Float64,Float64}[]
for ef in SWEEP
    times = Float64[]
    for _ in 1:REPS
        GC.gc()
        t0 = time_ns()
        MA.query(idx_mann, data_mat, queries_mat, K; ef_search=ef)
        push!(times, (time_ns() - t0) / 1e9)
    end
    res = MA.query(idx_mann, data_mat, queries_mat, K; ef_search=ef)
    ids = [[n.id for n in r] for r in res]
    r_at_k = recall_vec(ids, gt_ids, K)
    qps = NQ / minimum(times)
    push!(mann_pareto, (ef, qps, r_at_k))
    @printf "%4d  %8.0f  %8.4f\n" ef qps r_at_k
end

#==============================================================================
# Sweep NND.jl max_candidates
==============================================================================#
println("\n[NND.jl sweep on max_candidates]")
@printf "%4s  %8s  %8s\n" "mc" "qps" "recall@$K"
nnd_pareto = Tuple{Int,Float64,Float64}[]
for mc in SWEEP
    times = Float64[]
    for _ in 1:REPS
        GC.gc()
        t0 = time_ns()
        NND.search(g_nnd, queries_cols, K; max_candidates=mc)
        push!(times, (time_ns() - t0) / 1e9)
    end
    ids_mat, _ = NND.search(g_nnd, queries_cols, K; max_candidates=mc)
    r_at_k = recall_mat(ids_mat, gt_ids, K)
    qps = NQ / minimum(times)
    push!(nnd_pareto, (mc, qps, r_at_k))
    @printf "%4d  %8.0f  %8.4f\n" mc qps r_at_k
end

#==============================================================================
# Matched-recall comparison: for each recall level, compare qps
==============================================================================#
function qps_at_recall(pareto, target_recall)
    # linearly interpolate qps at target recall, or return nothing if outside
    sorted = sort(pareto; by = p -> p[3])
    if target_recall < sorted[1][3] || target_recall > sorted[end][3]
        return nothing
    end
    for i in 1:length(sorted)-1
        r0, r1 = sorted[i][3], sorted[i+1][3]
        if r0 <= target_recall <= r1
            q0, q1 = sorted[i][2], sorted[i+1][2]
            if r0 == r1
                return q0
            end
            t = (target_recall - r0) / (r1 - r0)
            return q0 + t * (q1 - q0)
        end
    end
    return nothing
end

println("\n[matched-recall qps]")
@printf "%9s  %12s  %12s  %s\n" "recall" "MANN qps" "NND.jl qps" "ratio (MANN/NND)"
for r in (0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99)
    qm = qps_at_recall(mann_pareto, r)
    qn = qps_at_recall(nnd_pareto, r)
    if qm !== nothing && qn !== nothing
        @printf "%9.2f  %12.0f  %12.0f  %.2fx\n" r qm qn (qm/qn)
    elseif qm !== nothing
        @printf "%9.2f  %12.0f  %12s  --\n" r qm "(out of range)"
    elseif qn !== nothing
        @printf "%9.2f  %12s  %12.0f  --\n" r "(out of range)" qn
    end
end

println("\nDone.")
