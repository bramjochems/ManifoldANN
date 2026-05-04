#!/usr/bin/env julia
# Compare NN-Descent build with max_iterations=10 (current default) vs the
# PyNNDescent-style adaptive max(5, round(log2(n))) at n ∈ {10k, 100k}.
# Reports build time, recall, and qps on fashion-mnist.
#
# Run: julia --project=benchmarking/julia -t auto scripts/perf/nndescent_max_iter_sweep.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "benchmarking", "julia"))

using Random, Printf, LinearAlgebra
using HDF5
using ManifoldANN
const MA = ManifoldANN

const DATA_PATH = joinpath(@__DIR__, "..", "..", "data", "fashion-mnist-784-euclidean.hdf5")
const K         = 10
const NQ        = 1000
const SEED      = 0xC0FFEE
const REPS      = 3
const EF_SWEEP  = [10, 20, 40, 80, 160]

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())

function load_fashion()
    h5open(DATA_PATH, "r") do f
        train = read(f["train"])  # (n, d) row-major
        test  = read(f["test"])
        gt    = haskey(f, "neighbors") ? read(f["neighbors"]) : nothing
        # Transpose to (d, n)
        train_cm = size(train, 1) > size(train, 2) ? permutedims(train) : train
        test_cm  = size(test,  1) > size(test,  2) ? permutedims(test)  : test
        if gt !== nothing
            gt_cm = size(gt, 1) > size(gt, 2) ? permutedims(gt) : gt
            gt_cm = Matrix{Int}(gt_cm) .+ 1  # 0-based -> 1-based
            return Float32.(train_cm), Float32.(test_cm), gt_cm
        end
        return Float32.(train_cm), Float32.(test_cm), nothing
    end
end

function brute_gt(data::Matrix{Float32}, queries::Matrix{Float32}, k::Int)
    d, n = size(data); _, nq = size(queries)
    gt_ids = Matrix{Int}(undef, k, nq)
    Threads.@threads for qi in 1:nq
        q = view(queries, :, qi)
        dists = Vector{Float32}(undef, n)
        @inbounds for j in 1:n
            s = 0.0f0
            @simd for kk in 1:d
                δ = data[kk, j] - q[kk]
                s += δ * δ
            end
            dists[j] = s
        end
        gt_ids[:, qi] = partialsortperm(dists, 1:k)
    end
    return gt_ids
end

function recall_vec(ids_vec, gt, k)
    nq = length(ids_vec); total = 0
    for qi in 1:nq
        s = Set(view(gt, 1:k, qi))
        for id in ids_vec[qi]
            id in s && (total += 1)
        end
    end
    return total / (nq * k)
end

println("loading fashion-mnist...")
train_full, test_full, _ = load_fashion()
println("  train: ", size(train_full), "  test: ", size(test_full))

# Use first NQ test queries
test = test_full[:, 1:min(NQ, size(test_full, 2))]
nq_actual = size(test, 2)

for n_target in (10_000, 100_000)
    n = min(n_target, size(train_full, 2))
    data = train_full[:, 1:n]
    adaptive_iter = max(5, round(Int, log2(n)))
    @printf "\n========== n=%d  k=%d  nq=%d ==========\n" n K nq_actual
    @printf "adaptive max_iter = max(5, round(log2(%d))) = %d\n" n adaptive_iter

    # Compute ground truth at this n (subset of train)
    @printf "computing brute-force ground truth...\n"
    gt_t = @elapsed gt_ids = brute_gt(data, test, K)
    @printf "  gt: %.2fs\n" gt_t

    for max_it in (10, adaptive_iter)
        @printf "\n--- max_iterations = %d ---\n" max_it
        build_t = @elapsed idx = MA.build_index(MA.NNDescentIndex, data;
            k=32, max_iterations=max_it, threaded=true,
            rng=MersenneTwister(SEED))
        @printf "build: %.3fs\n" build_t

        # Warm
        MA.query(idx, data, view(test, :, 1:min(8, nq_actual)), K; ef_search=40)
        GC.gc()

        @printf "%6s  %8s  %8s\n" "ef" "qps" "recall@$K"
        for ef in EF_SWEEP
            times = Float64[]
            for _ in 1:REPS
                GC.gc()
                t0 = time_ns()
                MA.query(idx, data, test, K; ef_search=ef)
                push!(times, (time_ns() - t0) / 1e9)
            end
            res = MA.query(idx, data, test, K; ef_search=ef)
            ids = [[nb.id for nb in r] for r in res]
            r_at_k = recall_vec(ids, gt_ids, K)
            qps = nq_actual / minimum(times)
            @printf "%6d  %8.0f  %8.4f\n" ef qps r_at_k
        end
    end
end

println("\nDone.")
