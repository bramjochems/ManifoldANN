#!/usr/bin/env julia
# Head-to-head Pareto sweep: MANN-HNSW (DiversifiedNeighborPolicy) vs HNSW.jl
# on the SIFT-128-Euclidean dataset (1M train, 10k test, 128-d).
#
# Each library is built across a small grid of (M, ef_construction) configs.
# For every build we sweep ef_search at query time (single rebuild per
# build config; ef is mutated between query rounds to avoid rebuilds).
#
# Run:
#   julia --project=benchmarking/julia -t 16 scripts/thesis/ann/hnsw_jl_pareto.jl
#
# Output:
#   docs/thesis/results/ann_pareto/hnsw_jl_sift.csv
#
# Discipline notes:
#   * Both libraries get the same -t threads. BLAS pinned to 1 to avoid
#     hidden parallelism on per-distance kernels.
#   * Warmup runs the build+query path on tiny dummy data before timing,
#     so the first timed config does not pay JIT cost.
#   * Euclidean distance on raw vectors; GT loaded from HDF5 `neighbors` key
#     (precomputed by ann-benchmarks, 0-indexed, converted to 1-indexed).
#   * Configs capped at M=16, ef_c=200 to keep HNSW.jl build times tractable
#     (HNSW.jl is single-threaded; heavier configs take hours at 1M points).
#   * Reps = 1 per config: this is a Pareto sweep, noise per point matters
#     less than coverage of the frontier.

using LinearAlgebra
using Random
using Printf
using Dates
using HDF5
using ManifoldANN
using HNSW
const MA = ManifoldANN

BLAS.set_num_threads(1)

# ----- Paths and constants -------------------------------------------------

const REPO       = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_PATH  = joinpath(REPO, "benchmarking", "data", "sift-128-euclidean.hdf5")
const OUT_DIR    = joinpath(REPO, "..", "..", "docs", "thesis", "results", "ann_pareto")
const OUT_CSV    = joinpath(OUT_DIR, "hnsw_jl_sift.csv")

const K_QUERY = 10
const SEED    = 0xC0FFEE

# Build-config grid: capped at M=16, ef_c=200 to keep HNSW.jl (single-threaded)
# tractable at 1M points.
const BUILD_CONFIGS = [
    (M= 8, ef_c=100),
    (M=12, ef_c=100),
    (M=16, ef_c=100),
    (M=16, ef_c=200),
]

# ef_search values swept per build (no rebuild needed: HNSW.jl mutates
# `idx.ef`; MANN HNSW takes `ef_search` as a query kwarg).
const EF_SEARCH_SWEEP = [25, 50, 100, 200]

# ----- Dataset loading -----------------------------------------------------

println("Loading dataset: $DATA_PATH")
const train, test, gt_raw = h5open(DATA_PATH, "r") do f
    # HDF5 stores as (n, d) row-major from Python; Julia reads as (d, n) column-major.
    # neighbors is (100, n_test) — already (k, n_test) in Julia.
    read(f["train"]), read(f["test"]), read(f["neighbors"])
end
const D = size(train, 1)
const N_TRAIN = size(train, 2)
const N_TEST  = size(test, 2)
@printf("  train: %d × %d, test: %d × %d, k=%d\n",
        D, N_TRAIN, D, N_TEST, K_QUERY)

# GT from file: 0-indexed, take top K_QUERY rows, convert to 1-indexed.
# SIFT is Euclidean so the precomputed GT matches our distance directly.
const gt_ids = Matrix{Int}(gt_raw[1:K_QUERY, :]) .+ 1
@printf("  ground truth: loaded from file (%d × %d)\n", K_QUERY, N_TEST)

# ----- Recall helper -------------------------------------------------------

function recall_at_k(predicted::Vector{<:Vector}, gt::Matrix{Int}, k::Int)
    nq = length(predicted)
    total = 0
    for qi in 1:nq
        s = Set(view(gt, 1:k, qi))
        for id in predicted[qi]
            id in s && (total += 1)
        end
    end
    return total / (nq * k)
end

function recall_at_k_mat(ids_mat::Matrix{Int}, gt::Matrix{Int}, k::Int)
    nq = size(ids_mat, 2)
    total = 0
    for qi in 1:nq
        s = Set(view(gt, 1:k, qi))
        @inbounds for r in 1:k
            ids_mat[r, qi] in s && (total += 1)
        end
    end
    return total / (nq * k)
end

# ----- Warmup --------------------------------------------------------------

println("Warming up Julia JIT for both libraries…")
let
    warm_d = D
    warm_n = 256
    warm_data = randn(Float32, warm_d, warm_n)
    warm_q    = warm_data[:, 1:8]

    # MANN warmup (default Euclidean)
    idx_mann = MA.build_index(MA.HNSWIndex, warm_data;
        M=8, ef_construction=40, ef_search=16,
        neighbor_policy=:diversified)
    MA.query(idx_mann, warm_data, warm_q, 5; ef_search=16)

    # HNSW.jl warmup (default Euclidean)
    warm_vov = [collect(warm_data[:, i]) for i in 1:warm_n]
    warm_q_vov = [collect(warm_q[:, i]) for i in 1:size(warm_q, 2)]
    idx_jl = HierarchicalNSW(warm_vov; metric=Euclidean(),
        M=8, efConstruction=40, ef=16)
    add_to_graph!(idx_jl)
    knn_search(idx_jl, warm_q_vov, 5)
end
println("  warmup done")

# ----- HNSW.jl input prep --------------------------------------------------
# HNSW.jl needs a Vector of Vectors for both data and queries.

println("Preparing HNSW.jl inputs (vector-of-vectors)…")
prep_t_jl = @elapsed begin
    train_vov = [collect(train[:, i]) for i in 1:N_TRAIN]
    test_vov  = [collect(test[:, i])  for i in 1:N_TEST]
end
@printf("  vov prep: %.2fs\n", prep_t_jl)

# ----- Sweep ---------------------------------------------------------------

mkpath(OUT_DIR)
io = open(OUT_CSV, "w")
println(io, "algorithm,M,ef_construction,ef_search,build_time,recall@$(K_QUERY),qps")

println("\n===== MANN-HNSW (diversified) sweep =====")
for cfg in BUILD_CONFIGS
    @printf("\n[MANN] M=%d ef_c=%d : building… (started %s)\n", cfg.M, cfg.ef_c, Dates.format(Dates.now(), "HH:MM:SS"))
    Random.seed!(SEED)
    build_t = @elapsed idx = MA.build_index(MA.HNSWIndex, train;
        M=cfg.M, ef_construction=cfg.ef_c,
        ef_search=EF_SEARCH_SWEEP[1],  # placeholder; queries override
        neighbor_policy=:diversified)
    @printf("       build: %.2fs\n", build_t)

    for ef in EF_SEARCH_SWEEP
        # Time only the batch query.
        local results
        q_t = @elapsed results = MA.query(idx, train, test, K_QUERY; ef_search=ef)
        # MANN returns Vector{Vector{Neighbor{S}}}; extract ids.
        pred_ids = [Int.(MA.neighbor_ids(r)) for r in results]
        rec = recall_at_k(pred_ids, gt_ids, K_QUERY)
        qps = N_TEST / q_t
        @printf("       ef=%3d : recall=%.4f qps=%.0f\n", ef, rec, qps)
        @printf(io, "MANN-HNSW-diversified,%d,%d,%d,%.4f,%.4f,%.2f\n",
                cfg.M, cfg.ef_c, ef, build_t, rec, qps)
        flush(io)
    end
end

println("\n===== HNSW.jl sweep =====")
for cfg in BUILD_CONFIGS
    @printf("\n[HNSW.jl] M=%d ef_c=%d : building… (started %s)\n", cfg.M, cfg.ef_c, Dates.format(Dates.now(), "HH:MM:SS"))
    Random.seed!(SEED)
    # ef placeholder; we mutate idx.ef per query round.
    build_t = @elapsed begin
        idx = HierarchicalNSW(train_vov; metric=Euclidean(),
            M=cfg.M, efConstruction=cfg.ef_c,
            ef=EF_SEARCH_SWEEP[1])
        add_to_graph!(idx)
    end
    @printf("          build: %.2fs\n", build_t)

    for ef in EF_SEARCH_SWEEP
        idx.ef = ef
        local ids_mat
        q_t = @elapsed begin
            ids_mat, _ = knn_search(idx, test_vov, K_QUERY)
        end
        # HNSW.jl returns Vector{Vector{UInt32}}, 1-indexed.
        pred_ids = [Int.(v) for v in ids_mat]
        rec = recall_at_k(pred_ids, gt_ids, K_QUERY)
        qps = N_TEST / q_t
        @printf("          ef=%3d : recall=%.4f qps=%.0f\n", ef, rec, qps)
        @printf(io, "HNSW.jl,%d,%d,%d,%.4f,%.4f,%.2f\n",
                cfg.M, cfg.ef_c, ef, build_t, rec, qps)
        flush(io)
    end
end

close(io)
@printf("\nWrote %s\n", OUT_CSV)
