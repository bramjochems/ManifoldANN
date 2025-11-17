#!/usr/bin/env julia
# Julia-only benchmarking runner to mirror the Python benchmark output for ManifoldANN algorithms.
#
# Example (matches benchmark.py default fashion-mnist config):
#   julia --project=benchmarking/julia benchmarking/julia_benchmark.jl fashion-mnist -k 10

using Pkg
Pkg.activate(joinpath(@__DIR__, "julia"))

using Distances
using HDF5
using ManifoldANN
using YAML
using Printf

struct BenchResult
    name::String
    source::String
    type::String
    build_time::Float64
    qps::Float64
    recall::Float64
end

function load_config(config_name::String)
    path = joinpath(@__DIR__, "configs", config_name * ".yaml")
    isfile(path) || error("Config not found: $path")
    return YAML.load_file(path)
end

function load_dataset(dataset::String, data_dir::String)
    path = joinpath(data_dir, dataset * ".hdf5")
    isfile(path) || error("Dataset file not found: $path")
    h5open(path, "r") do f
        train = read(f["train"])
        test = read(f["test"])
        gt = haskey(f, "neighbors") ? read(f["neighbors"]) : nothing
        return train, test, gt
    end
end

# Ensure data is column-major (d × n) for Julia; datasets are stored row-major (n × d)
function to_column_major(data)
    ndims(data) == 2 || error("Expected 2D data, got $(ndims(data))D")
    # If data arrives as (n, d), transpose to (d, n)
    return size(data, 1) > size(data, 2) ? permutedims(data) : data
end

# Normalize ground-truth neighbors to (k × n) 1-based IDs
function normalize_ground_truth(gt)
    isnothing(gt) && return nothing
    ndims(gt) == 2 || error("Expected 2D ground truth, got $(ndims(gt))D")
    # If stored as (n_test, k), transpose to (k, n_test)
    gt_mat = size(gt, 1) > size(gt, 2) ? permutedims(gt) : gt
    # Ensure 1-based indexing
    gt_mat = gt_mat .+ 1  # force 1-based
    return Matrix{Int}(gt_mat)
end

function compute_recall(predictions::Matrix{Int}, ground_truth::Matrix{Int}, k::Int)
    n = size(predictions, 2)
    hits = 0
    for j in 1:n
        preds = predictions[:, j]
        gts = ground_truth[:, j]
        # treat ids as 0-based in ground truth? Python uses 0-based; stored gt is likely 0-based.
        predset = Set(preds)
        gtset = Set(gts)
        hits += length(intersect(predset, gtset))
    end
    return hits / (n * k)
end

function recompute_ground_truth(train::Matrix{Float32}, test::Matrix{Float32}, metric::String, k::Int)
    dist_metric = metric == "angular" ? Distances.CosineDist() : Distances.Euclidean()
    # pairwise with dims=2 computes column-wise distances: size(train,2) × size(test,2)
    D = pairwise(dist_metric, train, test; dims=2)
    n_test = size(test, 2)
    k_eff = min(k, size(train, 2))
    neighbors = Array{Int}(undef, k_eff, n_test)
    @inbounds for j in 1:n_test
        col = view(D, :, j)
        idxs = partialsortperm(col, 1:k_eff)
        neighbors[:, j] = idxs
    end
    return neighbors
end

function measure_ivf(config, X, Q, k; metric)
    nlist = Int(config["algorithms"]["MANN-IVF-HNSW"]["nlist"])
    routing_k = Int(config["algorithms"]["MANN-IVF-HNSW"]["routing_k"])
    M = Int(config["algorithms"]["MANN-IVF-HNSW"]["M"])
    efc = Int(config["algorithms"]["MANN-IVF-HNSW"]["ef_construction"])
    efs = Int(config["algorithms"]["MANN-IVF-HNSW"]["ef_search"])
    neighbor_policy = Symbol(config["algorithms"]["MANN-IVF-HNSW"]["neighbor_policy"])

    distance = metric == "angular" ? ManifoldANN.squared_cosine_distance : ManifoldANN.default_distance
    kmeans_dist = metric == "angular" ? Distances.CosineDist() : Distances.Euclidean()

    build_time = @elapsed begin
        global ivf_index = build_ivf_hnsw_index(
            X;
            nlist=nlist,
            routing_k=routing_k,
            kmeans_distance=kmeans_dist,
            kmeans_init=:kmeans_plus_plus,
            kmeans_max_iters=5,
            kmeans_tol=1e-4,
            hnsw_M=M,
            hnsw_ef_construction=efc,
            hnsw_ef_search=efs,
            hnsw_neighbor_policy=neighbor_policy,
            distance=distance,
        )
    end

    query_time = @elapsed begin
        global preds = query(ivf_index, X, Q, k; ef_search=efs)
    end

    # Convert Vector{Vector{Neighbor}} to Matrix ids (1-based)
    pred_ids = [n.id for batch in preds for n in batch]
    pred_mat = reshape(pred_ids, k, :)
    qps = size(Q, 2) / query_time

    return build_time, query_time, qps, pred_mat
end

function main()
    config_name = length(ARGS) >= 1 ? ARGS[1] : "fashion-mnist"
    k = 10
    for arg in ARGS[2:end]
        if startswith(arg, "-k")
            k = parse(Int, split(arg, "=")[end])
        end
    end

    config = load_config(config_name)
    dataset = config["dataset"]
    metric = config["metric"]
    n_train = Int(config["n_train"])
    n_test = Int(config["n_test"])

    data_dir = joinpath(@__DIR__, "data")
    train, test, ground_truth = load_dataset(dataset, data_dir)
    @printf("Dataset shapes: train %s, test %s, gt %s\n", size(train), size(test), isa(ground_truth, Nothing) ? "none" : string(size(ground_truth)))

    X_full = to_column_major(train)
    Q_full = to_column_major(test)
    X = Matrix{Float32}(X_full[:, 1:n_train])
    Q = Matrix{Float32}(Q_full[:, 1:n_test])
    gt = normalize_ground_truth(ground_truth)

    # If we subsample train or test, recompute ground truth on the subset
    if gt !== nothing
        if size(train, 2) > n_train || size(test, 2) > n_test
            @printf("Recomputing ground truth for subset: n_train=%d, n_test=%d\n", n_train, n_test)
            gt = recompute_ground_truth(X, Q, metric, k)
            # recompute_ground_truth returns 1-based ids already; ensure shape (k × n_test)
            gt = gt[1:min(k, size(gt, 1)), 1:n_test]
        else
            gt = gt[1:min(k, size(gt, 1)), 1:n_test]
        end
    end
    if gt !== nothing
        @printf("GT stats: size %s, min %d, max %d\n", size(gt), minimum(gt), maximum(gt))
    end

    results = BenchResult[]

    # IVF-HNSW
    build_time, query_time, qps, pred_mat = measure_ivf(config, X, Q, k; metric=metric)
    recall = gt === nothing ? NaN : compute_recall(pred_mat, gt, k)
    push!(results, BenchResult("MANN-IVF-HNSW", "ManifoldANN", "quantization", build_time, qps, recall))

    println("algorithm                     source         type             build_time    qps        recall")
    for r in results
        @printf("%-30s %-13s %-15s %8.2f %10.0f %11.4f\n", r.name, r.source, r.type, r.build_time, r.qps, r.recall)
    end
end

main()
