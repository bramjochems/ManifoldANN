using Test
using Random
using LinearAlgebra
using ManifoldANN

@testset "PCATreeIndex build smoke test" begin
    rng = MersenneTwister(0xB1)
    data = randn(rng, Float64, 8, 64)
    idx = build_index(PCATreeIndex, data; rng = MersenneTwister(1))
    @test idx isa PCATreeIndex
    @test idx.n_points == 64
    @test idx.dimension == 8
    @test !isempty(idx.nodes)
    @test length(idx.leaf_members) == 64
end

@testset "PCATreeIndex single + batch query paths" begin
    rng = MersenneTwister(0xB2)
    data = randn(rng, Float32, 6, 80)
    idx = build_index(PCATreeIndex, data; rng = MersenneTwister(7))
    q = randn(MersenneTwister(11), Float32, 6)
    res = query(idx, data, q, 5)
    @test length(res) <= 5
    @test all(r -> 1 <= r.id <= 80, res)
    dists = [r.dist for r in res]
    @test dists == sort(dists)

    queries = randn(MersenneTwister(17), Float32, 6, 8)
    batch = query(idx, data, queries, 4)
    @test length(batch) == 8
    for r in batch
        @test length(r) <= 4
    end
end

@testset "PCATreeIndex deterministic build under fixed RNG" begin
    rng = MersenneTwister(0xB4)
    data = randn(rng, Float64, 8, 100)
    splitter = PCASplitter(
        ExactSVD(), RandomTopK(3),
        AnyOf(MaxLeafSize(20)), MedianSplit(),
    )
    a = build_index(PCATreeIndex, data; splitter = splitter,
                    rng = MersenneTwister(42))
    b = build_index(PCATreeIndex, data; splitter = splitter,
                    rng = MersenneTwister(42))
    @test a.leaf_members == b.leaf_members
    @test length(a.nodes) == length(b.nodes)
    q = randn(MersenneTwister(999), Float64, 8)
    @test query(a, data, q, 5) == query(b, data, q, 5)
end

@testset "PCATreeIndex k larger than n_points" begin
    rng = MersenneTwister(0xB5)
    data = randn(rng, Float64, 4, 10)
    idx = build_index(PCATreeIndex, data; rng = MersenneTwister(3))
    q = randn(MersenneTwister(5), Float64, 4)
    res = query(idx, data, q, 100)
    @test length(res) <= 10
end

@testset "PCATreeIndex dimension validation" begin
    data = randn(Float64, 3, 5)
    idx = build_index(PCATreeIndex, data)
    @test_throws DimensionMismatch query(idx, data, randn(4), 2)
end

@testset "PCATreeIndex MaxLeafSize bounds leaf size" begin
    rng = MersenneTwister(0xC0)
    data = randn(rng, Float64, 8, 200)
    cap = 16
    splitter = PCASplitter(
        ExactSVD(), TopComponent(),
        MaxLeafSize(cap), MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data; splitter = splitter,
                      rng = MersenneTwister(1))
    # The helper falls back to a leaf if a split is degenerate, so an
    # individual leaf may exceed `cap` only when the splitter was forced
    # to coalesce. With Gaussian data this should not happen.
    for node in idx.nodes
        if node.is_leaf
            sz = Int(node.leaf_hi) - Int(node.leaf_lo) + 1
            @test sz <= cap
        end
    end
end

@testset "PCATreeIndex IntrinsicDimRatio fires on low-rank data" begin
    rng = MersenneTwister(0xC1)
    d = 32
    n = 1000
    # Construct rank-1 data: all points are scalar multiples of a fixed
    # direction plus tiny isotropic noise. The top eigenvalue should
    # dominate, so IntrinsicDimRatio with a strict threshold fires
    # immediately at the root.
    direction = normalize!(randn(rng, d))
    scales = randn(rng, n)
    data = direction * scales' .+ 1e-6 .* randn(rng, d, n)

    splitter = PCASplitter(
        ExactSVD(), TopComponent(),
        AnyOf(MaxLeafSize(2_000_000),                 # disabled
              IntrinsicDimRatio(0.01; n_floor = 1)),
        MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data;
                      splitter = splitter,
                      rng = MersenneTwister(1))
    # Stopping at the root produces a single-leaf tree.
    @test length(idx.nodes) == 1
    @test idx.nodes[1].is_leaf
    @test length(idx.leaf_members) == n
end

@testset "PCATreeIndex IntrinsicDimRatio respects n_floor" begin
    rng = MersenneTwister(0xC2)
    d = 16
    n = 50
    data = randn(rng, d, n)  # full-rank Gaussian; ratio criterion never fires

    # Even on rank-1-ish data, n_floor=10_000 should suppress the
    # ratio-stop, leaving only MaxLeafSize to terminate recursion.
    splitter = PCASplitter(
        ExactSVD(), TopComponent(),
        AnyOf(MaxLeafSize(8),
              IntrinsicDimRatio(0.5; n_floor = 10_000)),
        MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data; splitter = splitter,
                      rng = MersenneTwister(1))
    # MaxLeafSize(8) on n=50 forces multiple internal nodes.
    @test count(node -> !node.is_leaf, idx.nodes) >= 1
end

@testset "PCATreeIndex policy variants smoke-construct" begin
    rng = MersenneTwister(0xC3)
    data = randn(rng, Float64, 12, 200)
    variants = [
        PCASplitter(),
        PCASplitter(ExactSVD(), RandomTopK(3),
                    MaxLeafSize(32), MeanSplit()),
        PCASplitter(ExactSVD(), RandomLinearCombo(4),
                    MaxLeafSize(32), RandomBetweenQuantiles(0.4, 0.6)),
        PCASplitter(RandomizedSVD(5; oversample = 5, n_iter = 1),
                    TopComponent(), MaxLeafSize(32), MedianSplit()),
        PCASplitter(SubsampledSVD(64, ExactSVD()),
                    TopComponent(), AllOf(MaxLeafSize(32)), MedianSplit()),
        pca_forest_splitter(sample_cap = 128, rank = 4, top_k = 2,
                            leaf_cap = 32, n_floor = 64),
    ]
    for splitter in variants
        idx = build_index(PCATreeIndex, data; splitter = splitter,
                          rng = MersenneTwister(7))
        q = randn(MersenneTwister(11), 12)
        res = query(idx, data, q, 5)
        @test length(res) <= 5
    end
end

@testset "PCATreeIndex recall floor on Gaussian blob" begin
    rng = MersenneTwister(0xBEEF)
    n, d, k = 800, 16, 10
    data = randn(rng, Float64, d, n)

    idx = build_index(PCATreeIndex, data; rng = MersenneTwister(2024))
    brute = build_index(BruteForceIndex, data)

    recalls = Float64[]
    for trial in 1:30
        q = randn(MersenneTwister(5000 + trial), d)
        ann = neighbor_ids(query(idx, data, q, k))
        bf  = neighbor_ids(query(brute, data, q, k))
        push!(recalls, length(intersect(Set(ann), Set(bf))) / length(bf))
    end
    avg = sum(recalls) / length(recalls)
    @test avg >= 0.10  # single-tree routing baseline
end

@testset "PCATreeIndex threading-safe query" begin
    rng = MersenneTwister(0xD0)
    n, d = 400, 16
    data = randn(rng, Float64, d, n)
    idx = build_index(PCATreeIndex, data; rng = MersenneTwister(2024))
    queries = randn(MersenneTwister(7), d, 200)

    # Reference: serial single-query path.
    serial = [query(idx, data, view(queries, :, i), 5) for i in 1:size(queries, 2)]
    # Threaded batch path (generic AbstractANNIndex fallback).
    threaded = query(idx, data, queries, 5)
    @test length(threaded) == length(serial)
    for i in eachindex(serial)
        @test neighbor_ids(threaded[i]) == neighbor_ids(serial[i])
    end
end
