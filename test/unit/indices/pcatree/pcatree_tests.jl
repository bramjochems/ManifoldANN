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
    # Rank-1+noise data so that, in the absence of `n_floor` gating,
    # `IntrinsicDimRatio(0.5)` *would* fire at the root. We probe two
    # configurations:
    #   (a) low n_floor: ratio fires early -> single-leaf or shallow tree
    #   (b) high n_floor: ratio is suppressed -> MaxLeafSize splits all
    #       the way down, producing many internal nodes.
    # If `n_floor` weren't gating in (b), the ratio criterion would
    # short-circuit the build the same way it does in (a) and the two
    # configurations would be indistinguishable.
    rng = MersenneTwister(0xC2)
    d = 16
    n = 200
    direction = normalize!(randn(rng, d))
    scales = randn(rng, n)
    data = direction * scales' .+ 1e-6 .* randn(rng, d, n)

    splitter_low = PCASplitter(
        ExactSVD(), TopComponent(),
        AnyOf(MaxLeafSize(8),
              IntrinsicDimRatio(0.5; n_floor = 1)),
        MedianSplit(),
    )
    splitter_high = PCASplitter(
        ExactSVD(), TopComponent(),
        AnyOf(MaxLeafSize(8),
              IntrinsicDimRatio(0.5; n_floor = 10_000)),
        MedianSplit(),
    )
    idx_low  = build_index(PCATreeIndex, data; splitter = splitter_low,
                           rng = MersenneTwister(1))
    idx_high = build_index(PCATreeIndex, data; splitter = splitter_high,
                           rng = MersenneTwister(1))

    n_internal_low  = count(node -> !node.is_leaf, idx_low.nodes)
    n_internal_high = count(node -> !node.is_leaf, idx_high.nodes)

    # Low n_floor: ratio fires immediately, single-leaf tree.
    @test n_internal_low == 0
    # High n_floor: ratio is gated off, MaxLeafSize(8) drives n=200 down
    # to many leaves -> many internal nodes.
    @test n_internal_high >= 5
    # The contrast is the test: n_floor *did* gate the ratio criterion.
    @test n_internal_high > n_internal_low
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
    # The PCA tree's `query` claims concurrent-safety. Two threads
    # calling it on the same index with different queries must produce
    # results identical to a serial loop. Mirrors the RPTreeForestIndex
    # re-entrancy test pattern. Under `julia -t 1` the threaded loop
    # degenerates to a serial sweep, so we skip with a warning rather
    # than report a green smoke test.
    if Threads.nthreads() >= 2
        rng = MersenneTwister(0xD0)
        n, d = 400, 16
        data = randn(rng, Float64, d, n)
        idx = build_index(PCATreeIndex, data; rng = MersenneTwister(2024))
        queries = [randn(MersenneTwister(0x100 + i), Float64, d) for i in 1:64]

        ref = [query(idx, data, q, 5) for q in queries]
        got = Vector{Vector{ManifoldANN.Neighbor{Float64}}}(undef, length(queries))
        Threads.@threads for i in eachindex(queries)
            got[i] = query(idx, data, queries[i], 5)
        end
        for i in eachindex(queries)
            @test neighbor_ids(got[i]) == neighbor_ids(ref[i])
        end
    else
        @warn "PCATreeIndex re-entrancy test requires Threads.nthreads() >= 2; skipping"
    end
end

@testset "PCATreeIndex elides SVD when MaxLeafSize fires on tiny nodes" begin
    # Fix #3: with the `!needs_spec` guard removed, the cheap-stop probe
    # `should_stop(stopping, n_node, nothing)` runs first; if it fires
    # (e.g. because `MaxLeafSize` says stop on a tiny node), we never
    # call the spectrum estimator. A composite stopping criterion that
    # *also* contains a spectrum-dependent rule used to defeat this; it
    # no longer does.
    mutable struct _CountingSVD <: ManifoldANN.AbstractSpectrumEstimator
        calls::Base.RefValue{Int}
    end
    _CountingSVD() = _CountingSVD(Ref(0))
    function ManifoldANN.estimate_spectrum(est::_CountingSVD, X::AbstractMatrix, rng::AbstractRNG)
        est.calls[] += 1
        return ManifoldANN.estimate_spectrum(ManifoldANN.ExactSVD(), X, rng)
    end

    rng = MersenneTwister(0xE0)
    d = 8
    n = 64
    data = randn(rng, Float64, d, n)

    estimator = _CountingSVD()
    # MaxLeafSize(8) fires on every node with <= 8 points; IntrinsicDimRatio
    # is spectrum-dependent. The cheap-stop probe must short-circuit on
    # tiny leaves before we ever touch the SVD.
    splitter = PCASplitter(
        estimator, TopComponent(),
        AnyOf(MaxLeafSize(8), IntrinsicDimRatio(0.01; n_floor = 1)),
        MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data; splitter = splitter,
                      rng = MersenneTwister(7))

    # Internal nodes are exactly the SVD-driven splits. Leaves where
    # MaxLeafSize fired must NOT have triggered an SVD call.
    n_internal = count(node -> !node.is_leaf, idx.nodes)
    @test estimator.calls[] == n_internal
    # Sanity: the tree actually split (so we exercised the
    # internal-node SVD path at least once and the leaf-elision path).
    @test n_internal >= 1
    @test count(node -> node.is_leaf, idx.nodes) >= 2
end

@testset "PCASplitter honors BPT non-degeneracy contract" begin
    # Fix #2: BPT no longer post-corrects empty-side splits; splitters
    # MUST emit BPTLeaf when a candidate split would collapse to one
    # side. Concentrate `n` co-located points so that the median split
    # produces an empty right side, forcing the splitter to fall back.
    n = 32
    d = 4
    data = zeros(Float64, d, n)  # all points identical -> spectrum is zero
    splitter = PCASplitter(
        ExactSVD(), TopComponent(),
        MaxLeafSize(2),  # ask for splitting; the splitter must still leaf
        MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data; splitter = splitter,
                      rng = MersenneTwister(1))
    # Either every point ends up under one leaf, or the tree is empty
    # of internal nodes — in any case, the union of leaf members covers
    # every point exactly once and no node leaks.
    members = Int[]
    for node in idx.nodes
        if node.is_leaf
            append!(members, idx.leaf_members[Int(node.leaf_lo):Int(node.leaf_hi)])
        end
    end
    @test sort(members) == collect(1:n)
end

@testset "AbstractRPSplitter honors BPT non-degeneracy contract" begin
    # Fix #2 mirror for the RP adapter: identical points cannot be split
    # by a two-point hyperplane (both selected points coincide -> zero
    # hyperplane). The splitter returns `nothing`, the adapter emits
    # BPTLeaf, and BPT does not post-correct.
    n = 16
    d = 4
    data = ones(Float64, d, n)
    idx = build_index(RPTreeIndex, data; leaf_cap = 2,
                      rng = MersenneTwister(1))
    members = Int[]
    for node in idx.tree.nodes
        if node.is_leaf
            append!(members, idx.tree.leaf_members[Int(node.leaf_lo):Int(node.leaf_hi)])
        end
    end
    @test sort(members) == collect(1:n)
end

@testset "PCATreeIndex RandomLinearCombo preserves Float32 eltype" begin
    # Regression: prior `randn(rng, k)` defaulted to Float64 and silently
    # promoted the direction vector to Float64 even for Float32 inputs.
    # The PCANodePayload constructor would coerce the result; this test
    # locks in type-stability of the direction itself.
    rng = MersenneTwister(0xF1)
    data = randn(rng, Float32, 10, 200)
    splitter = PCASplitter(
        ExactSVD(), RandomLinearCombo(3),
        MaxLeafSize(32), MedianSplit(),
    )
    idx = build_index(PCATreeIndex, data; splitter = splitter,
                      rng = MersenneTwister(2))
    for node in idx.nodes
        if !node.is_leaf
            @test eltype(node.payload.direction) === Float32
        end
    end
end
