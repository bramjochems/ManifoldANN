using Test
using Random
using ManifoldANN
using Distances

# Regression test: the multilevel builder must thread `rng` through to
# `fit!(transform, X)` so that two builds with the same seed produce
# identical inner KMeansTransform centroids. Without rng plumbing,
# threaded child builds race on Random.default_rng() and centroids
# diverge run-to-run.
@testset "MultiLevelIndex builder rng determinism" begin
    n_dims = 6
    n_points = 200
    rng_data = MersenneTwister(20251103)
    X = rand(rng_data, Float32, n_dims, n_points)

    # Two-level config: outer KMeans -> per-bucket KMeans -> brute force
    # leaf. Inner KMeans is the load-bearing case (it lives under the
    # threaded child loop, where the legacy default_rng() race lived).
    function make_config()
        TransformedConfig(
            KMeansTransform(k = 4, distance = Euclidean(), init = :kmeans_plus_plus),
            TopKRouting(2),
            TransformedConfig(
                KMeansTransform(k = 3, distance = Euclidean(), init = :kmeans_plus_plus),
                TopKRouting(1),
                TerminalConfig(BruteForceIndex, NamedTuple()),
            ),
        )
    end

    seed = 42
    idx_a = build_index(MultiLevelIndex, X, make_config(); rng = MersenneTwister(seed))
    idx_b = build_index(MultiLevelIndex, X, make_config(); rng = MersenneTwister(seed))

    # Outer-level centroids must match exactly.
    @test idx_a.root.transform.centroids == idx_b.root.transform.centroids

    # Inner-level (per-bucket) centroids must also match. This is the
    # case the threaded loop used to race on.
    @test length(idx_a.root.indices) == length(idx_b.root.indices)
    for (child_a, child_b) in zip(idx_a.root.indices, idx_b.root.indices)
        @test child_a.transform.centroids == child_b.transform.centroids
    end
end

# Companion to the KMeans test: verify the builder also threads `rng`
# through to RandomProjectionTransform.fit!. RP is preserves_data=false
# and non-bucketing, so it lands on the builder's single-child branch
# (line 195+ of builder.jl) — a different code path from KMeans's
# bucketed branch. A refactor that drops the RP overload of
# _fit_transform! would not be caught by the KMeans test alone.
@testset "MultiLevelIndex builder rng determinism (RandomProjection)" begin
    n_dims = 32
    n_points = 200
    rng_data = MersenneTwister(20251104)
    X = rand(rng_data, Float64, n_dims, n_points)

    function make_rp_config()
        TransformedConfig(
            RandomProjectionTransform(target_dim = 8, projection_type = :gaussian),
            TopKRouting(1),
            TerminalConfig(BruteForceIndex, NamedTuple()),
        )
    end

    seed = 7
    idx_a = build_index(MultiLevelIndex, X, make_rp_config(); rng = MersenneTwister(seed))
    idx_b = build_index(MultiLevelIndex, X, make_rp_config(); rng = MersenneTwister(seed))

    @test idx_a.root.transform.projection == idx_b.root.transform.projection

    idx_c = build_index(MultiLevelIndex, X, make_rp_config(); rng = MersenneTwister(seed + 1))
    @test idx_a.root.transform.projection != idx_c.root.transform.projection
end
