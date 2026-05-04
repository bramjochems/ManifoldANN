using Test
using Random
using Distances
using ManifoldANN

@testset "KDTreeIndex exact queries" begin
    rng = MersenneTwister(1234)
    dimension = 4
    n_points = 40
    data = randn(rng, dimension, n_points)

    kd_index = build_index(KDTreeIndex, data)
    brute_index = build_index(BruteForceIndex, data)

    for k in 1:5
        for _ in 1:15
            query_point = randn(rng, dimension)
            kd_ids = neighbor_ids(query(kd_index, data, query_point, k))
            brute_ids = neighbor_ids(query(brute_index, data, query_point, k))
            @test kd_ids == brute_ids
        end
    end
end

@testset "Axis selector variants stay consistent" begin
    rng = MersenneTwister(2024)
    data = randn(rng, 3, 20)
    variance_tree = build_index(KDTreeIndex, data; axis_selector = :variance)
    cyclic_tree = build_index(KDTreeIndex, data; axis_selector = :cyclic)

    for k in 1:3, _ in 1:10
        q = randn(rng, 3)
        @test neighbor_ids(query(variance_tree, data, q, k)) ==
            neighbor_ids(query(cyclic_tree, data, q, k))
    end
end

@testset "Dimension validation" begin
    data = randn(2, 5)
    index = build_index(KDTreeIndex, data)
    bad_data = randn(3, 5)
    bad_query = randn(3)
    @test_throws DimensionMismatch query(index, bad_data, randn(3), 2)
    @test_throws DimensionMismatch query(index, data, bad_query, 2)
end

# --- Rolling-bound prune correctness (FBF77) -----------------------------
#
# The rolling cell-distance bound must be a *valid lower bound* on
# dist(q, p) for any p in the cell. If it isn't, the prune is unsafe and
# the kdtree returns wrong neighbour IDs vs brute force. These tests are
# the property check: any disagreement with brute force means the bound
# is buggy.

@testset "KDTreeIndex with SqEuclidean (re-admitted)" begin
    rng = MersenneTwister(98765)
    data = randn(rng, 5, 60)
    kd = build_index(KDTreeIndex, data; distance = SqEuclidean())
    brute = build_index(BruteForceIndex, data; distance = SqEuclidean())

    for k in 1:6, _ in 1:20
        q = randn(rng, 5)
        kd_ids = neighbor_ids(query(kd, data, q, k))
        brute_ids = neighbor_ids(query(brute, data, q, k))
        @test kd_ids == brute_ids
    end
end

@testset "KDTreeIndex Euclidean and SqEuclidean agree on neighbour IDs" begin
    # Same data, same query: neighbor identity is invariant under
    # monotone transform of the metric. So Euclidean and SqEuclidean
    # MUST return the same IDs in the same order; only the distance
    # values differ by a sqrt.
    rng = MersenneTwister(424242)
    data = randn(rng, 7, 80)
    kd_l2 = build_index(KDTreeIndex, data; distance = Euclidean())
    kd_sq = build_index(KDTreeIndex, data; distance = SqEuclidean())

    for k in 1:5, _ in 1:25
        q = randn(rng, 7)
        ids_l2 = neighbor_ids(query(kd_l2, data, q, k))
        ids_sq = neighbor_ids(query(kd_sq, data, q, k))
        @test ids_l2 == ids_sq
    end
end

@testset "KDTreeIndex query outside data bbox" begin
    # Initial cell_dist > 0 because q is outside the global span. The
    # rolling bound must still match brute force.
    rng = MersenneTwister(13579)
    data = randn(rng, 3, 50)
    kd_l2 = build_index(KDTreeIndex, data; distance = Euclidean())
    kd_sq = build_index(KDTreeIndex, data; distance = SqEuclidean())
    brute_l2 = build_index(BruteForceIndex, data; distance = Euclidean())
    brute_sq = build_index(BruteForceIndex, data; distance = SqEuclidean())

    for k in 1:4, _ in 1:15
        # A point well outside the data envelope (ten std-devs out).
        q = 10.0 .+ randn(rng, 3)
        @test neighbor_ids(query(kd_l2, data, q, k)) ==
              neighbor_ids(query(brute_l2, data, q, k))
        @test neighbor_ids(query(kd_sq, data, q, k)) ==
              neighbor_ids(query(brute_sq, data, q, k))
    end
end

@testset "KDTreeIndex tiny tree (single leaf, no recursion)" begin
    # n_points <= leafsize: root is a leaf, no internal nodes — exercises
    # the no-recursion path through both metrics.
    rng = MersenneTwister(2468)
    data = randn(rng, 4, 5)
    for metric in (Euclidean(), SqEuclidean())
        kd = build_index(KDTreeIndex, data; distance = metric)
        brute = build_index(BruteForceIndex, data; distance = metric)
        for _ in 1:8
            q = randn(rng, 4)
            @test neighbor_ids(query(kd, data, q, 3)) ==
                  neighbor_ids(query(brute, data, q, 3))
        end
    end
end

@testset "KDTreeIndex Cityblock and Minkowski rolling bound" begin
    rng = MersenneTwister(31337)
    data = randn(rng, 4, 50)
    for metric in (Cityblock(), Minkowski(3.0))
        kd = build_index(KDTreeIndex, data; distance = metric)
        brute = build_index(BruteForceIndex, data; distance = metric)
        for k in 1:4, _ in 1:12
            q = randn(rng, 4)
            @test neighbor_ids(query(kd, data, q, k)) ==
                  neighbor_ids(query(brute, data, q, k))
        end
    end
end

@testset "KDTreeIndex concurrent batch queries (per-call descent stack)" begin
    # The rolling-bound descent uses per-call cell_lo/cell_hi vectors;
    # threaded batch queries must not corrupt one another. This test
    # asserts batch query results match the serial baseline.
    rng = MersenneTwister(55555)
    data = randn(rng, 6, 200)
    queries = randn(rng, 6, 64)
    for metric in (Euclidean(), SqEuclidean())
        kd = build_index(KDTreeIndex, data; distance = metric)
        brute = build_index(BruteForceIndex, data; distance = metric)
        batch = query(kd, data, queries, 5)
        for i in 1:size(queries, 2)
            @test neighbor_ids(batch[i]) ==
                  neighbor_ids(query(brute, data, view(queries, :, i), 5))
        end
    end
end

@testset "KDTreeIndex Float32 + SqEuclidean" begin
    rng = MersenneTwister(7777)
    data = randn(rng, Float32, 4, 40)
    kd = build_index(KDTreeIndex, data; distance = SqEuclidean())
    brute = build_index(BruteForceIndex, data; distance = SqEuclidean())
    for _ in 1:15
        q = randn(rng, Float32, 4)
        @test neighbor_ids(query(kd, data, q, 4)) ==
              neighbor_ids(query(brute, data, q, 4))
    end
end
