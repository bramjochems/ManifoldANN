using Test
using Random
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
            kd_ids = query(kd_index, data, query_point, k)
            brute_ids = query(brute_index, data, query_point, k)
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
        @test query(variance_tree, data, q, k) == query(cyclic_tree, data, q, k)
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
