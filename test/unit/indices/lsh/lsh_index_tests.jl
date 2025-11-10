using Test
using ManifoldANN
using Random
using LinearAlgebra

@testset "LSHIndex build and query" begin
    rng = MersenneTwister(1337)
    dimension = 3
    n_points = 8
    data = randn(rng, dimension, n_points)

    index = build_index(
        LSHIndex,
        data;
        n_tables = 6,
        hash_length = 12,
        rng = rng,
        hash_factory = make_random_hyperplane_hash,
    )

    @test index.n_points == n_points
    @test index.dimension == dimension

    for i in 1:n_points
        ids = query(index, data, data[:, i], 1)
        @test !isempty(ids)
        @test ids[1] == i
    end
end

@testset "LSHIndex query validation" begin
    rng = MersenneTwister(2024)
    data = randn(rng, 3, 20)
    index = build_index(
        LSHIndex,
        data;
        n_tables = 2,
        hash_length = 4,
        rng = rng,
        hash_factory = make_random_hyperplane_hash,
    )

    bad_q = randn(rng, 2)
    @test_throws DimensionMismatch query(index, data, bad_q, 1)
    @test query(index, data, data[:, 1], 0) == Int[]
end

@testset "LSHIndex determinism and candidate cap" begin
    rng_data = MersenneTwister(55)
    data = randn(rng_data, 4, 40)
    q = randn(rng_data, 4)

    rng1 = MersenneTwister(88)
    rng2 = MersenneTwister(88)

    index1 = build_index(
        LSHIndex,
        data;
        n_tables = 5,
        hash_length = 6,
        rng = rng1,
        hash_factory = make_random_hyperplane_hash,
    )
    index2 = build_index(
        LSHIndex,
        data;
        n_tables = 5,
        hash_length = 6,
        rng = rng2,
        hash_factory = make_random_hyperplane_hash,
    )

    neighbors1 = query(index1, data, q, 10)
    neighbors2 = query(index2, data, q, 10)
    @test neighbors1 == neighbors2

    capped = query(index1, data, q, 10; candidate_cap = 3)
    @test length(capped) <= 3
end

@testset "LSHIndex insertion" begin
    rng = MersenneTwister(99)
    data = randn(rng, 2, 5)
    index = build_index(
        LSHIndex,
        data;
        n_tables = 4,
        hash_length = 10,
        rng = rng,
        hash_factory = make_random_hyperplane_hash,
    )

    new_point = randn(rng, 2)
    data = hcat(data, new_point)
    insert!(index, new_point)

    result = query(index, data, new_point, 1)
    @test result[1] == size(data, 2)

    batch = randn(rng, 2, 2)
    data = hcat(data, batch)
    insert!(index, batch)
    @test index.n_points == size(data, 2)
end

@testset "Binning hash support" begin
    rng = MersenneTwister(101)
    data = randn(rng, 3, 6)
    index = build_index(
        LSHIndex,
        data;
        n_tables = 3,
        hash_length = 6,
        rng = rng,
        hash_factory = make_binning_hash,
        bin_width = 0.5,
        use_offset = true,
    )

    ids = query(index, data, data[:, 1], 2)
    @test !isempty(ids)
    @test ids[1] == 1
end
