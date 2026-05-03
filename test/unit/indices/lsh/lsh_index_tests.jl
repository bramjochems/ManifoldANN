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
        neighbors = query(index, data, data[:, i], 1)
        ids = neighbor_ids(neighbors)
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
    @test neighbor_ids(query(index, data, data[:, 1], 0)) == Int[]
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

    neighbors1 = neighbor_ids(query(index1, data, q, 10))
    neighbors2 = neighbor_ids(query(index2, data, q, 10))
    @test neighbors1 == neighbors2

    capped = query(index1, data, q, 10; candidate_cap = 3)
    @test length(capped) <= 3
end

@testset "LSHIndex candidate dedup and unbiased cap sampling" begin
    rng = MersenneTwister(424242)
    dimension = 4
    n_points = 200
    data = randn(rng, dimension, n_points)
    index = build_index(
        LSHIndex,
        data;
        n_tables = 8,
        hash_length = 4,
        rng = rng,
        hash_factory = make_random_hyperplane_hash,
    )
    q = randn(rng, dimension)

    # Dedup: returned neighbor IDs must be unique.
    full = query(index, data, q, 50)
    ids = neighbor_ids(full)
    @test length(ids) == length(unique(ids))

    # Sampling: with a tight candidate_cap, repeated queries with different RNG
    # must not all return the same low-numbered IDs. The old sort+resize
    # implementation deterministically returned the lowest IDs in the candidate
    # set. We expect the union over many trials to span a wide ID range.
    seen_ids = Set{Int}()
    for s in 1:200
        sampled = query(
            index, data, q, 1;
            candidate_cap = 1,
            rng = MersenneTwister(s),
        )
        for nid in neighbor_ids(sampled)
            push!(seen_ids, nid)
        end
    end
    # If sampling were biased toward low IDs we'd see only IDs near 1; require
    # at least one ID above the median index of the dataset.
    @test any(id -> id > n_points ÷ 2, seen_ids)
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

    result = neighbor_ids(query(index, data, new_point, 1))
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

    ids = neighbor_ids(query(index, data, data[:, 1], 2))
    @test !isempty(ids)
    @test ids[1] == 1
end
