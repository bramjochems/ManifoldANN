using Test
using ManifoldANN

@testset "BruteForceIndex build" begin
    data = reshape(Float64.(1:12), 3, 4)
    index = build_index(BruteForceIndex, data)
    @test index.dimension == 3
    @test index.n_points == 4
end

@testset "BruteForceIndex query" begin
    data = [0.0 1.0 2.0; 0.0 0.0 0.0]
    index = build_index(BruteForceIndex, data)
    q = [0.25, 0.0]
    neighbors = query(index, data, q, 2)
    @test neighbors == [1, 2]

    # Reduce data columns to trigger consistency error.
    smaller = data[:, 1:2]
    @test_throws ArgumentError query(index, smaller, q, 2)

    bad_q = [0.0, 0.0, 0.0]
    @test_throws DimensionMismatch query(index, data, bad_q, 1)
end

@testset "BruteForceIndex insert" begin
    data = [1.0 2.0; 3.0 4.0]
    index = build_index(BruteForceIndex, data)

    new_point = [10.0, 20.0]
    insert!(index, new_point)
    @test index.n_points == 3

    new_points = [30.0 40.0 50.0 60.0; 11.0 12.0 13.0 14.0]
    insert!(index, new_points)
    @test index.n_points == 7

    bad_point = [1.0, 2.0, 3.0]
    @test_throws DimensionMismatch insert!(index, bad_point)

    bad_matrix = reshape(collect(1.0:3.0), 3, 1)
    @test_throws DimensionMismatch insert!(index, bad_matrix)

    # Query after inserts requires expanded data matrix.
    expanded = hcat(data, new_point, new_points)
    q = [0.5, 0.5]
    neighbors = query(index, expanded, q, 3)
    @test length(neighbors) == 3
end
