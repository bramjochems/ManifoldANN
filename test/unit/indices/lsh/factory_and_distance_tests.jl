using Test
using ManifoldANN
using Random
using LinearAlgebra

@testset "Hash factory validation" begin
    rng = MersenneTwister(77)
    hf = make_random_hyperplane_hash(5, 7; rng = rng)
    @test size(hf.projections) == (7, 5)

    hf_bin = make_binning_hash(4, 3; bin_width = 2.0, rng = rng, use_offset = true)
    @test size(hf_bin.projections) == (3, 4)
    @test length(hf_bin.offsets) == 3
    @test hf_bin.bin_width == 2.0

    @test_throws ArgumentError make_binning_hash(4, 3; bin_width = 0.0)
end

@testset "Distance helpers" begin
    x = [1.0, 0.0, 0.0]
    y = [0.0, 1.0, 0.0]
    @test ManifoldANN.cosine_distance(x, y) ≈ 1.0
    @test ManifoldANN.euclidean_distance([1.0, 2.0], [4.0, 6.0]) ≈ 5.0
end
