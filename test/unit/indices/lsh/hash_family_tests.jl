using Test
using ManifoldANN
using Random

@testset "Hyperplane hash basics" begin
    projections = [1.0 0.0; 0.0 1.0]
    hf = RandomHyperplaneHash{Float64}(projections)

    @test ManifoldANN.hash_point(hf, [1.0, 1.0]) == 0x03
    @test ManifoldANN.hash_point(hf, [-1.0, -1.0]) == 0x00
    @test ManifoldANN.hash_point(hf, [1.0, -1.0]) == 0x01

    data = hcat([1.0, 1.0], [-1.0, -1.0])
    hashes = ManifoldANN.hash_batch(hf, data)
    @test hashes == [0x03, 0x00]
end

@testset "Binning hash basics" begin
    rng = MersenneTwister(1)
    hf = make_binning_hash(2, 2; bin_width = 1.0, use_offset = false, rng = rng)
    x = [0.1, 0.6]
    h = ManifoldANN.hash_point(hf, x)
    @test h isa UInt64
    hashes = ManifoldANN.hash_batch(hf, reshape(x, :, 1))
    @test hashes[1] == h
end
