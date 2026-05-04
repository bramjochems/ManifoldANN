using Test
using ManifoldANN
using Random

@testset "RandomProjectionTransform RNG reproducibility" begin
    d_ambient, n = 64, 50
    X = randn(Float64, d_ambient, n)

    @testset "Gaussian: same seed -> identical projections" begin
        rp1 = RandomProjectionTransform(target_dim=16, projection_type=:gaussian)
        rp2 = RandomProjectionTransform(target_dim=16, projection_type=:gaussian)

        fit!(rp1, X; rng=MersenneTwister(2024))
        fit!(rp2, X; rng=MersenneTwister(2024))

        @test rp1.projection == rp2.projection
    end

    @testset "Gaussian: different seeds -> different projections" begin
        rp1 = RandomProjectionTransform(target_dim=16, projection_type=:gaussian)
        rp2 = RandomProjectionTransform(target_dim=16, projection_type=:gaussian)

        fit!(rp1, X; rng=MersenneTwister(1))
        fit!(rp2, X; rng=MersenneTwister(2))

        @test rp1.projection != rp2.projection
    end

    @testset "Sparse: same seed -> identical projections" begin
        rp1 = RandomProjectionTransform(target_dim=20, projection_type=:sparse, density=1/3)
        rp2 = RandomProjectionTransform(target_dim=20, projection_type=:sparse, density=1/3)

        fit!(rp1, X; rng=MersenneTwister(99))
        fit!(rp2, X; rng=MersenneTwister(99))

        @test rp1.projection == rp2.projection
    end

    @testset "Sparse: different seeds -> different projections" begin
        rp1 = RandomProjectionTransform(target_dim=20, projection_type=:sparse, density=1/3)
        rp2 = RandomProjectionTransform(target_dim=20, projection_type=:sparse, density=1/3)

        fit!(rp1, X; rng=MersenneTwister(1))
        fit!(rp2, X; rng=MersenneTwister(2))

        @test rp1.projection != rp2.projection
    end
end
