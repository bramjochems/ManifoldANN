using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "PCATransform" begin
    @testset "Construction" begin
        # Default construction
        pca = PCATransform()
        @test pca isa PCATransform{Float64}
        @test isnothing(pca.target_dim)
        @test pca.variance_threshold == 0.95
        @test !pca.fitted

        # Fixed dimension
        pca = PCATransform(target_dim=10)
        @test pca.target_dim == 10
        @test !pca.fitted

        # Custom variance threshold
        pca = PCATransform(variance_threshold=0.99)
        @test pca.variance_threshold == 0.99

        # Invalid parameters
        @test_throws ArgumentError PCATransform(target_dim=0)
        @test_throws ArgumentError PCATransform(target_dim=-1)
        @test_throws ArgumentError PCATransform(variance_threshold=0.0)
        @test_throws ArgumentError PCATransform(variance_threshold=1.5)
    end

    @testset "Fitting and transformation - fixed dimension" begin
        rng = MersenneTwister(42)

        # Create 100-dimensional data with only 5 significant dimensions
        n_samples = 1000
        n_ambient = 100
        n_intrinsic = 5

        # Generate data with structure in first 5 dimensions
        X_low = randn(rng, n_intrinsic, n_samples) .* [10.0, 8.0, 6.0, 4.0, 2.0]
        # Pad with noise in remaining dimensions
        X_noise = randn(rng, n_ambient - n_intrinsic, n_samples) .* 0.1
        X = vcat(X_low, X_noise)

        # Fit PCA to 10 dimensions
        pca = PCATransform(target_dim=10)
        @test !pca.fitted
        fit!(pca, X)
        @test pca.fitted
        @test target_dimension(pca) == 10

        # Transform a point
        x = X[:, 1]
        result = transform(pca, x)
        @test result isa TransformResult
        @test length(result.data) == 10
        @test result.assignment === nothing

        # Check reconstruction
        x_reconstructed = inverse_transform(pca, result.data)
        @test length(x_reconstructed) == n_ambient

        # Reconstruction error should be small
        error = norm(x - x_reconstructed)
        @test error < 1.0  # Should be small due to low noise
    end

    @testset "Fitting and transformation - variance threshold" begin
        rng = MersenneTwister(43)

        # Create data on a 2D plane in 10D space
        n_samples = 500
        # Generate 2D data
        X_2d = randn(rng, 2, n_samples) .* 5.0
        # Embed in 10D with small noise
        embedding = randn(rng, 10, 2)
        X = embedding * X_2d .+ randn(rng, 10, n_samples) .* 0.01

        # Fit PCA with 95% variance threshold
        pca = PCATransform(variance_threshold=0.95)
        fit!(pca, X)
        @test pca.fitted

        # Should detect ~2 dimensions
        d = target_dimension(pca)
        @test d >= 2
        @test d <= 4  # Might pick up some noise dimensions

        # Check explained variance
        var_ratio = explained_variance_ratio(pca)
        @test sum(var_ratio) ≈ 1.0
        @test all(var_ratio .>= 0)
        @test issorted(var_ratio, rev=true)  # Decreasing order
    end

    @testset "Type stability" begin
        rng = MersenneTwister(44)
        X = randn(rng, 50, 100)

        # Float64
        pca64 = PCATransform{Float64}(target_dim=10)
        fit!(pca64, X)
        x = X[:, 1]
        result = transform(pca64, x)
        @test eltype(result.data) == Float64

        # Float32
        X32 = Float32.(X)
        pca32 = PCATransform{Float32}(target_dim=10)
        fit!(pca32, X32)
        x32 = X32[:, 1]
        result32 = transform(pca32, x32)
        @test eltype(result32.data) == Float32
    end

    @testset "Edge cases" begin
        rng = MersenneTwister(45)

        # Too few samples
        X_small = randn(rng, 10, 1)
        pca = PCATransform(target_dim=5)
        @test_throws ArgumentError fit!(pca, X_small)

        # Target dimension larger than ambient
        X = randn(rng, 10, 100)
        pca = PCATransform(target_dim=20)
        fit!(pca, X)  # Should not error
        @test target_dimension(pca) == 10  # Should cap at ambient dimension

        # Transform before fitting
        pca = PCATransform(target_dim=5)
        x = randn(rng, 10)
        @test_throws ErrorException transform(pca, x)
    end

    @testset "Preserves data check" begin
        pca = PCATransform(target_dim=10)
        @test !preserves_data(pca)  # PCA changes the representation
    end

    @testset "Batch transformation" begin
        rng = MersenneTwister(46)
        X = randn(rng, 50, 200)

        pca = PCATransform(target_dim=10)
        fit!(pca, X)

        # Transform all points
        X_transformed = hcat([transform(pca, X[:, i]).data for i in 1:size(X, 2)]...)

        @test size(X_transformed) == (10, 200)

        # Check that inverse transformation approximately recovers original
        X_recovered = hcat([inverse_transform(pca, X_transformed[:, i]) for i in 1:size(X_transformed, 2)]...)

        # Not exact due to dimensionality reduction
        # Relative error depends on intrinsic dimensionality of random data
        relative_error = norm(X - X_recovered) / norm(X)
        @test relative_error < 1.0  # Should preserve at least some information
    end
end
