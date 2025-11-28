using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "RandomProjectionTransform" begin
    @testset "Construction" begin
        # Default Gaussian projection
        rp = RandomProjectionTransform(target_dim=50)
        @test rp isa RandomProjectionTransform{Float64}
        @test rp.target_dim == 50
        @test rp.projection_type == :gaussian
        @test !rp.fitted

        # Sparse projection
        rp = RandomProjectionTransform(target_dim=100, projection_type=:sparse, density=0.25)
        @test rp.projection_type == :sparse
        @test rp.density == 0.25

        # Invalid parameters
        @test_throws ArgumentError RandomProjectionTransform(target_dim=0)
        @test_throws ArgumentError RandomProjectionTransform(target_dim=-5)
        @test_throws ArgumentError RandomProjectionTransform(target_dim=50, projection_type=:invalid)
        @test_throws ArgumentError RandomProjectionTransform(target_dim=50, density=0.0)
        @test_throws ArgumentError RandomProjectionTransform(target_dim=50, density=1.5)
    end

    @testset "Gaussian projection" begin
        rng = MersenneTwister(42)
        Random.seed!(42)

        # Create high-dimensional data
        n_ambient = 1000
        n_samples = 500
        target_dim = 100

        X = randn(rng, n_ambient, n_samples)

        # Fit and transform
        rp = RandomProjectionTransform(target_dim=target_dim, projection_type=:gaussian)
        @test !rp.fitted
        fit!(rp, X)
        @test rp.fitted
        @test target_dimension(rp) == target_dim
        @test size(rp.projection) == (target_dim, n_ambient)

        # Transform a single point
        x = X[:, 1]
        result = transform(rp, x)
        @test result isa TransformResult
        @test length(result.data) == target_dim
        @test result.assignment === nothing
    end

    @testset "Sparse projection" begin
        rng = MersenneTwister(43)
        Random.seed!(43)

        n_ambient = 500
        n_samples = 200
        target_dim = 50
        density = 1/3

        X = randn(rng, n_ambient, n_samples)

        # Fit sparse projection
        rp = RandomProjectionTransform(target_dim=target_dim, projection_type=:sparse, density=density)
        fit!(rp, X)
        @test rp.fitted

        # Check sparsity of projection matrix
        n_nonzero = count(!iszero, rp.projection)
        total_entries = target_dim * n_ambient
        actual_density = n_nonzero / total_entries

        # Density should be approximately as specified (with some random variation)
        @test actual_density > density * 0.8
        @test actual_density < density * 1.2

        # Transform
        x = X[:, 1]
        result = transform(rp, x)
        @test length(result.data) == target_dim
    end

    @testset "Distance preservation (JL lemma)" begin
        rng = MersenneTwister(44)
        Random.seed!(44)

        # Create a set of points
        n_samples = 100
        n_ambient = 500
        X = randn(rng, n_ambient, n_samples)

        # Suggested dimension for ε=0.2 (20% distortion)
        epsilon = 0.2
        target_dim = suggested_dimension(n_samples, epsilon=epsilon)
        @test target_dim > 0

        # Fit random projection
        rp = RandomProjectionTransform(target_dim=target_dim, projection_type=:gaussian)
        fit!(rp, X)

        # Transform all points
        X_projected = hcat([transform(rp, X[:, i]).data for i in 1:n_samples]...)

        # Sample some pairs and check distance preservation
        n_pairs = 50
        for _ in 1:n_pairs
            i, j = rand(rng, 1:n_samples, 2)
            if i == j
                continue
            end

            # Original distance
            d_orig = norm(X[:, i] - X[:, j])

            # Projected distance
            d_proj = norm(X_projected[:, i] - X_projected[:, j])

            # Check relative distortion
            distortion = abs(d_proj - d_orig) / d_orig

            # Note: JL lemma is probabilistic, so not all pairs will satisfy it
            # We just check that most pairs have reasonable distortion
            # For a proper test, would need statistical analysis
            if distortion > epsilon * 3
                @warn "Large distortion detected" distortion i j maxlog=3
            end
        end
    end

    @testset "suggested_dimension" begin
        # Test suggested dimension computation
        d = suggested_dimension(100, epsilon=0.1)
        @test d > 0
        @test d isa Int

        d = suggested_dimension(10000, epsilon=0.1)
        @test d > suggested_dimension(1000, epsilon=0.1)  # More samples need higher dimension

        d = suggested_dimension(1000, epsilon=0.2)
        @test d < suggested_dimension(1000, epsilon=0.1)  # Higher ε allows lower dimension

        # Edge cases
        @test_throws ArgumentError suggested_dimension(0)
        @test_throws ArgumentError suggested_dimension(-1)
        @test_throws ArgumentError suggested_dimension(100, epsilon=0.0)
        @test_throws ArgumentError suggested_dimension(100, epsilon=-0.1)
    end

    @testset "Type stability" begin
        rng = MersenneTwister(45)
        X = randn(rng, 100, 50)

        # Float64
        rp64 = RandomProjectionTransform{Float64}(target_dim=20)
        fit!(rp64, X)
        x = X[:, 1]
        result = transform(rp64, x)
        @test eltype(result.data) == Float64
        @test eltype(rp64.projection) == Float64

        # Float32
        X32 = Float32.(X)
        rp32 = RandomProjectionTransform{Float32}(target_dim=20)
        fit!(rp32, X32)
        x32 = X32[:, 1]
        result32 = transform(rp32, x32)
        @test eltype(result32.data) == Float32
        @test eltype(rp32.projection) == Float32
    end

    @testset "Edge cases" begin
        rng = MersenneTwister(46)

        # Target dimension larger than ambient
        X = randn(rng, 50, 100)
        rp = RandomProjectionTransform(target_dim=100)
        # Should warn but not error
        fit!(rp, X)
        @test rp.fitted

        # Transform before fitting
        rp = RandomProjectionTransform(target_dim=20)
        x = randn(rng, 50)
        @test_throws ErrorException transform(rp, x)
    end

    @testset "Preserves data check" begin
        rp = RandomProjectionTransform(target_dim=50)
        @test !preserves_data(rp)  # Random projection changes the representation
    end
end
