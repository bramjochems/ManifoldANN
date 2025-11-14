using Test
using ManifoldANN
using ManifoldANN: pairwise_distances!, pairwise_euclidean!, pairwise_generic!, compute_distances
using Distances
using Random
using LinearAlgebra

@testset "KMeans Optimizations" begin
    @testset "pairwise_euclidean! correctness" begin
        Random.seed!(42)
        d, n, k = 10, 100, 5
        X = rand(Float32, d, n)
        centroids = rand(Float32, d, k)

        # Compute with optimized version
        D_fast = Matrix{Float32}(undef, k, n)
        pairwise_euclidean!(D_fast, X, centroids)

        # Compute with generic version
        D_slow = Matrix{Float32}(undef, k, n)
        pairwise_generic!(D_slow, X, centroids, Distances.Euclidean())

        # Should match within floating point tolerance
        @test D_fast ≈ D_slow rtol=1e-5
    end

    @testset "pairwise_distances! dispatch" begin
        Random.seed!(42)
        d, n, k = 10, 50, 3
        X = rand(Float32, d, n)
        centroids = rand(Float32, d, k)

        # Test Euclidean dispatch
        D_euclidean = Matrix{Float32}(undef, k, n)
        pairwise_distances!(D_euclidean, X, centroids, Distances.Euclidean())

        # Test Manhattan fallback to threaded
        D_manhattan = Matrix{Float32}(undef, k, n)
        pairwise_distances!(D_manhattan, X, centroids, Distances.Cityblock())

        # Euclidean should be close to manually computed
        for i in 1:k, j in 1:n
            expected = norm(centroids[:, i] - X[:, j])
            @test D_euclidean[i, j] ≈ expected rtol=1e-5
        end

        # Manhattan should use correct metric
        for i in 1:k, j in 1:n
            expected = sum(abs, centroids[:, i] - X[:, j])
            @test D_manhattan[i, j] ≈ expected rtol=1e-5
        end
    end

    @testset "compute_distances optimization" begin
        Random.seed!(42)
        d, k = 128, 100
        x = rand(Float32, d)
        centroids = rand(Float32, d, k)

        # Optimized version
        distances_fast = compute_distances(x, centroids, Distances.Euclidean())

        # Reference version
        distances_ref = [norm(centroids[:, i] - x) for i in 1:k]

        @test distances_fast ≈ distances_ref rtol=1e-5
    end

    @testset "KMeans fit! still works" begin
        Random.seed!(42)
        d, n = 10, 1000
        k = 5
        X = rand(Float32, d, n)

        # Fit with optimized code
        transform = KMeansTransform(k=k, distance=Distances.Euclidean(), init=:kmeans_plus_plus)
        fit!(transform, X)

        # Should have learned centroids
        @test transform.centroids !== nothing
        @test size(transform.centroids) == (d, k)

        # Transform a point
        result = ManifoldANN.transform(transform, X[:, 1])
        @test result.data == X[:, 1]
        @test length(result.assignment.distances) == k
        @test all(result.assignment.distances .>= 0)
    end

    @testset "Edge cases" begin
        # Small dataset
        X_small = rand(Float32, 5, 3)
        centroids_small = rand(Float32, 5, 2)
        D_small = Matrix{Float32}(undef, 2, 3)
        pairwise_euclidean!(D_small, X_small, centroids_small)
        @test size(D_small) == (2, 3)
        @test all(isfinite, D_small)

        # Single point
        x = rand(Float32, 10)
        c = rand(Float32, 10, 1)
        d = compute_distances(x, c, Distances.Euclidean())
        @test length(d) == 1
        @test isfinite(d[1])
    end

    @testset "Negative squared distance clamping" begin
        # Test that floating point errors don't cause negative sqrt
        d, n, k = 100, 10, 5
        X = rand(Float32, d, n)
        centroids = copy(X[:, 1:k])  # Use actual data points as centroids

        D = Matrix{Float32}(undef, k, n)
        pairwise_euclidean!(D, X, centroids)

        # Distances to self should be ~0, not negative
        # Allow small tolerance for floating point accumulation
        for i in 1:k
            @test D[i, i] ≈ 0.0f0 atol=0.01f0
            @test D[i, i] >= 0.0f0
        end
    end
end
