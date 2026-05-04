using Test
using ManifoldANN
using ManifoldANN: lloyd!, KMeansTransform, fit!
using Distances
using Random

@testset "KMeans empty-cluster reassignment" begin
    # Construct a degenerate scenario forcing ≥2 empty clusters in a single
    # iteration. With initial centroids placed extremely far from all data
    # points, the assignment step funnels every point into the centroid that
    # happens to be closest, leaving the others empty. We then verify that
    # distinct points are promoted (the bug previously caused the same
    # farthest data point to be reassigned to every empty cluster).
    @testset "≥2 empty clusters get distinct centroids" begin
        # 5 data points in 2D, all near origin
        X = Float64[
            0.0  1.0  2.0  3.0  4.0;
            0.0  0.0  0.0  0.0  0.0
        ]
        d, n = size(X)
        k = 4

        # Initial centroids: one near data, three identical and absurdly far.
        # The E-step assigns all points to the nearest centroid (the one near
        # the data). The other three centroids end up empty in this iteration.
        centroids = Float64[
            0.0   1e9   1e9   1e9;
            0.0   1e9   1e9   1e9
        ]

        # Run a single iteration of Lloyd's so the empty-cluster pass kicks in.
        # max_iters=1 ensures the empty-cluster repair logic is the only thing
        # producing the final state for the empty cluster slots.
        out_centroids, _, _ = lloyd!(copy(centroids), X, Euclidean(); max_iters=1)

        # The first centroid is the mean of all data points; the other three
        # were repaired by promoting "farthest" points. Verify the three
        # repaired centroids are pairwise distinct (this is what the bug
        # broke: all three would be identical to one another).
        repaired = [out_centroids[:, 2], out_centroids[:, 3], out_centroids[:, 4]]
        @test repaired[1] != repaired[2]
        @test repaired[1] != repaired[3]
        @test repaired[2] != repaired[3]

        # Each repaired centroid must equal one of the data points.
        data_cols = [X[:, j] for j in 1:n]
        for r in repaired
            @test any(r == c for c in data_cols)
        end
    end
end

@testset "KMeansTransform RNG reproducibility" begin
    d, n, k = 8, 200, 5
    X = randn(Float32, d, n)

    @testset "Same seed -> identical centroids" begin
        t1 = KMeansTransform(k=k, distance=Euclidean(), init=:kmeans_plus_plus)
        t2 = KMeansTransform(k=k, distance=Euclidean(), init=:kmeans_plus_plus)

        rng1 = MersenneTwister(12345)
        rng2 = MersenneTwister(12345)
        fit!(t1, X; rng=rng1)
        fit!(t2, X; rng=rng2)

        @test t1.centroids == t2.centroids
    end

    @testset "Different seeds -> different centroids" begin
        t1 = KMeansTransform(k=k, distance=Euclidean(), init=:kmeans_plus_plus)
        t2 = KMeansTransform(k=k, distance=Euclidean(), init=:kmeans_plus_plus)

        fit!(t1, X; rng=MersenneTwister(1))
        fit!(t2, X; rng=MersenneTwister(2))

        @test t1.centroids != t2.centroids
    end

    @testset "Reproducibility with subsample_size + :random init" begin
        t1 = KMeansTransform(k=k, distance=Euclidean(), init=:random,
                             subsample_size=80)
        t2 = KMeansTransform(k=k, distance=Euclidean(), init=:random,
                             subsample_size=80)

        fit!(t1, X; rng=MersenneTwister(7))
        fit!(t2, X; rng=MersenneTwister(7))

        @test t1.centroids == t2.centroids
    end
end
