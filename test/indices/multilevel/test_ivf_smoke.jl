using Test
using ManifoldANN
using Distances
using LinearAlgebra

@testset "IVF Smoke Test" begin
    # Small dataset for quick testing
    n_points = 100
    n_dims = 10
    k_clusters = 5
    k_neighbors = 3

    # Generate random data
    X = rand(Float32, n_dims, n_points)

    @testset "Build IVF index with HNSW" begin
        # Create IVF configuration: KMeans(5) → HNSW per cluster
        config = TransformedConfig(
            KMeansTransform(k=k_clusters, distance=Euclidean(), init=:kmeans_plus_plus),
            TopKRouting(2),  # Probe 2 nearest clusters
            TerminalConfig(HNSWIndex, (M=8, ef_construction=50))
        )

        # Build index
        index = build_index(MultiLevelIndex, X, config)

        @test index isa MultiLevelIndex
        @test index.root isa TransformedIndex
        @test length(index.root.indices) == k_clusters

        # Verify each child is an HNSW index
        for child in index.root.indices
            @test child isa HNSWIndex
        end
    end

    @testset "Query IVF index" begin
        config = TransformedConfig(
            KMeansTransform(k=k_clusters, distance=Euclidean(), init=:kmeans_plus_plus),
            TopKRouting(2),
            TerminalConfig(HNSWIndex, (M=8, ef_construction=50))
        )

        index = build_index(MultiLevelIndex, X, config)

        # Query with a random point
        q = rand(Float32, n_dims)
        neighbors = query(index, X, q, k_neighbors)

        @test length(neighbors) <= k_neighbors
        @test all(1 .<= neighbors .<= n_points)
        @test length(unique(neighbors)) == length(neighbors)  # No duplicates
    end

    @testset "Identity transform pass-through" begin
        # Identity → Exhaustive → HNSW (should behave like plain HNSW)
        config = TransformedConfig(
            IdentityTransform(),
            ExhaustiveRouting(),
            TerminalConfig(HNSWIndex, (M=8, ef_construction=50))
        )

        index = build_index(MultiLevelIndex, X, config)

        @test index isa MultiLevelIndex
        @test length(index.root.indices) == 1  # Single child

        q = rand(Float32, n_dims)
        neighbors = query(index, X, q, k_neighbors)

        @test length(neighbors) <= k_neighbors
        @test all(1 .<= neighbors .<= n_points)
    end

    @testset "Two-level hierarchy" begin
        # KMeans(3) → Identity → HNSW
        config = TransformedConfig(
            KMeansTransform(k=3, distance=Euclidean(), init=:random),
            TopKRouting(2),
            TransformedConfig(
                IdentityTransform(),
                ExhaustiveRouting(),
                TerminalConfig(HNSWIndex, (M=8, ef_construction=50))
            )
        )

        index = build_index(MultiLevelIndex, X, config)

        @test index isa MultiLevelIndex
        @test index.root isa TransformedIndex
        @test length(index.root.indices) == 3

        # Each child should also be a TransformedIndex
        for child in index.root.indices
            @test child isa TransformedIndex
            @test length(child.indices) == 1
            @test child.indices[1] isa HNSWIndex
        end

        q = rand(Float32, n_dims)
        neighbors = query(index, X, q, k_neighbors)

        @test length(neighbors) <= k_neighbors
    end

    @testset "build_ivf_hnsw_index convenience" begin
        ivf = build_ivf_hnsw_index(
            X;
            nlist = 4,
            routing_k = 2,
            kmeans_distance = Euclidean(),
            hnsw_M = 8,
            hnsw_ef_construction = 50,
            hnsw_ef_search = 24,
        )

        @test ivf isa MultiLevelIndex
        q = rand(Float32, n_dims)
        neighbors = query(ivf, X, q, k_neighbors)
        @test length(neighbors) <= k_neighbors

        queries = rand(Float32, n_dims, 5)
        batch = query(ivf, X, queries, k_neighbors)
        @test length(batch) == size(queries, 2)
        @test all(length(ans) <= k_neighbors for ans in batch)

        vec_queries = [rand(Float32, n_dims) for _ in 1:3]
        vec_batch = query(ivf, X, vec_queries, k_neighbors)
        @test length(vec_batch) == length(vec_queries)
    end
end
