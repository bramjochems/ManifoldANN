using Test
using ManifoldANN
using Distances
using Random
using LinearAlgebra

@testset "Multi-level hierarchy deepcopy fix" begin
    rng = MersenneTwister(999)
    data = randn(rng, Float32, 20, 200)

    # Two-level hierarchy: KMeans(5) → KMeans(3) → HNSW
    config = TransformedConfig(
        KMeansTransform(k=5, distance=Euclidean(), init=:random),
        TopKRouting(3),
        TransformedConfig(
            KMeansTransform(k=3, distance=Euclidean(), init=:random),
            TopKRouting(2),
            TerminalConfig(HNSWIndex, (M=8, ef_construction=50))
        )
    )

    println("Building two-level hierarchy...")
    index = build_index(MultiLevelIndex, data, config)

    @test index isa MultiLevelIndex
    @test length(index.root.indices) == 5

    # Verify each coarse cluster has its own fine clustering with different centroids
    println("\nVerifying each cluster has unique transform parameters:")
    first_centroids = nothing
    all_different = true

    for (i, child) in enumerate(index.root.indices)
        @test child isa TransformedIndex
        @test length(child.indices) == 3  # 3 fine clusters

        if i == 1
            first_centroids = copy(child.transform.centroids)
            println("  Cluster 1: baseline centroids stored")
        else
            # Verify centroids are different from first cluster
            same = all(first_centroids .== child.transform.centroids)
            @test !same  # Should NOT be the same (this would fail without deepcopy)
            if !same
                println("  Cluster $i: ✓ centroids differ from cluster 1")
            else
                println("  Cluster $i: ✗ centroids IDENTICAL to cluster 1 (BUG)")
                all_different = false
            end
        end
    end

    @test all_different

    # Test query functionality
    q = randn(rng, Float32, 20)
    neighbors = query(index, data, q, 5)
    neighbor_ids_list = neighbor_ids(neighbors)

    @test length(neighbor_ids_list) <= 5
    @test all(1 .<= neighbor_ids_list .<= 200)
    @test length(unique(neighbor_ids_list)) == length(neighbor_ids_list)  # No duplicates

    # Check reasonable recall
    brute = build_index(BruteForceIndex, data)
    truth = neighbor_ids(query(brute, data, q, 5))
    recall = length(intersect(Set(neighbor_ids_list), Set(truth))) / length(truth)
    println("\nQuery test: recall = $(round(recall * 100, digits=1))%")
    @test recall >= 0.2  # At least 20% recall (very lenient, just checking it works)

    println("\n✓ All deepcopy tests passed")
end
