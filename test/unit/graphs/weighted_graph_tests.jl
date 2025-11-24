using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "WeightedKNNGraph construction" begin
    rng = MersenneTwister(42)

    @testset "basic construction from graph" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=5)

        method = PCAMethod(intrinsic_dim=2)
        wg = build_weighted_graph(method, graph, data)

        @test wg isa WeightedKNNGraph
        @test length(wg) == n
        @test length(wg.geometries) == n
        @test length(wg.edge_weights) == n
        @test all(length.(wg.edge_weights) .== 5)
    end

    @testset "convenience construction from index" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        wg = build_weighted_graph(method, index, data; k=5)

        @test wg isa WeightedKNNGraph
        @test length(wg) == n
        @test configured_k(wg) == 5
    end

    @testset "geometries are correct type" begin
        n = 15
        data = randn(rng, 4, n)

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=4)

        method = PCAMethod(intrinsic_dim=3)
        wg = build_weighted_graph(method, graph, data)

        @test all(g -> g isa PCAGeometry, wg.geometries)
        @test all(g -> intrinsic_dimension(g) == 3, wg.geometries)
    end

    @testset "edge weights are positive" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=5)

        method = PCAMethod(intrinsic_dim=2)
        wg = build_weighted_graph(method, graph, data)

        for i in 1:length(wg)
            for w in neighbor_weights(wg, i)
                @test w >= 0
            end
        end
    end
end

@testset "WeightedKNNGraph accessors" begin
    rng = MersenneTwister(123)
    n = 15
    k = 4
    data = randn(rng, 3, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)
    wg = build_weighted_graph(method, graph, data)

    @testset "node_geometry" begin
        for i in 1:n
            geom = node_geometry(wg, i)
            @test geom isa PCAGeometry
            @test center(geom) ≈ data[:, i]
        end
    end

    @testset "neighbors" begin
        for i in 1:n
            nbrs = neighbors(wg, i)
            @test length(nbrs) == k
            @test all(idx -> 1 <= idx <= n, nbrs)
            @test i ∉ nbrs  # self not included by default
        end
    end

    @testset "edge_weight" begin
        for i in 1:n
            for j in 1:k
                w = edge_weight(wg, i, j)
                @test w >= 0
                @test w isa Real
            end
        end
    end

    @testset "neighbor_weights" begin
        for i in 1:n
            weights = neighbor_weights(wg, i)
            @test length(weights) == k
            @test all(w -> w >= 0, weights)
        end
    end

    @testset "neighbors_with_weights" begin
        for i in 1:n
            pairs = collect(neighbors_with_weights(wg, i))
            @test length(pairs) == k

            for (neighbor_idx, weight) in pairs
                @test 1 <= neighbor_idx <= n
                @test weight >= 0
            end
        end
    end

    @testset "configured_k" begin
        @test configured_k(wg) == k
    end
end

@testset "WeightedKNNGraph Base interface" begin
    rng = MersenneTwister(456)
    n = 10
    k = 3
    data = randn(rng, 2, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)
    wg = build_weighted_graph(method, graph, data)

    @testset "length" begin
        @test length(wg) == n
    end

    @testset "getindex" begin
        for i in 1:n
            @test wg[i] == graph[i]
        end
    end

    @testset "iteration" begin
        count = 0
        for neighbor_list in wg
            count += 1
            @test neighbor_list isa Vector{Int}
            @test length(neighbor_list) == k
        end
        @test count == n
    end
end

@testset "WeightedKNNGraph metadata passthrough" begin
    rng = MersenneTwister(789)
    n = 10
    k = 3
    data = randn(rng, 2, n)
    labels = ["point_$i" for i in 1:n]

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k, metadata=labels)
    method = PCAMethod(intrinsic_dim=2)
    wg = build_weighted_graph(method, graph, data)

    @test has_metadata(wg)
    @test graph_metadata(wg) == labels

    for i in 1:n
        @test node_metadata(wg, i) == "point_$i"
    end
end

@testset "WeightedKNNGraph statistics" begin
    rng = MersenneTwister(111)
    n = 20
    k = 5
    data = randn(rng, 3, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)
    wg = build_weighted_graph(method, graph, data)

    @testset "total_edge_weight" begin
        total = total_edge_weight(wg)
        @test total >= 0

        # Manual calculation
        manual_total = sum(sum(wg.edge_weights[i]) for i in 1:n)
        @test total ≈ manual_total
    end

    @testset "mean_edge_weight" begin
        mean_w = mean_edge_weight(wg)
        @test mean_w >= 0
        @test mean_w ≈ total_edge_weight(wg) / (n * k)
    end

    @testset "edge_weight_statistics" begin
        stats = edge_weight_statistics(wg)

        @test stats.min >= 0
        @test stats.max >= stats.min
        @test stats.mean >= 0
        @test stats.total >= 0
        @test stats.n_edges == n * k
    end
end

@testset "WeightedKNNGraph integration" begin
    rng = MersenneTwister(222)

    @testset "2D plane in 3D" begin
        # Points on a 2D plane
        n = 30
        xy = randn(rng, 2, n)
        z = 0.5 .* xy[1, :] .+ 0.3 .* xy[2, :]
        data = vcat(xy, z')

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=8)
        method = PCAMethod(intrinsic_dim=2)
        wg = build_weighted_graph(method, graph, data)

        @test length(wg) == n

        # All geometries should be 2D
        for i in 1:n
            @test intrinsic_dimension(node_geometry(wg, i)) == 2
        end
    end

    @testset "circle manifold" begin
        # Points on a circle (1D manifold in 2D)
        n = 50
        t = range(0, 2π, length=n+1)[1:end-1]
        data = vcat(cos.(t)', sin.(t)')

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=6)
        method = PCAMethod(intrinsic_dim=1)
        wg = build_weighted_graph(method, graph, data)

        @test length(wg) == n

        # All geometries should be 1D
        for i in 1:n
            @test intrinsic_dimension(node_geometry(wg, i)) == 1
        end

        # Edge weights should be reasonable (arc length approximation)
        # Adjacent points on circle should have small weights
        stats = edge_weight_statistics(wg)
        @test stats.min > 0
        @test stats.max < 2.0  # Max distance on unit circle
    end

    @testset "different index types" begin
        n = 30
        data = randn(rng, 3, n)
        method = PCAMethod(intrinsic_dim=2)

        # Test with BruteForceIndex
        bf_index = build_index(BruteForceIndex, data)
        bf_graph = build_knn_graph(bf_index, data; k=5)
        bf_wg = build_weighted_graph(method, bf_graph, data)
        @test length(bf_wg) == n

        # Test with HNSWIndex
        hnsw_index = build_index(HNSWIndex, data; M=8, ef_construction=50)
        hnsw_graph = build_knn_graph(hnsw_index, data; k=5)
        hnsw_wg = build_weighted_graph(method, hnsw_graph, data)
        @test length(hnsw_wg) == n
    end
end
