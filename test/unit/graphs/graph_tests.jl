using Test
using ManifoldANN

@testset "KNN graph construction" begin
    data = reshape(collect(1:12), 3, 4)
    index = build_index(BruteForceIndex, data)

    graph = build_knn_graph(index, data; k = 2)
    @test graph.k == 2
    @test graph.include_self === false
    @test length(graph) == size(data, 2)
    @test eltype(graph) == Vector{Int}
    @test collect(graph) == graph.neighbors

    for (i, neighbors) in enumerate(graph.neighbors)
        @test length(neighbors) == 2
        @test all(1 .<= neighbors .<= size(data, 2))
        @test !(i in neighbors)
    end

    enumerated = collect(enumerate(graph))
    @test all(first(pair) == idx && last(pair) === graph.neighbors[idx] for (idx, pair) in enumerate(enumerated))

    graph_with_self = build_knn_graph(index, data; k = 3, include_self = true)
    for (i, neighbors) in enumerate(graph_with_self.neighbors)
        @test length(neighbors) == 3
        @test any(==(i), neighbors)
    end

    @test_throws ArgumentError build_knn_graph(index, data; k = 0)
    @test_throws ArgumentError build_knn_graph(index, data; k = 4, include_self = false)
end

@testset "Configured k validation" begin
    struct FixedKIndex <: AbstractANNIndex
        max_k::Int
    end

    ManifoldANN.configured_k(index::FixedKIndex) = index.max_k

    function ManifoldANN.query(index::FixedKIndex, data, q, k; kwargs...)
        n_points = size(data, 2)
        return collect(1:min(k, n_points))
    end

    data = randn(2, 3)
    index = FixedKIndex(2)

    graph = build_knn_graph(index, data; k = 2)
    @test graph.k == 2

    @test_throws ArgumentError build_knn_graph(index, data; k = 3)

    # Omitting k should fall back to the configured value.
    graph2 = build_knn_graph(index, data)
    @test graph2.k == 2
end
