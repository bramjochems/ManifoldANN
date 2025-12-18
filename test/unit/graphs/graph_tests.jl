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

    metadata = ["p$(i)" for i in 1:size(data, 2)]
    graph_with_metadata = build_knn_graph(index, data; k = 2, metadata = metadata)
    @test has_metadata(graph_with_metadata)
    @test graph_metadata(graph_with_metadata).node_metadata == metadata
    @test graph_metadata(graph_with_metadata).original_k == 2
    @test graph_metadata(graph_with_metadata).directed == true
    for i in 1:length(graph_with_metadata)
        @test node_metadata(graph_with_metadata, i) == metadata[i]
    end

    @test !has_metadata(graph)
    @test graph_metadata(graph).node_metadata === nothing
    @test graph_metadata(graph).original_k == 2
    @test_throws ArgumentError node_metadata(graph, 1)

    @test_throws ArgumentError build_knn_graph(index, data; k = 0)
    @test_throws ArgumentError build_knn_graph(index, data; k = 4, include_self = false)
    @test_throws ArgumentError build_knn_graph(index, data; k = 2, metadata = ["oops"])
end

@testset "Configured k validation" begin
    struct FixedKIndex <: AbstractANNIndex
        max_k::Int
    end

    ManifoldANN.configured_k(index::FixedKIndex) = index.max_k

    function ManifoldANN.query(index::FixedKIndex, data, q, k; kwargs...)
        n_points = size(data, 2)
        result_k = min(k, n_points)
        T = float(eltype(data))
        neighbors = Vector{Neighbor{T}}(undef, result_k)
        @inbounds for i in 1:result_k
            neighbors[i] = Neighbor{T}(i, zero(T))
        end
        return neighbors
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
