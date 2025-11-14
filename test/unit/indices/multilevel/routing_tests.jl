using Test
using ManifoldANN: AbstractANNIndex, KMeansAssignment, TopKRouting, select_indices

struct DummyRoutingIndex <: AbstractANNIndex
    id::Int
end

@testset "TopKRouting skips missing buckets" begin
    indices = [DummyRoutingIndex(i) for i in 1:5]
    routing = TopKRouting(3)

    # Distances favor buckets that have no corresponding child indices
    distances = Float32[0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.2]
    assignment = KMeansAssignment(distances)

    selected = select_indices(routing, assignment, indices)

    @test length(selected) == 3
    @test selected == [5, 4, 3]
end

@testset "TopKRouting clamps to available children" begin
    indices = [DummyRoutingIndex(i) for i in 1:4]
    routing = TopKRouting(10)  # Request more probes than children present

    distances = Float32[0.4, 0.3, 0.2, 0.1]
    assignment = KMeansAssignment(distances)

    selected = select_indices(routing, assignment, indices)

    @test length(selected) == 4
    @test selected == [4, 3, 2, 1]
end

@testset "TopKRouting handles empty indices" begin
    indices = DummyRoutingIndex[]
    routing = TopKRouting(3)

    distances = Float32[0.1, 0.2]
    assignment = KMeansAssignment(distances)

    selected = select_indices(routing, assignment, indices)

    @test selected isa Vector{Int}
    @test isempty(selected)
end

@testset "TopKRouting handles empty distances" begin
    indices = [DummyRoutingIndex(i) for i in 1:3]
    routing = TopKRouting(5)

    # Empty distances array - should throw
    distances = Float32[]
    assignment = KMeansAssignment(distances)

    @test_throws ArgumentError select_indices(routing, assignment, indices)
end
