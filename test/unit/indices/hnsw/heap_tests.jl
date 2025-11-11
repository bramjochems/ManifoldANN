using Test
using ManifoldANN

const NC = ManifoldANN.NeighborCandidate
const Heap = ManifoldANN.NeighborMinHeap

@testset "NeighborMinHeap maintains ordering" begin
    heap = Heap(NC(1, 0.4f0))
    push!(heap, NC(2, 0.2f0))
    push!(heap, NC(3, 0.8f0))
    push!(heap, NC(4, 0.6f0))

    popped = Float32[]
    while !isempty(heap)
        cand = popfirst!(heap)
        push!(popped, cand.dist)
    end

    @test popped == sort(popped)
end

@testset "NeighborMinHeap rejects empty pops" begin
    heap = Heap{Float32}()
    @test_throws ArgumentError popfirst!(heap)
end
