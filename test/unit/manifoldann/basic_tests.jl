using Test
using ManifoldANN

@testset "Module loads" begin
    @test isdefined(ManifoldANN, :AbstractANNIndex)
    @test isdefined(ManifoldANN, :BruteForceIndex)
end
