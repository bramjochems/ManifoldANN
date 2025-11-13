using Test
using Random
using ManifoldANN

@inline function l1_distance(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    dist = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        dist += abs(x[i] - y[i])
    end
    return dist
end

@testset "MultiLevelIndex respects child distances" begin
    rng = MersenneTwister(123)
    data = rand(rng, Float32, 8, 64)

    config = TransformedConfig(
        IdentityTransform(),
        ExhaustiveRouting(),
        TerminalConfig(BruteForceIndex, (distance = l1_distance,))
    )

    multi = build_index(MultiLevelIndex, data, config; distance = l1_distance)
    brute = build_index(BruteForceIndex, data; distance = l1_distance)

    for trial in 1:5
        q = rand(rng, Float32, size(data, 1))
        multi_neighbors = query(multi, data, q, 7)
        brute_neighbors = query(brute, data, q, 7)
        @test multi_neighbors == brute_neighbors
    end
end
