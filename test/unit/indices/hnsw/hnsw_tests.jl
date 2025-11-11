using Test
using Random
using ManifoldANN

@testset "HNSWIndex basic build/query" begin
    rng = MersenneTwister(2025)
    data = randn(rng, 4, 60)
    hnsw = build_index(HNSWIndex, data; M = 8, ef_construction = 80, ef_search = 50, rng = rng)
    brute = build_index(BruteForceIndex, data)

    for k in 1:5
        recalls = Float64[]
        for _ in 1:10
            q = randn(rng, 4)
            approx = query(hnsw, data, q, k; ef_search = 80)
            truth = query(brute, data, q, k)
            @test length(approx) == k
            overlap = length(intersect(Set(approx), Set(truth))) / k
            push!(recalls, overlap)
        end
        avg_recall = sum(recalls) / length(recalls)
        @test avg_recall >= 0.6
    end
end

@testset "HNSW mutation and dimension checks" begin
    rng = MersenneTwister(7)
    data = randn(rng, 3, 5)
    hnsw = build_index(HNSWIndex, data; M = 4, ef_construction = 20, ef_search = 20, rng = rng)

    new_point = randn(rng, 3)
    data = hcat(data, new_point)
    insert!(hnsw, data, new_point; rng = rng)
    @test hnsw.n_points == 6
    @test supports_mutation(hnsw)

    bad_point = randn(rng, 2)
    @test_throws DimensionMismatch insert!(hnsw, data, bad_point)

    bad_data = randn(rng, 4, 6)
    @test_throws DimensionMismatch query(hnsw, bad_data, randn(rng, 4), 2)
end
