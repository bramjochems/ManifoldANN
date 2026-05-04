using Test
using ManifoldANN
using LinearAlgebra

# Direct unit tests for `compute_shortest_paths`. Built on hand-constructable
# graphs where shortest-path values can be derived by hand, so a future swap
# (e.g. Floyd-Warshall → per-source Dijkstra) has an algorithm-independent
# correctness check rather than just "agrees with itself on random data".
#
# `compute_shortest_paths` is internal — accessed via the qualified name.

const csp = ManifoldANN.compute_shortest_paths

# Build a KNNGraph directly from an adjacency list, bypassing the index +
# build_knn_graph path. Lets us hand-design exactly the edges we want.
function _make_graph(adjacency::Vector{Vector{Int}}; directed::Bool, original_k::Int)
    n = length(adjacency)
    k = directed ? original_k : maximum(length, adjacency; init = 0)
    metadata = (original_k = original_k, directed = directed, node_metadata = nothing)
    return ManifoldANN.KNNGraph(adjacency, k, false, metadata)
end

@testset "compute_shortest_paths" begin
    @testset ":unit weights — directed line graph" begin
        # 1 → 2 → 3 → 4 → 5 (one-way chain)
        adj = [[2], [3], [4], [5], Int[]]
        g = _make_graph(adj; directed = true, original_k = 1)
        data = Float64.(reshape(1:5, 1, 5))  # 1 × 5

        dist = csp(g, data, :unit)

        @test size(dist) == (5, 5)
        for i in 1:5
            @test dist[i, i] == 0.0
        end
        # Forward distances = hop counts
        @test dist[1, 2] == 1.0
        @test dist[1, 3] == 2.0
        @test dist[1, 5] == 4.0
        @test dist[2, 5] == 3.0
        # Reverse direction has no edges → unreachable
        @test dist[5, 1] == Inf
        @test dist[3, 1] == Inf
    end

    @testset ":unit weights — undirected line graph" begin
        # Same chain but undirected: every forward edge implies a reverse edge.
        adj = [[2], [1, 3], [2, 4], [3, 5], [4]]
        g = _make_graph(adj; directed = false, original_k = 1)
        data = Float64.(reshape(1:5, 1, 5))

        dist = csp(g, data, :unit)

        @test dist[1, 5] == 4.0
        @test dist[5, 1] == 4.0  # reachable now
        @test dist[2, 4] == 2.0
        @test dist[4, 2] == 2.0
        @test issymmetric(dist)
    end

    @testset ":euclidean weights — undirected triangle (triangle inequality)" begin
        # Three points in 1D: 0, 3, 5. Edges: 1-2, 2-3 (no direct 1-3).
        # Shortest 1↔3 must go via 2: distance = 3 + 2 = 5.
        adj = [[2], [1, 3], [2]]
        g = _make_graph(adj; directed = false, original_k = 1)
        data = reshape(Float64[0.0, 3.0, 5.0], 1, 3)

        dist = csp(g, data, :euclidean)

        @test dist[1, 2] ≈ 3.0
        @test dist[2, 3] ≈ 2.0
        @test dist[1, 3] ≈ 5.0
        @test dist[3, 1] ≈ 5.0
        @test issymmetric(dist)
    end

    @testset ":euclidean weights — undirected square with diagonal shortcut" begin
        # 4 corners of a unit square + a node at the centre. Adjacencies set so
        # going round the perimeter is longer than going through the centre.
        # Layout: 1=(0,0), 2=(1,0), 3=(1,1), 4=(0,1), 5=(0.5,0.5).
        # Edges: perimeter (1-2-3-4-1) and every corner to centre.
        adj = [
            [2, 4, 5],
            [1, 3, 5],
            [2, 4, 5],
            [1, 3, 5],
            [1, 2, 3, 4],
        ]
        g = _make_graph(adj; directed = false, original_k = 3)
        data = Float64[
            0.0 1.0 1.0 0.0 0.5;
            0.0 0.0 1.0 1.0 0.5
        ]

        dist = csp(g, data, :euclidean)

        # Adjacent corners share an edge (length 1)
        @test dist[1, 2] ≈ 1.0
        @test dist[2, 3] ≈ 1.0
        # Diagonal: direct edge does not exist; via centre = 2 * sqrt(0.5) ≈ √2;
        # via perimeter = 2.0. Shortest path picks the centre route.
        @test dist[1, 3] ≈ sqrt(2) atol = 1e-12
        @test dist[2, 4] ≈ sqrt(2) atol = 1e-12
        @test issymmetric(dist)
    end

    @testset ":unit weights — disconnected components stay at Inf" begin
        # Two disjoint edges: {1-2} and {3-4}. No path between components.
        adj = [[2], [1], [4], [3]]
        g = _make_graph(adj; directed = false, original_k = 1)
        data = Float64.(reshape(1:4, 1, 4))

        dist = csp(g, data, :unit)

        @test dist[1, 2] == 1.0
        @test dist[3, 4] == 1.0
        @test dist[1, 3] == Inf
        @test dist[2, 4] == Inf
        @test dist[1, 4] == Inf
    end

    @testset ":normalized weights — agree with effective_epsilon on direct edges" begin
        # `:normalized` uses `effective_epsilon(i, j, graph, data)` per edge.
        # Hand-deriving eff-eps values is fragile (depends on profile +
        # slice rules), so the test instead asserts the contract that
        # `compute_shortest_paths(:normalized)` threads through:
        #   - direct edge (i, j): dist[i, j] == effective_epsilon(i, j)
        #   - two-hop path:       dist[i, k] == eff_eps(i, j) + eff_eps(j, k)
        #     (when no shorter route exists)
        # This still catches a Dijkstra/FW divergence — both must respect
        # the same per-edge weights and the same path-summation semantics.
        adj = [[2], [1, 3], [2, 4], [3]]
        g = _make_graph(adj; directed = false, original_k = 1)
        # Spread points unevenly so eff_eps differs across edges.
        data = reshape(Float64[0.0, 1.0, 3.0, 7.0], 1, 4)

        dist = csp(g, data, :normalized)
        e_fn = ManifoldANN.effective_epsilon

        # Direct edges: shortest path = the edge weight itself.
        @test isapprox(dist[1, 2], e_fn(1, 2, g, data); atol = 1e-12)
        @test isapprox(dist[2, 3], e_fn(2, 3, g, data); atol = 1e-12)
        @test isapprox(dist[3, 4], e_fn(3, 4, g, data); atol = 1e-12)

        # Two-hop: 1↔3 must equal eff_eps(1,2) + eff_eps(2,3).
        @test isapprox(
            dist[1, 3],
            e_fn(1, 2, g, data) + e_fn(2, 3, g, data);
            atol = 1e-12,
        )
        # Three-hop: 1↔4 = sum along the chain.
        @test isapprox(
            dist[1, 4],
            e_fn(1, 2, g, data) + e_fn(2, 3, g, data) + e_fn(3, 4, g, data);
            atol = 1e-12,
        )
        # Symmetry on undirected graph.
        @test issymmetric(dist)
        # Self-distance is 0.
        for i in 1:4
            @test dist[i, i] == 0.0
        end
    end

    @testset "unknown weight_type errors" begin
        adj = [[2], [1]]
        g = _make_graph(adj; directed = false, original_k = 1)
        data = reshape(Float64[0.0, 1.0], 1, 2)
        @test_throws ErrorException csp(g, data, :not_a_real_weight)
    end
end
