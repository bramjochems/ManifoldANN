using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "Graph Refinement" begin
    @testset "NodeNeighborhood construction" begin
        # Basic construction
        neighbors = [2, 3, 4, 5]
        probs = fill(0.25, 4)
        node_nb = NodeNeighborhood(1, neighbors, probs)

        @test node_nb.node_id == 1
        @test node_nb.neighbors == neighbors
        @test node_nb.probabilities == probs

        # Uniform neighborhood helper
        node_nb = uniform_neighborhood(1, neighbors, Float64)
        @test all(p ≈ 0.25 for p in node_nb.probabilities)
        @test sum(node_nb.probabilities) ≈ 1.0

        # Invalid: probabilities don't sum to 1
        @test_throws ArgumentError NodeNeighborhood(1, neighbors, [0.1, 0.2, 0.3, 0.3])

        # Invalid: length mismatch
        @test_throws ArgumentError NodeNeighborhood(1, neighbors, [0.5, 0.5])
    end

    @testset "EdgeNeighborhoodView creation" begin
        # Create two neighborhoods
        x_neighbors = [3, 4, 5, 6]
        y_neighbors = [4, 5, 7, 8]

        x_nb = uniform_neighborhood(1, x_neighbors, Float64)
        y_nb = uniform_neighborhood(2, y_neighbors, Float64)

        # Create edge view
        edge_dist = 1.5
        edge_view = create_edge_view(x_nb, y_nb, edge_dist)

        @test edge_view.x_id == 1
        @test edge_view.y_id == 2
        @test edge_view.edge_distance == 1.5

        # Check decomposition
        @test Set(edge_view.shared) == Set([4, 5])
        @test Set(edge_view.unique_x) == Set([3, 6])
        @test Set(edge_view.unique_y) == Set([7, 8])

        # Check probabilities
        @test edge_view.x_probs[3] ≈ 0.25
        @test edge_view.y_probs[7] ≈ 0.25
    end

    @testset "CurvatureResult" begin
        result = CurvatureResult{Float64}(
            1, 2,
            0.5,  # curvature
            0.75,  # wasserstein
            1.5,  # edge distance
            :fast_matching
        )

        @test result.x_id == 1
        @test result.y_id == 2
        @test result.curvature == 0.5
        @test is_positive_curvature(result)
        @test passes_threshold(result, 0.0)
        @test passes_threshold(result, 0.5)
        @test !passes_threshold(result, 0.6)

        # Negative curvature
        result_neg = CurvatureResult{Float64}(1, 2, -0.2, 1.2, 1.0, :brute_force)
        @test !is_positive_curvature(result_neg)
        @test !passes_threshold(result_neg, 0.0)
    end

    @testset "HungarianSolver - can_handle" begin
        solver = HungarianSolver()

        # Case 1: Same degree, uniform measures - CAN handle
        x_neighbors = [3, 4, 5, 6]
        y_neighbors = [5, 6, 7, 8]
        x_nb = uniform_neighborhood(1, x_neighbors, Float64)
        y_nb = uniform_neighborhood(2, y_neighbors, Float64)
        edge_view = create_edge_view(x_nb, y_nb, 1.0)

        @test can_handle(solver, edge_view)

        # Case 2: Different degrees - CANNOT handle
        x_neighbors = [3, 4, 5]
        y_neighbors = [5, 6, 7, 8]
        x_nb = uniform_neighborhood(1, x_neighbors, Float64)
        y_nb = uniform_neighborhood(2, y_neighbors, Float64)
        edge_view = create_edge_view(x_nb, y_nb, 1.0)

        @test !can_handle(solver, edge_view)

        # Case 3: Same degree, non-uniform measures - CANNOT handle
        x_neighbors = [3, 4, 5, 6]
        y_neighbors = [5, 6, 7, 8]
        x_probs = [0.1, 0.2, 0.3, 0.4]
        y_probs = fill(0.25, 4)
        x_nb = NodeNeighborhood(1, x_neighbors, x_probs)
        y_nb = NodeNeighborhood(2, y_neighbors, y_probs)
        edge_view = create_edge_view(x_nb, y_nb, 1.0)

        @test !can_handle(solver, edge_view)
    end

    @testset "Curvature computation - simple case" begin
        # Create a simple metric: nodes on a line with unit spacing
        # Nodes: 1 -- 2 -- 3 -- 4 -- 5
        # Distance function: |i - j|
        distance_fn = (i, j) -> abs(Float64(i - j))

        # Consider edge (2, 3)
        # N(2) = [1, 3, 4] (k=3)
        # N(3) = [2, 4, 5] (k=3)
        x_neighbors = [1, 3, 4]
        y_neighbors = [2, 4, 5]

        x_nb = uniform_neighborhood(2, x_neighbors, Float64)
        y_nb = uniform_neighborhood(3, y_neighbors, Float64)

        edge_dist = distance_fn(2, 3)  # = 1.0
        edge_view = create_edge_view(x_nb, y_nb, edge_dist)

        # Decomposition:
        # shared: [4]
        # unique_x: [1, 3]
        # unique_y: [2, 5]
        @test Set(edge_view.shared) == Set([4])
        @test Set(edge_view.unique_x) == Set([1, 3])
        @test Set(edge_view.unique_y) == Set([2, 5])

        # Use HungarianSolver
        solver = HungarianSolver()
        @test can_handle(solver, edge_view)

        result = compute_curvature(solver, edge_view, distance_fn)

        @test result isa CurvatureResult{Float64}
        @test result.x_id == 2
        @test result.y_id == 3
        @test result.edge_distance ≈ 1.0
        @test result.solver_type == :hungarian  # FastMatchingSolver now returns HungarianSolver

        # Wasserstein distance computation:
        # Shared: node 4 has zero cost
        # Unique matching:
        #   1 -> 2: cost 1.0
        #   3 -> 5: cost 2.0
        # or
        #   1 -> 5: cost 4.0
        #   3 -> 2: cost 1.0
        # Best is 1->2, 3->5 with total cost 3.0
        # Scaled by 1/3: W1 = 1.0
        # Curvature: κ = 1 - 1.0/1.0 = 0.0
        @test result.wasserstein_distance ≈ 1.0
        @test result.curvature ≈ 0.0
    end

    @testset "GenericOTSolver" begin
        # Test with non-uniform measures
        x_neighbors = [3, 4, 5]
        y_neighbors = [4, 6, 7]
        x_probs = [0.5, 0.3, 0.2]
        y_probs = [0.4, 0.35, 0.25]

        x_nb = NodeNeighborhood(1, x_neighbors, x_probs)
        y_nb = NodeNeighborhood(2, y_neighbors, y_probs)

        distance_fn = (i, j) -> abs(Float64(i - j))
        edge_dist = distance_fn(1, 2)
        edge_view = create_edge_view(x_nb, y_nb, edge_dist)

        # Use GenericOTSolver with greedy method
        solver = GenericOTSolver(method=:greedy)
        result = compute_curvature(solver, edge_view, distance_fn)

        @test result isa CurvatureResult{Float64}
        @test result.solver_type == :greedy  # GenericOTSolver now returns specific solver type
        @test result.wasserstein_distance >= 0
        @test result.edge_distance == edge_dist
    end

    @testset "GenericOTSolver - Sinkhorn method" begin
        # Test Sinkhorn algorithm
        x_neighbors = [3, 4, 5]
        y_neighbors = [4, 6, 7]
        x_probs = [0.5, 0.3, 0.2]
        y_probs = [0.4, 0.35, 0.25]

        x_nb = NodeNeighborhood(1, x_neighbors, x_probs)
        y_nb = NodeNeighborhood(2, y_neighbors, y_probs)

        distance_fn = (i, j) -> abs(Float64(i - j))
        edge_dist = distance_fn(1, 2)
        edge_view = create_edge_view(x_nb, y_nb, edge_dist)

        # Use Sinkhorn solver
        solver_sinkhorn = GenericOTSolver(method=:sinkhorn, sinkhorn_reg=0.01)
        result_sinkhorn = compute_curvature(solver_sinkhorn, edge_view, distance_fn)

        # Compare with LP (exact) solver
        solver_lp = GenericOTSolver(method=:lp)
        result_lp = compute_curvature(solver_lp, edge_view, distance_fn)

        @test result_sinkhorn isa CurvatureResult{Float64}
        @test result_sinkhorn.solver_type == :sinkhorn  # GenericOTSolver now returns specific solver type
        @test result_sinkhorn.wasserstein_distance >= 0

        # Sinkhorn should be close to exact LP solution (within regularization tolerance)
        @test abs(result_sinkhorn.curvature - result_lp.curvature) < 0.01
        @test abs(result_sinkhorn.wasserstein_distance - result_lp.wasserstein_distance) < 0.01
    end

    @testset "filter_graph - basic functionality" begin
        rng = MersenneTwister(42)

        # Create simple data: points on a 2D grid
        n = 20
        data = Float64.(hcat([[i, j] for i in 1:4 for j in 1:5]...))

        # Build a kNN graph
        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=5)

        @test length(graph) == n
        @test graph.k == 5

        # Filter with threshold 0.0 (remove negative curvature edges)
        filtered = filter_graph(
            graph, data,
            curvature_threshold=0.0,
            min_neighbors=2,
            verbose=false
        )

        @test filtered isa KNNGraph
        @test length(filtered) == n

        # Each node should have at least min_neighbors
        for i in 1:n
            @test length(filtered[i]) >= 2
        end

        # Filtered graph should generally have fewer edges
        total_edges_orig = sum(length(graph[i]) for i in 1:n)
        total_edges_filtered = sum(length(filtered[i]) for i in 1:n)

        # This is probabilistic but usually holds for random graphs
        # (some edges will have negative curvature)
        @test total_edges_filtered <= total_edges_orig
    end

    @testset "compute_all_curvatures" begin
        rng = MersenneTwister(43)

        # Small graph for testing
        n = 10
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=4)

        curvatures = compute_all_curvatures(graph, data)

        @test curvatures isa Dict{Tuple{Int,Int}, CurvatureResult{Float64}}
        @test !isempty(curvatures)

        # Check that we have results for edges in the graph
        for i in 1:n
            for j in graph[i]
                @test haskey(curvatures, (i, j))
            end
        end

        # Test statistics
        stats = curvature_statistics(curvatures)
        @test !isnan(stats.mean)
        @test !isnan(stats.std)
        @test !isnan(stats.min)
        @test !isnan(stats.max)
        @test stats.min <= stats.mean <= stats.max
        @test stats.n_positive + stats.n_negative == length(curvatures)
    end

    @testset "curvature_statistics - edge cases" begin
        # Empty curvatures
        empty_curvatures = Dict{Tuple{Int,Int}, CurvatureResult{Float64}}()
        stats = curvature_statistics(empty_curvatures)

        @test isnan(stats.mean)
        @test isnan(stats.std)
        @test stats.n_positive == 0
        @test stats.n_negative == 0
    end

    @testset "_create_precomputed_distance_fn - eltype propagation (Float32)" begin
        # Build a Float32 EdgeNeighborhoodView and verify the precomputed
        # distance closure returns Float32 values (not promoted to Float64).
        x_neighbors = [3, 4, 5, 6]
        y_neighbors = [4, 5, 7, 8]

        x_nb = uniform_neighborhood(1, x_neighbors, Float32)
        y_nb = uniform_neighborhood(2, y_neighbors, Float32)
        edge_view = create_edge_view(x_nb, y_nb, 1.5f0)

        @test edge_view isa ManifoldANN.EdgeNeighborhoodView{Float32}

        # Stub data and a Float32-returning dist_fn. The precomputed matrix
        # should be Matrix{Float32}; the closure should return Float32.
        data = randn(Float32, 3, 10)
        dist_fn = (i::Int, j::Int) -> norm(data[:, i] - data[:, j])

        pre_fn = ManifoldANN._create_precomputed_distance_fn(edge_view, data, dist_fn)

        # In-neighborhood lookups return Float32 with the matched value.
        for ni in x_neighbors, nj in y_neighbors
            d = pre_fn(ni, nj)
            @test d isa Float32
            @test d ≈ Float32(dist_fn(ni, nj))
        end
    end

    @testset "_create_precomputed_distance_fn - errors on out-of-neighborhood node" begin
        # The closure must raise rather than silently fall back to dist_fn,
        # so that an upstream logic bug surfaces immediately.
        x_nb = uniform_neighborhood(1, [3, 4], Float64)
        y_nb = uniform_neighborhood(2, [4, 5], Float64)
        edge_view = create_edge_view(x_nb, y_nb, 1.0)

        data = randn(2, 10)
        dist_fn = (i::Int, j::Int) -> norm(data[:, i] - data[:, j])

        pre_fn = ManifoldANN._create_precomputed_distance_fn(edge_view, data, dist_fn)

        # Querying a node outside the neighborhood raises.
        @test_throws ErrorException pre_fn(99, 4)
        @test_throws ErrorException pre_fn(3, 99)

        # In-neighborhood query still works.
        @test pre_fn(3, 4) ≈ dist_fn(3, 4)
    end
end
