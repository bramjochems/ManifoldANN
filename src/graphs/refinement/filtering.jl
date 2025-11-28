using LinearAlgebra

"""
    filter_graph(graph, data; curvature_threshold=0.0, solver=FastMatchingSolver(),
                 fallback_solver=GenericOTSolver(), distance_fn=nothing,
                 min_neighbors=1, verbose=false)

Filter a kNN graph by removing edges with curvature below threshold.
"""
function filter_graph(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    curvature_threshold::Real=0.0,
    solver::AbstractCurvatureSolver=FastMatchingSolver(),
    fallback_solver::AbstractCurvatureSolver=GenericOTSolver(),
    distance_fn::Union{Nothing,Function}=nothing,
    min_neighbors::Int=1,
    verbose::Bool=false
) where {T}
    n_nodes = length(graph)
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be >= 1"))

    dist_fn = distance_fn === nothing ? (i, j) -> norm(data[:, i] - data[:, j]) : distance_fn

    verbose && println("Building node neighborhoods...")
    neighborhoods = Dict{Int,NodeNeighborhood{Float64}}(
        i => uniform_neighborhood(i, graph[i], Float64) for i in 1:n_nodes
    )

    verbose && println("Computing edge curvatures...")
    edge_curvatures = Dict{Tuple{Int,Int},CurvatureResult{Float64}}()

    for x in 1:n_nodes
        for y in graph[x]
            haskey(edge_curvatures, (x, y)) && continue

            edge_dist = dist_fn(x, y)
            edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], edge_dist)

            active_solver = can_handle(solver, edge_view) ? solver : fallback_solver
            result = compute_curvature(active_solver, edge_view, dist_fn)
            edge_curvatures[(x, y)] = result

            if y in graph[x] && x in graph[y]
                edge_curvatures[(y, x)] = CurvatureResult{Float64}(
                    y, x, result.curvature, result.wasserstein_distance,
                    result.edge_distance, result.solver_type
                )
            end
        end
        verbose && (x % 100 == 0) && println("  Processed $x / $n_nodes nodes")
    end

    verbose && println("Filtering edges...")
    filtered_neighbors = Vector{Vector{Int}}(undef, n_nodes)
    n_removed = 0

    for i in 1:n_nodes
        neighbor_curvatures = [
            (j, edge_curvatures[(i, j)].curvature)
            for j in graph[i] if haskey(edge_curvatures, (i, j))
        ]

        sort!(neighbor_curvatures, by=x -> x[2], rev=true)

        kept = Int[]
        for (j, curv) in neighbor_curvatures
            if curv >= curvature_threshold || length(kept) < min_neighbors
                push!(kept, j)
            else
                n_removed += 1
            end
        end
        filtered_neighbors[i] = kept
    end

    verbose && println("Removed $n_removed edges total")

    new_k = maximum(length(neighbors) for neighbors in filtered_neighbors)
    KNNGraph(filtered_neighbors, new_k, graph.include_self, graph.metadata)
end

"""
    compute_all_curvatures(graph, data; solver=FastMatchingSolver(),
                          fallback_solver=GenericOTSolver(), distance_fn=nothing)

Compute curvatures for all edges without filtering.
"""
function compute_all_curvatures(
    graph::KNNGraph,
    data::AbstractMatrix{T};
    solver::AbstractCurvatureSolver=FastMatchingSolver(),
    fallback_solver::AbstractCurvatureSolver=GenericOTSolver(),
    distance_fn::Union{Nothing,Function}=nothing
) where {T}
    n_nodes = length(graph)
    dist_fn = distance_fn === nothing ? (i, j) -> norm(data[:, i] - data[:, j]) : distance_fn

    neighborhoods = Dict{Int,NodeNeighborhood{Float64}}(
        i => uniform_neighborhood(i, graph[i], Float64) for i in 1:n_nodes
    )

    results = Dict{Tuple{Int,Int},CurvatureResult{Float64}}()

    for x in 1:n_nodes
        for y in graph[x]
            haskey(results, (x, y)) && continue

            edge_dist = dist_fn(x, y)
            edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], edge_dist)

            active_solver = can_handle(solver, edge_view) ? solver : fallback_solver
            results[(x, y)] = compute_curvature(active_solver, edge_view, dist_fn)
        end
    end

    return results
end

"""
    curvature_statistics(curvatures)

Compute summary statistics of curvature distribution.
"""
function curvature_statistics(curvatures::Dict{Tuple{Int,Int}, CurvatureResult{T}}) where {T}
    curv_values = [result.curvature for result in Base.values(curvatures)]

    if isempty(curv_values)
        return (mean=NaN, std=NaN, min=NaN, max=NaN, median=NaN, n_positive=0, n_negative=0)
    end

    sorted_values = sort(curv_values)
    n = length(curv_values)

    (
        mean = sum(curv_values) / n,
        std = sqrt(sum((v - sum(curv_values)/n)^2 for v in curv_values) / n),
        min = minimum(curv_values),
        max = maximum(curv_values),
        median = n % 2 == 1 ? sorted_values[div(n, 2) + 1] : (sorted_values[div(n, 2)] + sorted_values[div(n, 2) + 1]) / 2,
        n_positive = count(v -> v > 0, curv_values),
        n_negative = count(v -> v < 0, curv_values)
    )
end
