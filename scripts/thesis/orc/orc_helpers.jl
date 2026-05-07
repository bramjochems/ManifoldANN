# ==============================================================================
# ORC Experiment Shared Helpers
#
# Shared evaluation logic for the unified ORC experiment (experiment_orc.jl).
#
# Contents:
#   - MinHeap (binary min-heap for Dijkstra)
#   - label_shortcuts, evaluate_at_threshold, best_f1_threshold
#   - curvature_stats_by_label
#   - dijkstra_from, graph_to_adj_weights, prune_graph (legacy)
#   - find_bridges, oracle_prune, rank_prune, random_prune
#   - spearman_rank_correlation
#   - geodesic_error_at_pairs
#   - CSV writing helpers
#   - Constants (THRESHOLD_GRID, PRUNE_FRACTIONS, N_GEO_PAIRS)
# ==============================================================================

using Printf
using Dates
using Statistics

# ==============================================================================
# Constants
# ==============================================================================

# κ sweep for best-F1 search
const THRESHOLD_GRID = -1.0:0.05:1.0

# κ thresholds for pruning in downstream geodesic error analysis (legacy)
const PRUNE_THRESHOLDS = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Fraction-based pruning (rank-based and random)
const PRUNE_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]

# Number of random node pairs to sample for geodesic error comparison
const N_GEO_PAIRS = 500

# ==============================================================================
# Binary min-heap for Dijkstra (Float64 key, Int payload)
# ==============================================================================

"""Minimal binary min-heap: stores (key::Float64, idx::Int) pairs."""
mutable struct MinHeap
    data::Vector{Tuple{Float64, Int}}
end
MinHeap() = MinHeap(Tuple{Float64, Int}[])

@inline function _heap_up!(h::MinHeap, i::Int)
    while i > 1
        p = i >> 1
        if h.data[p][1] > h.data[i][1]
            h.data[p], h.data[i] = h.data[i], h.data[p]
            i = p
        else
            break
        end
    end
end

@inline function _heap_down!(h::MinHeap, i::Int)
    n = length(h.data)
    while true
        l, r = 2i, 2i + 1
        m = i
        l <= n && h.data[l][1] < h.data[m][1] && (m = l)
        r <= n && h.data[r][1] < h.data[m][1] && (m = r)
        m == i && break
        h.data[m], h.data[i] = h.data[i], h.data[m]
        i = m
    end
end

function heap_push!(h::MinHeap, key::Float64, idx::Int)
    push!(h.data, (key, idx))
    _heap_up!(h, length(h.data))
end

function heap_pop!(h::MinHeap)
    n = length(h.data)
    top = h.data[1]
    h.data[1] = h.data[n]
    resize!(h.data, n - 1)
    n > 1 && _heap_down!(h, 1)
    return top
end

@inline Base.isempty(h::MinHeap) = isempty(h.data)

# ==============================================================================
# F1 evaluation helpers
# ==============================================================================

"""
    label_shortcuts(ratios, tau) → Dict{Tuple{Int,Int}, Bool}

Apply threshold τ to pre-computed ratios to get ground-truth labels.
"""
function label_shortcuts(ratios::Dict, tau::Float64)
    Dict(edge => (r < tau) for (edge, r) in ratios)
end

"""
    evaluate_at_threshold(curvatures, labels, κ_threshold)

Predict shortcut when κ < κ_threshold. Returns (tp, fp, tn, fn, precision, recall, f1).
"""
function evaluate_at_threshold(curvatures::Dict, labels::Dict, κ_threshold::Float64)
    tp = fp = tn = fn = 0
    for (edge, result) in curvatures
        haskey(labels, edge) || continue
        is_sc   = labels[edge]
        pred_sc = result.curvature < κ_threshold
        if pred_sc && is_sc;       tp += 1
        elseif pred_sc && !is_sc;  fp += 1
        elseif !pred_sc && !is_sc; tn += 1
        else;                      fn += 1
        end
    end
    precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    f1        = (precision + recall) > 0 ? 2precision * recall / (precision + recall) : 0.0
    return (tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1)
end

"""
    best_f1_threshold(curvatures, labels) → (κ, precision, recall, f1)

Sweep κ over THRESHOLD_GRID and return the configuration achieving the highest F1.
"""
function best_f1_threshold(curvatures, labels)
    best = (κ=0.0, precision=0.0, recall=0.0, f1=-1.0)
    for κ in THRESHOLD_GRID
        m = evaluate_at_threshold(curvatures, labels, Float64(κ))
        if m.f1 > best.f1
            best = (κ=Float64(κ), precision=m.precision, recall=m.recall, f1=m.f1)
        end
    end
    return best
end

"""
    curvature_stats_by_label(curvatures, labels)

Mean κ split by ground-truth shortcut / non-shortcut label.
"""
function curvature_stats_by_label(curvatures, labels)
    sc_κ  = Float64[]
    nsc_κ = Float64[]
    for (edge, result) in curvatures
        haskey(labels, edge) || continue
        labels[edge] ? push!(sc_κ, result.curvature) :
                       push!(nsc_κ, result.curvature)
    end
    (mean_shortcut     = isempty(sc_κ)  ? NaN : mean(sc_κ),
     mean_non_shortcut = isempty(nsc_κ) ? NaN : mean(nsc_κ),
     n_shortcut        = length(sc_κ),
     n_non_shortcut    = length(nsc_κ))
end

# ==============================================================================
# Graph Dijkstra helpers (for downstream geodesic error analysis)
# ==============================================================================

"""
    dijkstra_from(adj, weights, source, n) → Vector{Float64}

Run Dijkstra's algorithm on a sparse graph given as adjacency list + weight dict.
Returns distances from source to all nodes.
"""
function dijkstra_from(adj::Vector{Vector{Int}},
                       weights::Dict{Tuple{Int,Int}, Float64},
                       source::Int, n::Int)
    dist = fill(Inf, n)
    dist[source] = 0.0
    visited = falses(n)

    heap = MinHeap()
    heap_push!(heap, 0.0, source)

    while !isempty(heap)
        d, u = heap_pop!(heap)
        visited[u] && continue
        visited[u] = true
        for v in adj[u]
            w = get(weights, (u, v), Inf)
            alt = dist[u] + w
            if alt < dist[v]
                dist[v] = alt
                heap_push!(heap, alt, v)
            end
        end
    end
    return dist
end

"""
    graph_to_adj_weights(graph, data) → (adj, weights)

Convert a KNNGraph to adjacency list + Euclidean edge weight dict.
"""
function graph_to_adj_weights(graph, data)
    n = length(graph)
    adj = [Int[] for _ in 1:n]
    weights = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        for j in graph[i]
            push!(adj[i], j)
            weights[(i, j)] = norm(data[:, i] - data[:, j])
        end
    end
    return adj, weights
end

"""
    prune_graph(adj, weights, curvatures, κ_threshold) → (adj_pruned, weights_pruned)

Remove edges with κ < κ_threshold. Keeps at least 1 neighbor per node.
"""
function prune_graph(adj, weights, curvatures, κ_threshold)
    n = length(adj)
    adj_p = [Int[] for _ in 1:n]
    w_p = Dict{Tuple{Int,Int}, Float64}()

    for i in 1:n
        neighbors_with_kappa = Tuple{Int, Float64}[]
        for j in adj[i]
            κ = haskey(curvatures, (i, j)) ? curvatures[(i, j)].curvature : 0.0
            push!(neighbors_with_kappa, (j, κ))
        end
        sort!(neighbors_with_kappa, by=x -> -x[2])  # highest κ first

        kept = 0
        for (j, κ) in neighbors_with_kappa
            if κ >= κ_threshold || kept < 1
                push!(adj_p[i], j)
                w_p[(i, j)] = weights[(i, j)]
                kept += 1
            end
        end
    end
    return adj_p, w_p
end

# ==============================================================================
# Bridge detection (Tarjan's algorithm)
# ==============================================================================

"""
    find_bridges(adj) → Set{Tuple{Int,Int}}

Find all bridge edges in an undirected graph using Tarjan's algorithm.
A bridge is an edge whose removal disconnects the graph.
Returns edges as canonical (min, max) pairs.
"""
function find_bridges(adj::Vector{Vector{Int}})
    n = length(adj)
    visited = falses(n)
    disc = zeros(Int, n)
    low = zeros(Int, n)
    parent = zeros(Int, n)
    bridges = Set{Tuple{Int,Int}}()
    timer = Ref(0)

    function dfs(u::Int)
        visited[u] = true
        timer[] += 1
        disc[u] = low[u] = timer[]
        for v in adj[u]
            if !visited[v]
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]
                    push!(bridges, minmax(u, v))
                end
            elseif v != parent[u]
                low[u] = min(low[u], disc[v])
            end
        end
    end

    for i in 1:n
        if !visited[i]
            dfs(i)
        end
    end
    return bridges
end

# ==============================================================================
# Pruning variants for geodesic error analysis
# ==============================================================================

"""
    oracle_prune(adj, weights, ratios, tau) → (adj_pruned, weights_pruned)

Remove ground-truth shortcut edges (those with r_ij < tau).
This gives the upper bound: perfect shortcut identification.
"""
function oracle_prune(adj::Vector{Vector{Int}},
                      weights::Dict{Tuple{Int,Int}, Float64},
                      ratios::Dict{Tuple{Int,Int}, Float64},
                      tau::Float64)
    n = length(adj)
    adj_p = [Int[] for _ in 1:n]
    w_p = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        for j in adj[i]
            r = get(ratios, (i, j), 1.0)
            if r >= tau  # keep non-shortcut edges
                push!(adj_p[i], j)
                w_p[(i, j)] = weights[(i, j)]
            end
        end
    end
    return adj_p, w_p
end

"""
    rank_prune(adj, weights, scores, fraction;
               bridges=nothing, ascending=true) → (adj_pruned, weights_pruned)

Remove the bottom (or top) `fraction` of edges ranked by `scores`.
- `scores`: Dict{Tuple{Int,Int}, Float64} mapping edges to their scores
- `ascending=true`: remove edges with lowest scores first (for κ: lowest = most shortcut-like)
- `ascending=false`: remove edges with highest scores first (for tangent angle: highest = most off-manifold)
- `bridges`: if provided (Set of canonical edge tuples), skip bridge edges to preserve connectivity
"""
function rank_prune(adj::Vector{Vector{Int}},
                    weights::Dict{Tuple{Int,Int}, Float64},
                    scores::Dict{Tuple{Int,Int}, Float64},
                    fraction::Float64;
                    bridges::Union{Nothing, Set{Tuple{Int,Int}}}=nothing,
                    ascending::Bool=true)
    # Collect all edges with scores
    edge_list = Tuple{Int, Int, Float64}[]
    for i in 1:length(adj)
        for j in adj[i]
            s = get(scores, (i, j), NaN)
            isnan(s) && continue
            push!(edge_list, (i, j, s))
        end
    end

    # Sort by score
    if ascending
        sort!(edge_list, by=x -> x[3])  # lowest first (remove these)
    else
        sort!(edge_list, by=x -> -x[3])  # highest first (remove these)
    end

    n_remove = round(Int, fraction * length(edge_list))
    remove_set = Set{Tuple{Int,Int}}()
    removed = 0
    for (i, j, _) in edge_list
        removed >= n_remove && break
        if bridges !== nothing
            canon = minmax(i, j)
            canon in bridges && continue
        end
        push!(remove_set, (i, j))
        removed += 1
    end

    # Build pruned graph
    n = length(adj)
    adj_p = [Int[] for _ in 1:n]
    w_p = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        for j in adj[i]
            if (i, j) ∉ remove_set
                push!(adj_p[i], j)
                w_p[(i, j)] = weights[(i, j)]
            end
        end
    end
    return adj_p, w_p
end

"""
    random_prune(adj, weights, fraction, rng) → (adj_pruned, weights_pruned)

Remove a random `fraction` of edges. Baseline for comparing to targeted pruning.
"""
function random_prune(adj::Vector{Vector{Int}},
                      weights::Dict{Tuple{Int,Int}, Float64},
                      fraction::Float64,
                      rng::AbstractRNG)
    # Collect all edges
    all_edges = Tuple{Int,Int}[]
    for i in 1:length(adj)
        for j in adj[i]
            push!(all_edges, (i, j))
        end
    end

    n_remove = round(Int, fraction * length(all_edges))
    remove_indices = sort(randperm(rng, length(all_edges))[1:min(n_remove, length(all_edges))])
    remove_set = Set{Tuple{Int,Int}}(all_edges[i] for i in remove_indices)

    n = length(adj)
    adj_p = [Int[] for _ in 1:n]
    w_p = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        for j in adj[i]
            if (i, j) ∉ remove_set
                push!(adj_p[i], j)
                w_p[(i, j)] = weights[(i, j)]
            end
        end
    end
    return adj_p, w_p
end

# compute_tangent_angles and compute_local_zscores now live in ManifoldANN.jl
# (src/graphs/detection_signals.jl and src/utils/edge_score_normalization.jl).

# ==============================================================================
# Spearman rank correlation
# ==============================================================================

"""
    spearman_rank_correlation(x, y) → Float64

Compute Spearman's rank correlation coefficient between vectors x and y.
Returns NaN if either vector has zero variance.
"""
function spearman_rank_correlation(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || error("Vectors must have same length")
    n = length(x)
    n < 2 && return NaN

    rx = _rank_vector(x)
    ry = _rank_vector(y)

    mx = mean(rx)
    my = mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in 1:n)
    dx = sqrt(sum((rx[i] - mx)^2 for i in 1:n))
    dy = sqrt(sum((ry[i] - my)^2 for i in 1:n))
    (dx < 1e-12 || dy < 1e-12) && return NaN
    return num / (dx * dy)
end

"""Compute ranks with average tie-breaking."""
function _rank_vector(v::AbstractVector)
    n = length(v)
    order = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[order[j+1]] == v[order[j]]
            j += 1
        end
        avg_rank = (i + j) / 2.0
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end
    return ranks
end

# ==============================================================================
# Geodesic error analysis
# ==============================================================================

"""
    geodesic_error_at_pairs(adj, weights, pairs, n) → (mean_rel_error, median_rel_error, n_disconnected, spearman)

Compute graph shortest-path distances for random pairs and compare to true geodesics.
`pairs` is a vector of (i, j, d_true) tuples.
"""
function geodesic_error_at_pairs(adj, weights, pairs, n)
    rel_errors = Float64[]
    d_graph_vec = Float64[]
    d_true_vec = Float64[]
    n_disc = 0

    # Cache Dijkstra: run from each unique source
    sources = unique([p[1] for p in pairs])
    dist_cache = Dict{Int, Vector{Float64}}()
    for s in sources
        dist_cache[s] = dijkstra_from(adj, weights, s, n)
    end

    for (i, j, d_true) in pairs
        d_graph = dist_cache[i][j]
        if isinf(d_graph) || d_true < 1e-12
            n_disc += 1
            continue
        end
        push!(rel_errors, abs(d_graph - d_true) / d_true)
        push!(d_graph_vec, d_graph)
        push!(d_true_vec, d_true)
    end

    if isempty(rel_errors)
        return (mean_rel_error=NaN, median_rel_error=NaN,
                n_disconnected=n_disc, spearman=NaN)
    end
    sp = spearman_rank_correlation(d_graph_vec, d_true_vec)
    return (mean_rel_error=mean(rel_errors), median_rel_error=median(rel_errors),
            n_disconnected=n_disc, spearman=sp)
end

# ==============================================================================
# CSV writing helpers
# ==============================================================================

"""
    write_raw_csv(results, filepath; extra_columns=Symbol[])

Write the raw tidy CSV with one row per experimental configuration.
`results` is a vector of NamedTuples. Columns are derived from the first element's keys.
"""
function write_raw_csv(results, filepath)
    isempty(results) && return
    keys = propertynames(first(results))
    open(filepath, "w") do io
        println(io, join(string.(keys), ","))
        for r in results
            vals = String[]
            for k in keys
                v = getproperty(r, k)
                if v isa Float64
                    push!(vals, isnan(v) ? "NaN" : @sprintf("%.6f", v))
                elseif v isa Integer
                    push!(vals, string(v))
                else
                    push!(vals, string(v))
                end
            end
            println(io, join(vals, ","))
        end
    end
end

"""
    write_pivot_csv(results, filepath, value_fn, filter_fn, n_grid, k_grid;
                    section_label_fn=nothing)

Write a pivoted CSV (rows=n, columns=k).
If section_label_fn is provided, it is called with the first result in each section
to produce a comment line.
"""
function write_pivot_csv(results, filepath, value_fn, sections, n_grid, k_grid;
                         section_label_fn=nothing)
    open(filepath, "w") do io
        first_section = true
        for (filter_fn, label) in sections
            subset = filter(filter_fn, results)
            isempty(subset) && continue

            first_section || println(io)
            first_section = false

            println(io, "# $label")
            println(io, join(["n"; string.(k_grid)], ","))

            for n in n_grid
                row = String[string(n)]
                for k in k_grid
                    m = filter(r -> r.n == n && r.k == k, subset)
                    push!(row, isempty(m) ? "NA" : @sprintf("%.4f", value_fn(m[1])))
                end
                println(io, join(row, ","))
            end
        end
    end
end

"""
    write_edges_csv(edge_results, filepath)

Write per-edge data (κ, ratio, shortcut labels) for AUROC computation.
"""
function write_edges_csv(edge_results, filepath)
    isempty(edge_results) && return
    open(filepath, "w") do io
        println(io, "manifold,n,k,noise_std,orc_variant,edge_i,edge_j,kappa,ratio,is_sc_05,is_sc_08")
        for r in edge_results
            println(io, join([
                r.manifold, r.n, r.k, r.noise_std, r.orc_variant,
                r.edge_i, r.edge_j,
                @sprintf("%.8f", r.kappa), @sprintf("%.6f", r.ratio),
                r.is_sc_05 ? 1 : 0, r.is_sc_08 ? 1 : 0,
            ], ","))
        end
    end
end

"""
    write_geoderror_csv(geoderror_results, filepath)

Write geodesic error data at pruning thresholds.
"""
function write_geoderror_csv(geoderror_results, filepath)
    isempty(geoderror_results) && return
    open(filepath, "w") do io
        println(io, "manifold,n,k,noise_std,orc_variant,kappa_thresh,n_edges_before,n_edges_after,frac_removed,mean_rel_error,median_rel_error,n_disconnected")
        for r in geoderror_results
            println(io, join([
                r.manifold, r.n, r.k, r.noise_std, r.orc_variant,
                @sprintf("%.4f", r.kappa_thresh),
                r.n_edges_before, r.n_edges_after,
                @sprintf("%.6f", r.frac_removed),
                @sprintf("%.6f", r.mean_rel_error),
                @sprintf("%.6f", r.median_rel_error),
                r.n_disconnected,
            ], ","))
        end
    end
end

# ==============================================================================
# Incremental CSV I/O helpers (for crash-safe writing + resume)
# ==============================================================================

"""
    init_csv(filepath, header::String) → filepath

Write the header line only if the file doesn't already exist.
"""
function init_csv(filepath::String, header::String)
    if !isfile(filepath)
        open(filepath, "w") do io
            println(io, header)
        end
    end
    return filepath
end

"""
    append_csv_rows(filepath, rows::Vector{String})

Open file in append mode, write all rows, and close. One open/close per batch.
"""
function append_csv_rows(filepath::String, rows::Vector{String})
    open(filepath, "a") do io
        for row in rows
            println(io, row)
        end
    end
end

"""
    load_completed_keys(filepath, key_columns::Vector{Symbol}) → Set{Tuple}

Read an existing CSV and extract the set of key-column value tuples already present.
Keys are stored as string tuples so the skip check just converts loop values via string().
Returns an empty set if the file doesn't exist or has no data rows.
"""
function load_completed_keys(filepath::String, key_columns::Vector{Symbol})
    completed = Set{Tuple}()
    isfile(filepath) || return completed
    lines = readlines(filepath)
    length(lines) < 2 && return completed
    header = Symbol.(split(lines[1], ","))
    col_indices = [findfirst(==(k), header) for k in key_columns]
    any(isnothing, col_indices) && return completed
    for line in lines[2:end]
        isempty(strip(line)) && continue
        startswith(line, "#") && continue
        fields = split(line, ",")
        length(fields) < maximum(col_indices) && continue
        key = Tuple(fields[i] for i in col_indices)
        push!(completed, key)
    end
    return completed
end

"""
    write_config_toml(filepath; kwargs...)

Write a config.toml file recording experiment parameters for reproducibility.
"""
function write_config_toml(filepath; kwargs...)
    open(filepath, "w") do io
        println(io, "# Experiment configuration — auto-generated")
        println(io, "timestamp = \"$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))\"")

        # Try to get git hash
        try
            hash = strip(read(`git -C $(dirname(filepath)) rev-parse HEAD`, String))
            println(io, "git_hash = \"$hash\"")
        catch
            println(io, "git_hash = \"unknown\"")
        end

        for (k, v) in kwargs
            if v isa AbstractString
                println(io, "$k = \"$v\"")
            elseif v isa AbstractVector
                println(io, "$k = [$(join(repr.(v), ", "))]")
            elseif v isa Bool
                println(io, "$k = $(v ? "true" : "false")")
            else
                println(io, "$k = $v")
            end
        end
    end
end
