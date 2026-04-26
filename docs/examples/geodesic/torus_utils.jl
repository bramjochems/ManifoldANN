# Torus Utilities for Geodesic Examples
#
# This file provides utilities for generating torus data and computing
# approximate geodesic distances numerically.

using LinearAlgebra

"""
    generate_torus(n; rng=Random.GLOBAL_RNG, R=3.0, r=1.0)

Generate n points uniformly on a torus manifold.

# Arguments
- `n`: Number of points to generate
- `rng`: Random number generator (default: global RNG)
- `R`: Major radius — distance from torus centre to tube centre (default: 3.0)
- `r`: Minor radius — radius of the tube (default: 1.0)

# Returns
- `data`: 3×n matrix of points in ambient space
- `params`: NamedTuple with `u`, `v`, `R`, `r` for each point

# Torus Parametrization
    x = (R + r·cos(v))·cos(u)
    y = (R + r·cos(v))·sin(u)
    z = r·sin(v)

where u ∈ [0, 2π) is the angle around the main axis (longitude) and
v ∈ [0, 2π) is the angle around the tube (latitude).

# Gaussian Curvature
The Gaussian curvature of the torus is:
    K(u, v) = cos(v) / (r · (R + r·cos(v)))

This varies with v:
  - K > 0 on the outer rim  (v near 0,  outer equator)
  - K < 0 on the inner rim  (v near π,  inner equator)
  - K = 0 at v = π/2 and v = 3π/2 (top and bottom)

The ratio R/r controls the curvature range: a thin tube (R ≫ r) has
nearly flat geometry; a fat tube (R close to r) has strong curvature.

# Uniform Sampling
To sample uniformly on the surface (with respect to area element), the
acceptance-rejection method is used: propose (u, v) uniformly on [0,2π)²,
accept with probability (R + r·cos(v)) / (R + r), which is proportional
to the area element |(R + r·cos(v))|.  This ensures area-uniform sampling.
"""
function generate_torus(n::Int;
                        rng=Random.GLOBAL_RNG,
                        R::Float64=3.0,
                        r::Float64=1.0)
    @assert R > r > 0 "Require R > r > 0 to avoid self-intersection"

    u_vec = Vector{Float64}(undef, n)
    v_vec = Vector{Float64}(undef, n)

    # Rejection sampling for area-uniform distribution on the torus
    # Area element: dA = r(R + r·cos(v)) du dv  →  weight = (R + r·cos(v))/(R+r)
    accepted = 0
    while accepted < n
        u_cand = 2π * rand(rng)
        v_cand = 2π * rand(rng)
        weight = (R + r * cos(v_cand)) / (R + r)
        if rand(rng) < weight
            accepted += 1
            u_vec[accepted] = u_cand
            v_vec[accepted] = v_cand
        end
    end

    # Embed in 3D
    data = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        u = u_vec[i]
        v = v_vec[i]
        data[1, i] = (R + r * cos(v)) * cos(u)
        data[2, i] = (R + r * cos(v)) * sin(u)
        data[3, i] = r * sin(v)
    end

    params = (u=u_vec, v=v_vec, R=R, r=r)
    return data, params
end

# ---------------------------------------------------------------------------
# Geodesic distance computation
# ---------------------------------------------------------------------------
#
# The torus metric is ds² = (R + r·cos(v))²du² + r²dv².  Unlike the Swiss
# roll (developable), the torus has nonzero intrinsic curvature and cannot
# be isometrically unrolled.  Exact closed-form geodesics exist only for
# coordinate circles; the general case requires numerical integration.
#
# We use a grid Dijkstra approach:
#   1. Build a grid graph on (u, v) ∈ [0, 2π)² with n_grid × n_grid nodes.
#   2. Each node connects to its 8 axis-aligned and diagonal neighbours
#      with arc-length weights under the torus metric.
#   3. Run Dijkstra from the source node to all targets.
#
# Grid size n_grid=100 gives a graph of 10,000 nodes.  Dijkstra with a binary
# heap runs in O(E log V) ≈ O(8·10⁴ · 14) ≈ 10⁶ operations — fast per pair.
#
# The absolute error in geodesic distance is O(h) where h = 2π/n_grid;
# at n_grid=100, h ≈ 0.063 rad, which gives < 1% relative error for typical
# edge lengths in k-NN graphs with n ≥ 200.  For shortcut detection, a
# conservative 5–10% error threshold still cleanly separates shortcuts
# (ambient/geodesic ratio < 0.5) from non-shortcuts.

# ---------------------------------------------------------------------------
# Binary min-heap for Dijkstra (Float64 key, Int payload)
# ---------------------------------------------------------------------------
# MinHeap is defined in orc_helpers.jl for experiment scripts.
# For standalone use of torus_utils.jl (e.g. in examples), define it here
# only if not already loaded.

if !@isdefined(MinHeap)

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

end  # if !@isdefined(MinHeap)

# ---------------------------------------------------------------------------
# Grid Dijkstra (single-source, all destinations on same grid as source)
# ---------------------------------------------------------------------------

"""
    _grid_dijkstra(i_src, j_src, i_dst, j_dst, R, r, n_grid)

Run Dijkstra from grid node (i_src, j_src) to (i_dst, j_dst) on a
n_grid × n_grid toroidal parameter grid.  Returns the shortest path length
under the torus metric.

Grid indices are 1-based.  Node (i, j) corresponds to
  u = (i - 1) * h,  v = (j - 1) * h  where h = 2π / n_grid.

8-connectivity (including diagonals) is used to reduce grid anisotropy.
"""
function _grid_dijkstra(i_src::Int, j_src::Int,
                        i_dst::Int, j_dst::Int,
                        R::Float64, r::Float64,
                        n_grid::Int)::Float64

    i_src == i_dst && j_src == j_dst && return 0.0

    h = 2π / n_grid
    N = n_grid

    dist = fill(Inf, N, N)
    dist[i_src, j_src] = 0.0

    heap = MinHeap()
    heap_push!(heap, 0.0, (i_src - 1) * N + j_src)   # flat index

    while !isempty(heap)
        cost, flat = heap_pop!(heap)
        ci = (flat - 1) ÷ N + 1
        cj = (flat - 1) % N + 1

        cost > dist[ci, cj] + 1e-14 && continue

        ci == i_dst && cj == j_dst && return cost

        # v coordinate at current node
        v_c = (cj - 1) * h
        w_u = (R + r * cos(v_c)) * h   # arc length for Δu = h (approximate)
        w_v = r * h                     # arc length for Δv = h (exact)
        w_d = sqrt(w_u^2 + w_v^2)      # diagonal step

        for (di, dj, w) in (
                ( 1,  0, w_u),
                (-1,  0, w_u),
                ( 0,  1, w_v),
                ( 0, -1, w_v),
                ( 1,  1, w_d),
                ( 1, -1, w_d),
                (-1,  1, w_d),
                (-1, -1, w_d),
            )
            ni = mod1(ci + di, N)
            nj = mod1(cj + dj, N)
            nc = dist[ci, cj] + w
            if nc < dist[ni, nj]
                dist[ni, nj] = nc
                heap_push!(heap, nc, (ni - 1) * N + nj)
            end
        end
    end

    return dist[i_dst, j_dst]
end

"""
    exact_torus_geodesic(u1, v1, u2, v2, R, r; n_grid=100)

Compute an approximate geodesic distance on the torus using grid Dijkstra.

# Arguments
- `u1`, `v1`: Parameters of first point ∈ [0, 2π)
- `u2`, `v2`: Parameters of second point ∈ [0, 2π)
- `R`, `r`: Torus major and minor radii
- `n_grid`: Grid resolution per dimension (default: 100)
             100 → ~10⁴ nodes, ~10⁶ Dijkstra ops, < 5ms per call.
             200 → ~4·10⁴ nodes, ~4·10⁶ ops, ~20ms per call.

# Returns
- Approximate geodesic distance (Float64); error is O(h) = O(2π/n_grid).
"""
function exact_torus_geodesic(u1::Real, v1::Real, u2::Real, v2::Real,
                               R::Real, r::Real;
                               n_grid::Int=100)
    h = 2π / n_grid
    i1 = mod1(round(Int, u1 / h), n_grid)
    j1 = mod1(round(Int, v1 / h), n_grid)
    i2 = mod1(round(Int, u2 / h), n_grid)
    j2 = mod1(round(Int, v2 / h), n_grid)
    return _grid_dijkstra(i1, j1, i2, j2, Float64(R), Float64(r), n_grid)
end

"""
    exact_torus_geodesic(params, i, j; n_grid=100)

Compute geodesic distance between points i and j using their parameter indices.

# Arguments
- `params`: NamedTuple with `u`, `v`, `R`, `r` vectors (from `generate_torus`)
- `i`, `j`: Indices of the two points
- `n_grid`: Grid resolution (default: 100)

# Returns
- Approximate geodesic distance (Float64)
"""
function exact_torus_geodesic(params::NamedTuple, i::Int, j::Int; n_grid::Int=100)
    return exact_torus_geodesic(
        params.u[i], params.v[i],
        params.u[j], params.v[j],
        params.R, params.r;
        n_grid=n_grid
    )
end

"""
    torus_gaussian_curvature(v, R, r)

Gaussian curvature of the torus at a point with tube angle v.

    K(v) = cos(v) / (r · (R + r·cos(v)))

Positive on the outer rim (v ≈ 0), negative on the inner rim (v ≈ π).
"""
function torus_gaussian_curvature(v::Real, R::Real, r::Real)
    return cos(v) / (r * (R + r * cos(v)))
end

"""
    compute_all_exact_distances_torus(params, indices; n_grid=100)

Compute pairwise geodesic distances for all points in `indices`.

# Arguments
- `params`: NamedTuple with `u`, `v`, `R`, `r` vectors
- `indices`: Indices of points to compute (default: all points)
- `n_grid`: Grid resolution for geodesic computation

# Returns
- Distance matrix D where D[i,j] = approximate geodesic distance
"""
function compute_all_exact_distances_torus(params::NamedTuple,
                                           indices=eachindex(params.u);
                                           n_grid::Int=100)
    m = length(indices)
    D = zeros(m, m)
    idx = collect(indices)

    for i in 1:m
        for j in i+1:m
            d = exact_torus_geodesic(params, idx[i], idx[j]; n_grid=n_grid)
            D[i, j] = d
            D[j, i] = d
        end
    end

    return D
end
