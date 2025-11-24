# Swiss Roll Utilities for Geodesic Examples
#
# This file provides utilities for generating Swiss roll data and computing
# exact geodesic distances analytically.

"""
    generate_swiss_roll(n; rng=Random.GLOBAL_RNG, t_min=1.5π, t_range=3π, h_scale=10.0)

Generate n points on a Swiss roll manifold.

# Arguments
- `n`: Number of points to generate
- `rng`: Random number generator (default: global RNG)
- `t_min`: Minimum value of parameter t (default: 1.5π)
- `t_range`: Range of parameter t (default: 3π, so t ∈ [1.5π, 4.5π])
- `h_scale`: Scale of height dimension (default: 10.0)

# Returns
- `data`: 3×n matrix of points in ambient space
- `params`: NamedTuple with `t` and `h` parameters for each point

# Swiss Roll Parametrization
The Swiss roll is parametrized as:
- x = t * cos(t)
- y = h
- z = t * sin(t)

where t is the angular parameter and h is the height.
"""
function generate_swiss_roll(n::Int;
                             rng=Random.GLOBAL_RNG,
                             t_min::Float64=1.5π,
                             t_range::Float64=3π,
                             h_scale::Float64=10.0)
    t = t_min .+ t_range .* rand(rng, n)
    h = h_scale .* rand(rng, n)

    data = vcat(
        (t .* cos.(t))',
        h',
        (t .* sin.(t))'
    )

    params = (t=t, h=h)

    return data, params
end

"""
    exact_swiss_roll_geodesic(t1, h1, t2, h2)

Compute exact geodesic distance on the Swiss roll between two points.

# Arguments
- `t1`, `h1`: Parameters of first point
- `t2`, `h2`: Parameters of second point

# Returns
- Exact geodesic distance

# Mathematical Background
The Swiss roll has metric ds² = (1 + t²)dt² + dh².

Since it's a developable surface (can be unrolled into a plane without distortion),
geodesics are straight lines in the unrolled (s, h) coordinates, where:

    s(t) = ∫₀ᵗ √(1 + τ²) dτ = (t√(1 + t²) + asinh(t)) / 2

This is the arc length along the spiral at height 0.

The geodesic distance between (t₁, h₁) and (t₂, h₂) is:

    d = √[(s(t₂) - s(t₁))² + (h₂ - h₁)²]

This is simply the Euclidean distance in the unrolled coordinates.

# Reference
The Swiss roll is isometric to a portion of the plane, making it ideal for
validating geodesic distance approximations.
"""
function exact_swiss_roll_geodesic(t1::Real, h1::Real, t2::Real, h2::Real)
    # Arc length function along the spiral
    # s(t) = ∫₀ᵗ √(1 + τ²) dτ
    # This has a closed form: s(t) = (t√(1+t²) + asinh(t)) / 2
    function arc_length(t)
        return (t * sqrt(1 + t^2) + asinh(t)) / 2
    end

    # Compute arc lengths
    s1 = arc_length(t1)
    s2 = arc_length(t2)

    # Distance in unrolled (s, h) coordinates is just Euclidean
    ds = s2 - s1
    dh = h2 - h1

    return sqrt(ds^2 + dh^2)
end

"""
    exact_swiss_roll_geodesic(params, i, j)

Compute exact geodesic distance between two points given their parameter indices.

# Arguments
- `params`: NamedTuple with `t` and `h` vectors (from `generate_swiss_roll`)
- `i`, `j`: Indices of the two points

# Returns
- Exact geodesic distance
"""
function exact_swiss_roll_geodesic(params::NamedTuple, i::Int, j::Int)
    return exact_swiss_roll_geodesic(params.t[i], params.h[i], params.t[j], params.h[j])
end

"""
    compute_all_exact_distances(params, indices=eachindex(params.t))

Compute exact pairwise geodesic distances for all points in `indices`.

# Arguments
- `params`: NamedTuple with `t` and `h` vectors
- `indices`: Indices of points to compute (default: all points)

# Returns
- Distance matrix D where D[i,j] = exact geodesic distance from i to j

# Warning
For large n, this computes n×n distances which can be memory intensive.
Use `indices` to compute only a subset.
"""
function compute_all_exact_distances(params::NamedTuple, indices=eachindex(params.t))
    n = length(indices)
    D = zeros(n, n)

    for i in 1:n
        for j in i+1:n
            d = exact_swiss_roll_geodesic(params, indices[i], indices[j])
            D[i, j] = d
            D[j, i] = d
        end
    end

    return D
end

"""
    ambient_distance(data, i, j)

Compute Euclidean distance in ambient 3D space between points i and j.

This is the "shortcut" distance that incorrectly goes through the roll.
"""
function ambient_distance(data::AbstractMatrix, i::Int, j::Int)
    return norm(data[:, i] - data[:, j])
end

"""
    compare_distance_estimates(data, params, model, pairs)

Compare different distance estimates for pairs of points.

# Arguments
- `data`: 3×n Swiss roll data matrix
- `params`: Swiss roll parameters (t, h)
- `model`: GeodesicDistanceModel
- `pairs`: Vector of (i, j) tuples to compare

# Returns
- DataFrame with columns: pair, exact, ambient, graph, relative_error_graph

# Example
```julia
data, params = generate_swiss_roll(100)
model = build_geodesic_model(...)
pairs = [(1, 50), (10, 80), (20, 90)]
df = compare_distance_estimates(data, params, model, pairs)
```
"""
function compare_distance_estimates(data, params, model, pairs)
    results = []

    for (i, j) in pairs
        d_exact = exact_swiss_roll_geodesic(params, i, j)
        d_ambient = ambient_distance(data, i, j)
        d_graph = geodesic_distance(model, data, i, j)

        rel_error_graph = abs(d_graph - d_exact) / d_exact

        push!(results, (
            pair = "($i, $j)",
            exact = d_exact,
            ambient = d_ambient,
            graph = d_graph,
            relative_error_graph = rel_error_graph
        ))
    end

    return results
end
