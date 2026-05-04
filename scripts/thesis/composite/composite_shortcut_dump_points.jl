# Dump point coordinates and intrinsic params for the three manifold cells
# used by composite_shortcut_validation. Reproduces the exact RNG procedure
# from experiment_orc.jl (MersenneTwister(SEED=42), then generate_*).
#
# Outputs CSVs with columns (x,y,z,p1,p2) where (p1,p2) are (t,h) for swiss
# variants and (u,v) for torus.

using Random
using DelimitedFiles

include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "torus_utils.jl"))

const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "benchmark_results", "composite_shortcut_validation")
mkpath(OUT_DIR)

const N = 2000
const SEED = 42

# Replicates the original swiss_roll_utils signature
function gen_swiss(n; t_min=1.5π, t_range=3π, h_scale=10.0, radial_scale=1.0)
    rng = MersenneTwister(SEED)
    if radial_scale == 1.0
        # Use the on-disk generator (no radial_scale kwarg)
        data, params = generate_swiss_roll(n; rng=rng, t_min=t_min, t_range=t_range, h_scale=h_scale)
        return data, params
    else
        # Inline the tight version with radial_scale (matches stash diff)
        t = t_min .+ t_range .* rand(rng, n)
        h = h_scale .* rand(rng, n)
        data = vcat((radial_scale .* t .* cos.(t))', h', (radial_scale .* t .* sin.(t))')
        params = (t=t, h=h, radial_scale=radial_scale)
        return data, params
    end
end

function dump_swiss(name; radial_scale=1.0)
    data, params = gen_swiss(N; radial_scale=radial_scale)
    out = hcat(data', params.t, params.h)
    open(joinpath(OUT_DIR, "points_$(name).csv"), "w") do io
        write(io, "x,y,z,t,h\n")
        writedlm(io, out, ',')
    end
    println("Wrote points_$(name).csv ", size(out))
end

function dump_torus(name; R=2.0, r=1.0)
    rng = MersenneTwister(SEED)
    data, params = generate_torus(N; rng=rng, R=R, r=r)
    out = hcat(data', params.u, params.v)
    open(joinpath(OUT_DIR, "points_$(name).csv"), "w") do io
        write(io, "x,y,z,u,v\n")
        writedlm(io, out, ',')
    end
    println("Wrote points_$(name).csv ", size(out))
end

dump_swiss("swiss_roll"; radial_scale=1.0)
dump_swiss("swiss_roll_tight"; radial_scale=0.05)
dump_torus("torus_R2r1"; R=2.0, r=1.0)
println("Done.")
