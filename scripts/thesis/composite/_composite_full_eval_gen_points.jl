
# Helper: regenerate (x,y,z) for a given (manifold-kind, params, n, noise_std)
# matching the procedure in experiment_orc.jl exactly.
using Random
using DelimitedFiles

include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "docs", "examples", "geodesic", "torus_utils.jl"))

function gen_swiss(n; t_min, t_range, h_scale, radial_scale)
    rng = MersenneTwister(42)
    if radial_scale == 1.0
        data, params = generate_swiss_roll(n; rng=rng, t_min=t_min, t_range=t_range, h_scale=h_scale)
    else
        t = t_min .+ t_range .* rand(rng, n)
        h = h_scale .* rand(rng, n)
        data = vcat((radial_scale .* t .* cos.(t))', h', (radial_scale .* t .* sin.(t))')
        params = (t=t, h=h, radial_scale=radial_scale)
    end
    return data, params, rng
end

function gen_torus(n; R, r)
    rng = MersenneTwister(42)
    data, params = generate_torus(n; rng=rng, R=R, r=r)
    return data, params, rng
end

function add_noise!(data, rng, noise_std)
    if noise_std > 0
        data .+= noise_std .* randn(rng, size(data))
    end
    return data
end

function main(args)
    out_path = args[1]
    kind = args[2]
    n = parse(Int, args[3])
    noise_std = parse(Float64, args[4])
    if kind == "swiss"
        t_min = parse(Float64, args[5])
        t_range = parse(Float64, args[6])
        h_scale = parse(Float64, args[7])
        radial_scale = parse(Float64, args[8])
        data, params, rng = gen_swiss(n; t_min=t_min, t_range=t_range, h_scale=h_scale, radial_scale=radial_scale)
        add_noise!(data, rng, noise_std)
        # Output: x,y,z,t,h
        out = hcat(data', params.t, params.h)
        open(out_path, "w") do io
            write(io, "x,y,z,p1,p2\n")
            writedlm(io, out, ',')
        end
    elseif kind == "torus"
        R = parse(Float64, args[5])
        r = parse(Float64, args[6])
        data, params, rng = gen_torus(n; R=R, r=r)
        add_noise!(data, rng, noise_std)
        out = hcat(data', params.u, params.v)
        open(out_path, "w") do io
            write(io, "x,y,z,p1,p2\n")
            writedlm(io, out, ',')
        end
    else
        error("unknown kind: $kind")
    end
    return nothing
end

main(ARGS)
