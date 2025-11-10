using Test
using ManifoldANN

unit_dir = normpath(joinpath(@__DIR__, "..", "test", "unit"))

files = String[]
for (root, _, entries) in walkdir(unit_dir)
    for entry in entries
        endswith(entry, ".jl") || continue
        push!(files, joinpath(root, entry))
    end
end

sort!(files)
@testset "ManifoldANN unit tests" begin
    for file in files
        include(file)
    end
end
