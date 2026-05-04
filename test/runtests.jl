using Test
using ManifoldANN

function _collect_test_files(root)
    files = String[]
    isdir(root) || return files
    for (dir, _, entries) in walkdir(root)
        for entry in entries
            endswith(entry, ".jl") || continue
            push!(files, joinpath(dir, entry))
        end
    end
    sort!(files)
    return files
end

# Walk a finalized testset tree counting (testsets, passes, fails, errors).
# Per-edge `@test`s inside loops inflate the raw pass count; the testset
# count is the meaningful signal.
function _summarize_testsets(t::Test.AbstractTestSet)
    # `DefaultTestSet` collapses individual `Pass` records into `n_passed`
    # at finalize and drops them from `results`; `Fail` / `Error` records
    # stay in `results` so they can be printed. Walk both.
    n_sets = 0
    n_pass = 0
    n_fail = 0
    n_err = 0
    function walk(node)
        if node isa Test.DefaultTestSet
            n_sets += 1
            n_pass += node.n_passed
            for r in node.results
                walk(r)
            end
        elseif node isa Test.Fail || node isa Test.Broken
            n_fail += 1
        elseif node isa Test.Error
            n_err += 1
        end
    end
    walk(t)
    return (testsets=n_sets, passes=n_pass, fails=n_fail, errors=n_err)
end

const TEST_ROOT = @__DIR__

outer = @testset "ManifoldANN" begin
    @testset "unit" begin
        for file in _collect_test_files(joinpath(TEST_ROOT, "unit"))
            include(file)
        end
    end

    @testset "regression" begin
        for file in _collect_test_files(joinpath(TEST_ROOT, "regression"))
            include(file)
        end
    end
end

let s = _summarize_testsets(outer)
    println()
    println("──────────────────────────────────────────────")
    println("Testset summary: $(s.testsets) testsets    " *
            "$(s.passes) passes, $(s.fails) failed, $(s.errors) errored")
    println("(per-edge `@test`s in loops inflate the pass count;")
    println(" the meaningful gate is `failed == 0 && errored == 0`.)")
    println("──────────────────────────────────────────────")
end
