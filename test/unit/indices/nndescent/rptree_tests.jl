using ManifoldANN
using Random
using Test

# Pull internal symbols (not exported; this is an internal helper module).
const _MA = ManifoldANN
const build_rptree = _MA.build_rptree
const build_rptree_forest = _MA.build_rptree_forest
const leaf_members_fn = _MA.leaf_members
const default_rptree_n_trees = _MA.default_rptree_n_trees
const default_rptree_leaf_cap = _MA.default_rptree_leaf_cap

# Walk all leaves of a tree, collecting (leaf_id => Vector{Int} members).
function _collect_leaves(tree)
    leaves = Vector{Vector{Int}}()
    for node in tree.nodes
        if node.left == 0  # leaf
            push!(leaves, collect(tree.leaf_members[Int(first(node.leaf_range)):Int(last(node.leaf_range))]))
        end
    end
    return leaves
end

@testset "RP-tree build: partition is a partition" begin
    rng = Random.MersenneTwister(0x5EED)
    data = randn(rng, Float32, 8, 200)
    tree = build_rptree(data, 20; rng = Random.MersenneTwister(1))
    leaves = _collect_leaves(tree)
    all_members = reduce(vcat, leaves; init = Int[])
    @test length(all_members) == 200
    @test sort(all_members) == collect(1:200)
end

@testset "RP-tree leaf cap respected" begin
    rng = Random.MersenneTwister(0x5EED)
    data = randn(rng, Float32, 8, 200)
    tree = build_rptree(data, 20; rng = Random.MersenneTwister(2))
    for leaf in _collect_leaves(tree)
        @test length(leaf) <= 20
    end
end

@testset "RP-tree forest: each tree returns a leaf for any query" begin
    rng = Random.MersenneTwister(0x5EED)
    data = randn(rng, Float32, 8, 200)
    forest = build_rptree_forest(data, 5, 20; rng = Random.MersenneTwister(3))
    @test length(forest) == 5
    for tree in forest
        for i in 1:size(data, 2)
            members = leaf_members_fn(tree, view(data, :, i))
            @test length(members) >= 1
            @test i in members  # data point must land in a leaf containing it
        end
    end
end

@testset "RP-tree determinism with fixed RNG" begin
    rng = Random.MersenneTwister(0x5EED)
    data = randn(rng, Float32, 8, 200)
    t1 = build_rptree(data, 20; rng = Random.MersenneTwister(42))
    t2 = build_rptree(data, 20; rng = Random.MersenneTwister(42))
    @test t1.leaf_members == t2.leaf_members
    @test length(t1.nodes) == length(t2.nodes)
    for i in eachindex(t1.nodes)
        @test t1.nodes[i].left == t2.nodes[i].left
        @test t1.nodes[i].right == t2.nodes[i].right
        @test t1.nodes[i].leaf_range == t2.nodes[i].leaf_range
    end
end

@testset "NN-Descent with RP-tree init achieves high recall" begin
    rng = Random.MersenneTwister(7)
    n, d, k = 500, 16, 15
    data = randn(rng, Float32, d, n)

    brute = build_index(BruteForceIndex, data)
    index = build_index(
        NNDescentIndex, data;
        k = k,
        max_iterations = 25,
        convergence_threshold = 0.0,
        sampling_policy = :uniform,
        rng = Random.MersenneTwister(0xC0DE),
        distance = default_squared_distance,
        init = :rptree,
    )

    # Recall vs brute force on 50 random query points.
    recalls = Float64[]
    for trial in 1:50
        q = randn(Random.MersenneTwister(2000 + trial), Float32, d)
        nn = query(index, data, q, k;
                   ef_search = 32,
                   rng = Random.MersenneTwister(3000 + trial))
        bf = query(brute, data, q, k)
        nn_ids = neighbor_ids(nn)
        bf_ids = neighbor_ids(bf)
        push!(recalls, length(intersect(Set(nn_ids), Set(bf_ids))) / length(bf_ids))
    end
    avg_recall = sum(recalls) / length(recalls)
    @test avg_recall >= 0.90
end

@testset "RP-tree default knobs" begin
    @test default_rptree_n_trees(100) >= 3
    @test default_rptree_n_trees(10_000_000) <= 12
    @test default_rptree_leaf_cap(15) == min(64, 75)
    @test default_rptree_leaf_cap(30) == 64
end
