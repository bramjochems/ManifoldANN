module ManifoldANN

include("ann_index.jl")
include("graphs/knn_graph.jl")
include("utils/random_utils.jl")
include("utils/validation.jl")
include("utils/eval_utils.jl")
include("indices/bruteforce.jl")
include("indices/lsh.jl")

export AbstractANNIndex,
       AbstractGraphIndex,
       AbstractLSHHash,
       BruteForceIndex,
       LSHIndex,
       KNNGraph,
       BinningHash,
       RandomHyperplaneHash,
       build_index,
       build_knn_graph,
       query,
       materialize_graph,
       ensure_graph,
       make_binning_hash,
       make_random_hyperplane_hash,
       configured_k,
       supports_layers,
       supports_metadata,
       supports_mutation,
       spawn_child_rngs,
       validate_index_dimensions,
       recall_at_k

end
