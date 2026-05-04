module ManifoldANN

include("ann_index.jl")
include("graphs/knn_graph.jl")
include("utils/constants.jl")
include("utils/distances.jl")
include("utils/random_utils.jl")
include("utils/validation.jl")
include("utils/eval_utils.jl")
include("utils/neighbor_heaps.jl")
include("utils/binary_partition_tree.jl")
include("utils/rptree.jl")
include("indices/bruteforce.jl")
include("indices/lsh.jl")
include("indices/kdtree.jl")
include("indices/hnsw.jl")
include("indices/nndescent.jl")
include("indices/rptree_forest.jl")
include("indices/pcatree.jl")
include("indices/pcatree_forest.jl")

# Transform module (must come before indices that depend on it)
include("transforms/transforms.jl")

# Preprocessing transforms (dimensionality reduction)
include("preprocessing/preprocessing.jl")

# Indices that depend on transforms
include("indices/faiss_ivf.jl")

# Multi-level indices
include("indices/multilevel/multilevel_index.jl")

# Local geometry estimation (for geodesic distance)
include("geometry/local_geometry.jl")
include("geometry/pca.jl")
include("geometry/criteria.jl")
include("geometry/neighborhood.jl")
include("geometry/estimator.jl")

# Per-edge weight abstraction (shared by weighted-graph and geodesic-model
# layers; Chapter 6 of the thesis defines the curvature-free symmetric
# weight, while the tangent-projection family is consumed both as ground
# costs for ORC and as per-edge geodesic-distance estimates).
include("graphs/edge_weight.jl")

# Weighted kNN graph (geodesic-aware edge weights)
include("graphs/weighted_knn_graph.jl")

# Graph curvature (Ollivier-Ricci curvature and filtering)
include("graphs/refinement/refinement.jl")

# Geodesic distance model
include("geodesic/geodesic_model.jl")
include("geodesic/refinement.jl")

export AbstractANNIndex,
       AbstractGraphIndex,
       AbstractLSHHash,
       BruteForceIndex,
       LSHIndex,
       KDTreeIndex,
       HNSWIndex,
       IVFFlatIndex,
       NNDescentIndex,
       RPTreeIndex,
       RPTreeForestIndex,
       PCATreeIndex,
       PCATreeForestIndex,
       PCASplitter,
       pca_forest_splitter,
       AbstractSpectrumEstimator,
       ExactSVD,
       RandomizedSVD,
       SubsampledSVD,
       AbstractSplitDirectionPolicy,
       TopComponent,
       RandomTopK,
       RandomLinearCombo,
       AbstractStoppingCriterion,
       MaxLeafSize,
       IntrinsicDimRatio,
       AnyOf,
       AllOf,
       AbstractSplitValuePolicy,
       MedianSplit,
       MeanSplit,
       RandomBetweenQuantiles,
       UniformPairSampling,
       FullSymmetry,
       PrunedSymmetry,
       NoSymmetry,
       KNNGraph,
       BinningHash,
       RandomHyperplaneHash,
       build_index,
       build_knn_graph,
       query,
       materialize_graph,
       ensure_graph,
       has_metadata,
       graph_metadata,
       node_metadata,
       make_binning_hash,
       make_random_hyperplane_hash,
       configured_k,
       index_distance,
       supports_layers,
       supports_metadata,
       supports_mutation,
       spawn_child_rngs,
       derive_child_seed,
       query_child_rng,
       validate_index_dimensions,
       validate_index_query_matrix,
       recall_at_k,
       default_distance,
       default_squared_distance,
       cosine_distance,
       squared_cosine_distance,
       # Transforms
       AbstractTransform,
       TransformResult,
       IdentityTransform,
       KMeansTransform,
       KMeansAssignment,
       fit!,
       transform,
       has_bucketing,
       get_bucket_assignment,
       partition_by_transform,
       apply_transform_batch,
       # Preprocessing transforms
       PCATransform,
       RandomProjectionTransform,
       inverse_transform,
       target_dimension,
       suggested_dimension,
       preserves_data,
       # Multi-level indices
       AbstractIndexConfig,
       TerminalConfig,
       TransformedConfig,
       AbstractRoutingStrategy,
       TopKRouting,
       ExhaustiveRouting,
       AbstractMergeStrategy,
       SimpleMerge,
       TransformedIndex,
       MultiLevelIndex,
       build_ivf_hnsw_index,
       Neighbor,
       neighbor_ids,
       # Local geometry estimation
       AbstractLocalGeometryMethod,
       AbstractLocalGeometry,
       PCAMethod,
       PCAGeometry,
       fit_geometry,
       local_distance,
       supports_projection,
       project,
       reconstruct,
       intrinsic_dimension,
       center,
       explained_variance_ratio,
       total_variance,
       fit_error,
       # Selection criteria for neighborhood strategies
       AbstractSelectionCriterion,
       FitErrorCriterion,
       DistortionCriterion,
       SubspaceAngleCriterion,
       evaluate_point,
       evaluate_neighborhood,
       compare_geometries,
       passes_threshold,
       subspace_angle,
       # Neighborhood strategies (composable with any geometry method)
       AbstractNeighborhoodStrategy,
       FixedNeighborhood,
       AdaptiveNeighborhood,
       ExpandingNeighborhood,
       NeighborhoodResult,
       select_neighbors,
       # Geometry estimator (composes strategy + method)
       LocalGeometryEstimator,
       EstimatedGeometry,
       unwrap_geometry,
       used_neighbor_count,
       refinement_iterations,
       max_reconstruction_error,
       recenter,
       # Weighted kNN graph
       WeightedKNNGraph,
       build_weighted_graph,
       # Per-edge weight abstraction (shared by build_weighted_graph and
       # build_geodesic_model)
       AbstractEdgeWeight,
       EuclideanChord,
       TangentProjectedSourceOnly,
       TangentProjectedSymmetricMean,
       TangentProjectedSymmetricMax,
       CurvatureFreeSymmetric,
       compute_edge_weight,
       EstimatorDiagnostics,
       diagnostics,
       # Tangent sharing modes
       AbstractTangentSharingMode,
       NoSharing,
       ShareSimilarTangents,
       node_geometry,
       edge_weight,
       neighbors,
       neighbor_weights,
       neighbors_with_weights,
       total_edge_weight,
       mean_edge_weight,
       edge_weight_statistics,
       unique_geometry_count,
       geometry_sharing_ratio,
       # Geodesic distance model
       GeodesicDistanceModel,
       build_geodesic_model,
       geodesic_distance,
       shortest_path_with_path,
       all_pairs_geodesic_distances,
       # Geodesic refinement
       AbstractGeodesicRefinement,
       RefinedPath,
       refine_path,
       NoRefinement,
       SubdivisionSmoothing,
       CurvatureCorrectedDistance,
       # Graph refinement (curvature-based filtering)
       NodeNeighborhood,
       EdgeNeighborhoodView,
       CurvatureResult,
       AbstractOTSolver,
       # Solvers
       HungarianSolver,
       SinkhornSolver,
       ClpSolver,
       LPReferenceSolver,
       GreedySolver,
       GenericOTSolver,
       # Functions
       uniform_neighborhood,
       create_edge_view,
       is_positive_curvature,
       passes_threshold,
       can_handle,
       compute_curvature,
       filter_graph,
       compute_all_curvatures,
       curvature_statistics,
       # ORC-ManL compatibility profiles (orcml replication preset)
       AbstractOrcMLCompatibilityProfile,
       ManifoldANNDefault,
       OrcmlExact,
       # ORC variant trait (StandardORC vs ORC-ManL)
       AbstractORCConfig,
       StandardORC,
       ORCManL,
       # Graph analysis utilities
       compute_jaccard_scores,
       compute_gabriel_mask

end
