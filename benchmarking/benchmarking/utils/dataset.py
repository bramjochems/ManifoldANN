"""Dataset downloading and loading utilities."""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from typing import Tuple, Optional

import numpy as np
import h5py


def download_dataset(dataset_name: str, data_dir: str = "data") -> str:
    """Download dataset if not already present.

    Args:
        dataset_name: Name of dataset (e.g., 'fashion-mnist-784-euclidean')
        data_dir: Directory to store datasets

    Returns:
        Path to downloaded dataset file

    Raises:
        SystemExit: If download fails
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, f"{dataset_name}.hdf5")

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
    print(f"Downloading {dataset_name}...")
    print("This may take a few minutes...")

    try:
        urlretrieve(url, dataset_path)
        print("Download complete!")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def load_dataset(
    dataset_path: str, n_train: Optional[int] = None, n_test: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from HDF5 file.

    Args:
        dataset_path: Path to HDF5 dataset file
        n_train: Maximum number of training points (None for all)
        n_test: Maximum number of test queries (None for all)

    Returns:
        Tuple of (train_data, test_queries, ground_truth_neighbors)
    """
    print(f"\nLoading dataset from {dataset_path}...")
    with h5py.File(dataset_path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
        distance_metric = f.attrs.get("distance", "unknown")

        print(f"Original train shape: {train.shape}")
        print(f"Original test shape: {test.shape}")
        print(f"Distance metric: {distance_metric}")

        # Limit training set if requested
        if n_train and train.shape[0] > n_train:
            print(f"\n⚠️  Limiting train set from {train.shape[0]} to {n_train} samples")
            print(f"   Recomputing ground truth neighbors for limited dataset...")
            train = train[:n_train]
            neighbors = _recompute_ground_truth(train, test, distance_metric, neighbors.shape[1])
            print(f"   Ground truth recomputed for {n_train} train samples")

        # Limit test set if requested
        if n_test and test.shape[0] > n_test:
            print(f"   Limiting test set from {test.shape[0]} to {n_test} queries")
            test = test[:n_test]
            neighbors = neighbors[:n_test]

        print(f"Final train shape: {train.shape}")
        print(f"Final test shape: {test.shape}")
        print(f"Ground truth shape: {neighbors.shape}")

        # Sanity check
        _verify_ground_truth(train, test, neighbors, distance_metric)

        return train, test, neighbors


def _recompute_ground_truth(
    train: np.ndarray, test: np.ndarray, distance_metric: str, k: int
) -> np.ndarray:
    """Recompute ground truth neighbors for a subset of training data.

    Args:
        train: Training data (subset)
        test: Test queries
        distance_metric: 'euclidean' or 'angular'
        k: Number of neighbors to find

    Returns:
        Ground truth neighbor indices (n_test, k)
    """
    print(f"   Computing ground truth using vectorized operations...")

    if distance_metric == "euclidean":
        # Vectorized: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        test_norm_sq = np.sum(test**2, axis=1, keepdims=True)  # (n_test, 1)
        train_norm_sq = np.sum(train**2, axis=1)  # (n_train,)
        distances = test_norm_sq + train_norm_sq - 2 * test @ train.T  # (n_test, n_train)
        distances = np.sqrt(np.maximum(distances, 0))  # Avoid numerical issues

    elif distance_metric == "angular":
        # Vectorized: cosine distance = 1 - cosine_similarity
        test_norms = np.linalg.norm(test, axis=1, keepdims=True)
        train_norms = np.linalg.norm(train, axis=1, keepdims=True)

        # Avoid division by zero
        test_norms[test_norms == 0] = 1.0
        train_norms[train_norms == 0] = 1.0

        test_normalized = test / test_norms
        train_normalized = train / train_norms

        similarities = test_normalized @ train_normalized.T  # (n_test, n_train)
        distances = 1 - similarities

    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Get k nearest neighbors using argpartition (O(n) vs O(n log n))
    k_actual = min(k, distances.shape[1])
    neighbors = np.argpartition(distances, k_actual - 1, axis=1)[:, :k_actual]

    # Sort the k nearest neighbors by distance
    for i in range(neighbors.shape[0]):
        neighbors[i] = neighbors[i, np.argsort(distances[i, neighbors[i]])]

    return neighbors


def _verify_ground_truth(
    train: np.ndarray, test: np.ndarray, neighbors: np.ndarray, distance_metric: str
) -> None:
    """Sanity check ground truth by verifying first query's nearest neighbor.

    Args:
        train: Training data
        test: Test queries
        neighbors: Ground truth neighbor indices
        distance_metric: 'euclidean' or 'angular'
    """
    if len(neighbors) == 0 or len(train) == 0:
        return

    first_query = test[0]
    expected_nn = neighbors[0][0]

    if distance_metric == "euclidean":
        dist_to_nn = np.linalg.norm(first_query - train[expected_nn])
        print(f"Sanity check: distance to 1st NN = {dist_to_nn:.4f}")

    elif distance_metric == "angular":
        q_norm = np.linalg.norm(first_query)
        nn_norm = np.linalg.norm(train[expected_nn])
        if q_norm > 0 and nn_norm > 0:
            cos_sim = np.dot(first_query, train[expected_nn]) / (q_norm * nn_norm)
            cos_dist = 1 - cos_sim
            print(f"Sanity check: cosine distance to 1st NN = {cos_dist:.4f}")
