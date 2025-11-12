"""Evaluation metrics for ANN algorithms."""

import numpy as np
from typing import List


def compute_recall(
    predicted: List[int], ground_truth: np.ndarray, k: int = None
) -> float:
    """Compute recall@k for a single query.

    Args:
        predicted: List of predicted neighbor indices
        ground_truth: Array of true neighbor indices
        k: Number of neighbors to consider (None = use len(predicted))

    Returns:
        Recall score between 0 and 1
    """
    if k is None:
        k = len(predicted)

    # Take only first k elements from each
    predicted_k = set(predicted[:k])
    true_k = set(ground_truth[:k])

    if len(true_k) == 0:
        return 0.0

    return len(predicted_k & true_k) / len(true_k)


def compute_recall_batch(
    predictions: List[List[int]], ground_truth: np.ndarray, k: int = None
) -> float:
    """Compute average recall@k across multiple queries.

    Args:
        predictions: List of predicted neighbor lists (one per query)
        ground_truth: Array of true neighbors (n_queries, k)
        k: Number of neighbors to consider (None = use prediction length)

    Returns:
        Average recall score between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    recalls = [
        compute_recall(pred, ground_truth[i], k) for i, pred in enumerate(predictions)
    ]
    return np.mean(recalls)
