from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================================================
# Utils
# =========================================================

def _to_numpy_y(y: Any) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if isinstance(y, pd.Series):
        return y.to_numpy().squeeze()
    return np.asarray(y).squeeze()


def _to_numpy_x(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def _get_feature_names(X: Any):
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    X_np = np.asarray(X)
    return [f"feature_{i}" for i in range(X_np.shape[1])]


# =========================================================
# Imbalance metrics
# =========================================================

def class_counts(y: Any) -> Counter:
    y_np = _to_numpy_y(y)
    return Counter(y_np)


def imbalance_ratio(y: Any) -> float:
    """
    Global IR = max_class_size / min_class_size
    """
    counts = class_counts(y)
    return max(counts.values()) / min(counts.values())


def imbalance_ratio_per_class(y: Any) -> pd.Series:
    """
    IR_i = N_max / N_i
    """
    counts = class_counts(y)
    max_count = max(counts.values())

    result = {cls: max_count / count for cls, count in counts.items()}
    return pd.Series(result, name="IR_per_class").sort_values(ascending=False)


# =========================================================
# Overlap metrics: F1 (Fisher)
# =========================================================

def fisher_feature_overlap(X: Any, y: Any) -> pd.Series:
    """
    F1: Fisher's discriminant ratio for each feature.
    Higher value -> better class separability.
    Lower value -> more overlap.
    """
    X_np = _to_numpy_x(X)
    y_np = _to_numpy_y(y)
    feature_names = _get_feature_names(X)

    classes = np.unique(y_np)
    overall_mean = np.mean(X_np, axis=0)

    numerator = np.zeros(X_np.shape[1], dtype=float)
    denominator = np.zeros(X_np.shape[1], dtype=float)

    for cls in classes:
        X_c = X_np[y_np == cls]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0, ddof=1) if X_c.shape[0] > 1 else np.zeros(X_np.shape[1])

        numerator += n_c * (mean_c - overall_mean) ** 2
        denominator += n_c * var_c

    fisher_ratio = numerator / (denominator + 1e-12)

    return pd.Series(fisher_ratio, index=feature_names, name="F1_fisher").sort_values(ascending=False)


def fisher_overlap_summary(X: Any, y: Any) -> Dict[str, float]:
    """
    Summary stats for Fisher overlap.
    """
    values = fisher_feature_overlap(X, y)
    return {
        "f1_fisher_mean": float(values.mean()),
        "f1_fisher_max": float(values.max()),
        "f1_fisher_min": float(values.min()),
    }


# =========================================================
# Overlap metrics: F2
# =========================================================

def volume_of_overlap_region(X: Any, y: Any) -> pd.Series:
    """
    F2: overlap by feature ranges.
    Higher value -> more overlap.
    """
    X_np = _to_numpy_x(X)
    y_np = _to_numpy_y(y)
    feature_names = _get_feature_names(X)

    classes = np.unique(y_np)
    results = []

    for j in range(X_np.shape[1]):
        pair_overlaps = []

        for c1, c2 in combinations(classes, 2):
            x1 = X_np[y_np == c1, j]
            x2 = X_np[y_np == c2, j]

            min1, max1 = np.min(x1), np.max(x1)
            min2, max2 = np.min(x2), np.max(x2)

            intersection = max(0.0, min(max1, max2) - max(min1, min2))
            union = max(max1, max2) - min(min1, min2)

            overlap = intersection / union if union > 0 else 0.0
            pair_overlaps.append(overlap)

        results.append(np.mean(pair_overlaps) if pair_overlaps else 0.0)

    return pd.Series(results, index=feature_names, name="F2_overlap").sort_values(ascending=False)


def f2_overlap_summary(X: Any, y: Any) -> Dict[str, float]:
    """
    Summary stats for F2 overlap.
    """
    values = volume_of_overlap_region(X, y)
    return {
        "f2_overlap_mean": float(values.mean()),
        "f2_overlap_max": float(values.max()),
        "f2_overlap_min": float(values.min()),
    }


# =========================================================
# Overlap metrics: N3
# =========================================================

def n3_error_rate(X: Any, y: Any) -> float:
    """
    N3: 1-NN leave-one-out style error using nearest neighbor.
    Higher value -> more local overlap.
    """
    X_np = _to_numpy_x(X)
    y_np = _to_numpy_y(y)

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_np)

    distances, indices = nn.kneighbors(X_np)
    nearest_neighbor_idx = indices[:, 1]
    nearest_neighbor_labels = y_np[nearest_neighbor_idx]

    return float(np.mean(nearest_neighbor_labels != y_np))


def n3_per_class(X: Any, y: Any) -> pd.Series:
    """
    N3 per class.
    """
    X_np = _to_numpy_x(X)
    y_np = _to_numpy_y(y)

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_np)

    distances, indices = nn.kneighbors(X_np)
    nearest_neighbor_idx = indices[:, 1]
    nearest_neighbor_labels = y_np[nearest_neighbor_idx]

    result = {}

    for cls in np.unique(y_np):
        mask = y_np == cls
        result[cls] = float(np.mean(nearest_neighbor_labels[mask] != y_np[mask]))

    return pd.Series(result, name="N3_per_class").sort_values(ascending=False)


# =========================================================
# Combined summary
# =========================================================

def data_complexity_summary(X: Any, y: Any) -> Dict[str, float]:
    """
    Main summary for experiments table.
    """
    result = {
        "ir": float(imbalance_ratio(y)),
        "n3": float(n3_error_rate(X, y)),
    }

    result.update(fisher_overlap_summary(X, y))
    result.update(f2_overlap_summary(X, y))

    return result