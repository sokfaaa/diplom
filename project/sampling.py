from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
    KMeansSMOTE,
)

import smote_variants as sv


# =========================================================
# Utility
# =========================================================

def _to_numpy_y(y: Any) -> np.ndarray:
    """Convert target to 1D numpy array."""
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if isinstance(y, pd.Series):
        return y.to_numpy().squeeze()
    return np.asarray(y).squeeze()


def _to_numpy_x(X: Any) -> np.ndarray:
    """Convert features to numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def _restore_x_type(
    X_original: Any,
    X_new: np.ndarray,
    keep_pandas: bool = True,
) -> Any:
    """Restore X to DataFrame if original X was DataFrame."""
    if keep_pandas and isinstance(X_original, pd.DataFrame):
        return pd.DataFrame(X_new, columns=X_original.columns)
    return X_new


def _restore_y_type(
    y_original: Any,
    y_new: np.ndarray,
    keep_pandas: bool = True,
) -> Any:
    """Restore y to Series if original y was pandas-like."""
    if keep_pandas and isinstance(y_original, (pd.Series, pd.DataFrame)):
        name = None
        if isinstance(y_original, pd.Series):
            name = y_original.name
        elif isinstance(y_original, pd.DataFrame) and y_original.shape[1] == 1:
            name = y_original.columns[0]
        return pd.Series(y_new, name=name)
    return y_new


def get_class_counts(y: Any) -> Counter:
    """Return class counts."""
    y_np = _to_numpy_y(y)
    return Counter(y_np)


def count_ir(y: Any) -> float:
    """Global imbalance ratio: max_class_size / min_class_size."""
    counts = get_class_counts(y)
    return max(counts.values()) / min(counts.values())


def _safe_neighbor_count(
    y: Any,
    requested_neighbors: int,
) -> int:
    """
    Clamp requested neighbors to a valid value based on the smallest class size.
    For SMOTE-like methods, max valid neighbors is min_class_size - 1.
    """
    counts = get_class_counts(y)
    min_class_size = min(counts.values())
    max_valid = min_class_size - 1

    if max_valid < 1:
        raise ValueError(
            "Слишком мало объектов в одном из классов для метода на основе соседей."
        )

    return min(requested_neighbors, max_valid)


# =========================================================
# Sampler builders: imbalanced-learn
# =========================================================

def build_imblearn_sampler(
    name: str,
    random_state: int = 42,
    sampling_strategy: Any = "not majority",
    **kwargs: Any,
):
    """
    Build an imbalanced-learn sampler.

    Supported:
    - SMOTE
    - BorderlineSMOTE
    - SVMSMOTE
    - ADASYN
    - KMeansSMOTE
    """
    lname = name.lower()

    if lname == "smote":
        k_neighbors = kwargs.get("k_neighbors", 5)
        return SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    if lname in {"borderlinesmote", "borderline-smote"}:
        k_neighbors = kwargs.get("k_neighbors", 5)
        m_neighbors = kwargs.get("m_neighbors", 10)
        kind = kwargs.get("kind", "borderline-1")
        return BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            kind=kind,
        )

    if lname in {"svmsmote", "svm-smote", "borderline-smote svm"}:
        k_neighbors = kwargs.get("k_neighbors", 5)
        m_neighbors = kwargs.get("m_neighbors", 10)
        return SVMSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
        )

    if lname == "adasyn":
        n_neighbors = kwargs.get("n_neighbors", 5)
        return ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )

    if lname in {"kmeanssmote", "k-means smote", "kmeans-smote"}:
        k_neighbors = kwargs.get("k_neighbors", 2)
        cluster_balance_threshold = kwargs.get("cluster_balance_threshold", "auto")
        density_exponent = kwargs.get("density_exponent", "auto")
        return KMeansSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            cluster_balance_threshold=cluster_balance_threshold,
            density_exponent=density_exponent,
        )

    raise ValueError(f"Неизвестный imbalanced-learn sampler: {name}")


# =========================================================
# Sampler builders: smote-variants
# =========================================================

def build_sv_sampler(
    name: str,
    random_state: int = 42,
    **kwargs: Any,
):
    """
    Build a smote-variants sampler.

    Supported:
    - distance_SMOTE
    - cluster_SMOTE
    - CBSO
    - AHC
    - DBSMOTE
    - MWMOTE
    """
    lname = name.lower()

    if lname in {"distance_smote", "distance-smote", "distance-based"}:
        return sv.distance_SMOTE(random_state=random_state, **kwargs)

    if lname in {"cluster_smote", "cluster-smote", "cbo"}:
        return sv.cluster_SMOTE(random_state=random_state, **kwargs)

    if lname == "cbso":
        return sv.CBSO(random_state=random_state, **kwargs)

    if lname == "ahc":
        return sv.AHC(random_state=random_state, **kwargs)

    if lname == "dbsmote":
        return sv.DBSMOTE(random_state=random_state, **kwargs)

    if lname == "mwmote":
        return sv.MWMOTE(random_state=random_state, **kwargs)

    raise ValueError(f"Неизвестный smote-variants sampler: {name}")


# =========================================================
# Apply samplers
# =========================================================

def apply_imblearn_sampler(
    X: Any,
    y: Any,
    name: str,
    random_state: int = 42,
    sampling_strategy: Any = "not majority",
    keep_pandas: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Apply an imbalanced-learn sampler with neighbor safety checks.
    """
    y_np = _to_numpy_y(y)

    lname = name.lower()

    if lname == "smote":
        requested = kwargs.get("k_neighbors", 5)
        safe_k = _safe_neighbor_count(y_np, requested)
        kwargs["k_neighbors"] = safe_k
        if verbose and safe_k != requested:
            print(f"[SMOTE] k_neighbors {requested} -> {safe_k}")

    elif lname in {"borderlinesmote", "borderline-smote"}:
        requested_k = kwargs.get("k_neighbors", 5)
        requested_m = kwargs.get("m_neighbors", 10)
        safe_k = _safe_neighbor_count(y_np, requested_k)
        safe_m = _safe_neighbor_count(y_np, requested_m)
        kwargs["k_neighbors"] = safe_k
        kwargs["m_neighbors"] = safe_m
        if verbose and (safe_k != requested_k or safe_m != requested_m):
            print(f"[BorderlineSMOTE] k_neighbors {requested_k}->{safe_k}, m_neighbors {requested_m}->{safe_m}")

    elif lname in {"svmsmote", "svm-smote", "borderline-smote svm"}:
        requested_k = kwargs.get("k_neighbors", 5)
        requested_m = kwargs.get("m_neighbors", 10)
        safe_k = _safe_neighbor_count(y_np, requested_k)
        safe_m = _safe_neighbor_count(y_np, requested_m)
        kwargs["k_neighbors"] = safe_k
        kwargs["m_neighbors"] = safe_m
        if verbose and (safe_k != requested_k or safe_m != requested_m):
            print(f"[SVMSMOTE] k_neighbors {requested_k}->{safe_k}, m_neighbors {requested_m}->{safe_m}")

    elif lname == "adasyn":
        requested = kwargs.get("n_neighbors", 5)
        safe_n = _safe_neighbor_count(y_np, requested)
        kwargs["n_neighbors"] = safe_n
        if verbose and safe_n != requested:
            print(f"[ADASYN] n_neighbors {requested} -> {safe_n}")

    elif lname in {"kmeanssmote", "k-means smote", "kmeans-smote"}:
        requested = kwargs.get("k_neighbors", 2)
        safe_k = _safe_neighbor_count(y_np, requested)
        kwargs["k_neighbors"] = safe_k
        if verbose and safe_k != requested:
            print(f"[KMeansSMOTE] k_neighbors {requested} -> {safe_k}")

    sampler = build_imblearn_sampler(
        name=name,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        **kwargs,
    )

    X_res, y_res = sampler.fit_resample(X, y_np)

    X_res = _restore_x_type(X, X_res, keep_pandas=keep_pandas)
    y_res = _restore_y_type(y, y_res, keep_pandas=keep_pandas)

    return X_res, y_res


def apply_sv_sampler(
    X: Any,
    y: Any,
    name: str,
    random_state: int = 42,
    keep_pandas: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Apply a smote-variants sampler.
    """
    X_np = _to_numpy_x(X)
    y_np = _to_numpy_y(y)

    sampler = build_sv_sampler(
        name=name,
        random_state=random_state,
        **kwargs,
    )

    X_res, y_res = sampler.sample(X_np, y_np)

    X_res = _restore_x_type(X, X_res, keep_pandas=keep_pandas)
    y_res = _restore_y_type(y, y_res, keep_pandas=keep_pandas)

    return X_res, y_res


def apply_sampler(
    X: Any,
    y: Any,
    name: str,
    random_state: int = 42,
    sampling_strategy: Any = "not majority",
    keep_pandas: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Unified sampler interface.

    Parameters
    ----------
    name : str
        Sampler name.

    imbalanced-learn names:
        SMOTE
        BorderlineSMOTE
        SVMSMOTE
        ADASYN
        KMeansSMOTE

    smote-variants names:
        distance_SMOTE
        cluster_SMOTE
        CBSO
        AHC
        DBSMOTE
        MWMOTE
    """
    imblearn_names = {
        "smote",
        "borderlinesmote",
        "borderline-smote",
        "svmsmote",
        "svm-smote",
        "borderline-smote svm",
        "adasyn",
        "kmeanssmote",
        "k-means smote",
        "kmeans-smote",
    }

    if name.lower() in imblearn_names:
        return apply_imblearn_sampler(
            X=X,
            y=y,
            name=name,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            keep_pandas=keep_pandas,
            verbose=verbose,
            **kwargs,
        )

    return apply_sv_sampler(
        X=X,
        y=y,
        name=name,
        random_state=random_state,
        keep_pandas=keep_pandas,
        **kwargs,
    )


# =========================================================
# Ready-to-use experiment lists
# =========================================================

def get_default_sampler_names() -> list[str]:
    """
    A compact and strong sampler set for experiments.
    """
    return [
        "SMOTE",
        "BorderlineSMOTE",
        "SVMSMOTE",
        "ADASYN",
        "KMeansSMOTE",
        "DBSMOTE",
        "MWMOTE",
    ]


def get_extended_sampler_names() -> list[str]:
    """
    Extended sampler set including extra cluster/distance-based methods.
    """
    return [
        "SMOTE",
        "BorderlineSMOTE",
        "SVMSMOTE",
        "ADASYN",
        "KMeansSMOTE",
        "distance_SMOTE",
        "cluster_SMOTE",
        "AHC",
        "DBSMOTE",
        "MWMOTE",
    ]