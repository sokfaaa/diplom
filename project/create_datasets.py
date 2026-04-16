from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from itertools import product

def make_synthetic_multiclass_dataset(
    n_samples=3000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_repeated=0,
    n_classes=4,
    weights=(0.50, 0.25, 0.15, 0.10),
    class_sep=1.0,
    flip_y=0.02,
    n_clusters_per_class=2,
    random_state=42,
    return_dataframe=True
):

    """
    Генерирует синтетический датасет для экспериментов с multiclass imbalance.

    Parameters
    ----------
    n_samples : int
        Число объектов.
    n_features : int
        Общее число признаков.
    n_informative : int
        Число информативных признаков.
    n_redundant : int
        Число линейно зависимых признаков.
    n_repeated : int
        Число повторяющихся признаков.
    n_classes : int
        Число классов.
    weights : tuple or list
        Доли классов.
    class_sep : float
        Разделимость классов. Чем меньше, тем больше overlap.
    flip_y : float
        Доля шумовых меток.
    n_clusters_per_class : int
        Число кластеров на класс.
    random_state : int
        Seed.
    return_dataframe : bool
        Если True, вернуть pandas DataFrame / Series.

    Returns
    -------
    X, y
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        weights=list(weights),
        class_sep=class_sep,
        flip_y=flip_y,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state
    )

    if return_dataframe:
        columns = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=columns)
        y = pd.Series(y, name="target")

    return X, y


def get_synthetic_dataset_configs():
    fixed_params = {
        "n_samples": 3000,
        "n_features": 20,
        "n_informative": 10,
        "n_redundant": 4,
        "n_repeated": 0,
        "n_classes": 4,
        "flip_y": 0.01,
    }

    ir_levels = {
        "low_ir": [0.35, 0.30, 0.20, 0.15],
        "medium_ir": [0.45, 0.25, 0.20, 0.10],
        "high_ir": [0.55, 0.25, 0.15, 0.05],
    }

    overlap_levels = {
        "low_overlap": 1.5,
        "medium_overlap": 1.0,
        "high_overlap": 0.6,
    }

    cluster_levels = {
        "low_clusters": 1,
        "medium_clusters": 2,
        "high_clusters": 3,
    }

    configs = []
    random_state = 42

    for (ir_name, weights), (ov_name, class_sep), (cl_name, n_clusters) in product(
        ir_levels.items(),
        overlap_levels.items(),
        cluster_levels.items()
    ):
        cfg = {
            "name": f"{ir_name}__{ov_name}__{cl_name}",
            "ir_level": ir_name,
            "overlap_level": ov_name,
            "cluster_level": cl_name,
            "weights": weights,
            "class_sep": class_sep,
            "n_clusters_per_class": n_clusters,
            "random_state": random_state,
        }
        cfg.update(fixed_params)
        configs.append(cfg)
        random_state += 1

    return configs

def generate_dataset_from_config(config):
    X, y = make_classification(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        n_informative=config["n_informative"],
        n_redundant=config["n_redundant"],
        n_repeated=config["n_repeated"],
        n_classes=config["n_classes"],
        n_clusters_per_class=config["n_clusters_per_class"],
        weights=config["weights"],
        class_sep=config["class_sep"],
        flip_y=config["flip_y"],
        random_state=config["random_state"]
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(config["n_features"])])
    y = pd.Series(y, name="target")

    return X, y