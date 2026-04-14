from config import DEFAULT_MINOR_PATTERN
from config import LEVELS

from tqdm import tqdm

import numpy as np
import pandas as pd
import math

from sklearn.datasets import make_classification



#генерация соотношения классов (они убывают)
def generate_weights(n_classes, major_weight, parameter_weights=DEFAULT_MINOR_PATTERN):
    if n_classes < 2:
        raise ValueError("n_classes должно быть >= 2")

    if not (0 < major_weight < 1):
        raise ValueError("major_weight должно быть между 0 и 1")

    if len(parameter_weights) < n_classes - 1:
        raise ValueError("В parameter_weights недостаточно коэффициентов")

    weights = [major_weight]
    remain = 1 - major_weight
    parameters = parameter_weights[:n_classes-1:]
    coeff = remain/sum(parameters)

    for parametr in parameters:
        weights.append(parametr * coeff)
    return weights


#подсчет n_informative, n_reduntant
def compute_feature_split(
    n_features,
    n_classes,
    n_clusters_per_class,
    informative_ratio=0.6,
    redundant_ratio=0.2,
):    
    # базовый расчет
    n_informative = max(2, int(round(n_features * informative_ratio)))
    n_redundant = int(round(n_features * redundant_ratio))
    #flip = int(round(n_features * flip_ration))

    # минимально допустимое число informative для make_classification
    min_informative = math.ceil(math.log2(n_classes * n_clusters_per_class))

    n_informative = max(n_informative, min_informative)

    # защита от переполнения
    if n_informative + n_redundant >= n_features:
        n_redundant = max(0, n_features - n_informative - 1)

    return n_informative, n_redundant

#подбор всех фич, все кроме test_feature остаются фиксированными 
def build_synthetic_config(
    n_samples='easy',
    n_features='easy',
    n_classes='easy',
    imbalance_level="easy",
    overlap_level="easy",
    noise_level="easy",
    cluster_level="easy",
    random_state=42
):
    n_samples = LEVELS["n_samples"][n_samples]
    n_features = LEVELS["n_features"][n_features]
    n_classes = LEVELS["n_classes"][n_classes]
    major_weight = LEVELS["major_weight"][imbalance_level]
    class_sep = LEVELS["overlap"][overlap_level]
    #flip_y = LEVELS["noise"][noise_level]
    n_clusters_per_class = LEVELS["n_clusters"][cluster_level]

    weights = generate_weights(n_classes, major_weight)

    n_informative, n_redundant = compute_feature_split(
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class
    )

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_redundant": n_redundant,
        "n_classes": n_classes,
        "weights": weights,
        "class_sep": class_sep,
        #"flip_y": flip_y,
        "n_clusters_per_class": n_clusters_per_class,
        "random_state": random_state
    }

#генерация синтетических данных
def generate_synthetic_dataset(
    n_samples = 2000, 
    n_features = 15, 
    n_informative=10,
    n_redundant=2,
    n_classes = 3,
    weights=0,
    class_sep = 1.0, 
    n_clusters_per_class = 1,
    random_state = 42):
    
    x,y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant, #процент от кол-ва признаков
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        class_sep=class_sep,
        flip_y=0.2,
        random_state=random_state
    )

    x = pd.DataFrame(x, columns = [f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    return x, y








