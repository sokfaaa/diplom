import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from itertools import product
from scipy import stats
from scipy.ndimage import rotate
from scipy.stats import multivariate_normal

# -------------------------------------------
# Новые генераторы нелинейных датасетов
# -------------------------------------------

def make_spiral_multiclass_dataset(
    n_samples=3000,
    n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    noise_std=0.15,          # радиальный/угловой шум
    flip_y=0.02,
    random_state=42,
    return_dataframe=True
):
    """
    Генерирует 4-классовый датасет на основе спиралей:
    - 4 ветви спирали Архимеда с разными начальными углами
    - каждая ветвь = один класс
    - классы пересекаются за счёт шума noise_std
    """
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    # Параметры спирали
    max_radius = 3.0
    n_turns = 2.5
    theta_max = 2 * np.pi * n_turns

    data_2d = []
    labels = []
    for class_idx in range(4):
        n = n_per_class[class_idx]
        # Случайные радиусы от почти 0 до max_radius
        radius = rng.uniform(0.2, max_radius, size=n)
        # Угол пропорционален радиусу (спираль) + начальное смещение
        base_angle = class_idx * (np.pi / 2)  # сдвиг на 90° для каждой ветви
        angle = base_angle + (radius / max_radius) * theta_max
        # Добавляем шум в угол и радиус
        if noise_std > 0:
            angle += rng.normal(0, noise_std, size=n)
            radius += rng.normal(0, noise_std * 0.5, size=n)
            radius = np.clip(radius, 0.1, max_radius + 0.5)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        data_2d.append(np.column_stack([x, y]))
        labels.append(np.full(n, class_idx))

    X_2d = np.vstack(data_2d)
    y = np.concatenate(labels)

    # Перемешиваем
    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    # Добавляем шумовые признаки
    if n_features > 2:
        extra_noise = rng.normal(0, 0.5, size=(len(y), n_features - 2))
        X = np.hstack([X_2d, extra_noise])
    else:
        X = X_2d[:, :n_features]

    # Зашумление меток
    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible_classes = np.arange(4)
        for i in flip_idx:
            new_class = rng.choice(possible_classes[possible_classes != y[i]])
            y[i] = new_class

    if return_dataframe:
        columns = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=columns)
        y = pd.Series(y, name="target")
    return X, y


def make_complex_nonlinear_dataset(
    n_samples=3000,
    n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    distortion=0.3,          # степень искажения/пересечения
    flip_y=0.02,
    random_state=42,
    return_dataframe=True
):
    """
    Генерирует датасет с нелинейными структурами, используя scipy:
    - класс 0: "лепестки" (полярные розы) с шумом
    - класс 1: две анизотропные гауссианы (овалы) с поворотом
    - класс 2: S-образная кривая (логистическая сигмоида)
    - класс 3: концентрические эллипсы с переменной плотностью
    Все 4 класса в одном 2D-пространстве, сильно нелинейные границы.
    """
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    def add_distortion(points, scale=1.0):
        """Добавляет нелинейное искажение координат."""
        noise = rng.normal(0, scale, size=points.shape)
        # небольшой сдвиг всего облака
        points = points + noise
        # нелинейное скручивание (через sin/cos)
        if scale > 0:
            points = np.column_stack([
                points[:, 0] + 0.2 * np.sin(points[:, 1] * 1.5),
                points[:, 1] + 0.2 * np.cos(points[:, 0] * 1.5)
            ])
        return points

    # Класс 0: полярная роза (4 лепестка) с вариацией радиуса
    n0 = n_per_class[0]
    theta0 = rng.uniform(0, 2*np.pi, n0)
    r0 = np.abs(np.cos(2 * theta0)) * 1.8 + rng.normal(0, distortion*0.3, n0)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Класс 1: две гауссианы, разнесённые и повёрнутые
    n1 = n_per_class[1]
    n1_a = n1 // 2
    n1_b = n1 - n1_a
    mean_a = [1.2, 0.8]
    mean_b = [-1.0, -1.0]
    cov_base = np.array([[0.4, 0.2], [0.2, 0.4]])  # эллиптическая форма
    # поворот
    angle1 = np.pi/5
    R = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    cov_a = R @ cov_base @ R.T
    cov_b = cov_base * 1.5  # другой овал
    points1a = multivariate_normal.rvs(mean=mean_a, cov=cov_a, size=n1_a, random_state=rng)
    points1b = multivariate_normal.rvs(mean=mean_b, cov=cov_b, size=n1_b, random_state=rng)
    points1 = np.vstack([points1a, points1b])

    # Класс 2: S-образная кривая (сигмоида) с разбросом
    n2 = n_per_class[2]
    x2 = rng.uniform(-2.5, 2.5, n2)
    # y = сигмоида от x с искажением
    y2 = 2.0 / (1 + np.exp(-3 * x2)) - 1.0
    # добавляем перпендикулярный шум
    y2 += rng.normal(0, distortion*0.4, n2)
    # лёгкое вращение
    coords2 = np.column_stack([x2, y2])
    angle2 = np.pi/8
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    coords2 = coords2 @ R2

    # Класс 3: концентрические эллипсы с радиальным шумом
    n3 = n_per_class[3]
    # берём несколько эллиптических слоёв
    inner = rng.uniform(0.4, 1.0, size=n3//2)
    outer = rng.uniform(1.5, 2.2, size=n3 - n3//2)
    radii = np.concatenate([inner, outer])
    angles3 = rng.uniform(0, 2*np.pi, n3)
    # эллипс: a=1.3, b=0.7
    x3 = 1.3 * radii * np.cos(angles3)
    y3 = 0.7 * radii * np.sin(angles3)
    coords3 = np.column_stack([x3, y3])
    # добавляем нелинейные искажения специфичные для класса 3
    coords3 = add_distortion(coords3, scale=distortion*0.2)

    # Собираем все классы
    X_2d = np.vstack([
        np.column_stack([x0, y0]),
        points1,
        coords2,
        coords3
    ])
    y = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
        np.full(n2, 2, dtype=int),
        np.full(n3, 3, dtype=int)
    ])

    # Общее нелинейное искажение всего пространства (смешивание)
    X_2d = add_distortion(X_2d, scale=distortion*0.15)

    # Перемешиваем
    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    # Добавляем шумовые признаки
    if n_features > 2:
        extra_noise = rng.normal(0, 0.5, size=(len(y), n_features - 2))
        X = np.hstack([X_2d, extra_noise])
    else:
        X = X_2d[:, :n_features]

    # Зашумление меток
    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible_classes = np.arange(4)
        for i in flip_idx:
            new_class = rng.choice(possible_classes[possible_classes != y[i]])
            y[i] = new_class

    if return_dataframe:
        columns = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=columns)
        y = pd.Series(y, name="target")
    return X, y


def make_nonlinear_multiclass_dataset(
    n_samples=3000,
    n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    noise_std=0.1,
    flip_y=0.02,
    random_state=42,
    return_dataframe=True
):
    """Исходный circles+moons (оставлен без изменений)"""
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    n_circles = n_per_class[0] + n_per_class[1]
    X_circ, y_circ = make_circles(
        n_samples=n_circles, noise=noise_std*0.7, factor=0.5,
        random_state=random_state
    )
    X0 = X_circ[y_circ == 0][:n_per_class[0]]
    X1 = X_circ[y_circ == 1][:n_per_class[1]]

    n_moons = n_per_class[2] + n_per_class[3]
    X_moon, y_moon = make_moons(
        n_samples=n_moons, noise=noise_std*0.5,
        random_state=random_state+1
    )
    X2 = X_moon[y_moon == 0][:n_per_class[2]]
    X3 = X_moon[y_moon == 1][:n_per_class[3]]

    shift_x, shift_y = 0.3, -0.2
    angle = np.pi/7
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X2 = X2 @ R + np.array([shift_x, shift_y])
    X3 = X3 @ R + np.array([shift_x, shift_y])

    X_2d = np.vstack([X0, X1, X2, X3])
    y = np.concatenate([
        np.zeros(len(X0)), np.ones(len(X1)),
        np.full(len(X2), 2), np.full(len(X3), 3)
    ])

    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    if n_features > 2:
        extra_noise = rng.normal(0, 0.5, size=(len(y), n_features-2))
        X = np.hstack([X_2d, extra_noise])
    else:
        X = X_2d[:, :n_features]

    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible_classes = np.arange(4)
        for i in flip_idx:
            new_class = rng.choice(possible_classes[possible_classes != y[i]])
            y[i] = new_class

    if return_dataframe:
        columns = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=columns)
        y = pd.Series(y, name="target")
    return X, y


# --------------------------------------------------
# Конфигурации
# --------------------------------------------------
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

    # ----- Линейные (как раньше) -----
    ir_levels_linear = {
        "low_ir": [0.35, 0.30, 0.20, 0.15],
        "medium_ir": [0.45, 0.25, 0.20, 0.10],
        "high_ir": [0.55, 0.25, 0.15, 0.05],
    }
    overlap_levels_linear = {
        "low_overlap": 1.5,
        "medium_overlap": 1.0,
        "high_overlap": 0.6,
    }
    cluster_levels_linear = {
        "low_clusters": 1,
        "medium_clusters": 2,
        "high_clusters": 3,
    }

    configs = []
    random_state = 42

    for (ir_name, weights), (ov_name, class_sep), (cl_name, n_clusters) in product(
        ir_levels_linear.items(),
        overlap_levels_linear.items(),
        cluster_levels_linear.items()
    ):
        cfg = {
            "name": f"linear__{ir_name}__{ov_name}__{cl_name}",
            "dataset_type": "linear",
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

    # ----- Нелинейные circles+moons -----
    nonlinear_ir = {
        "low_ir": [0.35, 0.30, 0.20, 0.15],
        "medium_ir": [0.45, 0.25, 0.20, 0.10],
        "high_ir": [0.55, 0.25, 0.15, 0.05],
    }
    nonlinear_overlap = {
        "low_overlap": 0.05,
        "medium_overlap": 0.15,
        "high_overlap": 0.3,
    }
    for (ir_name, weights), (ov_name, noise_std) in product(
        nonlinear_ir.items(), nonlinear_overlap.items()
    ):
        cfg = {
            "name": f"nonlinear__{ir_name}__{ov_name}",
            "dataset_type": "nonlinear",
            "ir_level": ir_name,
            "overlap_level": ov_name,
            "weights": weights,
            "noise_std": noise_std,
            "random_state": random_state,
        }
        for k, v in fixed_params.items():
            if k not in cfg:
                cfg[k] = v
        configs.append(cfg)
        random_state += 1

    # ----- Спиральные (spiral) -----
    spiral_overlap = {
        "low_overlap": 0.05,
        "medium_overlap": 0.2,
        "high_overlap": 0.35,
    }
    for (ir_name, weights), (ov_name, noise_std) in product(
        nonlinear_ir.items(), spiral_overlap.items()
    ):
        cfg = {
            "name": f"spiral__{ir_name}__{ov_name}",
            "dataset_type": "spiral",
            "ir_level": ir_name,
            "overlap_level": ov_name,
            "weights": weights,
            "noise_std": noise_std,
            "random_state": random_state,
        }
        for k, v in fixed_params.items():
            if k not in cfg:
                cfg[k] = v
        configs.append(cfg)
        random_state += 1

    # ----- Сложные (complex) с distortion -----
    complex_overlap = {
        "low_overlap": 0.1,
        "medium_overlap": 0.25,
        "high_overlap": 0.45,
    }
    for (ir_name, weights), (ov_name, distortion) in product(
        nonlinear_ir.items(), complex_overlap.items()
    ):
        cfg = {
            "name": f"complex__{ir_name}__{ov_name}",
            "dataset_type": "complex",
            "ir_level": ir_name,
            "overlap_level": ov_name,
            "weights": weights,
            "distortion": distortion,
            "random_state": random_state,
        }
        for k, v in fixed_params.items():
            if k not in cfg:
                cfg[k] = v
        configs.append(cfg)
        random_state += 1

    return configs


def generate_dataset_from_config(config):
    dtype = config.get("dataset_type", "linear")
    if dtype == "linear":
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
    elif dtype == "nonlinear":
        X, y = make_nonlinear_multiclass_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            noise_std=config.get("noise_std", 0.1),
            flip_y=config.get("flip_y", 0.01),
            random_state=config["random_state"]
        )
    elif dtype == "spiral":
        X, y = make_spiral_multiclass_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            noise_std=config.get("noise_std", 0.15),
            flip_y=config.get("flip_y", 0.01),
            random_state=config["random_state"]
        )
    elif dtype == "complex":
        X, y = make_complex_nonlinear_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            distortion=config.get("distortion", 0.3),
            flip_y=config.get("flip_y", 0.01),
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dtype}")
    return X, y