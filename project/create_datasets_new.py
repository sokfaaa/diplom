import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import norm, t, gamma, beta
from scipy.ndimage import rotate
from scipy.stats import multivariate_normal
from sklearn.datasets import make_moons, make_circles, make_classification

# ------------------------------------------------------------
# Генератор коррелированных шумовых признаков
# ------------------------------------------------------------
def generate_noise_features(
    n_samples, n_noise,
    correlation=0.0,
    distribution='gaussian',
    random_state=42
):
    """
    Создаёт n_noise признаков с заданной корреляцией и маргинальными распределениями.
    Использует копулу Гаусса: от многомерного нормального распределения
    переходим к нужным квантилям.
    """
    rng = np.random.RandomState(random_state)
    if n_noise == 0:
        return np.empty((n_samples, 0))

    # Корреляционная матрица (Тёплиц)
    if correlation == 0.0:
        cov = np.eye(n_noise)
    else:
        base = np.arange(n_noise)
        cov = correlation ** np.abs(np.subtract.outer(base, base))

    # Многомерное нормальное (копула)
    mean = np.zeros(n_noise)
    Z = rng.multivariate_normal(mean, cov, size=n_samples)

    if distribution == 'gaussian':
        return Z

    # Преобразование: нормальный CDF -> (0,1) -> квантиль нужного распределения
    U = norm.cdf(Z)

    if distribution == 't':
        return t.ppf(U, df=3)
    elif distribution == 'gamma':
        # shape=2, scale=1 даёт скошенное распределение
        return gamma.ppf(U, a=2, scale=1)
    elif distribution == 'beta':
        # beta(2,5) растянуто и центрировано примерно от -3 до 3
        return beta.ppf(U, a=2, b=5) * 6 - 3
    elif distribution == 'mixed':
        # Каждая фича получает своё распределение из пула (циклически)
        pool = ['gaussian', 't', 'gamma', 'beta']
        noise = np.zeros_like(Z)
        for i in range(n_noise):
            dist = pool[i % len(pool)]
            Ui = U[:, i]
            if dist == 'gaussian':
                noise[:, i] = Z[:, i]
            elif dist == 't':
                noise[:, i] = t.ppf(Ui, df=3)
            elif dist == 'gamma':
                noise[:, i] = gamma.ppf(Ui, a=2, scale=1)
            elif dist == 'beta':
                noise[:, i] = beta.ppf(Ui, a=2, b=5) * 6 - 3
        return noise
    else:
        return Z


# ------------------------------------------------------------
# Нелинейные генераторы (4 класса)
# ------------------------------------------------------------
def make_nonlinear_multiclass_dataset(
    n_samples=3000, n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    noise_std=0.1, flip_y=0.02,
    noise_correlation=0.0,
    noise_distribution='gaussian',
    random_state=42, return_dataframe=True
):
    """Круги + полумесяцы (как раньше, но с настраиваемым шумом)."""
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    # Круги
    n_circles = n_per_class[0] + n_per_class[1]
    X_circ, y_circ = make_circles(
        n_samples=n_circles, noise=noise_std * 0.7, factor=0.5,
        random_state=random_state
    )
    X0 = X_circ[y_circ == 0][:n_per_class[0]]
    X1 = X_circ[y_circ == 1][:n_per_class[1]]

    # Полумесяцы
    n_moons = n_per_class[2] + n_per_class[3]
    X_moon, y_moon = make_moons(
        n_samples=n_moons, noise=noise_std * 0.5,
        random_state=random_state + 1
    )
    X2 = X_moon[y_moon == 0][:n_per_class[2]]
    X3 = X_moon[y_moon == 1][:n_per_class[3]]

    # Сдвиг/поворот полумесяцев
    shift = np.array([0.3, -0.2])
    angle = np.pi / 7
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X2 = X2 @ R + shift
    X3 = X3 @ R + shift

    X_2d = np.vstack([X0, X1, X2, X3])
    y = np.concatenate([
        np.zeros(len(X0)), np.ones(len(X1)),
        np.full(len(X2), 2), np.full(len(X3), 3)
    ])

    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    # Добавляем шумовые признаки (остальные n_features-2)
    n_noise = max(0, n_features - 2)
    extra = generate_noise_features(
        len(y), n_noise,
        correlation=noise_correlation,
        distribution=noise_distribution,
        random_state=random_state + 2
    )
    X = np.hstack([X_2d[:, :2], extra]) if n_noise > 0 else X_2d[:, :n_features]

    # Зашумление меток
    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible = np.arange(4)
        for i in flip_idx:
            y[i] = rng.choice(possible[possible != y[i]])

    if return_dataframe:
        cols = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(y, name="target")
    return X, y


def make_spiral_multiclass_dataset(
    n_samples=3000, n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    noise_std=0.15, flip_y=0.02,
    noise_correlation=0.0,
    noise_distribution='gaussian',
    random_state=42, return_dataframe=True
):
    """Спиральные ветви (4 класса)."""
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    max_radius = 3.0
    theta_max = 2 * np.pi * 2.5
    data_2d, labels = [], []
    for k in range(4):
        n = n_per_class[k]
        r = rng.uniform(0.2, max_radius, n)
        base_angle = k * np.pi / 2
        theta = base_angle + (r / max_radius) * theta_max
        if noise_std > 0:
            theta += rng.normal(0, noise_std, n)
            r += rng.normal(0, noise_std * 0.5, n)
            r = np.clip(r, 0.1, max_radius + 0.5)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data_2d.append(np.column_stack([x, y]))
        labels.append(np.full(n, k))

    X_2d = np.vstack(data_2d)
    y = np.concatenate(labels)
    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    n_noise = max(0, n_features - 2)
    extra = generate_noise_features(len(y), n_noise,
                                    correlation=noise_correlation,
                                    distribution=noise_distribution,
                                    random_state=random_state + 2)
    X = np.hstack([X_2d[:, :2], extra]) if n_noise > 0 else X_2d[:, :n_features]

    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible = np.arange(4)
        for i in flip_idx:
            y[i] = rng.choice(possible[possible != y[i]])

    if return_dataframe:
        cols = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(y, name="target")
    return X, y


def make_complex_nonlinear_dataset(
    n_samples=3000, n_features=20,
    weights=(0.50, 0.25, 0.15, 0.10),
    distortion=0.3, flip_y=0.02,
    noise_correlation=0.0,
    noise_distribution='gaussian',
    random_state=42, return_dataframe=True
):
    """Сложный датасет: лепестки, овалы, S-кривая, эллипсы (использует scipy)."""
    rng = np.random.RandomState(random_state)
    n_per_class = np.round(np.array(weights) * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    def _distort(p, scale=1.0):
        p = p + rng.normal(0, scale, size=p.shape)
        if scale > 0:
            p = np.column_stack([
                p[:, 0] + 0.2 * np.sin(p[:, 1] * 1.5),
                p[:, 1] + 0.2 * np.cos(p[:, 0] * 1.5)
            ])
        return p

    # Класс 0: полярная роза
    n0 = n_per_class[0]
    theta = rng.uniform(0, 2*np.pi, n0)
    r0 = np.abs(np.cos(2*theta)) * 1.8 + rng.normal(0, distortion*0.3, n0)
    xy0 = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta)])

    # Класс 1: две гауссианы
    n1 = n_per_class[1]
    n1a = n1 // 2
    n1b = n1 - n1a
    cov_base = np.array([[0.4, 0.2], [0.2, 0.4]])
    R = np.array([[np.cos(np.pi/5), -np.sin(np.pi/5)],
                  [np.sin(np.pi/5),  np.cos(np.pi/5)]])
    cov_a = R @ cov_base @ R.T
    p1a = multivariate_normal.rvs(mean=[1.2, 0.8], cov=cov_a, size=n1a, random_state=rng)
    p1b = multivariate_normal.rvs(mean=[-1.0, -1.0], cov=cov_base*1.5, size=n1b, random_state=rng)
    xy1 = np.vstack([p1a, p1b])

    # Класс 2: S-образная кривая
    n2 = n_per_class[2]
    x2 = rng.uniform(-2.5, 2.5, n2)
    y2 = 2.0 / (1 + np.exp(-3 * x2)) - 1.0
    y2 += rng.normal(0, distortion*0.4, n2)
    xy2 = np.column_stack([x2, y2])
    angle = np.pi/8
    R2 = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle),  np.cos(angle)]])
    xy2 = xy2 @ R2

    # Класс 3: эллипсы
    n3 = n_per_class[3]
    inner = rng.uniform(0.4, 1.0, size=n3//2)
    outer = rng.uniform(1.5, 2.2, size=n3 - n3//2)
    radii = np.concatenate([inner, outer])
    ang = rng.uniform(0, 2*np.pi, n3)
    xy3 = np.column_stack([1.3 * radii * np.cos(ang), 0.7 * radii * np.sin(ang)])
    xy3 = _distort(xy3, scale=distortion*0.2)

    X_2d = np.vstack([xy0, xy1, xy2, xy3])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int),
                        np.full(n2, 2, dtype=int), np.full(n3, 3, dtype=int)])

    X_2d = _distort(X_2d, scale=distortion*0.15)
    idx = rng.permutation(len(y))
    X_2d, y = X_2d[idx], y[idx]

    n_noise = max(0, n_features - 2)
    extra = generate_noise_features(len(y), n_noise,
                                    correlation=noise_correlation,
                                    distribution=noise_distribution,
                                    random_state=random_state + 2)
    X = np.hstack([X_2d[:, :2], extra]) if n_noise > 0 else X_2d[:, :n_features]

    if flip_y > 0:
        n_flip = int(flip_y * len(y))
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        possible = np.arange(4)
        for i in flip_idx:
            y[i] = rng.choice(possible[possible != y[i]])

    if return_dataframe:
        cols = [f"feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(y, name="target")
    return X, y


# ------------------------------------------------------------
# Конфигурации
# ------------------------------------------------------------
def get_synthetic_dataset_configs(extended=False):
    """
    Если extended=False (по умолчанию) — классические 54 датасета.
    Если extended=True — добавляет варьирование flip_y, n_features,
    noise_correlation и noise_distribution для нелинейных датасетов,
    а также flip_y и n_features для линейных.
    """
    # Базовые фиксированные параметры (будут переопределяться в циклах)
    base_fixed = {
        "n_samples": 3000,
        "n_features": 20,
        "n_informative": 10,
        "n_redundant": 4,
        "n_repeated": 0,
        "n_classes": 4,
        "flip_y": 0.01,
    }

    # Уровни дисбаланса
    ir_levels = {
        "low_ir": [0.35, 0.30, 0.20, 0.15],
        "medium_ir": [0.45, 0.25, 0.20, 0.10],
        "high_ir": [0.55, 0.25, 0.15, 0.05],
    }

    configs = []
    rs = 42

    # ----- ЛИНЕЙНЫЕ ДАТАСЕТЫ -----
    overlap_lin = {"low_overlap": 1.5, "medium_overlap": 1.0, "high_overlap": 0.6}
    cluster_lin = {"low_clusters": 1, "medium_clusters": 2, "high_clusters": 3}

    # Дополнительные варьируемые параметры для extended
    if extended:
        flip_y_vals = [0.0, 0.01, 0.03, 0.05]
        n_feat_vals_lin = [10, 20, 50, 100]   # для линейных общее число признаков
        # Для линейных не будем менять correlation/distribution (make_classification сам генерит)
        for (ir_n, w), (ov_n, sep), (cl_n, ncl), flip_y, nf in product(
                ir_levels.items(), overlap_lin.items(), cluster_lin.items(),
                flip_y_vals, n_feat_vals_lin):
            # Подгоняем информативные/избыточные, чтобы сохранить структуру
            n_inf = min(max(2, int(nf * 0.5)), nf - 2)
            n_red = min(int(nf * 0.2), nf - n_inf - 1)
            cfg = {
                "name": f"linear__{ir_n}__{ov_n}__{cl_n}__flip{flip_y}__f{nf}",
                "dataset_type": "linear",
                "ir_level": ir_n,
                "overlap_level": ov_n,
                "cluster_level": cl_n,
                "weights": w,
                "class_sep": sep,
                "n_clusters_per_class": ncl,
                "flip_y": flip_y,
                "n_features": nf,
                "n_informative": n_inf,
                "n_redundant": n_red,
                "n_repeated": 0,
                "n_samples": 3000,
                "n_classes": 4,
                "random_state": rs,
            }
            configs.append(cfg)
            rs += 1
    else:  # обычный набор (как раньше)
        for (ir_n, w), (ov_n, sep), (cl_n, ncl) in product(
                ir_levels.items(), overlap_lin.items(), cluster_lin.items()):
            cfg = {
                "name": f"linear__{ir_n}__{ov_n}__{cl_n}",
                "dataset_type": "linear",
                "ir_level": ir_n, "overlap_level": ov_n, "cluster_level": cl_n,
                "weights": w, "class_sep": sep, "n_clusters_per_class": ncl,
                "random_state": rs,
            }
            cfg.update(base_fixed)
            configs.append(cfg)
            rs += 1

    # ----- НЕЛИНЕЙНЫЕ (nonlinear, spiral, complex) -----
    # ----- НЕЛИНЕЙНЫЕ (nonlinear, spiral, complex) -----
    nonlinear_overlap = {"low_overlap": 0.05, "medium_overlap": 0.15, "high_overlap": 0.3}
    spiral_overlap = {"low_overlap": 0.05, "medium_overlap": 0.2, "high_overlap": 0.35}
    complex_overlap = {"low_overlap": 0.1, "medium_overlap": 0.25, "high_overlap": 0.45}

    type_configs = [
        ("nonlinear", make_nonlinear_multiclass_dataset, nonlinear_overlap, "noise_std"),
        ("spiral", make_spiral_multiclass_dataset, spiral_overlap, "noise_std"),
        ("complex", make_complex_nonlinear_dataset, complex_overlap, "distortion"),
    ]

    if extended:
        flip_y_vals_nl = [0.0, 0.01, 0.05]
        n_feat_vals_nl = [10, 20, 50, 100]
        corr_vals = [0.0, 0.3, 0.7]
        dist_vals = ['gaussian', 't', 'gamma', 'mixed']

        for dtype, _, ov_dict, param_name in type_configs:
            for (ir_n, w), (ov_n, ov_val) in product(ir_levels.items(), ov_dict.items()):
                for flip_y in flip_y_vals_nl:
                    for nf in n_feat_vals_nl:
                        for corr in corr_vals:
                            for dist in dist_vals:
                                if nf < 2:
                                    continue
                                cfg = {
                                    "name": f"{dtype}__{ir_n}__{ov_n}__flip{flip_y}__f{nf}__corr{corr}__dist{dist}",
                                    "dataset_type": dtype,
                                    "ir_level": ir_n,
                                    "overlap_level": ov_n,
                                    "weights": w,
                                    param_name: ov_val,
                                    "flip_y": flip_y,
                                    "n_features": nf,
                                    "noise_correlation": corr,
                                    "noise_distribution": dist,
                                    "n_samples": 3000,
                                    "random_state": rs,
                                }
                                configs.append(cfg)
                                rs += 1
    else:
        for dtype, _, ov_dict, param_name in type_configs:
            for (ir_n, w), (ov_n, ov_val) in product(ir_levels.items(), ov_dict.items()):
                cfg = {
                    "name": f"{dtype}__{ir_n}__{ov_n}",
                    "dataset_type": dtype,
                    "ir_level": ir_n,
                    "overlap_level": ov_n,
                    "weights": w,
                    param_name: ov_val,
                    "flip_y": 0.01,
                    "n_features": 20,
                    "random_state": rs,
                }
                configs.append(cfg)
                rs += 1

    return configs


def generate_dataset_from_config(config):
    dtype = config.get("dataset_type", "linear")
    if dtype == "linear":
        X, y = make_classification(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            n_informative=config.get("n_informative", 10),
            n_redundant=config.get("n_redundant", 4),
            n_repeated=config.get("n_repeated", 0),
            n_classes=config["n_classes"],
            n_clusters_per_class=config.get("n_clusters_per_class", 1),
            weights=config["weights"],
            class_sep=config.get("class_sep", 1.0),
            flip_y=config.get("flip_y", 0.01),
            random_state=config["random_state"]
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(config["n_features"])])
        y = pd.Series(y, name="target")
        return X, y

    elif dtype == "nonlinear":
        return make_nonlinear_multiclass_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            noise_std=config.get("noise_std", 0.1),
            flip_y=config.get("flip_y", 0.01),
            noise_correlation=config.get("noise_correlation", 0.0),
            noise_distribution=config.get("noise_distribution", "gaussian"),
            random_state=config["random_state"]
        )
    elif dtype == "spiral":
        return make_spiral_multiclass_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            noise_std=config.get("noise_std", 0.15),
            flip_y=config.get("flip_y", 0.01),
            noise_correlation=config.get("noise_correlation", 0.0),
            noise_distribution=config.get("noise_distribution", "gaussian"),
            random_state=config["random_state"]
        )
    elif dtype == "complex":
        return make_complex_nonlinear_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            weights=config["weights"],
            distortion=config.get("distortion", 0.3),
            flip_y=config.get("flip_y", 0.01),
            noise_correlation=config.get("noise_correlation", 0.0),
            noise_distribution=config.get("noise_distribution", "gaussian"),
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dtype}")