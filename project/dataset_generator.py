"""
Генератор датасетов для исследования методов борьбы с дисбалансом
в многоклассовой классификации.

Структура выходных данных:
    datasets/
        <dataset_name>/
            X_train.npy, X_test.npy, y_train.npy, y_test.npy
            meta.json  <- параметры генерации (для метамодели)

Запуск:
    python dataset_generator.py                  # все датасеты
    python dataset_generator.py --only moons     # только группа moons
    python dataset_generator.py --list           # список всех конфигов
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy import ndimage
from scipy.stats import pareto
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (
    make_classification,
    make_circles,
    make_gaussian_quantiles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    group: str                          # для фильтрации --only
    base_type: str                      # 'moons' | 'circles' | 'gaussian' | 's_curve' | 'swiss_roll' | 'classification'
    n_samples: int = 3000
    n_features: int = 10               # итоговая размерность (после добавления шумовых признаков)
    n_classes: int = 3
    # дисбаланс: вектор долей (будет нормализован); None = балансированный
    class_weights: Optional[list] = None
    # overlap / сложность
    noise: float = 0.0                 # flip_y / gaussian noise уровень
    overlap: float = 0.0              # расстояние/cluster_std — интерпретируется по типу
    # усложнения
    noise_type: Optional[Literal["gaussian", "t", "gamma", "pareto"]] = None
    noise_scale: float = 0.1
    outlier_frac: float = 0.0          # доля выбросов (pareto)
    spatial_distortion: bool = False    # scipy.ndimage искажение
    # воспроизводимость
    random_state: int = 42
    # дополнительные поля для метамодели
    extra_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Паттерны дисбаланса (для 3/4/5 классов)
# ---------------------------------------------------------------------------

IMBALANCE_PATTERNS = {
    "balanced":     None,
    "mild":         [0.60, 0.25, 0.15],
    "moderate":     [0.70, 0.20, 0.10],
    "severe":       [0.80, 0.15, 0.05],
    "power_law":    [0.60, 0.25, 0.10, 0.05],      # 4 класса
    "long_tail":    [0.50, 0.25, 0.15, 0.07, 0.03],# 5 классов
    "dominant_one": [0.90, 0.05, 0.05],
}

# ---------------------------------------------------------------------------
# Конфигурации датасетов
# ---------------------------------------------------------------------------

CONFIGS: list[DatasetConfig] = []
_seen_names: set[str] = set()
_RS_COUNTER = 10  # глобальный счётчик — каждый _add() получает уникальный rs


def _add(cfg: DatasetConfig):
    """Добавляет конфиг с гарантированно уникальным именем и random_state."""
    global _RS_COUNTER

    # Уникальный random_state — не зависит от того что написано в конфиге
    cfg = DatasetConfig(**{**asdict(cfg), "random_state": _RS_COUNTER})
    _RS_COUNTER += 1

    # Дедупликация имён: если такое имя уже есть — добавляем суффикс
    name = cfg.name
    if name in _seen_names:
        suffix = 2
        while f"{name}_v{suffix}" in _seen_names:
            suffix += 1
        cfg = DatasetConfig(**{**asdict(cfg), "name": f"{name}_v{suffix}"})

    _seen_names.add(cfg.name)
    CONFIGS.append(cfg)


# ── 1. MOONS (чистый) ─────────────────────────────────────────────────────
for imb_name, weights in IMBALANCE_PATTERNS.items():
    n_cls = len(weights) if weights else 3
    if n_cls > 2:
        continue  # make_moons — бинарный, мультикласс через стэкинг ниже
    _add(DatasetConfig(
        name=f"moons_{imb_name}",
        group="moons",
        base_type="moons",
        n_classes=2,
        class_weights=weights,
        noise=0.1,
        n_features=2,
        random_state=10,
    ))

# ── 2. MOONS + шум и выбросы ──────────────────────────────────────────────
for noise_t in ["gaussian", "t", "gamma"]:
    _add(DatasetConfig(
        name=f"moons_noise_{noise_t}",
        group="moons",
        base_type="moons",
        n_classes=2,
        class_weights=[0.75, 0.25],
        noise=0.15,
        n_features=2,
        noise_type=noise_t,
        noise_scale=0.2,
        outlier_frac=0.03,
        random_state=11,
    ))

# ── 3. MOONS + make_classification (гибрид) ───────────────────────────────
for n_feat in [10, 50, 100]:
    _add(DatasetConfig(
        name=f"moons_x_classification_f{n_feat}",
        group="moons_hybrid",
        base_type="moons+classification",
        n_classes=2,
        class_weights=[0.70, 0.30],
        n_features=n_feat,
        noise=0.1,
        random_state=20,
    ))

# ── 3b. MOONS × CIRCLES ───────────────────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate", [0.70, 0.30], 21),
    ("severe",   [0.85, 0.15], 22),
]:
    _add(DatasetConfig(
        name=f"moons_x_circles_{imb_name}",
        group="moons_circles",
        base_type="moons|circles",
        n_classes=2,
        class_weights=weights,
        n_features=4,
        noise=0.1,
        random_state=rs,
        extra_meta={"circles_factor": 0.5},
    ))

# ── 3c. MOONS × GAUSSIAN_QUANTILES (3 класса) ─────────────────────────────
for imb_name, weights, rs in [
    ("moderate",     [0.70, 0.20, 0.10], 23),
    ("severe",       [0.80, 0.15, 0.05], 24),
    ("dominant_one", [0.90, 0.05, 0.05], 25),
]:
    _add(DatasetConfig(
        name=f"moons_x_gaussian_{imb_name}",
        group="moons_gaussian",
        base_type="moons|gaussian_quantiles",
        n_classes=3,
        class_weights=weights,
        n_features=10,
        noise=0.1,
        random_state=rs,
    ))

# ── 3d. MOONS × S_CURVE ───────────────────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate", [0.70, 0.20, 0.10], 26),
    ("severe",   [0.80, 0.15, 0.05], 27),
]:
    _add(DatasetConfig(
        name=f"moons_x_s_curve_{imb_name}",
        group="moons_s_curve",
        base_type="moons|s_curve",
        n_classes=3,
        class_weights=weights,
        n_features=5,
        noise=0.1,
        random_state=rs,
    ))

# ── 3e. MOONS × SWISS_ROLL ────────────────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate", [0.70, 0.20, 0.10], 28),
    ("severe",   [0.80, 0.15, 0.05], 29),
]:
    _add(DatasetConfig(
        name=f"moons_x_swiss_roll_{imb_name}",
        group="moons_swiss_roll",
        base_type="moons|swiss_roll",
        n_classes=3,
        class_weights=weights,
        n_features=5,
        noise=0.1,
        noise_scale=0.1,
        random_state=rs,
    ))

# ── 4. CIRCLES (чистый) ───────────────────────────────────────────────────
for factor in [0.3, 0.5, 0.7]:   # overlap через factor
    _add(DatasetConfig(
        name=f"circles_factor{int(factor*10)}",
        group="circles",
        base_type="circles",
        n_classes=2,
        class_weights=[0.70, 0.30],
        overlap=factor,           # = factor в make_circles
        noise=0.05,
        n_features=2,
        random_state=30,
    ))

# ── 5. CIRCLES + искажение ────────────────────────────────────────────────
_add(DatasetConfig(
    name="circles_distorted_imbalanced",
    group="circles",
    base_type="circles",
    n_classes=2,
    class_weights=[0.80, 0.20],
    overlap=0.5,
    noise=0.1,
    spatial_distortion=True,
    n_features=2,
    random_state=31,
))

# ── 6. CIRCLES + make_classification ─────────────────────────────────────
for n_feat in [10, 50]:
    _add(DatasetConfig(
        name=f"circles_x_classification_f{n_feat}",
        group="circles_hybrid",
        base_type="circles+classification",
        n_classes=2,
        class_weights=[0.75, 0.25],
        n_features=n_feat,
        noise=0.1,
        random_state=32,
    ))

# ── 6b. CIRCLES × GAUSSIAN_QUANTILES ────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate",     [0.70, 0.20, 0.10], 33),
    ("severe",       [0.80, 0.15, 0.05], 34),
    ("dominant_one", [0.90, 0.05, 0.05], 35),
]:
    _add(DatasetConfig(
        name=f"circles_x_gaussian_{imb_name}",
        group="circles_gaussian",
        base_type="circles|gaussian_quantiles",
        n_classes=3,
        class_weights=weights,
        n_features=10,
        noise=0.08,
        overlap=0.5,
        random_state=rs,
    ))

# ── 6c. CIRCLES × S_CURVE ────────────────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate", [0.70, 0.20, 0.10], 36),
    ("severe",   [0.80, 0.15, 0.05], 37),
]:
    _add(DatasetConfig(
        name=f"circles_x_s_curve_{imb_name}",
        group="circles_s_curve",
        base_type="circles|s_curve",
        n_classes=3,
        class_weights=weights,
        n_features=5,
        noise=0.08,
        overlap=0.5,
        random_state=rs,
    ))

# ── 6d. CIRCLES × SWISS_ROLL ─────────────────────────────────────────────
for imb_name, weights, rs in [
    ("moderate", [0.70, 0.20, 0.10], 38),
    ("severe",   [0.80, 0.15, 0.05], 39),
]:
    _add(DatasetConfig(
        name=f"circles_x_swiss_roll_{imb_name}",
        group="circles_swiss_roll",
        base_type="circles|swiss_roll",
        n_classes=3,
        class_weights=weights,
        n_features=5,
        noise=0.08,
        noise_scale=0.1,
        overlap=0.5,
        random_state=rs,
    ))

# ── 7. GAUSSIAN QUANTILES (многоклассовый) ───────────────────────────────
for n_cls, imb_name in [(3, "moderate"), (3, "severe"), (4, "power_law"), (5, "long_tail")]:
    weights = IMBALANCE_PATTERNS[imb_name]
    if len(weights) != n_cls:
        continue
    _add(DatasetConfig(
        name=f"gaussian_q{n_cls}_{imb_name}",
        group="gaussian",
        base_type="gaussian_quantiles",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        random_state=40,
        extra_meta={"cluster_std": 1.0 + 0.5 * n_cls},
    ))

# ── 8. GAUSSIAN QUANTILES + make_classification ───────────────────────────
for n_feat in [10, 50, 100]:
    _add(DatasetConfig(
        name=f"gaussian_x_classification_f{n_feat}",
        group="gaussian_hybrid",
        base_type="gaussian_quantiles+classification",
        n_classes=3,
        class_weights=IMBALANCE_PATTERNS["severe"],
        n_features=n_feat,
        noise=0.05,
        random_state=41,
    ))

# ── 8b. GAUSSIAN × S_CURVE ───────────────────────────────────────────────
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 42),
    ("severe",    [0.80, 0.15, 0.05], 3, 43),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 44),
]:
    _add(DatasetConfig(
        name=f"gaussian_x_s_curve_c{n_cls}_{imb_name}",
        group="gaussian_s_curve",
        base_type="gaussian_quantiles|s_curve",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        noise_scale=0.05,
        random_state=rs,
    ))

# ── 8c. GAUSSIAN × SWISS_ROLL ─────────────────────────────────────────────
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 45),
    ("severe",    [0.80, 0.15, 0.05], 3, 46),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 47),
]:
    _add(DatasetConfig(
        name=f"gaussian_x_swiss_roll_c{n_cls}_{imb_name}",
        group="gaussian_swiss_roll",
        base_type="gaussian_quantiles|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        noise_scale=0.1,
        random_state=rs,
    ))

# ── 9. S_CURVE (многообразие) ─────────────────────────────────────────────
for n_cls in [3, 5]:
    _add(DatasetConfig(
        name=f"s_curve_{n_cls}cls",
        group="s_curve",
        base_type="s_curve",
        n_classes=n_cls,
        class_weights=IMBALANCE_PATTERNS["moderate"] if n_cls == 3 else IMBALANCE_PATTERNS["long_tail"],
        n_features=3,
        noise_type="gaussian",
        noise_scale=0.05,
        random_state=50,
    ))

# ── 10. S_CURVE + make_classification ────────────────────────────────────
for n_feat in [10, 50]:
    _add(DatasetConfig(
        name=f"s_curve_x_classification_f{n_feat}",
        group="s_curve_hybrid",
        base_type="s_curve+classification",
        n_classes=3,
        class_weights=IMBALANCE_PATTERNS["severe"],
        n_features=n_feat,
        random_state=51,
    ))

# ── 10b. S_CURVE × SWISS_ROLL ────────────────────────────────────────────
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 52),
    ("severe",    [0.80, 0.15, 0.05], 3, 53),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 54),
    ("long_tail", [0.50, 0.25, 0.15, 0.07, 0.03], 5, 55),
]:
    _add(DatasetConfig(
        name=f"s_curve_x_swiss_roll_c{n_cls}_{imb_name}",
        group="s_curve_swiss_roll",
        base_type="s_curve|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=6,
        noise_scale=0.1,
        random_state=rs,
    ))

# ── 11. SWISS ROLL (многообразие 3D) ─────────────────────────────────────
for n_cls in [3, 4]:
    _add(DatasetConfig(
        name=f"swiss_roll_{n_cls}cls",
        group="swiss_roll",
        base_type="swiss_roll",
        n_classes=n_cls,
        class_weights=IMBALANCE_PATTERNS["moderate"] if n_cls == 3 else IMBALANCE_PATTERNS["power_law"],
        n_features=3,
        noise_type="gaussian",
        noise_scale=0.1,
        outlier_frac=0.02,
        random_state=60,
    ))

# ── 12. SWISS ROLL + make_classification ─────────────────────────────────
for n_feat in [10, 50]:
    _add(DatasetConfig(
        name=f"swiss_roll_x_classification_f{n_feat}",
        group="swiss_roll_hybrid",
        base_type="swiss_roll+classification",
        n_classes=3,
        class_weights=IMBALANCE_PATTERNS["severe"],
        n_features=n_feat,
        random_state=61,
    ))

# ── 12b. ТРОЙНЫЕ ГИБРИДЫ ─────────────────────────────────────────────────
# moons | circles | gaussian_quantiles
for imb_name, weights, n_cls, rs in [
    ("moderate",     [0.70, 0.20, 0.10], 3, 62),
    ("severe",       [0.80, 0.15, 0.05], 3, 63),
    ("dominant_one", [0.90, 0.05, 0.05], 3, 64),
]:
    _add(DatasetConfig(
        name=f"moons_circles_gaussian_c{n_cls}_{imb_name}",
        group="triple_hybrid",
        base_type="moons|circles|gaussian_quantiles",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        noise=0.08,
        overlap=0.5,
        random_state=rs,
    ))

# moons | s_curve | swiss_roll
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 65),
    ("severe",    [0.80, 0.15, 0.05], 3, 66),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 67),
]:
    _add(DatasetConfig(
        name=f"moons_s_curve_swiss_roll_c{n_cls}_{imb_name}",
        group="triple_hybrid",
        base_type="moons|s_curve|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=8,
        noise=0.1,
        noise_scale=0.1,
        random_state=rs,
    ))

# circles | gaussian_quantiles | s_curve
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 68),
    ("severe",    [0.80, 0.15, 0.05], 3, 69),
    ("long_tail", [0.50, 0.25, 0.15, 0.07, 0.03], 5, 70),
]:
    _add(DatasetConfig(
        name=f"circles_gaussian_s_curve_c{n_cls}_{imb_name}",
        group="triple_hybrid",
        base_type="circles|gaussian_quantiles|s_curve",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        noise=0.08,
        overlap=0.5,
        noise_scale=0.05,
        random_state=rs,
    ))

# gaussian_quantiles | s_curve | swiss_roll
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 71),
    ("severe",    [0.80, 0.15, 0.05], 3, 72),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 73),
]:
    _add(DatasetConfig(
        name=f"gaussian_s_curve_swiss_roll_c{n_cls}_{imb_name}",
        group="triple_hybrid",
        base_type="gaussian_quantiles|s_curve|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=10,
        noise_scale=0.1,
        random_state=rs,
    ))

# moons | circles | s_curve | swiss_roll  (четвёрная)
for imb_name, weights, n_cls, rs in [
    ("severe",    [0.80, 0.15, 0.05], 3, 74),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 75),
]:
    _add(DatasetConfig(
        name=f"moons_circles_s_curve_swiss_roll_c{n_cls}_{imb_name}",
        group="quad_hybrid",
        base_type="moons|circles|s_curve|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=11,
        noise=0.08,
        overlap=0.4,
        noise_scale=0.1,
        random_state=rs,
    ))

# Полный ансамбль: все 5 источников
for imb_name, weights, n_cls, rs in [
    ("moderate",  [0.70, 0.20, 0.10], 3, 76),
    ("severe",    [0.80, 0.15, 0.05], 3, 77),
    ("power_law", [0.60, 0.25, 0.10, 0.05], 4, 78),
    ("long_tail", [0.50, 0.25, 0.15, 0.07, 0.03], 5, 79),
]:
    _add(DatasetConfig(
        name=f"all5_sources_c{n_cls}_{imb_name}",
        group="all5_hybrid",
        base_type="moons|circles|gaussian_quantiles|s_curve|swiss_roll",
        n_classes=n_cls,
        class_weights=weights,
        n_features=13,
        noise=0.08,
        overlap=0.5,
        noise_scale=0.1,
        outlier_frac=0.02,
        random_state=rs,
    ))

# ── 13. make_classification (чистый, разные конфигурации) ─────────────────
for n_feat, n_info, n_cls, imb_name in [
    (10,  5,  3, "mild"),
    (10,  5,  3, "severe"),
    (50,  20, 3, "moderate"),
    (50,  20, 5, "long_tail"),
    (100, 30, 3, "dominant_one"),
    (100, 40, 4, "power_law"),
]:
    weights = IMBALANCE_PATTERNS[imb_name]
    if len(weights) != n_cls:
        continue
    _add(DatasetConfig(
        name=f"classification_f{n_feat}_c{n_cls}_{imb_name}",
        group="classification",
        base_type="classification",
        n_classes=n_cls,
        class_weights=weights,
        n_features=n_feat,
        noise=0.05,
        random_state=70,
        extra_meta={"n_informative": n_info},
    ))

# ── 14. make_classification + выбросы и шум ──────────────────────────────
for noise_t, outlier_frac in [("t", 0.03), ("pareto", 0.05), ("gamma", 0.02)]:
    _add(DatasetConfig(
        name=f"classification_outliers_{noise_t}",
        group="classification",
        base_type="classification",
        n_classes=3,
        class_weights=IMBALANCE_PATTERNS["severe"],
        n_features=20,
        noise=0.1,
        noise_type=noise_t,
        noise_scale=0.3,
        outlier_frac=outlier_frac,
        random_state=71,
        extra_meta={"n_informative": 10},
    ))


# ---------------------------------------------------------------------------
# Движок генерации
# ---------------------------------------------------------------------------

def _normalize_weights(weights: list, n_samples: int) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w /= w.sum()
    counts = (w * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()   # компенсируем округление
    return counts


def _add_noise(X: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.noise_type is None:
        return X
    n, d = X.shape
    if cfg.noise_type == "gaussian":
        noise = rng.normal(0, cfg.noise_scale, (n, d))
    elif cfg.noise_type == "t":
        from scipy.stats import t as t_dist
        noise = t_dist.rvs(df=3, scale=cfg.noise_scale, size=(n, d), random_state=int(rng.integers(1e6)))
    elif cfg.noise_type == "gamma":
        noise = rng.gamma(shape=2, scale=cfg.noise_scale / 2, size=(n, d)) - cfg.noise_scale
    elif cfg.noise_type == "pareto":
        noise = (pareto.rvs(b=1.5, size=(n, d), random_state=int(rng.integers(1e6))) - 1) * cfg.noise_scale
    else:
        noise = np.zeros((n, d))
    return X + noise


def _add_outliers(X: np.ndarray, y: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator):
    if cfg.outlier_frac <= 0:
        return X, y
    n_out = max(1, int(len(X) * cfg.outlier_frac))
    outliers = rng.uniform(X.min(axis=0) * 3, X.max(axis=0) * 3, (n_out, X.shape[1]))
    out_labels = rng.integers(0, cfg.n_classes, n_out)
    return np.vstack([X, outliers]), np.concatenate([y, out_labels])


def _spatial_distort(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Применяет случайное пространственное искажение через ndimage."""
    result = X.copy()
    for col in range(X.shape[1]):
        col_2d = result[:, col].reshape(1, -1)
        sigma = rng.uniform(1, 3)
        distorted = ndimage.gaussian_filter(col_2d, sigma=sigma)
        result[:, col] = distorted.ravel()
    return result


def _generate_single_source(
    source: str, n_samples: int, n_classes: int,
    cfg: DatasetConfig, rng: np.random.Generator, rs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует X, y для одного источника.
    Все многообразия нарезаются на n_classes классов по квантилям параметра t.
    Метки сдвигаются так, чтобы каждый источник давал непересекающиеся классы —
    потом они объединяются и ремаппируются в [0, n_classes-1].
    """
    n = n_samples
    if source == "moons":
        X, y = make_moons(n_samples=n, noise=cfg.noise, random_state=rs)
    elif source == "circles":
        factor = max(0.01, min(0.99, cfg.overlap if cfg.overlap > 0 else 0.5))
        X, y = make_circles(n_samples=n, noise=cfg.noise, factor=factor, random_state=rs)
    elif source == "gaussian_quantiles":
        cov = cfg.extra_meta.get("cluster_std", 1.5)
        X, y = make_gaussian_quantiles(
            n_samples=n, n_features=max(2, cfg.n_features // 3),
            n_classes=n_classes, cov=cov, random_state=rs,
        )
    elif source == "s_curve":
        X_3d, t = make_s_curve(n_samples=n, noise=cfg.noise_scale, random_state=rs)
        quantiles = np.quantile(t, np.linspace(0, 1, n_classes + 1))
        y = np.digitize(t, quantiles[1:-1])
        X = X_3d
    elif source == "swiss_roll":
        X_3d, t = make_swiss_roll(n_samples=n, noise=cfg.noise_scale, random_state=rs)
        quantiles = np.quantile(t, np.linspace(0, 1, n_classes + 1))
        y = np.digitize(t, quantiles[1:-1])
        X = X_3d
    else:
        raise ValueError(f"Неизвестный источник: {source}")
    return X, y


def _generate_multi_source(
    cfg: DatasetConfig, sources: list[str], rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает датасет из нескольких источников.

    Стратегия:
      1. Каждый источник генерирует n_samples точек.
      2. Метки каждого источника маппируются в [0, n_classes-1]
         через modulo, чтобы все классы присутствовали в каждом источнике.
      3. Признаки горизонтально стэкируются (разные пространства).
      4. Дисбаланс применяется единожды поверх объединённого датасета.
    """
    n_cls = cfg.n_classes
    Xs, ys = [], []

    for i, source in enumerate(sources):
        rs_i = int(rng.integers(1e6))
        X_s, y_s = _generate_single_source(source, cfg.n_samples, n_cls, cfg, rng, rs_i)
        # ремаппинг меток в [0, n_classes-1]
        unique = np.unique(y_s)
        label_map = {old: old % n_cls for old in unique}
        y_s = np.array([label_map[v] for v in y_s])
        Xs.append(X_s)
        ys.append(y_s)

    # Мягкое голосование меток: берём моду по источникам для каждой точки
    y_stack = np.stack(ys, axis=1)   # (n_samples, n_sources)
    from scipy.stats import mode as scipy_mode
    y_combined = scipy_mode(y_stack, axis=1, keepdims=False).mode.ravel()

    # Горизонтальный стэк признаков
    X_combined = np.hstack(Xs)

    # Если итоговая размерность > cfg.n_features — PCA-подрезаем
    if X_combined.shape[1] > cfg.n_features:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=cfg.n_features, random_state=cfg.random_state)
        X_combined = pca.fit_transform(X_combined)

    return X_combined, y_combined


def _generate_base(cfg: DatasetConfig, rng: np.random.Generator):
    """Генерирует базовые X, y в зависимости от base_type."""
    rs = int(rng.integers(1e6))
    n = cfg.n_samples
    bt = cfg.base_type

    # ── чистые базовые генераторы ──────────────────────────────────────
    if bt == "moons":
        X, y = make_moons(n_samples=n, noise=cfg.noise, random_state=rs)

    elif bt == "circles":
        factor = max(0.01, min(0.99, cfg.overlap if cfg.overlap > 0 else 0.5))
        X, y = make_circles(n_samples=n, noise=cfg.noise, factor=factor, random_state=rs)

    elif bt == "gaussian_quantiles":
        cov = cfg.extra_meta.get("cluster_std", 1.0)
        X, y = make_gaussian_quantiles(
            n_samples=n, n_features=cfg.n_features,
            n_classes=cfg.n_classes, cov=cov, random_state=rs,
        )

    elif bt == "s_curve":
        X_3d, t = make_s_curve(n_samples=n, noise=cfg.noise_scale, random_state=rs)
        # разбиваем на классы по значению t
        quantiles = np.quantile(t, np.linspace(0, 1, cfg.n_classes + 1))
        y = np.digitize(t, quantiles[1:-1])
        X = X_3d

    elif bt == "swiss_roll":
        X_3d, t = make_swiss_roll(n_samples=n, noise=cfg.noise_scale, random_state=rs)
        quantiles = np.quantile(t, np.linspace(0, 1, cfg.n_classes + 1))
        y = np.digitize(t, quantiles[1:-1])
        X = X_3d

    elif bt == "classification":
        n_info = cfg.extra_meta.get("n_informative", max(2, cfg.n_features // 3))
        n_red = min(2, cfg.n_features - n_info - 1)
        weights = [w / sum(cfg.class_weights) for w in cfg.class_weights] if cfg.class_weights else None
        X, y = make_classification(
            n_samples=n,
            n_features=cfg.n_features,
            n_informative=n_info,
            n_redundant=n_red,
            n_classes=cfg.n_classes,
            weights=weights,
            flip_y=cfg.noise,
            random_state=rs,
        )
        return X, y   # дисбаланс уже встроен

    # ── гибрид: make_X + make_classification (старый синтаксис "+") ───
    elif "+" in bt and "|" not in bt:
        left, right = bt.split("+")
        left_cfg = DatasetConfig(**{**asdict(cfg), "base_type": left, "n_features": 2})
        X_left, y_left = _generate_base(left_cfg, rng)
        n_extra = max(cfg.n_features - X_left.shape[1], 2)
        n_info = max(2, n_extra // 2)
        n_red = min(2, n_extra - n_info)
        X_right, _ = make_classification(
            n_samples=len(X_left),
            n_features=n_extra,
            n_informative=n_info,
            n_redundant=n_red,
            n_classes=max(2, cfg.n_classes),
            random_state=rs + 1,
        )
        X = np.hstack([X_left, X_right])
        y = y_left
        return X, y

    # ── гибрид: произвольные комбинации через "|" ──────────────────────
    elif "|" in bt:
        sources = bt.split("|")
        return _generate_multi_source(cfg, sources, rng)

    else:
        raise ValueError(f"Неизвестный base_type: {bt}")

    return X, y


def _apply_imbalance(X: np.ndarray, y: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator):
    """Субсэмплирует классы согласно заданным весам."""
    if cfg.class_weights is None:
        return X, y
    classes = np.unique(y)
    n_cls = len(classes)
    weights = cfg.class_weights[:n_cls]
    counts = _normalize_weights(weights, cfg.n_samples)

    X_parts, y_parts = [], []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        if cnt >= len(idx):
            chosen = idx
        else:
            chosen = rng.choice(idx, cnt, replace=False)
        X_parts.append(X[chosen])
        y_parts.append(y[chosen])

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def generate_dataset(cfg: DatasetConfig, output_dir: Path) -> dict:
    """Генерирует один датасет, сохраняет файлы, возвращает мета-словарь."""
    rng = np.random.default_rng(cfg.random_state)

    # 1. Базовая структура
    X, y = _generate_base(cfg, rng)

    # 2. Дисбаланс (для генераторов, где он не встроен)
    if cfg.base_type != "classification":
        X, y = _apply_imbalance(X, y, cfg, rng)

    # 3. Дополнительный шум признаков
    X = _add_noise(X, cfg, rng)

    # 4. Выбросы
    X, y = _add_outliers(X, y, cfg, rng)

    # 5. Пространственное искажение
    if cfg.spatial_distortion:
        X = _spatial_distort(X, rng)

    # 6. Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 7. Train / test split (стратифицированный)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=cfg.random_state
        )
    except ValueError:
        # fallback без стратификации, если какой-то класс слишком мал
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=cfg.random_state
        )

    # 8. Сохранение
    ds_dir = output_dir / cfg.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "X_train.npy", X_train)
    np.save(ds_dir / "X_test.npy",  X_test)
    np.save(ds_dir / "y_train.npy", y_train)
    np.save(ds_dir / "y_test.npy",  y_test)

    # 9. Метаданные
    classes, train_counts = np.unique(y_train, return_counts=True)
    actual_ir = float(train_counts.max() / train_counts.min()) if len(train_counts) > 1 else 1.0
    actual_weights = (train_counts / train_counts.sum()).tolist()

    meta = {
        # --- параметры генерации ---
        "name":              cfg.name,
        "group":             cfg.group,
        "base_type":         cfg.base_type,
        "random_state":      cfg.random_state,
        # --- размерность ---
        "n_samples_total":   int(len(y)),
        "n_samples_train":   int(len(y_train)),
        "n_samples_test":    int(len(y_test)),
        "n_features":        int(X.shape[1]),
        "n_classes":         int(len(classes)),
        # --- дисбаланс ---
        "target_weights":    cfg.class_weights,
        "actual_weights":    actual_weights,
        "imbalance_ratio":   actual_ir,       # max_class / min_class
        "class_counts_train": train_counts.tolist(),
        # --- сложность ---
        "noise":             cfg.noise,
        "overlap":           cfg.overlap,
        "noise_type":        cfg.noise_type,
        "noise_scale":       cfg.noise_scale,
        "outlier_frac":      cfg.outlier_frac,
        "spatial_distortion": cfg.spatial_distortion,
        # --- доп. поля ---
        **cfg.extra_meta,
    }

    with open(ds_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ---------------------------------------------------------------------------
# EXTREME CONFIGS — датасеты, покрывающие "хвосты" пространства мета-признаков
# Именно они будут выбросами на PCA-графике и дадут метамодели сигнал из разных зон
# ---------------------------------------------------------------------------

# ── E1. Очень высокая размерность (n_features = 200, 500) ─────────────────
for n_feat, rs in [(200, 200), (500, 201)]:
    for imb_name, weights, n_cls in [
        ("severe",       [0.80, 0.15, 0.05], 3),
        ("dominant_one", [0.90, 0.05, 0.05], 3),
        ("power_law",    [0.60, 0.25, 0.10, 0.05], 4),
    ]:
        _add(DatasetConfig(
            name=f"extreme_hd_f{n_feat}_c{n_cls}_{imb_name}",
            group="extreme_high_dim",
            base_type="classification",
            n_samples=5000,
            n_features=n_feat,
            n_classes=n_cls,
            class_weights=weights,
            noise=0.05,
            random_state=rs,
            extra_meta={"n_informative": n_feat // 5},
        ))

# ── E2. Очень высокая размерность + гибрид ────────────────────────────────
for n_feat, base, rs in [
    (200, "gaussian_quantiles|s_curve|swiss_roll",               202),
    (200, "moons|circles|gaussian_quantiles|s_curve|swiss_roll", 203),
    (500, "gaussian_quantiles|swiss_roll",                       204),
    (500, "moons|circles|gaussian_quantiles|s_curve|swiss_roll", 205),
]:
    _add(DatasetConfig(
        name=f"extreme_hd_f{n_feat}_{base.replace('|','_')[:30]}",
        group="extreme_high_dim_hybrid",
        base_type=base,
        n_samples=5000,
        n_features=n_feat,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        noise=0.08,
        noise_scale=0.1,
        random_state=rs,
    ))

# ── E3. Экстремальный дисбаланс IR > 20 ──────────────────────────────────
_EXTREME_WEIGHTS = {
    2: [
        [0.95, 0.05],   # IR≈19
        [0.97, 0.03],   # IR≈32
        [0.99, 0.01],   # IR≈99
    ],
    3: [
        [0.90, 0.07, 0.03],   # IR=30
        [0.93, 0.05, 0.02],   # IR=46
        [0.95, 0.04, 0.01],   # IR=95
    ],
    4: [
        [0.85, 0.08, 0.05, 0.02],   # IR=42
        [0.90, 0.06, 0.03, 0.01],   # IR=90
    ],
    5: [
        [0.80, 0.10, 0.05, 0.03, 0.02],  # IR=40
        [0.88, 0.06, 0.03, 0.02, 0.01],  # IR=88
    ],
}

_rs_e3 = 210
for n_cls, weight_list in _EXTREME_WEIGHTS.items():
    for weights in weight_list:
        for base_type, n_feat, group_suffix in [
            ("classification",   20, "clf"),
            ("gaussian_quantiles", 10, "gauss"),
            ("moons|circles",     4, "pair"),
            ("moons|circles|gaussian_quantiles|s_curve|swiss_roll", 13, "all5"),
        ]:
            ir_approx = int(max(weights) / min(weights))
            _add(DatasetConfig(
                name=f"extreme_ir{ir_approx}_c{n_cls}_{group_suffix}_rs{_rs_e3}",
                group="extreme_imbalance",
                base_type=base_type,
                n_samples=5000,
                n_features=n_feat,
                n_classes=n_cls,
                class_weights=weights,
                noise=0.05,
                noise_scale=0.05,
                random_state=_rs_e3,
                extra_meta={"n_informative": max(2, n_feat // 3)},
            ))
            _rs_e3 += 1

# ── E4. Большие датасеты (n_samples = 10k, 20k) ───────────────────────────
for n_samp, rs in [(10_000, 300), (20_000, 301)]:
    for base_type, n_feat, n_cls, weights in [
        ("classification",  20, 3, [0.80, 0.15, 0.05]),
        ("classification",  50, 3, [0.90, 0.05, 0.05]),
        ("classification", 100, 4, [0.60, 0.25, 0.10, 0.05]),
        ("gaussian_quantiles", 10, 3, [0.80, 0.15, 0.05]),
        ("moons|gaussian_quantiles", 10, 3, [0.70, 0.20, 0.10]),
        ("gaussian_quantiles|swiss_roll", 10, 4, [0.60, 0.25, 0.10, 0.05]),
        ("moons|circles|gaussian_quantiles|s_curve|swiss_roll", 13, 3, [0.80, 0.15, 0.05]),
        ("moons|circles|gaussian_quantiles|s_curve|swiss_roll", 13, 5,
         [0.50, 0.25, 0.15, 0.07, 0.03]),
    ]:
        ir_approx = int(max(weights) / min(weights))
        _add(DatasetConfig(
            name=f"extreme_large_n{n_samp}_ir{ir_approx}_{base_type.replace('|','_')[:25]}_c{n_cls}",
            group="extreme_large",
            base_type=base_type,
            n_samples=n_samp,
            n_features=n_feat,
            n_classes=n_cls,
            class_weights=weights,
            noise=0.05,
            noise_scale=0.1,
            random_state=rs,
            extra_meta={"n_informative": max(2, n_feat // 3)},
        ))

# ── E5. Экстремальный шум и выбросы ──────────────────────────────────────
for noise_type, noise_scale, outlier_frac, rs in [
    ("t",      0.5, 0.10, 310),
    ("pareto", 0.8, 0.15, 311),
    ("gamma",  0.6, 0.12, 312),
    ("t",      1.0, 0.20, 313),
    ("pareto", 1.5, 0.25, 314),
]:
    for base_type, n_feat, n_cls, weights in [
        ("classification",    20, 3, [0.80, 0.15, 0.05]),
        ("gaussian_quantiles", 10, 3, [0.70, 0.20, 0.10]),
        ("moons|circles|gaussian_quantiles|s_curve|swiss_roll", 13, 3, [0.80, 0.15, 0.05]),
    ]:
        _add(DatasetConfig(
            name=f"extreme_noise_{noise_type}_sc{int(noise_scale*10)}_out{int(outlier_frac*100)}_{base_type[:10]}_c{n_cls}_rs{rs}",
            group="extreme_noise",
            base_type=base_type,
            n_samples=3000,
            n_features=n_feat,
            n_classes=n_cls,
            class_weights=weights,
            noise=0.15,
            noise_type=noise_type,
            noise_scale=noise_scale,
            outlier_frac=outlier_frac,
            spatial_distortion=(rs % 2 == 0),
            random_state=rs,
            extra_meta={"n_informative": max(2, n_feat // 3)},
        ))

# ── E6. Комбо: высокая размерность + экстремальный IR + шум ──────────────
_COMBO_SPECS = [
    # n_feat, n_samp, n_cls, weights, noise_type, noise_sc, out_frac, base, rs
    (200, 5000,  3, [0.95, 0.04, 0.01],             "pareto", 0.5, 0.10,
     "moons|circles|gaussian_quantiles|s_curve|swiss_roll", 320),
    (500, 8000,  3, [0.93, 0.05, 0.02],             "t",      0.8, 0.08,
     "classification", 321),
    (200, 10000, 4, [0.85, 0.08, 0.05, 0.02],       "gamma",  0.4, 0.05,
     "gaussian_quantiles|s_curve|swiss_roll", 322),
    (100, 20000, 5, [0.80, 0.10, 0.05, 0.03, 0.02], "t",      0.3, 0.04,
     "moons|circles|gaussian_quantiles|s_curve|swiss_roll", 323),
    (500, 5000,  3, [0.97, 0.02, 0.01],             "pareto", 1.0, 0.15,
     "classification", 324),
    (200, 5000,  5, [0.88, 0.06, 0.03, 0.02, 0.01], "t",      0.6, 0.10,
     "gaussian_quantiles|swiss_roll", 325),
]
for n_feat, n_samp, n_cls, weights, noise_type, noise_sc, out_frac, base, rs in _COMBO_SPECS:
    ir_approx = int(max(weights) / min(weights))
    _add(DatasetConfig(
        name=f"extreme_combo_f{n_feat}_n{n_samp}_ir{ir_approx}_c{n_cls}_rs{rs}",
        group="extreme_combo",
        base_type=base,
        n_samples=n_samp,
        n_features=n_feat,
        n_classes=n_cls,
        class_weights=weights,
        noise=0.15,
        noise_type=noise_type,
        noise_scale=noise_sc,
        outlier_frac=out_frac,
        spatial_distortion=True,
        random_state=rs,
        extra_meta={"n_informative": max(2, n_feat // 5)},
    ))

# ── E7. Много классов (7, 8, 10) ─────────────────────────────────────────
for n_cls, weights, base_type, n_feat, rs in [
    (7,  [0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03],
     "gaussian_quantiles", 15, 330),
    (8,  [0.35, 0.18, 0.14, 0.11, 0.09, 0.07, 0.04, 0.02],
     "classification", 20, 331),
    (10, [0.30, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03, 0.01],
     "classification", 30, 332),
    (7,  [0.50, 0.15, 0.12, 0.09, 0.07, 0.05, 0.02],
     "gaussian_quantiles|s_curve|swiss_roll", 15, 333),
    (10, [0.40, 0.14, 0.11, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02, 0.01],
     "moons|circles|gaussian_quantiles|s_curve|swiss_roll", 20, 334),
]:
    ir_approx = int(max(weights) / min(weights))
    _add(DatasetConfig(
        name=f"extreme_manycls{n_cls}_ir{ir_approx}_{base_type[:12]}_f{n_feat}",
        group="extreme_many_classes",
        base_type=base_type,
        n_samples=max(5000, n_cls * 500),
        n_features=n_feat,
        n_classes=n_cls,
        class_weights=weights,
        noise=0.05,
        noise_scale=0.1,
        random_state=rs,
        extra_meta={"n_informative": max(2, n_feat // 3)},
    ))

# ── E8. Малая размерность + экстремальный IR ──────────────────────────────
_rs_e8 = 340
for n_feat in [2, 3]:
    for weights, n_cls in [
        ([0.95, 0.05],       2),
        ([0.97, 0.03],       2),
        ([0.90, 0.07, 0.03], 3),
        ([0.95, 0.04, 0.01], 3),
    ]:
        for base_type, bt_short in [
            ("moons",         "moons"),
            ("circles",       "circles"),
            ("moons|circles", "moons_circles"),
        ]:
            if n_cls > 2 and base_type in ("moons", "circles"):
                continue
            ir_approx = int(max(weights) / min(weights))
            _add(DatasetConfig(
                name=f"extreme_lowd_f{n_feat}_ir{ir_approx}_{bt_short}_c{n_cls}",
                group="extreme_low_dim_high_ir",
                base_type=base_type,
                n_samples=3000,
                n_features=n_feat,
                n_classes=n_cls,
                class_weights=weights,
                noise=0.1,
                random_state=_rs_e8,
            ))
            _rs_e8 += 1

# ── E9. Разреженные: много шумовых признаков, мало информативных ─────────
for n_feat, n_info, rs in [(100, 3, 350), (200, 5, 351), (500, 5, 352)]:
    for weights, n_cls in [
        ([0.80, 0.15, 0.05],       3),
        ([0.90, 0.05, 0.05],       3),
        ([0.60, 0.25, 0.10, 0.05], 4),
    ]:
        _add(DatasetConfig(
            name=f"extreme_sparse_f{n_feat}_info{n_info}_c{n_cls}",
            group="extreme_sparse",
            base_type="classification",
            n_samples=5000,
            n_features=n_feat,
            n_classes=n_cls,
            class_weights=weights,
            noise=0.05,
            random_state=rs,
            extra_meta={"n_informative": n_info},
        ))

_n_extreme = sum(1 for c in CONFIGS if c.group.startswith("extreme"))
print(f"[INFO] Фиксированных конфигов: {len(CONFIGS)}  (extreme: {_n_extreme})")

# ---------------------------------------------------------------------------
# Случайный сэмплер конфигураций
# ---------------------------------------------------------------------------

# Все доступные атомарные источники
_SOURCES = ["moons", "circles", "gaussian_quantiles", "s_curve", "swiss_roll"]

# Пространство параметров для случайного сэмплинга
# Расширено экстремальными значениями — покрываем хвосты распределения
_PARAM_SPACE = {
    "n_samples":         [1000, 2000, 3000, 5000, 10_000, 20_000],
    "n_features":        [2, 3, 5, 10, 20, 50, 100, 200, 500],
    "n_classes":         [2, 3, 4, 5, 7, 10],
    "noise":             [0.0, 0.05, 0.1, 0.15, 0.2],
    "overlap":           [0.0, 0.3, 0.5, 0.7],
    "noise_type":        [None, "gaussian", "t", "gamma", "pareto"],
    "noise_scale":       [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5],
    "outlier_frac":      [0.0, 0.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.25],
    "spatial_distortion": [False, False, False, True],
}

# Паттерны весов — расширены экстремальными IR для каждого числа классов
_WEIGHT_TEMPLATES = {
    2: [
        None,
        [0.5, 0.5],
        [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.85, 0.15],
        [0.9, 0.1], [0.95, 0.05], [0.97, 0.03], [0.99, 0.01],
    ],
    3: [
        None,
        [0.5, 0.3, 0.2], [0.6, 0.25, 0.15], [0.7, 0.2, 0.1],
        [0.8, 0.15, 0.05], [0.85, 0.1, 0.05], [0.9, 0.05, 0.05],
        [0.6, 0.3, 0.1], [0.5, 0.4, 0.1],
        [0.90, 0.07, 0.03], [0.93, 0.05, 0.02], [0.95, 0.04, 0.01],
    ],
    4: [
        None,
        [0.5, 0.25, 0.15, 0.1], [0.6, 0.2, 0.15, 0.05],
        [0.7, 0.15, 0.1, 0.05], [0.8, 0.1, 0.06, 0.04],
        [0.4, 0.3, 0.2, 0.1],
        [0.85, 0.08, 0.05, 0.02], [0.90, 0.06, 0.03, 0.01],
    ],
    5: [
        None,
        [0.5, 0.2, 0.15, 0.1, 0.05], [0.6, 0.15, 0.12, 0.08, 0.05],
        [0.4, 0.25, 0.2, 0.1, 0.05], [0.7, 0.1, 0.08, 0.07, 0.05],
        [0.5, 0.25, 0.15, 0.07, 0.03],
        [0.80, 0.10, 0.05, 0.03, 0.02], [0.88, 0.06, 0.03, 0.02, 0.01],
    ],
    7: [
        [0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03],
        [0.50, 0.15, 0.12, 0.09, 0.07, 0.05, 0.02],
        [0.60, 0.12, 0.09, 0.07, 0.06, 0.04, 0.02],
        [0.70, 0.10, 0.07, 0.05, 0.04, 0.03, 0.01],
    ],
    10: [
        [0.30, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03, 0.01],
        [0.40, 0.14, 0.11, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02, 0.01],
        [0.50, 0.12, 0.09, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01],
    ],
}


def _sample_base_type(rng: np.random.Generator) -> str:
    """
    Случайно выбирает base_type:
      - 40% — один источник
      - 35% — пара
      - 20% — тройка
      -  5% — 4–5 источников
    """
    n_sources = rng.choice([1, 2, 3, 4, 5], p=[0.40, 0.35, 0.20, 0.03, 0.02])
    chosen = rng.choice(_SOURCES, size=int(n_sources), replace=False).tolist()
    if len(chosen) == 1:
        return chosen[0]
    return "|".join(chosen)


def sample_random_configs(n: int, seed: int = 0) -> list[DatasetConfig]:
    """
    Генерирует n случайных DatasetConfig, равномерно покрывая пространство параметров.
    Использует латинский гиперкуб по ключевым осям (n_classes, n_features, IR, n_sources)
    чтобы избежать кластеризации конфигов в одной зоне.
    """
    rng = np.random.default_rng(seed)
    configs = []

    # Сетка для равномерного покрытия: делим [0,1] на n частей по 4 ключевым осям
    # и случайно перемешиваем — это даёт Latin Hypercube Sampling
    def lhs_grid(size):
        grid = np.linspace(0, 1, size, endpoint=False) + (1 / size) * rng.random(size)
        rng.shuffle(grid)
        return grid

    n_classes_grid  = lhs_grid(n)   # → [2,5]
    n_features_grid = lhs_grid(n)   # → выбор из списка
    ir_grid         = lhs_grid(n)   # → влияет на выбор весов
    noise_grid      = lhs_grid(n)   # → noise + noise_scale

    for i in range(n):
        # n_classes: LHS по расширенному набору [2,3,4,5,7,10]
        # Маппируем равномерный [0,1] → индекс в списке классов
        # Чаще берём 2-5 (80%), реже 7,10 (20%)
        _cls_options = [2, 3, 3, 4, 4, 5, 5, 7, 10]
        cls_idx = int(n_classes_grid[i] * len(_cls_options))
        n_cls = _cls_options[min(cls_idx, len(_cls_options) - 1)]

        # n_features: логарифмически через LHS, включая 200 и 500
        # Экстремальные значения — с меньшей вероятностью (в конце списка)
        feat_options = [2, 3, 5, 10, 10, 20, 20, 50, 100, 200, 500]
        feat_idx = int(n_features_grid[i] * len(feat_options))
        n_feat = feat_options[min(feat_idx, len(feat_options) - 1)]

        # n_samples: больше для высокой размерности и многих классов
        if n_feat >= 200 or n_cls >= 7:
            n_samp = int(rng.choice([5000, 10_000, 20_000]))
        elif n_feat >= 50:
            n_samp = int(rng.choice([2000, 3000, 5000, 10_000]))
        else:
            n_samp = int(rng.choice([1000, 2000, 3000, 5000]))

        # class_weights: через ir_grid — низкие значения → ближе к balanced
        templates = _WEIGHT_TEMPLATES[n_cls]
        # ir_grid[i] ближе к 1 → balanced (None или равные), к 0 → экстремальный дисбаланс
        ir_percentile = ir_grid[i]
        if ir_percentile < 0.15:
            weights = None   # балансированный
        else:
            # выбираем из templates, исключая None, пропорционально ir_percentile
            non_none = [w for w in templates if w is not None]
            idx = int(ir_percentile * len(non_none))
            idx = min(idx, len(non_none) - 1)
            weights = non_none[idx]

        # noise — шире диапазон, включая экстремальные
        noise_level = noise_grid[i]
        noise       = float(round(noise_level * 0.2, 3))
        # noise_scale: логарифмически, редко экстремальные
        _ns_options = [0.05, 0.1, 0.1, 0.2, 0.2, 0.3, 0.5, 1.0, 1.5]
        ns_idx = int(noise_level * len(_ns_options))
        noise_scale = _ns_options[min(ns_idx, len(_ns_options) - 1)]

        # outlier_frac: чаще 0, иногда экстремальный
        outlier_frac = float(rng.choice(
            [0.0, 0.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.25],
            p=[0.40, 0.15, 0.10, 0.10, 0.10, 0.07, 0.05, 0.03],
        ))
        noise_type         = rng.choice([None, "gaussian", "t", "gamma", "pareto"],
                                        p=[0.35, 0.25, 0.15, 0.15, 0.10])
        spatial_distortion = bool(rng.random() < 0.08)
        overlap           = float(rng.choice([0.0, 0.3, 0.5, 0.7]))
        rs                = int(rng.integers(1, 100_000))

        base_type = _sample_base_type(rng)

        # группа для CLI-фильтрации
        n_src = len(base_type.split("|"))
        if n_src == 1:
            group = f"rand_{base_type}"
        elif n_src == 2:
            group = "rand_pair"
        elif n_src == 3:
            group = "rand_triple"
        else:
            group = "rand_multi"

        name = f"rand_{i:04d}_{base_type.replace('|','_')}_c{n_cls}_f{n_feat}"

        configs.append(DatasetConfig(
            name=name,
            group=group,
            base_type=base_type,
            n_samples=n_samp,
            n_features=n_feat,
            n_classes=n_cls,
            class_weights=weights,
            noise=noise,
            overlap=overlap,
            noise_type=noise_type,
            noise_scale=noise_scale,
            outlier_frac=outlier_frac,
            spatial_distortion=spatial_distortion,
            random_state=rs,
            extra_meta={"sampler": "random_lhs", "lhs_index": i},
        ))

    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Генератор датасетов для исследования дисбаланса классов",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--output",  default="datasets",
                        help="Папка для сохранения (default: datasets)")
    parser.add_argument("--only",    default=None,
                        help="Фильтр по группе — только фиксированные конфиги")
    parser.add_argument("--list",    action="store_true",
                        help="Показать список фиксированных конфигов и выйти")
    parser.add_argument("--random",  type=int, default=0, metavar="N",
                        help="Сгенерировать N случайных конфигов (LHS-сэмплинг)\n"
                             "  python dataset_generator.py --random 500\n"
                             "  python dataset_generator.py --random 500 --fixed  # + 80 фиксированных\n"
                             "  python dataset_generator.py --random 420 --seed 7  # другой seed")
    parser.add_argument("--fixed",   action="store_true",
                        help="Добавить фиксированные 80 конфигов к случайным")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Random seed для --random (default: 0)")
    args = parser.parse_args()

    # ── --list ────────────────────────────────────────────────────────
    if args.list:
        groups = {}
        for c in CONFIGS:
            groups.setdefault(c.group, []).append(c.name)
        for g, names in groups.items():
            print(f"\n[{g}]")
            for n in names:
                print(f"  {n}")
        print(f"\nФиксированных: {len(CONFIGS)} датасетов")
        print(f"\nДля случайных: python dataset_generator.py --random 500")
        return

    # ── Собираем список конфигов ───────────────────────────────────────
    configs: list[DatasetConfig] = []

    # Фиксированные
    if args.random == 0 or args.fixed:
        fixed = CONFIGS
        if args.only:
            fixed = [c for c in CONFIGS if c.group == args.only]
            if not fixed:
                print(f"Группа '{args.only}' не найдена. Доступные: "
                      f"{sorted({c.group for c in CONFIGS})}")
                return
        configs.extend(fixed)

    # Случайные
    if args.random > 0:
        print(f"Сэмплирую {args.random} случайных конфигов (seed={args.seed})...")
        rand_cfgs = sample_random_configs(args.random, seed=args.seed)
        configs.extend(rand_cfgs)
        print(f"Итого конфигов к генерации: {len(configs)}\n")

    if not configs:
        print("Нет конфигов для генерации. Используй --random N или --fixed.")
        return

    # ── Генерация ──────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Дозаписываем к существующему all_meta.json если он есть
    summary_path = output_dir / "all_meta.json"
    all_meta: list[dict] = []
    if summary_path.exists():
        with open(summary_path) as f:
            all_meta = json.load(f)
        existing_names = {m["name"] for m in all_meta}
        before = len(configs)
        configs = [c for c in configs if c.name not in existing_names]
        if skipped := before - len(configs):
            print(f"  Пропускаем {skipped} уже существующих датасетов.")

    ok = err = 0
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"[{i:>4}/{total}] {cfg.name[:55]:<55}", end=" ", flush=True)
        try:
            meta = generate_dataset(cfg, output_dir)
            all_meta.append(meta)
            print(
                f"OK  "
                f"({meta['n_samples_train']}+{meta['n_samples_test']}"
                f"×{meta['n_features']})  "
                f"IR={meta['imbalance_ratio']:.1f}  "
                f"cls={meta['n_classes']}"
            )
            ok += 1
        except Exception as e:
            print(f"ОШИБКА: {e}")
            err += 1

        # Пишем промежуточный all_meta каждые 50 датасетов
        if i % 50 == 0:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, indent=2, ensure_ascii=False)

    # Финальная запись
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Готово. OK={ok}  Ошибок={err}  Всего в all_meta={len(all_meta)}")
    print(f"  Сводный файл: {summary_path}")


if __name__ == "__main__":
    main()