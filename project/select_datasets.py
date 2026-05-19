"""
Кластеризация датасетов по мета-признакам и отбор по одному
ближайшему к центроиду из каждого кластера.

Алгоритм:
  1. Загружаем metafeatures.csv
  2. Чистим: дропаем gen_* колонки, NaN-колонки, масштабируем RobustScaler
  3. Снижаем размерность через PCA (сохраняем 95% дисперсии)
  4. KMeans(n_clusters=120, n_init=20) в PCA-пространстве
  5. Для каждого кластера находим датасет с минимальным евклидовым
     расстоянием до центроида → это и есть представитель
  6. Сохраняем:
       selected_datasets.csv   — 120 строк с полными мета-признаками
       clustering_report.csv   — все датасеты с меткой кластера и дистанцией
       cluster_summary.csv     — сводка по кластерам (размер, inertia, представитель)
       plots/                  — PCA-2D визуализация кластеров

Запуск:
  python select_datasets.py                           # стандартный
  python select_datasets.py --input my_mf.csv         # другой входной файл
  python select_datasets.py --clusters 80             # другое число кластеров
  python select_datasets.py --no-plots                # без графиков
  python select_datasets.py --seed 42                 # другой random seed
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Загрузка и очистка
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> pd.DataFrame:
    """
    Загружает файл с мета-признаками.
    Поддерживает:
      - CSV  (.csv)
      - JSON (.json) — список словарей (формат all_meta.json из compute_metafeatures)
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Иногда JSON может быть словарём {name: {...}, ...}
            df = pd.DataFrame.from_dict(data, orient="index")
        else:
            raise ValueError(f"Неподдерживаемая структура JSON в {path}")
        print(f"  Формат: JSON  ({len(df)} записей)")

    elif suffix == ".csv":
        df = pd.read_csv(path)
        print(f"  Формат: CSV  ({len(df)} строк)")

    else:
        # Пробуем угадать по содержимому
        try:
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
            print(f"  Формат: JSON (определён по содержимому)")
        except Exception:
            df = pd.read_csv(path)
            print(f"  Формат: CSV (определён по содержимому)")

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит колонки к единому виду независимо от источника:
      - compute_metafeatures.py → колонка 'dataset_name'
      - all_meta.json генератора → колонка 'name'
      - clustering_report.csv   → колонка 'dataset'
    Всегда гарантирует наличие колонки 'dataset_name'.
    """
    # Унифицируем имя колонки с названием датасета
    if "dataset_name" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "dataset_name"})
        elif "dataset" in df.columns:
            df = df.rename(columns={"dataset": "dataset_name"})

    # Унифицируем колонку группы
    if "group" not in df.columns and "base_type" in df.columns:
        df["group"] = df["base_type"]

    # Разворачиваем list-колонки (target_weights, actual_weights, class_counts_train)
    # — они не нужны для кластеризации, но мешают обработке числовых колонок
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(3)
            if sample.apply(lambda x: isinstance(x, (list, dict))).any():
                df = df.drop(columns=[col])

    return df


def load_and_clean(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Возвращает:
      df_full   — исходный DataFrame (все строки и колонки)
      X_scaled  — нормализованная матрица мета-признаков (только числовые MF)
      mf_cols   — список использованных колонок мета-признаков
    """
    print(f"Загружаю: {input_path}")
    df = _load_file(input_path)
    df = _normalize_columns(df)
    print(f"Загружено: {len(df)} датасетов, {df.shape[1]} колонок")

    # Колонки, которые точно не являются мета-признаками
    NON_MF_PREFIXES = ("dataset_", "group", "gen_", "__")
    NON_MF_EXACT = {
        "name", "dataset", "dataset_name", "group", "base_type",
        "random_state", "noise_type", "spatial_distortion", "sampler",
        "status", "error", "is_representative",
        # из all_meta.json генератора — параметры генерации, не мета-признаки
        "n_samples_total", "n_samples_train", "n_samples_test",
        "target_weights", "actual_weights", "class_counts_train",
        "circles_factor", "cluster_std", "n_informative",
        "lhs_index", "sampler",
    }

    mf_cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in NON_MF_PREFIXES):
            continue
        if c in NON_MF_EXACT:
            continue
        if df[c].dtype == object:
            continue
        mf_cols.append(c)

    if not mf_cols:
        raise ValueError(
            "Не найдено числовых мета-признаков для кластеризации.\n"
            "Убедитесь, что передаёте файл из compute_metafeatures.py "
            "(metafeatures.csv / metafeatures.json), а не all_meta.json генератора."
        )

    # Предупреждение: если мало признаков — вероятно подан all_meta.json генератора
    PYMFE_EXPECTED_MIN = 20  # compute_metafeatures даёт 40+
    if len(mf_cols) < PYMFE_EXPECTED_MIN:
        print(
            f"\n  ⚠️  ВНИМАНИЕ: найдено только {len(mf_cols)} мета-признаков.\n"
            f"     Для качественной кластеризации нужны pymfe-признаки (40+).\n"
            f"     Сначала запустите:\n"
            f"       python compute_metafeatures.py --datasets datasets/\n"
            f"     Затем передайте результат:\n"
            f"       python select_datasets.py --input metafeatures.csv\n"
            f"     Продолжаем на {len(mf_cols)} признаках (результат будет менее точным).\n"
        )

    X_raw = df[mf_cols].copy()

    # ── Замена inf / -inf на NaN (приходят из pymfe на вырожденных данных) ──
    n_inf = np.isinf(X_raw.values).sum()
    if n_inf > 0:
        print(f"  Заменяем {n_inf} значений inf → NaN")
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # ── Дроп колонок с >40% NaN ───────────────────────────────────────
    nan_frac = X_raw.isna().mean()
    drop_cols = nan_frac[nan_frac > 0.40].index.tolist()
    if drop_cols:
        print(f"  Дропаем {len(drop_cols)} колонок с >40% NaN: {drop_cols}")
        X_raw = X_raw.drop(columns=drop_cols)
        mf_cols = [c for c in mf_cols if c not in drop_cols]

    # ── Заполняем оставшиеся NaN медианой ────────────────────────────
    nan_remaining = X_raw.isna().sum().sum()
    if nan_remaining > 0:
        print(f"  Заполняем {nan_remaining} NaN медианами")
        X_raw = X_raw.fillna(X_raw.median())

    # ── Clip экстремальных значений (защита от overflow в PCA/scaler) ─
    # Считаем IQR и обрезаем всё за пределами median ± 10*IQR
    q1  = X_raw.quantile(0.25)
    q3  = X_raw.quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 10 * iqr
    hi  = q3 + 10 * iqr
    clipped = ((X_raw < lo) | (X_raw > hi)).sum().sum()
    if clipped > 0:
        print(f"  Clip {clipped} экстремальных значений (±10 IQR)")
        X_raw = X_raw.clip(lower=lo, upper=hi, axis=1)

    # ── Дроп признаков с нулевой дисперсией ──────────────────────────
    zero_var = X_raw.std() < 1e-10
    if zero_var.any():
        drop_zv = zero_var[zero_var].index.tolist()
        print(f"  Дропаем {len(drop_zv)} константных признаков: {drop_zv}")
        X_raw = X_raw.drop(columns=drop_zv)
        mf_cols = [c for c in mf_cols if c not in drop_zv]

    print(f"  Итого мета-признаков для кластеризации: {len(mf_cols)}")

    # ── Логарифмирование тяжёлохвостых признаков ─────────────────────
    LOG_TRANSFORM_PATTERNS = [
        "nr_inst", "nr_attr", "eq_num_attr",
        "IR", "HDB_mean_f1", "HDB_std_f1",
        "eigenvalues", "cov",
    ]
    log_applied = []
    for col in X_raw.columns:
        if any(pat in col for pat in LOG_TRANSFORM_PATTERNS):
            col_min = X_raw[col].min()
            if col_min >= 0:
                # Стандартный log1p для неотрицательных
                X_raw[col] = np.log1p(X_raw[col])
            else:
                # Shifted log для колонок с отрицательными значениями
                shift = -col_min + 1e-9
                X_raw[col] = np.log1p(X_raw[col] + shift)
            log_applied.append(col)
    if log_applied:
        print(f"  log1p-преобразование: {log_applied}")

    # ── Финальная проверка: убираем любые оставшиеся inf/nan ─────────
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    if X_raw.isna().any().any():
        X_raw = X_raw.fillna(X_raw.median())
    # Последний clip после log-трансформа (на случай если log дал выброс)
    X_raw = X_raw.clip(
        lower=X_raw.quantile(0.001),
        upper=X_raw.quantile(0.999),
        axis=1,
    )

    # RobustScaler — устойчив к выбросам (характерно для мета-признаков)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        index=df.index,
        columns=mf_cols,
    )

    # Финальный guard: sklearn выбрасывает ValueError на inf/overflow
    if not np.isfinite(X_scaled.values).all():
        n_bad = (~np.isfinite(X_scaled.values)).sum()
        print(f"  ⚠️  После скейлинга осталось {n_bad} нефинитных значений — заменяем 0")
        X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, X_scaled, mf_cols


# ---------------------------------------------------------------------------
# 2. Снижение размерности: PCA (для кластеризации) + UMAP + t-SNE (для визуализации)
# ---------------------------------------------------------------------------

def apply_pca(
    X_scaled: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, PCA]:
    """PCA для кластеризации — сохраняет нужную долю дисперсии."""
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled.values)
    n_comp = pca.n_components_
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {X_scaled.shape[1]} признаков → {n_comp} компонент ({explained:.1%} дисперсии)")
    return X_pca, pca


def apply_umap(X_scaled: pd.DataFrame, seed: int = 42) -> np.ndarray | None:
    """UMAP до 2D для визуализации. Возвращает None если не установлен."""
    try:
        import umap as umap_lib
    except ImportError:
        print("  UMAP не установлен: pip install umap-learn")
        return None

    import time
    t0 = time.time()
    n = len(X_scaled)
    # n_neighbors: больше → глобальная структура, меньше → локальная
    # для 200 датасетов оптимально 15-20
    n_neighbors = min(15, n - 1)
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,       # плотность кластеров: 0 = плотные, 1 = размытые
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    X_2d = reducer.fit_transform(X_scaled.values)
    print(f"  UMAP: готово за {time.time()-t0:.1f}s")
    return X_2d


def apply_tsne(X_scaled: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """t-SNE до 2D для визуализации."""
    from sklearn.manifold import TSNE
    import time
    t0 = time.time()
    n = len(X_scaled)
    perplexity = min(30, n // 3)  # перплексия < n/3 — стандартное правило

    # t-SNE лучше работает на уже сниженной размерности
    # Сначала PCA до 50 компонент если признаков больше
    X_in = X_scaled.values
    if X_in.shape[1] > 50:
        pca_pre = PCA(n_components=50, random_state=seed)
        X_in = pca_pre.fit_transform(X_in)

    import sklearn
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    # n_iter переименован в max_iter в sklearn >= 1.5
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_version >= (1, 5):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    tsne = TSNE(**tsne_kwargs)
    X_2d = tsne.fit_transform(X_in)
    print(f"  t-SNE: готово за {time.time()-t0:.1f}s  (perplexity={perplexity})")
    return X_2d


# ---------------------------------------------------------------------------
# 3. KMeans
# ---------------------------------------------------------------------------

def run_kmeans(
    X_pca: np.ndarray,
    n_clusters: int,
    seed: int,
) -> KMeans:
    print(f"\nKMeans(n_clusters={n_clusters}, n_init=20, seed={seed})...")
    km = KMeans(
        n_clusters=n_clusters,
        n_init=20,           # много инициализаций → стабильный результат
        max_iter=500,
        random_state=seed,
    )
    km.fit(X_pca)

    inertia = km.inertia_
    # Silhouette на подвыборке (дорогая метрика)
    sample_size = min(len(X_pca), 2000)
    idx = np.random.default_rng(seed).choice(len(X_pca), sample_size, replace=False)
    sil = silhouette_score(X_pca[idx], km.labels_[idx], metric="euclidean")

    print(f"  Inertia:    {inertia:.2f}")
    print(f"  Silhouette: {sil:.4f}  (на подвыборке {sample_size})")

    return km


# ---------------------------------------------------------------------------
# 4. Отбор представителей кластеров
# ---------------------------------------------------------------------------

def select_representatives(
    df_full: pd.DataFrame,
    X_pca: np.ndarray,
    km: KMeans,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Для каждого кластера выбирает датасет с минимальным евклидовым
    расстоянием до центроида кластера.

    Возвращает:
      df_report     — все датасеты + cluster + dist_to_centroid + is_representative
      df_selected   — только представители (120 строк)
    """
    labels    = km.labels_
    centroids = km.cluster_centers_  # shape: (n_clusters, n_pca_components)

    # Расстояние каждой точки до центроида своего кластера
    dists = np.linalg.norm(X_pca - centroids[labels], axis=1)

    df_report = df_full.copy()
    df_report["cluster"]           = labels
    df_report["dist_to_centroid"]  = dists
    df_report["is_representative"] = False

    representatives: list[int] = []
    for cluster_id in range(km.n_clusters):
        mask = labels == cluster_id
        if not mask.any():
            continue
        cluster_indices = np.where(mask)[0]
        # Ближайший к центроиду
        best_local = np.argmin(dists[cluster_indices])
        best_global = cluster_indices[best_local]
        df_report.at[best_global, "is_representative"] = True
        representatives.append(best_global)

    df_selected = df_report.loc[representatives].copy().reset_index(drop=True)
    df_selected = df_selected.sort_values("cluster").reset_index(drop=True)

    print(f"\nПредставителей отобрано: {len(df_selected)}")

    # Статистика по размерам кластеров
    sizes = pd.Series(labels).value_counts()
    print(f"  Размер кластеров: min={sizes.min()}  median={sizes.median():.0f}"
          f"  max={sizes.max()}  (всего датасетов: {len(labels)})")

    return df_report, df_selected


# ---------------------------------------------------------------------------
# 5. Сводка по кластерам
# ---------------------------------------------------------------------------

def build_cluster_summary(
    df_report: pd.DataFrame,
    df_selected: pd.DataFrame,
    mf_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for cluster_id, rep in df_selected[["cluster", "dataset_name"]].iterrows():
        cluster_mask = df_report["cluster"] == rep["cluster"]
        cluster_df   = df_report[cluster_mask]
        row = {
            "cluster":           rep["cluster"],
            "representative":    rep["dataset_name"],
            "cluster_size":      cluster_mask.sum(),
            "dist_to_centroid":  df_report.loc[
                df_report["dataset_name"] == rep["dataset_name"],
                "dist_to_centroid"
            ].values[0],
        }
        # Среднее по ключевым мета-признакам внутри кластера
        for col in ["IR", "CV", "n_classes" if "n_classes" in df_report else "nr_class",
                    "nr_attr", "nr_inst"]:
            if col in df_report.columns:
                row[f"cluster_mean_{col}"] = cluster_df[col].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Вспомогательные функции проекции для визуализации
# ---------------------------------------------------------------------------

def apply_umap(
    X_scaled: pd.DataFrame,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray | None:
    """
    UMAP — нелинейная проекция в 2D.
    Сохраняет и локальную структуру (кластеры) и глобальную (расстояния между ними).
    n_neighbors: сколько соседей учитывать (больше = глобальнее картина, меньше = локальнее)
    min_dist: минимальное расстояние между точками (меньше = плотнее кластеры)
    """
    try:
        import umap
        print("  UMAP...", end=" ", flush=True)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
            verbose=False,
        )
        X_2d = reducer.fit_transform(X_scaled.values)
        print("готово")
        return X_2d
    except ImportError:
        print("  UMAP пропущен (pip install umap-learn)")
        return None
    except Exception as e:
        print(f"  UMAP ошибка: {e}")
        return None


def apply_tsne(
    X_scaled: pd.DataFrame,
    seed: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray | None:
    """
    t-SNE — нелинейная проекция в 2D.
    Хорошо разделяет кластеры визуально, но не сохраняет глобальные расстояния.
    perplexity: ~число ближайших соседей (5–50, больше для крупных наборов)
    """
    try:
        from sklearn.manifold import TSNE
        import sklearn
        print("  t-SNE...", end=" ", flush=True)
        perp = min(perplexity, max(5.0, len(X_scaled) // 5 - 1))
        # n_iter переименован в max_iter в sklearn >= 1.5
        sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        iter_kwarg = {"max_iter" if sk_version >= (1, 5) else "n_iter": 1000}
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=seed,
            verbose=0,
            **iter_kwarg,
        )
        X_2d = reducer.fit_transform(X_scaled.values)
        print("готово")
        return X_2d
    except Exception as e:
        print(f"  t-SNE ошибка: {e}")
        return None


def apply_kernel_pca(
    X_scaled: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray | None:
    """
    Kernel PCA с RBF-ядром — нелинейное обобщение PCA.
    Не требует доп. зависимостей, промежуточный вариант между PCA и UMAP.
    """
    try:
        from sklearn.decomposition import KernelPCA
        print("  Kernel PCA...", end=" ", flush=True)
        reducer = KernelPCA(
            n_components=2,
            kernel="rbf",
            random_state=seed,
        )
        X_2d = reducer.fit_transform(X_scaled.values)
        print("готово")
        return X_2d
    except Exception as e:
        print(f"  Kernel PCA ошибка: {e}")
        return None




# ---------------------------------------------------------------------------
# 6. Визуализация
# ---------------------------------------------------------------------------

def _draw_scatter(
    ax,
    X_2d: np.ndarray,
    labels: np.ndarray,
    is_rep: np.ndarray,
    palette: list,
    n_clusters: int,
    title: str,
    xlabel: str,
    ylabel: str,
):
    """Рисует один scatter-plot с кластерами и представителями."""
    for cid in range(n_clusters):
        mask = labels == cid
        if not mask.any():
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=palette[cid], s=22, alpha=0.6, linewidths=0,
        )
    ax.scatter(
        X_2d[is_rep, 0], X_2d[is_rep, 1],
        c="white", s=100, edgecolors="black", linewidths=1.3,
        zorder=5, label="Представитель",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc="upper right")


def make_plots(
    X_pca: np.ndarray,
    X_scaled: pd.DataFrame,
    km: KMeans,
    df_report: pd.DataFrame,
    output_dir: Path,
    n_clusters: int,
    seed: int = 42,
):
    output_dir.mkdir(exist_ok=True)

    labels = km.labels_
    is_rep = df_report["is_representative"].values
    palette = sns.color_palette("husl", n_clusters)

    # ── 2D-проекции ───────────────────────────────────────────────────
    print("\n  Считаем проекции для графиков...")

    # PCA — уже есть, берём первые 2 компоненты
    if X_pca.shape[1] >= 2:
        X_pca2d = X_pca[:, :2]
    else:
        X_pca2d = np.hstack([X_pca, np.zeros((len(X_pca), 1))])

    X_kpca  = apply_kernel_pca(X_scaled, seed=seed)
    X_umap  = apply_umap(X_scaled, seed=seed)
    X_tsne  = apply_tsne(X_scaled, seed=seed)

    # Собираем все доступные проекции
    projections = [
        (X_pca2d, "PCA",        "PC1",     "PC2",
         "Линейный. Хорошо показывает глобальную структуру,\nплохо — нелинейные кластеры."),
        (X_kpca,  "Kernel PCA", "KPC1",    "KPC2",
         "Нелинейный PCA (RBF). Промежуточный вариант —\nлучше PCA, но мягче UMAP."),
        (X_umap,  "UMAP",       "UMAP-1",  "UMAP-2",
         "Лучший для кластеризации: сохраняет\nи локальную, и глобальную структуру."),
        (X_tsne,  "t-SNE",      "t-SNE-1", "t-SNE-2",
         "Хорошо разделяет кластеры визуально,\nно расстояния между ними не информативны."),
    ]
    available = [(X2d, name, xl, yl, note)
                 for X2d, name, xl, yl, note in projections
                 if X2d is not None]

    # ── График 1: все проекции рядом (сравнение) ─────────────────────
    n_plots = len(available)
    ncols = min(n_plots, 2)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 8 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, (X2d, name, xl, yl, note) in enumerate(available):
        _draw_scatter(
            axes_flat[i], X2d, labels, is_rep, palette, n_clusters,
            title=f"{name}\n{note}",
            xlabel=xl, ylabel=yl,
        )

    # Скрываем пустые ячейки если нечётное число графиков
    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Сравнение методов проекции  |  {n_clusters} кластеров  |  {len(labels)} датасетов",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    path_compare = output_dir / "clusters_comparison.png"
    fig.savefig(path_compare, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён: {path_compare}")

    # ── Графики 2–5: каждый метод отдельно (крупнее) ─────────────────
    for X2d, name, xl, yl, note in available:
        fig, ax = plt.subplots(figsize=(12, 9))
        _draw_scatter(
            ax, X2d, labels, is_rep, palette, n_clusters,
            title=(f"Кластеризация датасетов по мета-признакам\n"
                   f"{name}  |  {n_clusters} кластеров  |  {note}"),
            xlabel=xl, ylabel=yl,
        )
        plt.tight_layout()
        fname = output_dir / f"clusters_{name.lower().replace(' ', '_')}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {fname}")

    # ── График 5: размеры кластеров ───────────────────────────────────
    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = [palette[i] for i in sizes.index]
    ax.bar(range(len(sizes)), sizes.values, color=colors, edgecolor="none", width=1.0)
    ax.axhline(sizes.mean(), color="black", linestyle="--", linewidth=1,
               label=f"Среднее: {sizes.mean():.1f}")
    ax.set_title("Размер кластеров", fontsize=12)
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Число датасетов")
    ax.legend()
    plt.tight_layout()
    path_sizes = output_dir / "cluster_sizes.png"
    fig.savefig(path_sizes, dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {path_sizes}")

    # ── График 6: heatmap мета-признаков представителей ──────────────
    rep_df = df_report[df_report["is_representative"]].copy()
    num_cols = rep_df.select_dtypes(include="number").columns.tolist()
    exclude = {"cluster", "dist_to_centroid", "is_representative"}
    num_cols = [c for c in num_cols if c not in exclude and not c.startswith("gen_")]
    if len(num_cols) > 1:
        variances = rep_df[num_cols].var().sort_values(ascending=False)
        top_cols = variances.head(15).index.tolist()
        heat_data = rep_df[top_cols].copy()
        heat_norm = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-9)
        heat_norm = heat_norm.sort_values(top_cols[0])

        fig, ax = plt.subplots(figsize=(14, max(6, len(heat_norm) * 0.15)))
        sns.heatmap(
            heat_norm.T, ax=ax, cmap="RdBu_r", center=0,
            xticklabels=False, yticklabels=True,
            linewidths=0, cbar_kws={"shrink": 0.6},
        )
        ax.set_title(
            "Мета-признаки представителей кластеров\n(z-score, топ-15 по дисперсии)",
            fontsize=12,
        )
        ax.set_xlabel("Датасет-представитель")
        plt.tight_layout()
        path_heat = output_dir / "representatives_heatmap.png"
        fig.savefig(path_heat, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path_heat}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Кластеризация датасетов по мета-признакам, отбор представителей"
    )
    parser.add_argument("--input",    default="metafeatures.csv",
                        help="Входной файл с мета-признаками: CSV или JSON\n"
                             "  metafeatures.csv  — выход compute_metafeatures.py\n"
                             "  metafeatures.json — выход compute_metafeatures.py (JSON-режим)\n"
                             "  (default: metafeatures.csv)")
    parser.add_argument("--output",   default=".",
                        help="Папка для выходных файлов (default: .)")
    parser.add_argument("--clusters", type=int, default=120,
                        help="Число кластеров KMeans (default: 120)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--variance", type=float, default=0.95,
                        help="Доля дисперсии для PCA (default: 0.95)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Не строить графики")
    args = parser.parse_args()

    input_csv  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if not input_csv.exists():
        print(f"Файл не найден: {input_csv}")
        return

    np.random.seed(args.seed)

    # ── 1. Загрузка ───────────────────────────────────────────────────
    df_full, X_scaled, mf_cols = load_and_clean(input_csv)

    n_clusters = args.clusters
    if n_clusters >= len(df_full):
        n_clusters = max(2, len(df_full) - 1)
        print(f"  n_clusters скорректировано до {n_clusters} (датасетов: {len(df_full)})")

    # ── 2. PCA ────────────────────────────────────────────────────────
    X_pca, pca = apply_pca(X_scaled, variance_threshold=args.variance)

    # ── 3. KMeans ─────────────────────────────────────────────────────
    km = run_kmeans(X_pca, n_clusters, args.seed)

    # ── 4. Отбор представителей ───────────────────────────────────────
    df_report, df_selected = select_representatives(df_full, X_pca, km)

    # ── 5. Сводка ─────────────────────────────────────────────────────
    df_summary = build_cluster_summary(df_report, df_selected, mf_cols)

    # ── 6. Сохранение ─────────────────────────────────────────────────
    selected_path = output_dir / "selected_datasets.csv"
    report_path   = output_dir / "clustering_report.csv"
    summary_path  = output_dir / "cluster_summary.csv"

    df_selected.to_csv(selected_path, index=False)
    df_report.to_csv(report_path,     index=False)
    df_summary.to_csv(summary_path,   index=False)

    print(f"\nФайлы сохранены:")
    print(f"  {selected_path}   ← {len(df_selected)} представителей для метамодели")
    print(f"  {report_path}   ← все датасеты с меткой кластера")
    print(f"  {summary_path}   ← сводка по кластерам")

    # ── 7. Графики ────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nСтроим графики...")
        plots_dir = output_dir / "plots"
        make_plots(X_pca, X_scaled, km, df_report, plots_dir, n_clusters, seed=args.seed)

    # ── 8. Итоговая сводка ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Готово!")
    print(f"  Всего датасетов:     {len(df_full)}")
    print(f"  Кластеров:           {n_clusters}")
    print(f"  Представителей:      {len(df_selected)}")
    print(f"  PCA-компонент:       {pca.n_components_}")
    print(f"  Мета-признаков:      {len(mf_cols)}")

    if "IR" in df_selected.columns:
        print(f"\n  IR у представителей:")
        print(f"    min={df_selected['IR'].min():.2f}  "
              f"median={df_selected['IR'].median():.2f}  "
              f"max={df_selected['IR'].max():.2f}")
    if "nr_class" in df_selected.columns:
        print(f"  n_classes: {sorted(df_selected['nr_class'].dropna().unique().astype(int).tolist())}")


if __name__ == "__main__":
    main()