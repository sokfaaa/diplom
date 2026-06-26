"""
Добавляет новые мета-признаки к существующему metafeatures.csv.

Новые признаки:
  [SSI — расширенные агрегации (Пинто, 2018)]
    Для всех векторных pymfe-признаков (f1, f2, attr_ent, eigenvalues и др.)
    добавляет: min, max, q1 (25%), q2/median (50%), q3 (75%)
    Итого: ~50 новых колонок из существующих .mean/.sd

  [Overlap / сложность]
    overlap_ratio        — доля объектов в зоне перекрытия классов (kNN)
    inter_intra_ratio    — межклассовое / внутриклассовое расстояние
    boundary_density     — плотность объектов вблизи границы классов

  [Minority-класс]
    minority_size        — размер наименьшего класса
    minority_frac        — доля наименьшего класса
    minority_nn_ratio    — доля соседей minority-точки из того же класса
    minority_isolation   — расстояние от minority до ближайшего majority

  [Multiclass-специфичные]
    macro_IR             — среднее IR по всем парам классов
    IR_variance          — дисперсия размеров классов (нормированная)
    n_minority_classes   — число классов с долей < 1/n_classes
    class_imbalance_skew — асимметрия распределения размеров классов

  [Landmarking — быстрые зонды]
    lm_decision_stump    — accuracy Decision Tree глубины 1
    lm_1nn               — accuracy 1-NN
    lm_naive_bayes       — accuracy Naive Bayes
    lm_lin_discr         — accuracy Linear Discriminant Analysis
    lm_stump_vs_1nn      — разница stump - 1nn (линейность задачи)

Запуск:
    python add_metafeatures.py
    python add_metafeatures.py --metafeatures metafeatures.csv --datasets datasets/
    python add_metafeatures.py --jobs 4
    python add_metafeatures.py --resume
"""

import argparse
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SUBSAMPLE = 2000   # максимум объектов для тяжёлых вычислений
KNN_K     = 5      # число соседей для overlap-метрик

# Базовые pymfe-признаки которые возвращаются как векторы
# (у них есть .mean и .sd — добавляем min, max, q1, q2, q3)
SSI_BASE_FEATURES = [
    "f1", "f2", "f3", "f4",          # complexity
    "n2", "n3",                        # complexity (kNN-based)
    "attr_ent", "joint_ent", "mut_inf", # info-theory
    "can_cor", "cov", "eigenvalues", "cor",  # statistical
    "HDB_mean_f1",                     # custom
]


# ── SSI расширенные агрегации ─────────────────────────────────────────────

def compute_ssi_aggregations(ds_dir: Path) -> dict:
    """
    Считает расширенный набор агрегаций для векторных pymfe-признаков:
      - min, max, median  — через pymfe summary
      - q1 (25%), q3 (75%) — вручную через повторный запуск с raw-значениями

    Итого ~33 новых колонки вида ssi_f1.min, ssi_f1.max, ssi_f1.median и т.д.
    """
    result = {}
    try:
        X = np.load(ds_dir / "X_train.npy")
        y = np.load(ds_dir / "y_train.npy")

        # Субсэмплинг
        if len(y) > SUBSAMPLE:
            rng = np.random.default_rng(42)
            classes = np.unique(y)
            idx = []
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                n_take  = max(2, int(SUBSAMPLE * len(cls_idx) / len(y)))
                idx.append(rng.choice(cls_idx, min(n_take, len(cls_idx)), replace=False))
            X_s = X[np.concatenate(idx)]
            y_s = y[np.concatenate(idx)]
        else:
            X_s, y_s = X, y

        from pymfe.mfe import MFE

        groups_feats = {
            "complexity":  ["f1", "f2", "f3", "f4"],
            "info-theory": ["attr_ent", "joint_ent", "mut_inf"],
            "statistical": ["can_cor", "cov", "eigenvalues", "cor"],
        }

        for group, features in groups_feats.items():
            try:
                # min, max, median — поддерживаются pymfe напрямую
                mfe = MFE(
                    groups=[group],
                    features=features,
                    summary=["min", "max", "median"],
                    suppress_warnings=True,
                    random_state=42,
                )
                mfe.fit(X_s, y_s, suppress_warnings=True)
                names, values = mfe.extract(suppress_warnings=True)
                for name, val in zip(names, values):
                    result[f"ssi_{name}"] = (
                        float(val) if val is not None and val == val
                        else np.nan
                    )

                # q1/q3 — через сырые значения (summary=None → список)
                mfe_raw = MFE(
                    groups=[group],
                    features=features,
                    summary=None,   # возвращает список значений для каждого признака
                    suppress_warnings=True,
                    random_state=42,
                )
                mfe_raw.fit(X_s, y_s, suppress_warnings=True)
                names_raw, vals_raw = mfe_raw.extract(suppress_warnings=True)

                for name, val in zip(names_raw, vals_raw):
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        arr = np.array(val, dtype=float)
                        arr = arr[~np.isnan(arr)]
                        if len(arr) > 0:
                            result[f"ssi_{name}.q1"] = float(np.percentile(arr, 25))
                            result[f"ssi_{name}.q3"] = float(np.percentile(arr, 75))
                        else:
                            result[f"ssi_{name}.q1"] = np.nan
                            result[f"ssi_{name}.q3"] = np.nan

            except Exception:
                pass

    except Exception as e:
        result["__ssi_error"] = str(e)[:100]

    return result


# ── Overlap / сложность ───────────────────────────────────────────────────

def compute_overlap(X: np.ndarray, y: np.ndarray) -> dict:
    """
    overlap_ratio      — доля объектов у которых хотя бы 1 из K соседей
                         принадлежит другому классу (зона границы)
    inter_intra_ratio  — среднее расстояние между центроидами классов /
                         средний внутриклассовый разброс
    boundary_density   — отношение граничных объектов к общему числу
    """
    result = {}
    classes = np.unique(y)

    # Субсэмплинг для скорости
    if len(y) > SUBSAMPLE:
        rng = np.random.default_rng(42)
        # Стратифицированный субсэмпл
        idx = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_take  = max(2, int(SUBSAMPLE * len(cls_idx) / len(y)))
            idx.append(rng.choice(cls_idx, min(n_take, len(cls_idx)), replace=False))
        idx = np.concatenate(idx)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    # Нормализуем
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_s)

    # kNN для overlap_ratio
    try:
        k = min(KNN_K, len(y_s) - 1)
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=1)
        knn.fit(X_sc, y_s)
        _, indices = knn.kneighbors(X_sc)

        boundary_mask = np.array([
            any(y_s[nb] != y_s[i] for nb in indices[i])
            for i in range(len(y_s))
        ])
        result["overlap_ratio"]   = float(boundary_mask.mean())
        result["boundary_density"] = float(boundary_mask.sum() / len(y_s))
    except Exception:
        result["overlap_ratio"]   = np.nan
        result["boundary_density"] = np.nan

    # inter / intra расстояния
    try:
        centroids = {c: X_sc[y_s == c].mean(axis=0) for c in classes}

        # Межклассовое
        inter_dists = []
        for c1, c2 in combinations(classes, 2):
            d = np.linalg.norm(centroids[c1] - centroids[c2])
            inter_dists.append(d)
        mean_inter = np.mean(inter_dists) if inter_dists else np.nan

        # Внутриклассовое
        intra_dists = []
        for c in classes:
            X_c = X_sc[y_s == c]
            if len(X_c) > 1:
                diffs = X_c - centroids[c]
                intra_dists.append(np.sqrt((diffs ** 2).sum(axis=1)).mean())
        mean_intra = np.mean(intra_dists) if intra_dists else np.nan

        result["inter_intra_ratio"] = (
            float(mean_inter / (mean_intra + 1e-9))
            if not np.isnan(mean_inter) and not np.isnan(mean_intra)
            else np.nan
        )
    except Exception:
        result["inter_intra_ratio"] = np.nan

    return result


# ── Minority-класс ────────────────────────────────────────────────────────

def compute_minority(X: np.ndarray, y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)

    min_idx  = np.argmin(counts)
    maj_idx  = np.argmax(counts)
    min_cls  = classes[min_idx]
    maj_cls  = classes[maj_idx]
    min_size = int(counts[min_idx])

    result = {
        "minority_size": min_size,
        "minority_frac": float(min_size / n),
    }

    # Нормализуем для расстояний
    try:
        scaler = RobustScaler()
        X_sc = scaler.fit_transform(X)

        X_min = X_sc[y == min_cls]
        X_maj = X_sc[y == maj_cls]

        if len(X_min) < 2 or len(X_maj) < 2:
            result["minority_nn_ratio"]  = np.nan
            result["minority_isolation"] = np.nan
            return result

        # Субсэмплинг majority для скорости
        if len(X_maj) > 500:
            rng = np.random.default_rng(42)
            X_maj = X_maj[rng.choice(len(X_maj), 500, replace=False)]

        # minority_nn_ratio: для каждой minority-точки
        # считаем долю ближайших соседей из того же класса
        k = min(KNN_K, len(X_sc) - 1)
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=1)
        knn.fit(X_sc, y)
        _, indices = knn.kneighbors(X_min)

        same_class_fracs = []
        for i in range(len(X_min)):
            nb_labels  = y[indices[i]]
            same_frac  = (nb_labels == min_cls).mean()
            same_class_fracs.append(same_frac)
        result["minority_nn_ratio"] = float(np.mean(same_class_fracs))

        # minority_isolation: среднее расстояние от minority до ближайшего majority
        from sklearn.metrics import pairwise_distances_argmin_min
        _, dists = pairwise_distances_argmin_min(X_min, X_maj, metric="euclidean")
        result["minority_isolation"] = float(dists.mean())

    except Exception:
        result["minority_nn_ratio"]  = np.nan
        result["minority_isolation"] = np.nan

    return result


# ── Multiclass-специфичные ────────────────────────────────────────────────

def compute_multiclass(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    K = len(classes)
    fracs = counts / n

    # macro_IR: среднее IR по всем парам классов
    pair_irs = []
    for i, j in combinations(range(K), 2):
        if counts[i] > 0 and counts[j] > 0:
            pair_irs.append(max(counts[i], counts[j]) / min(counts[i], counts[j]))
    macro_ir = float(np.mean(pair_irs)) if pair_irs else 1.0

    # IR_variance: нормированная дисперсия размеров классов
    ir_var = float(np.std(counts) / (np.mean(counts) + 1e-9))

    # n_minority_classes: классов с долей < 1/K (меньше ожидаемого)
    expected_frac  = 1.0 / K
    n_minority_cls = int((fracs < expected_frac).sum())

    # class_imbalance_skew: асимметрия распределения размеров
    from scipy.stats import skew
    try:
        skewness = float(skew(counts.astype(float)))
    except Exception:
        skewness = np.nan

    return {
        "macro_IR":            macro_ir,
        "IR_variance":         ir_var,
        "n_minority_classes":  n_minority_cls,
        "class_imbalance_skew": skewness,
    }


# ── Landmarking ───────────────────────────────────────────────────────────

def compute_landmarking(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Быстрые зондирующие классификаторы — 3-fold CV accuracy.
    Субсэмплинг до 1500 для скорости.
    """
    # Субсэмплинг
    if len(y) > 1500:
        rng = np.random.default_rng(42)
        classes = np.unique(y)
        idx = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_take  = max(2, int(1500 * len(cls_idx) / len(y)))
            idx.append(rng.choice(cls_idx, min(n_take, len(cls_idx)), replace=False))
        idx = np.concatenate(idx)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_s)

    min_cls_cnt = pd.Series(y_s).value_counts().min()
    n_splits    = min(3, min_cls_cnt)

    if n_splits < 2:
        return {
            "lm_decision_stump": np.nan,
            "lm_1nn":            np.nan,
            "lm_naive_bayes":    np.nan,
            "lm_lin_discr":      np.nan,
            "lm_stump_vs_1nn":   np.nan,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "lm_decision_stump": DecisionTreeClassifier(max_depth=1, random_state=42),
        "lm_1nn":            KNeighborsClassifier(n_neighbors=1),
        "lm_naive_bayes":    GaussianNB(),
        "lm_lin_discr":      LinearDiscriminantAnalysis(),
    }

    result = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_sc, y_s, cv=cv,
                                     scoring="accuracy", n_jobs=1)
            result[name] = float(scores.mean())
        except Exception:
            result[name] = np.nan

    # Разница stump vs 1nn
    if not np.isnan(result.get("lm_decision_stump", np.nan)) and \
       not np.isnan(result.get("lm_1nn", np.nan)):
        result["lm_stump_vs_1nn"] = float(
            result["lm_decision_stump"] - result["lm_1nn"]
        )
    else:
        result["lm_stump_vs_1nn"] = np.nan

    return result


# ── Вычисление для одного датасета ────────────────────────────────────────

def compute_one(ds_dir: Path, already_done: set[str]) -> dict | None:
    name = ds_dir.name
    if name in already_done:
        return {"dataset_name": name, "__skip": True}

    try:
        X = np.load(ds_dir / "X_train.npy")
        y = np.load(ds_dir / "y_train.npy")

        row = {"dataset_name": name}
        row.update(compute_ssi_aggregations(ds_dir))  # min/max/q1/q2/q3
        row.update(compute_overlap(X, y))
        row.update(compute_minority(X, y))
        row.update(compute_multiclass(y))
        row.update(compute_landmarking(X, y))
        return row

    except Exception as e:
        return {"dataset_name": name, "__error": str(e)[:150]}


# ── Основной пайплайн ─────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Добавляет новые мета-признаки к metafeatures.csv",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--metafeatures", default="metafeatures.csv",
                    help="Существующий CSV с мета-признаками (default: metafeatures.csv)")
    pa.add_argument("--datasets",     default="datasets",
                    help="Папка с датасетами (default: datasets)")
    pa.add_argument("--output",       default=None,
                    help="Выходной файл (default: перезаписать --metafeatures)")
    pa.add_argument("--jobs",         type=int, default=1,
                    help="Параллельные процессы (default: 1)")
    pa.add_argument("--resume",       action="store_true",
                    help="Пропустить датасеты у которых уже есть новые признаки")
    args = pa.parse_args()

    mf_path  = Path(args.metafeatures)
    ds_dir   = Path(args.datasets)
    out_path = Path(args.output) if args.output else mf_path

    # Загружаем существующий файл
    print(f"Загружаю: {mf_path}")
    df_existing = pd.read_csv(mf_path)
    ds_col = next((c for c in ["dataset_name","name","dataset"]
                   if c in df_existing.columns), df_existing.columns[0])
    df_existing = df_existing.rename(columns={ds_col: "dataset_name"})
    print(f"  Датасетов: {len(df_existing)}  Колонок: {df_existing.shape[1]}")

    NEW_COLS_FIXED = [
        "overlap_ratio", "inter_intra_ratio", "boundary_density",
        "minority_size", "minority_frac", "minority_nn_ratio", "minority_isolation",
        "macro_IR", "IR_variance", "n_minority_classes", "class_imbalance_skew",
        "lm_decision_stump", "lm_1nn", "lm_naive_bayes", "lm_lin_discr",
        "lm_stump_vs_1nn",
    ]
    # SSI-колонки определяются динамически после первого вычисления
    # Шаблон: ssi_{feature}.{aggregation}
    SSI_AGGREGATIONS = ["min", "max", "q1", "median", "q3"]
    SSI_FEATURES_BASE = [
        "f1", "f2", "f3", "f4", "n2", "n3",
        "attr_ent", "joint_ent", "mut_inf",
        "can_cor", "cov", "eigenvalues", "cor",
    ]
    # Примерный список (реальный — после вычисления)
    ssi_expected = [f"ssi_{f}.{a}"
                    for f in SSI_FEATURES_BASE
                    for a in SSI_AGGREGATIONS]
    NEW_COLS = NEW_COLS_FIXED + ssi_expected
    print(f"  Новых признаков (план): {len(NEW_COLS_FIXED)} фиксированных "
          f"+ ~{len(ssi_expected)} SSI-агрегаций")

    # Resume — проверяем по одной фиксированной колонке
    already_done: set[str] = set()
    if args.resume and "lm_decision_stump" in df_existing.columns:
        done_mask    = df_existing["lm_decision_stump"].notna()
        already_done = set(df_existing.loc[done_mask, "dataset_name"])
        print(f"  Resume: уже посчитано {len(already_done)} датасетов")

    # Находим папки датасетов
    ds_dirs = sorted([
        d for d in ds_dir.iterdir()
        if d.is_dir() and (d / "X_train.npy").exists()
    ])
    # Оставляем только те что есть в metafeatures
    known = set(df_existing["dataset_name"])
    ds_dirs = [d for d in ds_dirs if d.name in known]
    print(f"\nДатасетов к обработке: {len(ds_dirs) - len(already_done)}/{len(ds_dirs)}")
    print(f"{'='*55}")

    # Вычисление
    if args.jobs == 1:
        results = []
        for i, d in enumerate(ds_dirs, 1):
            name = d.name
            skip_str = " (пропуск)" if name in already_done else ""
            print(f"[{i:>4}/{len(ds_dirs)}] {name[:55]:<55}{skip_str}",
                  end="" if name in already_done else " ", flush=True)
            row = compute_one(d, already_done)
            if row:
                results.append(row)
                if not row.get("__skip") and "__error" not in row:
                    lm = row.get("lm_decision_stump", float("nan"))
                    ov = row.get("overlap_ratio",     float("nan"))
                    mi = row.get("minority_nn_ratio", float("nan"))
                    print(f"lm_stump={lm:.3f}  overlap={ov:.3f}  min_nn={mi:.3f}")
                elif "__error" in row:
                    print(f"ERR: {row['__error'][:50]}")
    else:
        print(f"Параллельный режим: {args.jobs} процессов")
        results = Parallel(n_jobs=args.jobs, verbose=2)(
            delayed(compute_one)(d, already_done) for d in ds_dirs
        )
        results = [r for r in results if r]

    # Собираем результаты в DataFrame
    new_rows = [r for r in results if r and not r.get("__skip") and "__error" not in r]
    err_rows = [r for r in results if r and "__error" in r]

    if err_rows:
        print(f"\nОшибок: {len(err_rows)}")
        for r in err_rows[:5]:
            print(f"  {r['dataset_name']}: {r['__error']}")

    if not new_rows:
        print("\nНет новых результатов для сохранения.")
        return

    df_new = pd.DataFrame(new_rows)

    # Определяем реальные SSI-колонки из результатов
    ssi_cols_actual = [c for c in df_new.columns if c.startswith("ssi_")]
    all_new_cols    = NEW_COLS_FIXED + ssi_cols_actual
    print(f"\nФактически получено признаков: "
          f"{len(NEW_COLS_FIXED)} фиксированных + {len(ssi_cols_actual)} SSI")

    # Мержим — удаляем старые версии этих колонок если есть
    cols_to_drop = [c for c in all_new_cols if c in df_existing.columns]
    if cols_to_drop:
        print(f"Обновляем {len(cols_to_drop)} существующих колонок")
        df_existing = df_existing.drop(columns=cols_to_drop)

    merge_cols = ["dataset_name"] + [c for c in all_new_cols if c in df_new.columns]
    df_merged = df_existing.merge(
        df_new[merge_cols],
        on="dataset_name",
        how="left",
    )

    df_merged.to_csv(out_path, index=False)

    # Итог
    added_cols = [c for c in all_new_cols if c in df_merged.columns]
    nan_counts = df_merged[added_cols].isna().mean()

    print(f"\n{'='*55}")
    print(f"Готово!")
    print(f"  Датасетов:              {len(df_merged)}")
    print(f"  Итого колонок:          {df_merged.shape[1]}")
    print(f"  Добавлено фиксированных: {len(NEW_COLS_FIXED)}")
    print(f"  Добавлено SSI:           {len(ssi_cols_actual)}")
    print(f"  Файл:                   {out_path}")
    print(f"\n  Качество фиксированных признаков (доля NaN):")
    for col in NEW_COLS_FIXED:
        if col in nan_counts:
            nan_pct = nan_counts[col]
            status  = "✓" if nan_pct < 0.05 else ("⚠" if nan_pct < 0.20 else "✗")
            print(f"    {status} {col:<30} {nan_pct:.1%} NaN")
    # SSI — только сводка
    if ssi_cols_actual:
        ssi_nan = df_merged[ssi_cols_actual].isna().mean().mean()
        print(f"\n  SSI ({len(ssi_cols_actual)} колонок): "
              f"среднее NaN = {ssi_nan:.1%}")
        print(f"  Примеры SSI-колонок: {ssi_cols_actual[:5]}")


if __name__ == "__main__":
    main()