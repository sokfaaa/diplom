"""
Бенчмарк методов борьбы с дисбалансом на наборе датасетов.

Для каждого датасета × сэмплер:
  1. Применяем сэмплер к train-части
  2. Обучаем RandomForest
  3. Считаем метрики на test-части: balanced_accuracy, F1-macro, G-mean

Выходные файлы:
  results_full.csv     — все результаты (датасет × сэмплер)
  results_pivot.csv    — сводная таблица: строки=датасеты, колонки=сэмплер×метрика
  results_summary.csv  — среднее по датасетам для каждого сэмплера (ранжирование)

Запуск:
  python benchmark_samplers.py                          # datasets/ → results/
  python benchmark_samplers.py --datasets my_ds/        # другая папка датасетов
  python benchmark_samplers.py --selected sel.csv       # только отобранные датасеты
  python benchmark_samplers.py --jobs 4                 # параллельно по датасетам
  python benchmark_samplers.py --resume                 # пропустить уже посчитанные
  python benchmark_samplers.py --samplers SMOTE,ADASYN  # только конкретные сэмплеры
"""

import argparse
import json
import logging
import os
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SVMSMOTE,
)
from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    NearMiss,
    TomekLinks,
    EditedNearestNeighbours,
    OneSidedSelection,
    NeighbourhoodCleaningRule,
)
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Патч для multi-imbalance (несовместимость с imblearn >= 0.11)
# ---------------------------------------------------------------------------

def _patch_mi(cls):
    """Добавляет заглушку _parameter_constraints, которую ждёт imblearn."""
    if not hasattr(cls, "_parameter_constraints"):
        cls._parameter_constraints = {}
    return cls


def _import_multi_imbalance():
    from multi_imbalance.resampling.soup import SOUP
    from multi_imbalance.resampling.global_cs import GlobalCS
    from multi_imbalance.resampling.spider import SPIDER3
    return _patch_mi(SOUP), _patch_mi(GlobalCS), _patch_mi(SPIDER3)


# ---------------------------------------------------------------------------
# Обёртка для smote_variants — они бинарные, нужен OvR для мультикласса
# ---------------------------------------------------------------------------

class SVMulticlassWrapper:
    """
    Оборачивает бинарный сэмплер smote_variants в схему One-vs-Rest:
    для каждого minority-класса делаем бинарную задачу (этот класс vs все),
    применяем сэмплер, добираем синтетические примеры этого класса,
    объединяем.
    """

    def __init__(self, sampler_cls, **kwargs):
        self.sampler_cls = sampler_cls
        self.kwargs = kwargs

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            # Бинарный случай — напрямую
            obj = self.sampler_cls(**self.kwargs)
            return obj.sample(X, y)

        # Мультикласс: определяем majority-класс
        maj_cls = classes[np.argmax(counts)]
        X_extra_list, y_extra_list = [], []

        for cls in classes:
            if cls == maj_cls:
                continue
            # Бинарный датасет: maj + текущий minority
            mask = (y == maj_cls) | (y == cls)
            X_bin = X[mask]
            y_bin = (y[mask] == cls).astype(int)   # minority=1, majority=0

            if y_bin.sum() < 2:
                continue  # слишком мало примеров

            try:
                obj = self.sampler_cls(**self.kwargs)
                X_res, y_res = obj.sample(X_bin, y_bin)
                # Берём только новые синтетические примеры minority-класса
                n_orig = mask.sum()
                X_new = X_res[n_orig:]
                y_new_real = np.full(len(X_new), cls, dtype=y.dtype)
                X_extra_list.append(X_new)
                y_extra_list.append(y_new_real)
            except Exception:
                continue

        if not X_extra_list:
            return X, y

        X_out = np.vstack([X] + X_extra_list)
        y_out = np.concatenate([y] + y_extra_list)
        return X_out, y_out


# ---------------------------------------------------------------------------
# Пакеты сэмплеров — запускай отдельно для контроля
# ---------------------------------------------------------------------------

SAMPLER_PACKS = {
    # Базовый — всегда запускать первым
    "baseline": [
        "Baseline",
    ],

    # Стандартные imblearn oversampling — быстрые, нативный multiclass
    "imblearn": [
        "RandomOverSampler",
        "SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "KMeansSMOTE",
        "SMOTEENN", "SMOTETomek",
    ],

    # Undersampling — отдельный пак (некоторые медленные: CNN, OSS)
    "undersampling": [
        "RandomUnderSampler",
        "TomekLinks",        # быстрый — просто удаляет граничные пары
        "ENN",               # средний — kNN для каждой точки
        "NCR",               # средний — расширенный ENN
        "NearMiss_v1",       # медленный — попарные расстояния
        "NearMiss_v2",       # медленный
        "NearMiss_v3",       # медленный
        "CNN",               # медленный — итеративный алгоритм
        "OSS",               # медленный — TomekLinks + CNN
    ],

    # Быстрый undersampling — без медленных методов
    "undersampling_fast": [
        "RandomUnderSampler",
        "TomekLinks",
        "ENN",
        "NCR",
    ],

    # smote_variants — OvA обёртка, медленные
    "sv": [
        "distance_SMOTE", "cluster_SMOTE",
        "CBSO", "DBSMOTE", "MWMOTE", "AHC",
    ],

    # multi-imbalance — нативно многоклассовые
    "multi": [
        "SOUP", "GlobalCS", "SPIDER3",
    ],

    # Всё кроме smote_variants и медленного undersampling
    "fast": [
        "Baseline",
        "RandomOverSampler", "SMOTE", "BorderlineSMOTE",
        "SVMSMOTE", "ADASYN", "KMeansSMOTE",
        "SMOTEENN", "SMOTETomek",
        "RandomUnderSampler", "TomekLinks", "ENN", "NCR",
        "SOUP", "GlobalCS", "SPIDER3",
    ],

    # Полный набор
    "all": [],  # [] = все доступные из реестра
}


def resolve_samplers(pack: str | None, samplers_arg: str | None) -> list[str] | None:
    """
    Возвращает список сэмплеров для запуска.
    Приоритет: --samplers > --pack > None (все)
    """
    if samplers_arg:
        return [s.strip() for s in samplers_arg.split(",")]
    if pack:
        pack = pack.lower()
        if pack not in SAMPLER_PACKS:
            print(f"Неизвестный пакет '{pack}'. Доступные: {list(SAMPLER_PACKS.keys())}")
            return None
        result = SAMPLER_PACKS[pack]
        return result if result else None  # [] → None → все сэмплеры
    return None  # None → все


def print_packs():
    """Выводит список доступных пакетов."""
    print("\nДоступные пакеты сэмплеров (--pack):")
    print("=" * 55)
    for pack_name, samplers in SAMPLER_PACKS.items():
        if samplers:
            print(f"\n  --pack {pack_name}")
            for s in samplers:
                print(f"    • {s}")
        else:
            print(f"\n  --pack {pack_name}  (все доступные сэмплеры)")
    print()
    print("Примеры запуска:")
    print("  # Шаг 1: Baseline")
    print("  python benchmark_samplers.py --pack baseline --resume")
    print()
    print("  # Шаг 2: Oversampling (imblearn, быстро)")
    print("  python benchmark_samplers.py --pack imblearn --resume --jobs 4")
    print()
    print("  # Шаг 3: Undersampling быстрые (RUS, TomekLinks, ENN, NCR)")
    print("  python benchmark_samplers.py --pack undersampling_fast --resume --jobs 4")
    print()
    print("  # Шаг 4: Undersampling медленные (CNN, NearMiss, OSS)")
    print("  python benchmark_samplers.py --pack undersampling --resume --jobs 2")
    print()
    print("  # Шаг 5: smote_variants (OvA, медленно)")
    print("  python benchmark_samplers.py --pack sv --resume --jobs 2")
    print()
    print("  # Шаг 6: multi-imbalance")
    print("  python benchmark_samplers.py --pack multi --resume")
    print()
    print("  # Или конкретные методы:")
    print("  python benchmark_samplers.py --samplers RandomUnderSampler,ENN,NCR")
    print("=" * 55)



def build_sampler_registry(random_state: int = 42) -> dict:
    """
    Возвращает словарь name → callable() → сэмплер с методом fit_resample.
    callable вместо объекта — чтобы каждый раз создавать свежий инстанс.
    """
    rs = random_state

    # ── imblearn ──────────────────────────────────────────────────────
    registry = {
        "Baseline":          lambda: None,
        # Oversampling
        "RandomOverSampler": lambda: RandomOverSampler(random_state=rs),
        "SMOTE":             lambda: SMOTE(random_state=rs),
        "BorderlineSMOTE":   lambda: BorderlineSMOTE(random_state=rs),
        "SVMSMOTE":          lambda: SVMSMOTE(random_state=rs),
        "ADASYN":            lambda: ADASYN(random_state=rs),
        "KMeansSMOTE":       lambda: KMeansSMOTE(
                                 random_state=rs,
                                 cluster_balance_threshold=0.0,
                             ),
        # Combine
        "SMOTEENN":          lambda: SMOTEENN(random_state=rs),
        "SMOTETomek":        lambda: SMOTETomek(random_state=rs),
        # Undersampling
        "RandomUnderSampler":        lambda: RandomUnderSampler(random_state=rs),
        "CNN":                       lambda: CondensedNearestNeighbour(random_state=rs, n_jobs=1),
        "NearMiss_v1":               lambda: NearMiss(version=1, n_jobs=1),
        "NearMiss_v2":               lambda: NearMiss(version=2, n_jobs=1),
        "NearMiss_v3":               lambda: NearMiss(version=3, n_jobs=1),
        "TomekLinks":                lambda: TomekLinks(n_jobs=1),
        "ENN":                       lambda: EditedNearestNeighbours(n_jobs=1),
        "OSS":                       lambda: OneSidedSelection(random_state=rs, n_jobs=1),
        "NCR":                       lambda: NeighbourhoodCleaningRule(n_jobs=1),
    }

    # ── smote_variants (OvR-обёртка) ─────────────────────────────────
    try:
        import smote_variants as sv
        sv_samplers = {
            "distance_SMOTE": sv.distance_SMOTE,
            "cluster_SMOTE":  sv.cluster_SMOTE,
            "CBSO":           sv.CBSO,
            "DBSMOTE":        sv.DBSMOTE,
            "MWMOTE":         sv.MWMOTE,
            "AHC":            sv.AHC,
        }
        for name, cls in sv_samplers.items():
            _cls = cls  # захват переменной для lambda
            registry[name] = lambda c=_cls: SVMulticlassWrapper(c)
    except ImportError:
        pass

    # ── multi-imbalance ───────────────────────────────────────────────
    try:
        SOUP, GlobalCS, SPIDER3 = _import_multi_imbalance()
        registry["SOUP"]     = lambda: SOUP()
        registry["GlobalCS"] = lambda: GlobalCS()
        registry["SPIDER3"]  = lambda: SPIDER3(k=5)
    except Exception:
        pass

    return registry


# ---------------------------------------------------------------------------
# Optuna: подбор гиперпараметров RF (один раз на датасет, без сэмплера)
# ---------------------------------------------------------------------------

# Константы по заданию
OPTUNA_TRIALS   = 20
INNER_CV_FOLDS  = 3   # внутренняя CV для оценки параметров RF
OUTER_CV_FOLDS  = 3   # внешняя CV для оценки сэмплера

RF_PARAM_SPACE = {
    "n_estimators":     (100, 400),
    "max_depth":        (5,   20),
    "min_samples_split":(5,   50),
    "min_samples_leaf": (2,   20),
    "max_features":     ["sqrt", "log2", 0.3, 0.5],
}


def tune_rf_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> dict:
    """
    Подбирает гиперпараметры RF через Optuna на исходных данных (без сэмплера).
    Использует внутреннюю {INNER_CV_FOLDS}-fold CV и Pruning для ранней остановки.

    Возвращает словарь лучших гиперпараметров.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from optuna.pruners import MedianPruner

    # Минимальный класс — ограничивает число фолдов
    min_cls = pd.Series(y_train).value_counts().min()
    n_splits = min(INNER_CV_FOLDS, min_cls)
    if n_splits < 2:
        # Слишком мало данных — возвращаем дефолтные параметры
        return {
            "n_estimators":      200,
            "max_depth":         10,
            "min_samples_split": 10,
            "min_samples_leaf":  4,
            "max_features":      "sqrt",
            "class_weight":      "balanced",
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",
                                     RF_PARAM_SPACE["n_estimators"][0],
                                     RF_PARAM_SPACE["n_estimators"][1]),
            "max_depth":         trial.suggest_int("max_depth",
                                     RF_PARAM_SPACE["max_depth"][0],
                                     RF_PARAM_SPACE["max_depth"][1]),
            "min_samples_split": trial.suggest_int("min_samples_split",
                                     RF_PARAM_SPACE["min_samples_split"][0],
                                     RF_PARAM_SPACE["min_samples_split"][1]),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",
                                     RF_PARAM_SPACE["min_samples_leaf"][0],
                                     RF_PARAM_SPACE["min_samples_leaf"][1]),
            "max_features":      trial.suggest_categorical(
                                     "max_features", RF_PARAM_SPACE["max_features"]),
            "class_weight":      "balanced",
            "random_state":      random_state,
            "n_jobs":            1,
        }
        rf = RandomForestClassifier(**params)

        # Оцениваем с Pruning — после каждого фолда проверяем перспективность
        fold_scores = []
        for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            rf_fold = RandomForestClassifier(**params)
            rf_fold.fit(X_train[tr_idx], y_train[tr_idx])
            score = balanced_accuracy_score(
                y_train[val_idx], rf_fold.predict(X_train[val_idx])
            )
            fold_scores.append(score)

            # Сообщаем Optuna промежуточный результат для Pruning
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)

    best = study.best_params.copy()
    best["class_weight"]  = "balanced"
    best["random_state"]  = random_state
    best["n_jobs"]        = 1
    return best


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "g_mean":            float(geometric_mean_score(y_true, y_pred,
                                                         average="multiclass")),
    }


# ---------------------------------------------------------------------------
# CV-оценка одного сэмплера (внешняя CV по заданию)
# ---------------------------------------------------------------------------

def evaluate_sampler_cv(
    X: np.ndarray,
    y: np.ndarray,
    sampler_name: str,
    sampler_factory,
    rf_params: dict,
    random_state: int,
) -> dict[str, float]:
    """
    Оценивает сэмплер через {OUTER_CV_FOLDS}-fold стратифицированную CV.
    На каждом фолде:
      1. Применяем сэмплер к train-фолду
      2. Обучаем RF с заранее подобранными гиперпараметрами
      3. Считаем метрики на val-фолде
    Возвращает средние метрики по фолдам.
    """
    from sklearn.model_selection import StratifiedKFold

    min_cls  = pd.Series(y).value_counts().min()
    n_splits = min(OUTER_CV_FOLDS, min_cls)
    if n_splits < 2:
        # Fallback: обучаем на всём без CV
        return _evaluate_sampler_simple(
            X, X, y, y, sampler_name, sampler_factory, rf_params, random_state
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = {k: [] for k in ["balanced_accuracy", "f1_macro", "g_mean"]}

    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        try:
            # Применяем сэмплер только к train-фолду
            if sampler_name == "Baseline" or sampler_factory is None:
                X_res, y_res = X_tr, y_tr
            else:
                sampler = sampler_factory()
                X_res, y_res = sampler.fit_resample(X_tr, y_tr)

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_res, y_res)
            y_pred = rf.predict(X_val)

            m = compute_metrics(y_val, y_pred)
            for k, v in m.items():
                fold_metrics[k].append(v)

        except Exception:
            # Если фолд упал — пропускаем
            continue

    if not fold_metrics["balanced_accuracy"]:
        return {"balanced_accuracy": np.nan, "f1_macro": np.nan, "g_mean": np.nan}

    return {k: float(np.mean(v)) for k, v in fold_metrics.items()}


def _evaluate_sampler_simple(
    X_train, X_test, y_train, y_test,
    sampler_name, sampler_factory, rf_params, random_state,
) -> dict[str, float]:
    """Простая оценка без CV (fallback для маленьких датасетов)."""
    try:
        if sampler_name == "Baseline" or sampler_factory is None:
            X_res, y_res = X_train, y_train
        else:
            sampler = sampler_factory()
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_res, y_res)
        return compute_metrics(y_test, rf.predict(X_test))
    except Exception:
        return {"balanced_accuracy": np.nan, "f1_macro": np.nan, "g_mean": np.nan}


# ---------------------------------------------------------------------------
# Один эксперимент: датасет × сэмплер
# ---------------------------------------------------------------------------

def run_one(
    ds_dir: Path,
    sampler_name: str,
    sampler_factory,
    rf_params: dict,
    random_state: int,
) -> dict:
    """
    Загружает датасет, оценивает сэмплер через внешнюю CV,
    использует rf_params подобранные Optuna заранее.
    """
    result = {
        "dataset":  ds_dir.name,
        "sampler":  sampler_name,
        "status":   "ok",
        "error":    "",
    }

    try:
        X_train = np.load(ds_dir / "X_train.npy")
        X_test  = np.load(ds_dir / "X_test.npy")
        y_train = np.load(ds_dir / "y_train.npy")
        y_test  = np.load(ds_dir / "y_test.npy")

        # Мета-информация
        meta_path = ds_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            result["n_classes"]  = meta.get("n_classes",    len(np.unique(y_train)))
            result["n_features"] = meta.get("n_features",   X_train.shape[1])
            result["n_train"]    = meta.get("n_samples_train", len(y_train))
            result["IR"]         = meta.get("imbalance_ratio", np.nan)
            result["base_type"]  = meta.get("base_type",    "unknown")

        # Объединяем train+test для внешней CV
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        # Внешняя CV для оценки сэмплера
        metrics = evaluate_sampler_cv(
            X_all, y_all,
            sampler_name, sampler_factory,
            rf_params, random_state,
        )
        result.update(metrics)
        result["n_train_resampled"] = len(y_train)  # приблизительно

    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"{type(e).__name__}: {str(e)[:200]}"
        result["balanced_accuracy"] = np.nan
        result["f1_macro"]          = np.nan
        result["g_mean"]            = np.nan

    return result


# ---------------------------------------------------------------------------
# Один датасет × все сэмплеры
# ---------------------------------------------------------------------------

def run_dataset(
    ds_dir: Path,
    registry: dict,
    rf_params: dict,           # дефолтные параметры (fallback если Optuna недоступна)
    random_state: int,
    skip_samplers: set[str],
) -> list[dict]:
    """
    Для одного датасета:
      1. Подбирает гиперпараметры RF через Optuna (один раз, на исходных данных)
      2. Оценивает каждый сэмплер через внешнюю CV с найденными параметрами
    """
    # ── Шаг 1: Optuna-тюнинг RF (один раз на датасет) ────────────────
    try:
        X_train = np.load(ds_dir / "X_train.npy")
        y_train = np.load(ds_dir / "y_train.npy")
        X_test  = np.load(ds_dir / "X_test.npy")
        y_test  = np.load(ds_dir / "y_test.npy")

        # Объединяем для CV
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        best_rf_params = tune_rf_optuna(X_all, y_all, random_state)
        print(f"    Optuna RF: n_est={best_rf_params['n_estimators']}  "
              f"depth={best_rf_params['max_depth']}  "
              f"feat={best_rf_params['max_features']}  "
              f"msl={best_rf_params['min_samples_leaf']}")
    except Exception as e:
        # Если Optuna не отработала — используем дефолтные параметры
        print(f"    Optuna fallback: {str(e)[:60]}")
        best_rf_params = rf_params.copy()

    # ── Шаг 2: оценка каждого сэмплера ───────────────────────────────
    rows = []
    for sampler_name, factory in registry.items():
        if sampler_name in skip_samplers:
            continue
        row = run_one(ds_dir, sampler_name, factory, best_rf_params, random_state)
        # Сохраняем лучшие RF-параметры в строку для прозрачности
        row["rf_n_estimators"]     = best_rf_params.get("n_estimators", np.nan)
        row["rf_max_depth"]        = best_rf_params.get("max_depth",    np.nan)
        row["rf_max_features"]     = str(best_rf_params.get("max_features", ""))
        row["rf_min_samples_leaf"] = best_rf_params.get("min_samples_leaf", np.nan)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run_benchmark(
    datasets_dir: Path,
    selected_csv: Path | None,
    output_dir: Path,
    n_jobs: int,
    resume: bool,
    random_state: int,
    rf_params: dict,
    samplers_filter: list[str] | None,
) -> pd.DataFrame:

    # Находим папки датасетов
    if selected_csv and selected_csv.exists():
        sel = pd.read_csv(selected_csv)
        ds_names = sel["dataset_name"].tolist()
        ds_dirs = [datasets_dir / name for name in ds_names
                   if (datasets_dir / name / "X_train.npy").exists()]
        print(f"Режим: отобранные датасеты из {selected_csv.name} → {len(ds_dirs)} найдено")
    else:
        ds_dirs = sorted([
            d for d in datasets_dir.iterdir()
            if d.is_dir() and (d / "X_train.npy").exists()
        ])
        print(f"Режим: все датасеты из {datasets_dir} → {len(ds_dirs)} найдено")

    if not ds_dirs:
        print("Датасетов не найдено.")
        return pd.DataFrame()

    # Реестр сэмплеров
    registry = build_sampler_registry(random_state)
    if samplers_filter:
        registry = {k: v for k, v in registry.items() if k in samplers_filter}
    print(f"Сэмплеров: {len(registry)}: {list(registry.keys())}")

    # Resume: загружаем уже посчитанные пары (датасет, сэмплер)
    full_path = output_dir / "results_full.csv"
    done_pairs: set[tuple] = set()
    existing_rows: list[dict] = []
    if resume and full_path.exists():
        df_ex = pd.read_csv(full_path)
        done_pairs = set(zip(df_ex["dataset"], df_ex["sampler"]))
        existing_rows = df_ex.to_dict("records")
        print(f"Resume: пропускаем {len(done_pairs)} уже посчитанных пар")

    # Считаем сколько осталось
    todo = []
    for ds_dir in ds_dirs:
        skip = {s for s in registry if (ds_dir.name, s) in done_pairs}
        if len(skip) < len(registry):
            todo.append((ds_dir, skip))

    total_pairs = sum(len(registry) - len(skip) for _, skip in todo)
    print(f"Осталось пар (датасет × сэмплер): {total_pairs}")

    all_rows = list(existing_rows)
    done_count = 0
    total = len(todo)

    if n_jobs == 1:
        # Последовательно — с подробным прогрессом и сохранением после каждого датасета
        for i, (ds_dir, skip) in enumerate(todo, 1):
            n_skip = len(skip)
            n_run  = len(registry) - n_skip
            print(f"\n[{i:>4}/{total}] {ds_dir.name[:50]:<50}  ({n_run} сэмплеров)")
            rows = run_dataset(ds_dir, registry, rf_params, random_state, skip)
            all_rows.extend(rows)
            done_count += len(rows)

            for row in rows:
                status = "✓" if row["status"] == "ok" else "✗"
                ba = row.get("balanced_accuracy", float("nan"))
                f1 = row.get("f1_macro",          float("nan"))
                gm = row.get("g_mean",             float("nan"))
                ba_str = f"{ba:.4f}" if not np.isnan(ba) else " nan "
                f1_str = f"{f1:.4f}" if not np.isnan(f1) else " nan "
                gm_str = f"{gm:.4f}" if not np.isnan(gm) else " nan "
                err = f"  [{row['error'][:60]}]" if row["status"] == "error" else ""
                print(f"  {status} {row['sampler']:<22} "
                      f"BA={ba_str}  F1={f1_str}  GM={gm_str}{err}")

            # Сохраняем после каждого датасета — так --resume всегда актуален
            _save_results(all_rows, output_dir)
            if i % 5 == 0:
                print(f"  → Сохранено ({len(all_rows)} строк, {i}/{total} датасетов)")
    else:
        # Параллельно — обрабатываем батчами по 10 датасетов
        # чтобы промежуточные результаты сохранялись
        BATCH_SIZE = 10
        batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        print(f"Параллельный режим: n_jobs={n_jobs}, батчей: {len(batches)}")

        processed = 0
        for b_idx, batch in enumerate(batches, 1):
            batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(run_dataset)(ds_dir, registry, rf_params, random_state, skip)
                for ds_dir, skip in batch
            )
            for rows in batch_results:
                all_rows.extend(rows)
            processed += len(batch)

            # Сохраняем после каждого батча
            _save_results(all_rows, output_dir)
            print(f"  → Батч {b_idx}/{len(batches)} готов "
                  f"({processed}/{total} датасетов, "
                  f"{len(all_rows)} строк сохранено)")

    return _save_results(all_rows, output_dir)


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def _save_results(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    output_dir.mkdir(exist_ok=True)
    metric_cols = ["balanced_accuracy", "f1_macro", "g_mean"]

    # ── results_full.csv ──────────────────────────────────────────────
    col_order = (
        ["dataset", "sampler", "status"]
        + [c for c in df.columns if c in
           ["base_type", "n_classes", "n_features", "n_train", "IR",
            "n_train_resampled"]]
        + metric_cols
        + ["rf_n_estimators", "rf_max_depth", "rf_max_features", "rf_min_samples_leaf"]
        + ["error"]
    )
    col_order = [c for c in col_order if c in df.columns]
    df_out = df[col_order]
    df_out.to_csv(output_dir / "results_full.csv", index=False)

    # ── results_pivot.csv — датасет × (сэмплер, метрика) ─────────────
    ok_df = df[df["status"] == "ok"].copy()
    if not ok_df.empty:
        pivot_parts = []
        for metric in metric_cols:
            piv = ok_df.pivot_table(
                index="dataset", columns="sampler",
                values=metric, aggfunc="first",
            )
            piv.columns = [f"{col}_{metric}" for col in piv.columns]
            pivot_parts.append(piv)
        pivot_df = pd.concat(pivot_parts, axis=1).reset_index()

        # Сортируем колонки: сначала все метрики одного сэмплера вместе
        samplers = ok_df["sampler"].unique().tolist()
        sorted_cols = ["dataset"]
        for s in samplers:
            for m in metric_cols:
                col = f"{s}_{m}"
                if col in pivot_df.columns:
                    sorted_cols.append(col)
        pivot_df = pivot_df[[c for c in sorted_cols if c in pivot_df.columns]]
        pivot_df.to_csv(output_dir / "results_pivot.csv", index=False)

    # ── results_summary.csv — среднее по датасетам, ранжирование ──────
    if not ok_df.empty:
        agg = ok_df.groupby("sampler")[metric_cols].agg(
            ["mean", "std", "median"]
        ).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        # Ранг по F1-macro (чем выше — тем лучше)
        if "f1_macro_mean" in agg.columns:
            agg["rank_f1_macro"] = agg["f1_macro_mean"].rank(ascending=False).astype(int)
        if "balanced_accuracy_mean" in agg.columns:
            agg["rank_balanced_accuracy"] = agg["balanced_accuracy_mean"].rank(ascending=False).astype(int)
        if "g_mean_mean" in agg.columns:
            agg["rank_g_mean"] = agg["g_mean_mean"].rank(ascending=False).astype(int)

        # Средний ранг по всем 3 метрикам
        rank_cols = [c for c in agg.columns if c.startswith("rank_")]
        if rank_cols:
            agg["rank_avg"] = agg[rank_cols].mean(axis=1).round(2)
            agg = agg.sort_values("rank_avg")

        agg.to_csv(output_dir / "results_summary.csv", index=False)

    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global OPTUNA_TRIALS
    parser = argparse.ArgumentParser(
        description="Бенчмарк сэмплеров на имбалансных датасетах",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--datasets",  default="datasets",
                        help="Папка с датасетами (default: datasets)")
    parser.add_argument("--selected",  default=None,
                        help="CSV с отобранными датасетами (selected_datasets.csv)")
    parser.add_argument("--output",    default="results",
                        help="Папка для результатов (default: results)")
    parser.add_argument("--jobs",      type=int, default=1,
                        help="Число параллельных процессов (default: 1)")
    parser.add_argument("--resume",    action="store_true",
                        help="Продолжить с места остановки")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--pack",       default=None,
                        help="Пакет сэмплеров (default: все):\n"
                             "  baseline  — только Baseline\n"
                             "  imblearn  — стандартные imblearn (быстро)\n"
                             "  sv        — smote_variants через OvA (медленно)\n"
                             "  multi     — multi-imbalance\n"
                             "  fast      — всё кроме smote_variants\n"
                             "  all       — полный набор\n"
                             "  Подробнее: --list-packs")
    parser.add_argument("--list-packs", action="store_true",
                        help="Показать список пакетов и примеры запуска")
    parser.add_argument("--samplers",  default=None,
                        help="Конкретные сэмплеры через запятую (приоритет над --pack)\n"
                             "  Пример: --samplers SMOTE,ADASYN,Baseline")
    parser.add_argument("--trials",    type=int, default=OPTUNA_TRIALS,
                        help=f"Число Optuna trials для подбора RF (default: {OPTUNA_TRIALS})")
    args = parser.parse_args()

    # ── --list-packs ──────────────────────────────────────────────────
    if args.list_packs:
        print_packs()
        return

    # Обновляем число trials если передан аргумент
    OPTUNA_TRIALS = args.trials

    datasets_dir = Path(args.datasets)
    selected_csv = Path(args.selected) if args.selected else None
    output_dir   = Path(args.output)

    # Дефолтные RF-параметры (используются как fallback если Optuna не справилась)
    rf_params_default = {
        "n_estimators": 200,
        "max_depth":    None,
        "n_jobs":       1,
        "class_weight": "balanced",
        "random_state": args.seed,
    }

    # Определяем набор сэмплеров
    samplers_filter = resolve_samplers(args.pack, args.samplers)
    pack_label = f"пакет '{args.pack}'" if args.pack else \
                 (f"сэмплеры: {samplers_filter}" if samplers_filter else "все сэмплеры")

    print(f"\n{'='*60}")
    print(f"  Датасеты:        {datasets_dir}")
    print(f"  Выбранные:       {selected_csv or 'все'}")
    print(f"  Результаты:      {output_dir}")
    print(f"  Сэмплеры:        {pack_label}")
    print(f"  Seed:            {args.seed}")
    print(f"\n  Optuna RF-тюнинг:")
    print(f"    Trials:          {OPTUNA_TRIALS}")
    print(f"    Внутренняя CV:   {INNER_CV_FOLDS} фолда (для подбора RF)")
    print(f"    Внешняя CV:      {OUTER_CV_FOLDS} фолда (для оценки сэмплера)")
    print(f"    Pruning:         MedianPruner")
    print(f"    Пространство:    n_est={RF_PARAM_SPACE['n_estimators']}  "
          f"depth={RF_PARAM_SPACE['max_depth']}  "
          f"feat={RF_PARAM_SPACE['max_features']}")
    print(f"{'='*60}\n")

    df = run_benchmark(
        datasets_dir=datasets_dir,
        selected_csv=selected_csv,
        output_dir=output_dir,
        n_jobs=args.jobs,
        resume=args.resume,
        random_state=args.seed,
        rf_params=rf_params_default,
        samplers_filter=samplers_filter,
    )

    # Итоговая сводка
    if df is not None and not df.empty:
        ok = df[df["status"] == "ok"]
        err = df[df["status"] != "ok"]
        print(f"\n{'='*60}")
        print(f"Готово!")
        print(f"  Всего пар:        {len(df)}")
        print(f"  Успешных:         {len(ok)}")
        print(f"  С ошибками:       {len(err)}")
        print(f"  Выходные файлы:   {output_dir}/")
        print(f"    results_full.csv    — все результаты")
        print(f"    results_pivot.csv   — широкая таблица (датасет × сэмплер)")
        print(f"    results_summary.csv — ранжирование сэмплеров")

        # Топ-5 сэмплеров по F1-macro
        summary_path = output_dir / "results_summary.csv"
        if summary_path.exists():
            summ = pd.read_csv(summary_path)
            if "f1_macro_mean" in summ.columns and "rank_avg" in summ.columns:
                print(f"\n  Топ-5 сэмплеров по среднему F1-macro:")
                top5 = summ.nsmallest(5, "rank_avg")[
                    ["sampler", "f1_macro_mean", "balanced_accuracy_mean",
                     "g_mean_mean", "rank_avg"]
                ]
                for _, row in top5.iterrows():
                    print(f"    {row['sampler']:<22} "
                          f"F1={row['f1_macro_mean']:.4f}  "
                          f"BA={row['balanced_accuracy_mean']:.4f}  "
                          f"GM={row['g_mean_mean']:.4f}  "
                          f"rank={row['rank_avg']:.1f}")


if __name__ == "__main__":
    main()