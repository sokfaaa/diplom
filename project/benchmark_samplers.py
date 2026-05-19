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
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # глушим логи smote_variants

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
# Реестр сэмплеров
# ---------------------------------------------------------------------------

def build_sampler_registry(random_state: int = 42) -> dict:
    """
    Возвращает словарь name → callable() → сэмплер с методом fit_resample.
    callable вместо объекта — чтобы каждый раз создавать свежий инстанс.
    """
    rs = random_state

    # ── imblearn ──────────────────────────────────────────────────────
    registry = {
        "Baseline": lambda: None,  # специальный случай — без сэмплинга
        "RandomOverSampler": lambda: RandomOverSampler(random_state=rs),
        "SMOTE":              lambda: SMOTE(random_state=rs),
        "BorderlineSMOTE":    lambda: BorderlineSMOTE(random_state=rs),
        "SVMSMOTE":           lambda: SVMSMOTE(random_state=rs),
        "ADASYN":             lambda: ADASYN(random_state=rs),
        "KMeansSMOTE":        lambda: KMeansSMOTE(
                                  random_state=rs,
                                  cluster_balance_threshold=0.0,
                              ),
        "SMOTEENN":           lambda: SMOTEENN(random_state=rs),
        "SMOTETomek":         lambda: SMOTETomek(random_state=rs),
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
    Загружает датасет, применяет сэмплер, обучает RF, считает метрики.
    Возвращает словарь с результатами.
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

        # Читаем мета-информацию для отчёта
        meta_path = ds_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            result["n_classes"]    = meta.get("n_classes", len(np.unique(y_train)))
            result["n_features"]   = meta.get("n_features", X_train.shape[1])
            result["n_train"]      = meta.get("n_samples_train", len(y_train))
            result["IR"]           = meta.get("imbalance_ratio", np.nan)
            result["base_type"]    = meta.get("base_type", "unknown")

        # Применяем сэмплер
        if sampler_name == "Baseline" or sampler_factory is None:
            X_res, y_res = X_train, y_train
        else:
            sampler = sampler_factory()
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        result["n_train_resampled"] = len(y_res)

        # RandomForest
        rf = RandomForestClassifier(random_state=random_state, **rf_params)
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test)

        # Метрики
        metrics = compute_metrics(y_test, y_pred)
        result.update(metrics)

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
    rf_params: dict,
    random_state: int,
    skip_samplers: set[str],
) -> list[dict]:
    rows = []
    for sampler_name, factory in registry.items():
        if sampler_name in skip_samplers:
            continue
        row = run_one(ds_dir, sampler_name, factory, rf_params, random_state)
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
        # Последовательно — с подробным прогрессом
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

            # Промежуточное сохранение каждые 5 датасетов
            if i % 5 == 0:
                _save_results(all_rows, output_dir)
                print(f"  → Промежуточное сохранение ({len(all_rows)} строк)")
    else:
        # Параллельно по датасетам
        print(f"Параллельный режим: n_jobs={n_jobs}")
        batch_results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(run_dataset)(ds_dir, registry, rf_params, random_state, skip)
            for ds_dir, skip in todo
        )
        for rows in batch_results:
            all_rows.extend(rows)

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
    parser.add_argument("--samplers",  default=None,
                        help="Запятая-разделённый список сэмплеров (default: все)\n"
                             "  Пример: --samplers SMOTE,ADASYN,Baseline")
    parser.add_argument("--rf-trees",  type=int, default=200,
                        help="Число деревьев RandomForest (default: 200)")
    parser.add_argument("--rf-depth",  type=int, default=None,
                        help="max_depth RandomForest (default: None)")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets)
    selected_csv = Path(args.selected) if args.selected else None
    output_dir   = Path(args.output)

    if not datasets_dir.exists():
        print(f"Папка датасетов не найдена: {datasets_dir}")
        return

    rf_params = {
        "n_estimators": args.rf_trees,
        "max_depth":    args.rf_depth,
        "n_jobs":       1,
        "class_weight": "balanced",   # RF сам учитывает дисбаланс — честный baseline
    }

    samplers_filter = None
    if args.samplers:
        samplers_filter = [s.strip() for s in args.samplers.split(",")]

    print(f"\n{'='*60}")
    print(f"  Датасеты:  {datasets_dir}")
    print(f"  Выбранные: {selected_csv or 'все'}")
    print(f"  Результаты: {output_dir}")
    print(f"  RF: n_estimators={args.rf_trees}, max_depth={args.rf_depth}")
    print(f"  Random seed: {args.seed}")
    print(f"{'='*60}\n")

    df = run_benchmark(
        datasets_dir=datasets_dir,
        selected_csv=selected_csv,
        output_dir=output_dir,
        n_jobs=args.jobs,
        resume=args.resume,
        random_state=args.seed,
        rf_params=rf_params,
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