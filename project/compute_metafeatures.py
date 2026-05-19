"""
Вычисление мета-признаков для всех датасетов из datasets/.

Выходной файл:
    metafeatures.csv  — одна строка на датасет, все мета-признаки + базовые мета из meta.json

Запуск:
    python compute_metafeatures.py                         # datasets/ → metafeatures.csv
    python compute_metafeatures.py --datasets my_ds/       # другая папка
    python compute_metafeatures.py --output results.csv    # другой выходной файл
    python compute_metafeatures.py --jobs 4                # параллельно на 4 ядрах
    python compute_metafeatures.py --resume                # пропустить уже посчитанные
"""

import argparse
import json
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymfe.mfe import MFE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Список pymfe-фичей — разделены на быстрые и медленные
# ---------------------------------------------------------------------------

# Быстрые — работают на любом датасете за секунды
PYMFE_FAST = {
    "general": [
        "nr_inst", "nr_attr", "nr_class",
    ],
    "info-theory": [
        "attr_ent", "class_ent", "joint_ent", "mut_inf", "eq_num_attr",
    ],
    "statistical": [
        "can_cor", "cov", "eigenvalues",
        "sd_ratio", "cor", "w_lambda", "p_trace", "lh_trace", "roy_root",
    ],
    "complexity": [
        # f1v пропускаем — постоянно NaN
        "c2", "f1", "f2", "f3", "f4", "lsc",
    ],
}

# Медленные — O(n²) или O(n·f²), запускаем только если датасет небольшой
# n1=MST, n2=kNN-граф, n3=kNN-ошибка, t1=гиперсфера
PYMFE_SLOW = {
    "complexity": ["n1", "n2", "n3", "t1"],
}

# Пороги для субсэмплинга и пропуска медленных фичей
MAX_SAMPLES_FULL   = 3_000   # при n > этого — субсэмплируем перед pymfe
MAX_SAMPLES_SLOW   = 2_000   # при n_subsampled > этого — пропускаем slow фичи
MAX_FEATURES_SLOW  = 100     # при f > этого — пропускаем slow фичи (kNN в 100d быстро)
SUBSAMPLE_TARGET   = 2_000   # до скольких сэмплов уменьшаем


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Стратифицированный субсэмплинг — сохраняет баланс классов.
    Если в каком-то классе < 2 примеров — берём все.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        # Пропорциональное число сэмплов для этого класса
        n_take = max(2, int(target * cnt / total))
        n_take = min(n_take, cnt)
        chosen = rng.choice(cls_idx, n_take, replace=False)
        indices.append(chosen)

    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Кастомные мета-признаки
# ---------------------------------------------------------------------------

def compute_custom_metafeatures(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    HDB, IR, CV, pIR_cv — реализованы вручную.
    """
    classes, counts = np.unique(y, return_counts=True)
    K = len(classes)
    result: dict[str, float] = {}

    # ── IR: max / min размер класса ──────────────────────────────────
    result["IR"] = float(counts.max() / counts.min())

    # ── CV: коэффициент вариации распределения классов ───────────────
    mu_counts = counts.mean()
    sigma_counts = np.sqrt(((counts - mu_counts) ** 2).mean())
    result["CV"] = float(sigma_counts / mu_counts) if mu_counts > 0 else np.nan

    # ── pIR_cv: CV попарных IR ───────────────────────────────────────
    if K >= 2:
        pair_irs = [
            max(counts[i], counts[j]) / min(counts[i], counts[j])
            for i, j in combinations(range(K), 2)
        ]
        pair_irs = np.array(pair_irs)
        mu_pair = pair_irs.mean()
        result["pIR_cv"] = float(pair_irs.std() / mu_pair) if mu_pair > 0 else 0.0
    else:
        result["pIR_cv"] = np.nan

    # ── HDB: гетерогенность границ ───────────────────────────────────
    # Для каждой пары классов (i, j) считаем Fisher ratio по признакам:
    # F1_f = (μi_f - μj_f)^2 / (σi_f^2 + σj_f^2)
    # F1_pair = max_f F1_f
    # HDB = std(F1_pairs) / mean(F1_pairs)
    if K >= 2:
        # Предвычислим mean и std каждого класса по каждому признаку
        class_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for cls in classes:
            mask = y == cls
            X_cls = X[mask]
            mu = X_cls.mean(axis=0)
            sigma = X_cls.std(axis=0)
            class_stats[cls] = (mu, sigma)

        f1_pairs: list[float] = []
        for i, j in combinations(classes, 2):
            mu_i, sigma_i = class_stats[i]
            mu_j, sigma_j = class_stats[j]
            denom = sigma_i ** 2 + sigma_j ** 2
            # Избегаем деления на ноль
            safe_denom = np.where(denom > 1e-12, denom, np.nan)
            f1_per_feat = (mu_i - mu_j) ** 2 / safe_denom
            f1_pair = float(np.nanmax(f1_per_feat))
            f1_pairs.append(f1_pair)

        f1_arr = np.array(f1_pairs)
        mu_f1 = np.nanmean(f1_arr)
        std_f1 = np.nanstd(f1_arr)
        result["HDB"] = float(std_f1 / mu_f1) if mu_f1 > 1e-12 else 0.0
        result["HDB_mean_f1"] = float(mu_f1)   # вспомогательный, полезен для метамодели
        result["HDB_std_f1"]  = float(std_f1)
    else:
        result["HDB"] = np.nan
        result["HDB_mean_f1"] = np.nan
        result["HDB_std_f1"]  = np.nan

    return result


# ---------------------------------------------------------------------------
# Основная функция извлечения для одного датасета
# ---------------------------------------------------------------------------

def extract_one(ds_dir: Path) -> dict | None:
    """
    Загружает датасет из ds_dir, считает все мета-признаки,
    возвращает словарь {feature_name: value, ...}.

    Стратегия ускорения:
      1. Субсэмплинг до SUBSAMPLE_TARGET если n > MAX_SAMPLES_FULL
      2. Медленные O(n²) фичи (n1,n2,n3,t1) пропускаются если
         n > MAX_SAMPLES_SLOW ИЛИ f > MAX_FEATURES_SLOW
         → заменяются на NaN (не теряем колонки для метамодели)
    """
    try:
        X_train = np.load(ds_dir / "X_train.npy")
        y_train = np.load(ds_dir / "y_train.npy")

        with open(ds_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        n_orig, f_orig = X_train.shape
        row: dict = {
            "dataset_name": meta["name"],
            "group":        meta.get("group", "unknown"),
        }

        # Базовые мета из meta.json
        for key in [
            "base_type", "n_samples_train", "n_samples_test",
            "n_features", "n_classes", "imbalance_ratio",
            "noise", "overlap", "noise_type", "noise_scale",
            "outlier_frac", "spatial_distortion",
        ]:
            row[f"gen_{key}"] = meta.get(key)

        # ── Субсэмплинг для pymfe ─────────────────────────────────────
        if n_orig > MAX_SAMPLES_FULL:
            X_mf, y_mf = _subsample(X_train, y_train, SUBSAMPLE_TARGET)
            row["gen_subsampled_to"] = len(y_mf)
        else:
            X_mf, y_mf = X_train, y_train
            row["gen_subsampled_to"] = n_orig

        n_mf = len(y_mf)
        use_slow = (n_mf <= MAX_SAMPLES_SLOW) and (f_orig <= MAX_FEATURES_SLOW)

        # ── Быстрые pymfe-признаки ────────────────────────────────────
        for group, features in PYMFE_FAST.items():
            try:
                mfe = MFE(
                    groups=[group],
                    features=features,
                    suppress_warnings=True,
                    random_state=42,
                )
                mfe.fit(X_mf, y_mf, suppress_warnings=True)
                names, values = mfe.extract(suppress_warnings=True)
                for name, val in zip(names, values):
                    row[name] = float(val) if val is not None and not isinstance(val, str) else np.nan
            except Exception as e:
                row[f"__error_{group}"] = str(e)[:120]

        # ── Медленные pymfe-признаки (с защитой) ─────────────────────
        for group, features in PYMFE_SLOW.items():
            if use_slow:
                try:
                    mfe = MFE(
                        groups=[group],
                        features=features,
                        suppress_warnings=True,
                        random_state=42,
                    )
                    mfe.fit(X_mf, y_mf, suppress_warnings=True)
                    names, values = mfe.extract(suppress_warnings=True)
                    for name, val in zip(names, values):
                        row[name] = float(val) if val is not None and not isinstance(val, str) else np.nan
                except Exception as e:
                    row[f"__error_slow_{group}"] = str(e)[:120]
            else:
                # Датасет слишком большой — ставим NaN, чтобы колонки сохранились
                for feat in features:
                    for suffix in [".mean", ".sd", ""]:
                        col = f"{feat}{suffix}"
                        # pymfe добавляет .mean/.sd только для агрегируемых фичей
                        # n1 и t1 — скаляры, n2/n3 — векторы с .mean/.sd
                        if feat in ("n1", "t1"):
                            row[feat] = np.nan
                        else:
                            row[f"{feat}.mean"] = np.nan
                            row[f"{feat}.sd"]   = np.nan

        # ── Кастомные мета-признаки (на полных данных — быстрые) ─────
        # Считаем на оригинальных данных — они не требуют O(n²)
        custom = compute_custom_metafeatures(X_train, y_train)
        row.update(custom)

        return row

    except Exception as e:
        return {"dataset_name": ds_dir.name, "__fatal_error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Параллельный запуск
# ---------------------------------------------------------------------------

def compute_all(
    datasets_dir: Path,
    output_csv: Path,
    n_jobs: int = 1,
    resume: bool = False,
) -> pd.DataFrame:

    # Найти все папки с датасетами (должны содержать X_train.npy)
    ds_dirs = sorted([
        d for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "X_train.npy").exists()
    ])

    if not ds_dirs:
        print(f"Не найдено датасетов в {datasets_dir}")
        return pd.DataFrame()

    # Resume: пропустить уже посчитанные
    done_names: set[str] = set()
    existing_df: pd.DataFrame | None = None
    if resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        done_names = set(existing_df["dataset_name"].tolist())
        print(f"Resume: пропускаем {len(done_names)} уже посчитанных датасетов.")

    todo = [d for d in ds_dirs if d.name not in done_names]
    print(f"Датасетов к обработке: {len(todo)} / {len(ds_dirs)}")

    if not todo:
        print("Всё уже посчитано.")
        return existing_df

    # Запуск
    if n_jobs == 1:
        # Последовательно — с прогрессом
        rows = []
        for i, ds_dir in enumerate(todo, 1):
            # Читаем shape заранее для информативного вывода
            try:
                shape_str = str(np.load(ds_dir / "X_train.npy").shape)
            except Exception:
                shape_str = "?"

            print(f"[{i:>4}/{len(todo)}] {ds_dir.name[:50]:<50} {shape_str:<14}",
                  end=" ", flush=True)
            row = extract_one(ds_dir)
            if row:
                rows.append(row)
                has_err = any(k.startswith("__") for k in row)
                status  = "ERR" if has_err else "OK "
                n_feat  = sum(1 for k in row if not k.startswith(("dataset_", "group", "gen_", "__")))
                sub     = row.get("gen_subsampled_to", "?")
                slow_ok = "slow✓" if row.get("n1") is not None and not (
                    isinstance(row.get("n1"), float) and np.isnan(row["n1"])
                ) else "slow–"
                print(f"{status}  mf={n_feat}  sub={sub}  {slow_ok}")
            else:
                print("NONE")

            # Промежуточное сохранение каждые 25 датасетов
            if i % 25 == 0:
                _save(rows, existing_df, output_csv)
                print(f"  → Промежуточное сохранение: {output_csv}")
    else:
        # Параллельно через joblib
        print(f"Параллельный режим: n_jobs={n_jobs}")
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(extract_one)(d) for d in todo
        )
        rows = [r for r in rows if r]

    # Финальное сохранение
    df = _save(rows, existing_df, output_csv)
    return df


def _save(
    new_rows: list[dict],
    existing_df: pd.DataFrame | None,
    output_csv: Path,
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    # Сортируем колонки: сначала идентификаторы, потом генераторные, потом мета-признаки
    id_cols   = ["dataset_name", "group"]
    gen_cols  = sorted([c for c in df.columns if c.startswith("gen_")])
    mf_cols   = sorted([c for c in df.columns
                         if c not in id_cols and not c.startswith("gen_") and not c.startswith("__")])
    err_cols  = sorted([c for c in df.columns if c.startswith("__")])

    ordered = id_cols + gen_cols + mf_cols + err_cols
    df = df[[c for c in ordered if c in df.columns]]

    df.to_csv(output_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Вычисление мета-признаков для всех датасетов (pymfe + custom)"
    )
    parser.add_argument("--datasets", default="datasets",
                        help="Папка с датасетами (default: datasets)")
    parser.add_argument("--output",   default="metafeatures.csv",
                        help="Выходной CSV (default: metafeatures.csv)")
    parser.add_argument("--jobs",     type=int, default=1,
                        help="Число параллельных процессов (default: 1)")
    parser.add_argument("--resume",   action="store_true",
                        help="Пропустить датасеты, уже присутствующие в output CSV")
    parser.add_argument("--test",     type=int, default=0, metavar="N",
                        help="Обработать только первые N датасетов (для отладки)")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets)
    output_csv   = Path(args.output)

    if not datasets_dir.exists():
        print(f"Папка {datasets_dir} не найдена.")
        return

    # Если --test: ограничиваем число датасетов
    if args.test > 0:
        ds_dirs = sorted([
            d for d in datasets_dir.iterdir()
            if d.is_dir() and (d / "X_train.npy").exists()
        ])[:args.test]
        print(f"[TEST] Обрабатываем только {len(ds_dirs)} датасетов.")
        rows = []
        for i, ds_dir in enumerate(ds_dirs, 1):
            print(f"[{i}/{len(ds_dirs)}] {ds_dir.name}", flush=True)
            row = extract_one(ds_dir)
            if row:
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\nГотово. Сохранено: {output_csv}")
        print(f"Датасетов: {len(df)}  Мета-признаков: {df.shape[1]}")
        _print_summary(df)
        return

    df = compute_all(
        datasets_dir=datasets_dir,
        output_csv=output_csv,
        n_jobs=args.jobs,
        resume=args.resume,
    )

    if df is not None and not df.empty:
        print(f"\n{'='*60}")
        print(f"Готово!")
        print(f"  Датасетов:       {len(df)}")
        print(f"  Всего колонок:   {df.shape[1]}")
        mf_cols = [c for c in df.columns
                   if not c.startswith(("dataset_", "group", "gen_", "__"))]
        print(f"  Мета-признаков:  {len(mf_cols)}")
        print(f"  Файл:            {output_csv}")
        _print_summary(df)


def _print_summary(df: pd.DataFrame):
    """Печатает краткую сводку по вычисленным мета-признакам."""
    mf_cols = [c for c in df.columns
               if not c.startswith(("dataset_", "group", "gen_", "__"))]
    if not mf_cols:
        return

    nan_frac = df[mf_cols].isna().mean()
    bad = nan_frac[nan_frac > 0.5]
    if not bad.empty:
        print(f"\n  Признаки с >50% NaN ({len(bad)}):")
        for name, frac in bad.items():
            print(f"    {name}: {frac:.0%} NaN")

    print(f"\n  Примеры значений (первая строка):")
    sample = df[mf_cols].iloc[0]
    for name in mf_cols[:10]:
        print(f"    {name:<30} {sample[name]:.4f}" if pd.notna(sample[name])
              else f"    {name:<30} NaN")
    if len(mf_cols) > 10:
        print(f"    ... и ещё {len(mf_cols) - 10} признаков")


if __name__ == "__main__":
    main()