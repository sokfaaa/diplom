"""
Поиск и скачивание реальных датасетов с OpenML и PMLB.

Датасеты сохраняются в том же формате что и синтетические:
    datasets/
        <dataset_name>/
            X_train.npy, X_test.npy, y_train.npy, y_test.npy
            meta.json

→ Сразу совместимы с compute_metafeatures.py и benchmark_samplers.py

Установка зависимостей:
    pip install openml pmlb scikit-learn numpy pandas

Запуск:
    # Скачать всё (OpenML + PMLB), целевое число = 150
    python fetch_real_datasets.py

    # Только одна платформа
    python fetch_real_datasets.py --sources openml
    python fetch_real_datasets.py --sources pmlb

    # Другая папка / число датасетов
    python fetch_real_datasets.py --output real_datasets/ --target 200

    # Пропустить уже скачанные
    python fetch_real_datasets.py --resume

    # Только посмотреть что найдено, не скачивать
    python fetch_real_datasets.py --dry-run
"""

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Критерии отбора датасетов
# ---------------------------------------------------------------------------

CRITERIA = {
    # Размер
    "min_samples":   200,       # слишком маленькие — нестабильные мета-признаки
    "max_samples":   100_000,   # слишком большие — долго считать
    "min_features":  2,
    "max_features":  500,
    # Классификация
    "min_classes":   3,
    "max_classes":   15,        # больше 15 — экзотика, сэмплеры не рассчитаны
    # Дисбаланс (IR = max_class / min_class)
    "min_ir":        1.0,       # балансированные тоже нужны
    "max_ir":        200.0,     # IR > 200 — почти вырожденный
    # Качество
    "max_missing_frac": 0.30,   # до 30% пропусков — заполним медианой
}

# Целевое покрытие по IR-зонам (для равномерного отбора)
IR_BINS = [
    (1.0,   2.0,  "balanced"),      # ~балансированные
    (2.0,   5.0,  "mild"),          # лёгкий дисбаланс
    (5.0,   15.0, "moderate"),      # умеренный
    (15.0,  50.0, "severe"),        # сильный
    (50.0,  200.0,"extreme"),       # экстремальный
]


# ---------------------------------------------------------------------------
# OpenML: поиск и скачивание
# ---------------------------------------------------------------------------

def search_openml(target_per_bin: int = 30) -> pd.DataFrame:
    """
    Запрашивает список датасетов OpenML с фильтрацией по числу классов
    и числу признаков. Возвращает DataFrame с кандидатами.
    """
    try:
        import openml
    except ImportError:
        print("  openml не установлен: pip install openml")
        return pd.DataFrame()

    print("OpenML: запрашиваю список датасетов...")
    try:
        # OpenML возвращает до 10000 за раз — берём все активные
        df = openml.datasets.list_datasets(
            output_format="dataframe",
            status="active",
            number_classes=None,   # все классы
        )
    except Exception as e:
        print(f"  Ошибка при запросе OpenML: {e}")
        return pd.DataFrame()

    print(f"  Получено записей: {len(df)}")

    # Фильтруем по доступным колонкам
    keep_cols = {
        "did": "openml_id",
        "name": "name",
        "NumberOfInstances": "n_samples",
        "NumberOfFeatures": "n_features",
        "NumberOfClasses": "n_classes",
        "NumberOfMissingValues": "n_missing",
        "NumberOfInstancesWithMissingValues": "n_rows_missing",
    }
    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Приводим к числовым
    for col in ["n_samples", "n_features", "n_classes", "n_missing"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Применяем фильтры
    c = CRITERIA
    mask = (
        df["n_samples"].between(c["min_samples"], c["max_samples"]) &
        df["n_features"].between(c["min_features"], c["max_features"]) &
        df["n_classes"].between(c["min_classes"], c["max_classes"])
    )
    if "n_missing" in df.columns:
        missing_frac = df["n_missing"] / (df["n_samples"] * df["n_features"])
        mask &= missing_frac.fillna(0) <= c["max_missing_frac"]

    df = df[mask].copy()
    df["source"] = "openml"
    df["ir_approx"] = np.nan  # посчитаем при скачивании

    print(f"  После фильтрации: {len(df)} датасетов")
    return df.reset_index(drop=True)


def download_openml(openml_id: int, name: str, output_dir: Path) -> dict | None:
    """Скачивает один датасет с OpenML, возвращает мета-словарь."""
    import openml
    try:
        dataset = openml.datasets.get_dataset(
            openml_id,
            download_data=True,
            download_qualities=True,
        )
        X, y, categorical_mask, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        if X is None or y is None:
            return None

        # Кодируем категориальные признаки
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "category":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Кодируем целевой класс
        if y.dtype == object or str(y.dtype) == "category":
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = LabelEncoder().fit_transform(y)

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        X_arr = X.values.astype(np.float64)
        y_arr = y.astype(np.int32)

        return _finalize(X_arr, y_arr, name=f"openml_{openml_id}_{name[:30]}",
                         source="openml", source_id=str(openml_id),
                         original_name=name, output_dir=output_dir)
    except Exception as e:
        return {"__error": str(e)[:200]}


# ---------------------------------------------------------------------------
# PMLB: поиск и скачивание
# ---------------------------------------------------------------------------

def search_pmlb() -> pd.DataFrame:
    """Читает локальный summary PMLB и возвращает отфильтрованных кандидатов."""
    try:
        import pmlb
        import os
        summary_path = os.path.join(os.path.dirname(pmlb.__file__), "all_summary_stats.tsv")
        df = pd.read_csv(summary_path, sep="\t")
    except Exception as e:
        print(f"  Ошибка чтения PMLB summary: {e}")
        return pd.DataFrame()

    # Только классификация
    df = df[df["task"] == "classification"].copy()

    # Переименовываем колонки
    df = df.rename(columns={
        "dataset":      "name",
        "n_instances":  "n_samples",
        "n_features":   "n_features",
        "n_classes":    "n_classes",
        "imbalance":    "pmlb_imbalance",  # [0,1] — выше = более сбалансированный
    })

    # IR приблизительно: imbalance=1/n_classes для балансированного
    # IR ≈ 1 / (imbalance * n_classes) — грубая оценка
    df["ir_approx"] = 1.0 / df["pmlb_imbalance"].replace(0, np.nan)

    # Фильтруем
    c = CRITERIA
    mask = (
        df["n_samples"].between(c["min_samples"], c["max_samples"]) &
        df["n_features"].between(c["min_features"], c["max_features"]) &
        df["n_classes"].between(c["min_classes"], c["max_classes"]) &
        df["ir_approx"].between(c["min_ir"], c["max_ir"])
    )
    df = df[mask].copy()
    df["source"] = "pmlb"
    df["openml_id"] = np.nan

    print(f"PMLB: {len(df)} датасетов после фильтрации")
    return df.reset_index(drop=True)


def download_pmlb(name: str, output_dir: Path, cache_dir: Path) -> dict | None:
    """Скачивает один датасет с PMLB."""
    try:
        import pmlb
        X, y = pmlb.fetch_data(
            name,
            return_X_y=True,
            local_cache_dir=str(cache_dir),
        )
        X_arr = np.array(X, dtype=np.float64)
        y_arr = LabelEncoder().fit_transform(y).astype(np.int32)

        # Пропуски
        if np.isnan(X_arr).any():
            col_medians = np.nanmedian(X_arr, axis=0)
            nan_mask = np.isnan(X_arr)
            X_arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        return _finalize(X_arr, y_arr, name=f"pmlb_{name[:40]}",
                         source="pmlb", source_id=name,
                         original_name=name, output_dir=output_dir)
    except Exception as e:
        return {"__error": str(e)[:200]}


# ---------------------------------------------------------------------------
# KEEL: поиск и скачивание
# ---------------------------------------------------------------------------

# Полный список KEEL imbalanced датасетов с известными IR
# Источник: sci2s.ugr.es/keel/imbalanced.php
# Формат URL: https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/...
KEEL_DATASETS = [
    # name,                                    ir_approx, n_cls
    # IR < 5 (mild)
    ("ecoli-0_vs_1",                            1.86,  2),
    ("new-thyroid1",                            5.14,  3),
    ("new-thyroid2",                            5.14,  3),
    ("cleveland-0_vs_4",                        5.55,  2),
    ("glass0",                                  2.06,  2),
    ("glass1",                                  1.82,  2),
    ("glass6",                                  6.38,  2),
    ("pima",                                    1.87,  2),
    ("haberman",                                2.78,  2),
    ("vehicle0",                                3.25,  2),
    ("vehicle1",                                2.90,  2),
    ("vehicle2",                                2.88,  2),
    ("vehicle3",                                2.99,  2),
    ("wisconsin",                               1.86,  2),
    ("yeast1",                                  2.46,  2),
    # IR 5–15 (moderate)
    ("ecoli1",                                  3.36,  2),
    ("ecoli2",                                  5.46,  2),
    ("ecoli3",                                  8.60,  2),
    ("glass-0-1-2-3_vs_4-5-6",                 3.20,  2),
    ("glass-0-1-6_vs_2",                        10.29, 2),
    ("glass-0-1-6_vs_5",                        19.44, 2),
    ("glass2",                                  11.59, 2),
    ("glass4",                                  15.46, 2),
    ("glass5",                                  22.78, 2),
    ("led7digit-0-2-4-5-6-7-8-9_vs_1",         10.97, 2),
    ("page-blocks-1-3_vs_4",                    15.86, 2),
    ("shuttle-c0-vs-c4",                        13.87, 2),
    ("vowel0",                                  9.98,  2),
    ("yeast-0-3-5-9_vs_7-8",                   9.12,  2),
    ("yeast-0-5-6-7-9_vs_4",                   9.35,  2),
    ("yeast1_vs_3",                             11.40, 2),
    ("yeast3",                                  8.10,  2),
    ("yeast4",                                  28.10, 2),
    ("yeast5",                                  32.73, 2),
    ("yeast6",                                  41.40, 2),
    # IR 15–50 (severe)
    ("abalone-17_vs_7-8-9-10",                 39.31, 2),
    ("abalone-19_vs_10-11-12-13",              49.69, 2),
    ("abalone-20_vs_8-9-10",                   72.69, 2),
    ("abalone-21_vs_8",                        40.50, 2),
    ("abalone9-18",                             16.40, 2),
    ("car-good",                                24.04, 2),
    ("car-vgood",                               25.58, 2),
    ("cleveland-0_vs_4",                        5.55,  2),
    ("dermatology-6",                           16.90, 2),
    ("ecoli-0-1-3-7_vs_2-6",                   39.14, 2),
    ("ecoli-0-1_vs_2-3-5",                     9.17,  2),
    ("ecoli-0-1_vs_5",                          11.00, 2),
    ("ecoli-0-2-3-4_vs_5",                     9.10,  2),
    ("ecoli-0-2-6-7_vs_3-5",                   9.18,  2),
    ("ecoli-0-3-4-6_vs_5",                     9.25,  2),
    ("ecoli-0-3-4-7_vs_5-6",                   10.59, 2),
    ("ecoli-0-3-4_vs_5",                        9.10,  2),
    ("ecoli-0-6-7_vs_3-5",                     9.18,  2),
    ("ecoli-0-6-7_vs_5",                        10.00, 2),
    ("ecoli-0_vs_1",                            1.86,  2),
    ("ecoli4",                                  15.80, 2),
    ("flare-F",                                 23.79, 2),
    ("glass-0-1-5_vs_2",                        9.12,  2),
    ("kddcup-buffer_overflow_vs_back",          73.43, 2),
    ("kddcup-guess_passwd_vs_satan",            6.36,  2),
    ("kddcup-land_vs_portsweep",               49.52, 2),
    ("kddcup-land_vs_satan",                    75.67, 2),
    ("kddcup-rootkit-imap_vs_back",            100.14, 2),
    ("kr-vs-k-one_vs_fifteen",                 2.56,  2),
    ("kr-vs-k-three_vs_eleven",                35.23, 2),
    ("kr-vs-k-zero_vs_eight",                  53.07, 2),
    ("kr-vs-k-zero_vs_fifteen",                80.22, 2),
    ("poker-8-9_vs_5",                         82.00, 2),
    ("poker-8-9_vs_6",                         58.40, 2),
    ("poker-8_vs_6",                           85.88, 2),
    ("poker-9_vs_7",                           29.50, 2),
    ("shuttle-2_vs_5",                         66.67, 2),
    ("shuttle-6_vs_2-3",                       22.00, 2),
    ("winequality-red-3_vs_5",                 68.10, 2),
    ("winequality-red-4",                      29.17, 2),
    ("winequality-red-8_vs_6",                 35.44, 2),
    ("winequality-red-8_vs_6-7",               46.50, 2),
    ("winequality-white-3-9_vs_5",             58.28, 2),
    ("winequality-white-3_vs_7",               44.00, 2),
    ("winequality-white-9_vs_4",               32.60, 2),
    ("yeast-0-2-5-6_vs_3-7-8-9",              9.14,  2),
    ("yeast-0-2-5-7-9_vs_3-6-7-8",           9.14,  2),
    ("yeast-0-3-5-9_vs_7-8",                  9.12,  2),
    ("yeast-1-2-8-9_vs_7",                    30.57, 2),
    ("yeast-1-4-5-8_vs_7",                    22.10, 2),
    ("yeast-1_vs_7",                           14.30, 2),
    ("yeast-2_vs_4",                           9.07,  2),
    ("yeast-2_vs_8",                           23.10, 2),
    ("yeast2vs4",                               9.08,  2),
]

# URL-шаблоны для скачивания KEEL датасетов
_KEEL_URL_TEMPLATES = [
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1/{name}.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9/{name}.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/{name}.zip",
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/{name}.zip",
]


def search_keel() -> pd.DataFrame:
    """Возвращает список KEEL датасетов как DataFrame кандидатов."""
    rows = []
    for name, ir_approx, n_cls in KEEL_DATASETS:
        rows.append({
            "name":       name,
            "source":     "keel",
            "n_samples":  0,
            "n_features": 0,
            "n_classes":  n_cls,
            "ir_approx":  ir_approx,
            "openml_id":  np.nan,
        })
    df = pd.DataFrame(rows)
    print(f"KEEL: {len(df)} датасетов в каталоге")
    return df


def _parse_keel_dat(content: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Парсит KEEL .dat / ARFF-подобный формат."""
    lines = content.split("\n")
    attr_names, attr_types = [], []
    data_start = False
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            aname = parts[1].strip("'\"")
            atype = parts[2].strip()
            attr_names.append(aname)
            attr_types.append("numeric" if any(
                t in atype.upper() for t in ["REAL", "INTEGER", "NUMERIC"]
            ) else "nominal")
        elif low.startswith("@data"):
            data_start = True
        elif data_start and line:
            data_rows.append(line)

    if not data_rows or len(attr_names) < 2:
        return None

    target_idx = len(attr_names) - 1
    X_rows, y_rows = [], []
    for row in data_rows:
        vals = [v.strip() for v in row.split(",")]
        if len(vals) != len(attr_names):
            continue
        try:
            x_vals = []
            for i, (v, t) in enumerate(zip(vals, attr_types)):
                if i == target_idx:
                    continue
                if t == "numeric":
                    x_vals.append(float(v) if v not in ("", "?") else np.nan)
                else:
                    x_vals.append(hash(v) % 1000)
            X_rows.append(x_vals)
            y_rows.append(vals[target_idx].strip("'\" "))
        except (ValueError, IndexError):
            continue

    if not X_rows:
        return None

    X = np.array(X_rows, dtype=np.float64)
    y = LabelEncoder().fit_transform(y_rows).astype(np.int32)
    return X, y


def download_keel(name: str, output_dir: Path, cache_dir: Path) -> dict | None:
    """
    Скачивает датасет KEEL.

    Стратегия (в порядке приоритета):
    1. Локальный кэш (.dat или .zip уже скачанные вручную в cache_dir/keel/)
    2. Прямая загрузка с сайта KEEL (может не работать — они блокируют боты)
    3. Зеркало через OpenML fetch_openml (некоторые KEEL датасеты там есть)

    Если KEEL не работает автоматически:
      Скачай zip вручную с https://sci2s.ugr.es/keel/imbalanced.php
      и положи .dat файл в папку: {cache_dir}/keel/{name}.dat
    """
    import requests, zipfile, io

    keel_cache = cache_dir / "keel"
    keel_cache.mkdir(parents=True, exist_ok=True)

    content = None

    # ── 1. Локальный .dat файл (скачан вручную) ───────────────────────
    local_dat = keel_cache / f"{name}.dat"
    if local_dat.exists():
        content = local_dat.read_text(encoding="latin-1", errors="replace")

    # ── 2. Локальный .zip файл (скачан вручную) ───────────────────────
    if content is None:
        local_zip = keel_cache / f"{name}.zip"
        if local_zip.exists():
            try:
                z = zipfile.ZipFile(local_zip)
                dat_files = [f for f in z.namelist()
                             if f.endswith(".dat") and "tra" not in f and "tst" not in f]
                if not dat_files:
                    dat_files = [f for f in z.namelist() if f.endswith(".dat")]
                if dat_files:
                    content = z.read(dat_files[0]).decode("latin-1", errors="replace")
            except Exception:
                pass

    # ── 3. Скачивание с KEEL сайта ────────────────────────────────────
    if content is None:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}
        for url_tmpl in _KEEL_URL_TEMPLATES:
            url = url_tmpl.format(name=name)
            try:
                r = requests.get(url, timeout=30, headers=headers)
                if r.status_code != 200:
                    continue
                z = zipfile.ZipFile(io.BytesIO(r.content))
                dat_files = [f for f in z.namelist()
                             if f.endswith(".dat") and "tra" not in f and "tst" not in f]
                if not dat_files:
                    dat_files = [f for f in z.namelist() if f.endswith(".dat")]
                if dat_files:
                    content = z.read(dat_files[0]).decode("latin-1", errors="replace")
                    # Кэшируем для следующего раза
                    local_dat.write_text(content, encoding="utf-8")
                    break
            except Exception:
                continue

    # ── 4. Fallback: пробуем через OpenML (некоторые KEEL там есть) ───
    if content is None:
        try:
            import openml
            # Ищем по имени
            results = openml.datasets.list_datasets(output_format="dataframe")
            match = results[results["name"].str.lower() == name.lower().replace("-", "_")]
            if not match.empty:
                did = int(match.iloc[0]["did"])
                ds = openml.datasets.get_dataset(did, download_data=True)
                X_oml, y_oml, _, _ = ds.get_data(
                    dataset_format="dataframe",
                    target=ds.default_target_attribute
                )
                if X_oml is not None and y_oml is not None:
                    X_arr = X_oml.apply(pd.to_numeric, errors="coerce").fillna(0).values
                    y_arr = LabelEncoder().fit_transform(y_oml.astype(str)).astype(np.int32)
                    return _finalize(X_arr, y_arr, name=f"keel_{name[:50]}",
                                     source="keel", source_id=name,
                                     original_name=name, output_dir=output_dir)
        except Exception:
            pass

    if content is None:
        return {
            "__error": (
                f"Не удалось скачать '{name}'. "
                f"Скачай вручную с sci2s.ugr.es/keel/imbalanced.php "
                f"и положи .dat в {keel_cache}/"
            )
        }

    result = _parse_keel_dat(content)
    if result is None:
        return {"__error": f"Не удалось распарсить {name}.dat"}

    X, y = result
    c = CRITERIA
    if len(y) < c["min_samples"] or X.shape[1] < c["min_features"]:
        return None

    return _finalize(X, y, name=f"keel_{name[:50]}",
                     source="keel", source_id=name,
                     original_name=name, output_dir=output_dir)


# ---------------------------------------------------------------------------
# HuggingFace: поиск и скачивание
# ---------------------------------------------------------------------------

# Только датасеты которые подтверждённо работают через HuggingFace datasets API.
# ВАЖНО: mstz/* датасеты удалены с HF Hub — они больше не доступны.
# Этот список содержит только проверенные рабочие датасеты.
HF_DATASETS: list[dict] = [
    # Проверенные датасеты — добавляй сюда новые после проверки:
    #   from datasets import load_dataset
    #   ds = load_dataset("author/name", split="train")
    #   print(ds.to_pandas().shape)
    #
    # Формат: id, subset (None если нет), split, target_col, n_approx, ir_approx, n_cls
    {
        "id": "scikit-learn/adult-census-income",
        "subset": None, "split": "train", "target": "income",
        "n": 48842, "ir": 3.2, "n_cls": 2,
    },
    {
        "id": "inria-soda/carte-blanche",
        "subset": None, "split": "train", "target": "fraud",
        "n": 20000, "ir": 50.0, "n_cls": 2,
    },
    # Добавь сюда датасеты которые работают в твоём окружении:
    # {"id": "...", "subset": None, "split": "train", "target": "...", "n": 0, "ir": 0, "n_cls": 2},
]


def search_hf() -> pd.DataFrame:
    """Возвращает список HuggingFace датасетов как DataFrame кандидатов."""
    rows = []
    for ds in HF_DATASETS:
        ir = ds.get("ir", float("nan"))
        n  = ds.get("n", 0)
        k  = ds.get("n_cls", 0)
        c  = CRITERIA
        if (n > c["max_samples"] or n < c["min_samples"] or
                k > c["max_classes"] or k < c["min_classes"] or
                (not np.isnan(ir) and ir > c["max_ir"])):
            continue
        rows.append({
            "name":      ds["id"].replace("/", "_"),
            "source":    "huggingface",
            "n_samples": n,
            "n_features": 0,
            "n_classes": k,
            "ir_approx": ir,
            "openml_id": np.nan,
            "hf_id":     ds["id"],
            "hf_subset": ds.get("subset"),
            "hf_split":  ds.get("split", "train"),
            "hf_target": ds.get("target"),
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"HuggingFace: {len(df)} датасетов в каталоге")
    if len(df) == 0:
        print("  ⚠️  Список HF датасетов пуст или все отфильтрованы.")
        print("  Чтобы добавить: запусти  python fetch_real_datasets.py --hf-check")
    return df


def download_hf(row: dict, output_dir: Path, cache_dir: Path) -> dict | None:
    """Скачивает один датасет с HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"__error": "pip install datasets"}

    hf_id    = row["hf_id"]
    hf_split = row.get("hf_split", "train")
    target   = row.get("hf_target")

    try:
        load_kwargs: dict = dict(
            path=hf_id,
            split=hf_split,
            cache_dir=str(cache_dir / "hf"),
        )
        if row.get("hf_subset"):
            load_kwargs["name"] = row["hf_subset"]

        ds = load_dataset(**load_kwargs)
        df = ds.to_pandas()

        # Целевая колонка
        if target and target in df.columns:
            target_col = target
        else:
            target_col = next(
                (c for c in df.columns
                 if c.lower() in ("class", "label", "target", "y")),
                df.columns[-1]
            )

        y_raw = df[target_col].values
        X_df  = df.drop(columns=[target_col])

        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.median())

        c = CRITERIA
        if X_df.shape[1] > c["max_features"] or X_df.shape[1] < c["min_features"]:
            return None

        X = X_df.values.astype(np.float64)
        y = LabelEncoder().fit_transform(y_raw).astype(np.int32)

        return _finalize(X, y,
                         name=f"hf_{hf_id.replace('/', '_')[:45]}",
                         source="huggingface", source_id=hf_id,
                         original_name=hf_id, output_dir=output_dir)

    except Exception as e:
        return {"__error": str(e)[:200]}


def check_hf_datasets(ids: list[str]):
    """
    Вспомогательная функция: проверяет список HF датасетов и выводит рабочие.
    Запуск: python fetch_real_datasets.py --hf-check
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        return

    print("Проверяю HuggingFace датасеты...")
    working = []
    for ds_id in ids:
        try:
            ds = load_dataset(ds_id, split="train")
            df = ds.to_pandas()
            print(f"  ✓ {ds_id:<45} shape={df.shape}  cols={list(df.columns)[:3]}")
            working.append(ds_id)
        except Exception as e:
            print(f"  ✗ {ds_id:<45} {str(e)[:60]}")
    print(f"\nРабочих: {len(working)}/{len(ids)}")
    if working:
        print("Добавь в HF_DATASETS:")
        for ds_id in working:
            print(f'  {{"id": "{ds_id}", "subset": None, "split": "train", "target": "???", "n": 0, "ir": 0, "n_cls": 0}},')


    """
    Парсит KEEL .dat формат (ARFF-подобный):
      @relation ...
      @attribute name TYPE
      ...
      @data
      v1,v2,...,class
    """
    lines = content.split("\n")

    attr_names = []
    attr_types = []  # "numeric" или "nominal"
    data_start = False
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        low = line.lower()

        if low.startswith("@attribute"):
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            aname = parts[1].strip("'\"")
            atype = parts[2].strip()
            attr_names.append(aname)
            # Числовой если REAL/INTEGER/NUMERIC, иначе nominal (включая Class)
            attr_types.append("numeric" if any(
                t in atype.upper() for t in ["REAL", "INTEGER", "NUMERIC"]
            ) else "nominal")

        elif low.startswith("@data"):
            data_start = True

        elif data_start and line:
            data_rows.append(line)

    if not data_rows or len(attr_names) < 2:
        return None

    # Последний атрибут — целевой класс
    target_idx = len(attr_names) - 1

    X_rows, y_rows = [], []
    for row in data_rows:
        vals = [v.strip() for v in row.split(",")]
        if len(vals) != len(attr_names):
            continue
        try:
            x_vals = []
            for i, (v, t) in enumerate(zip(vals, attr_types)):
                if i == target_idx:
                    continue
                if t == "numeric":
                    x_vals.append(float(v) if v not in ("", "?") else np.nan)
                else:
                    # Nominal → label encode как число
                    x_vals.append(hash(v) % 1000)
            X_rows.append(x_vals)
            y_rows.append(vals[target_idx].strip("'\" "))
        except (ValueError, IndexError):
            continue

    if not X_rows:
        return None

    X = np.array(X_rows, dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(y_rows).astype(np.int32)
    return X, y


def download_keel(name: str, output_dir: Path, cache_dir: Path) -> dict | None:
    """Скачивает один датасет KEEL, парсит .dat файл."""
    import requests, zipfile, io

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"keel_{name}.dat"

    # Пробуем загрузить из кэша
    if cache_file.exists():
        content = cache_file.read_text(encoding="latin-1", errors="replace")
    else:
        content = None
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

        for url_tmpl in _KEEL_URL_TEMPLATES:
            url = url_tmpl.format(name=name)
            try:
                r = requests.get(url, timeout=30, headers=headers)
                if r.status_code != 200:
                    continue

                # Распаковываем zip
                z = zipfile.ZipFile(io.BytesIO(r.content))
                dat_files = [f for f in z.namelist() if f.endswith(".dat")]
                if not dat_files:
                    continue

                # Берём файл с данными (не заголовочный)
                dat_name = next(
                    (f for f in dat_files if "tra" not in f and "tst" not in f),
                    dat_files[0]
                )
                content = z.read(dat_name).decode("latin-1", errors="replace")
                cache_file.write_text(content, encoding="utf-8")
                break

            except Exception:
                continue

        if content is None:
            return {"__error": f"Не удалось скачать {name} ни по одному URL"}

    # Парсим
    result = _parse_keel_dat(content)
    if result is None:
        return {"__error": f"Не удалось распарсить {name}.dat"}

    X, y = result

    # Проверка размера
    c = CRITERIA
    if len(y) < c["min_samples"] or X.shape[1] < c["min_features"]:
        return None

    return _finalize(
        X, y,
        name=f"keel_{name[:50]}",
        source="keel",
        source_id=name,
        original_name=name,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# HuggingFace: поиск и скачивание
# ---------------------------------------------------------------------------

# Кураторский список проверенных табличных датасетов на HuggingFace
# с известными характеристиками для задачи дисбаланса классов
HF_DATASETS: list[dict] = [
    # dataset_id,            subset, split,  target_col,         n_approx, ir_approx, n_cls
    {"id": "mstz/ecoli",             "subset": None,    "split": "train", "target": "class",          "n": 336,   "ir": 8.6,  "n_cls": 8},
    {"id": "mstz/yeast",             "subset": None,    "split": "train", "target": "class",          "n": 1484,  "ir": 28.1, "n_cls": 10},
    {"id": "mstz/abalone",           "subset": None,    "split": "train", "target": "class",          "n": 4177,  "ir": 49.7, "n_cls": 3},
    {"id": "mstz/page-blocks",       "subset": None,    "split": "train", "target": "class",          "n": 5473,  "ir": 175,  "n_cls": 5},
    {"id": "mstz/letter",            "subset": None,    "split": "train", "target": "class",          "n": 20000, "ir": 1.6,  "n_cls": 26},
    {"id": "mstz/satimage",          "subset": None,    "split": "train", "target": "class",          "n": 6435,  "ir": 9.3,  "n_cls": 6},
    {"id": "mstz/vowel",             "subset": None,    "split": "train", "target": "class",          "n": 990,   "ir": 1.0,  "n_cls": 11},
    {"id": "mstz/car-evaluation",    "subset": None,    "split": "train", "target": "class",          "n": 1728,  "ir": 25.6, "n_cls": 4},
    {"id": "mstz/mammography",       "subset": None,    "split": "train", "target": "class",          "n": 11183, "ir": 42.0, "n_cls": 2},
    {"id": "mstz/diabetes",          "subset": None,    "split": "train", "target": "class",          "n": 768,   "ir": 1.9,  "n_cls": 2},
    {"id": "mstz/heart_failure",     "subset": None,    "split": "train", "target": "class",          "n": 299,   "ir": 1.9,  "n_cls": 2},
    {"id": "mstz/segment",           "subset": None,    "split": "train", "target": "class",          "n": 2310,  "ir": 1.0,  "n_cls": 7},
    {"id": "mstz/splice",            "subset": None,    "split": "train", "target": "class",          "n": 3190,  "ir": 1.0,  "n_cls": 3},
    {"id": "mstz/waveform-noise",    "subset": None,    "split": "train", "target": "class",          "n": 5000,  "ir": 1.0,  "n_cls": 3},
    {"id": "mstz/phoneme",           "subset": None,    "split": "train", "target": "class",          "n": 5404,  "ir": 2.4,  "n_cls": 2},
    {"id": "mstz/magic",             "subset": None,    "split": "train", "target": "class",          "n": 19020, "ir": 1.8,  "n_cls": 2},
    {"id": "mstz/spambase",          "subset": None,    "split": "train", "target": "class",          "n": 4601,  "ir": 1.5,  "n_cls": 2},
    {"id": "mstz/electricity",       "subset": None,    "split": "train", "target": "class",          "n": 45312, "ir": 1.4,  "n_cls": 2},
    {"id": "mstz/bank-marketing",    "subset": None,    "split": "train", "target": "class",          "n": 45211, "ir": 7.6,  "n_cls": 2},
    {"id": "mstz/eeg-eye-state",     "subset": None,    "split": "train", "target": "class",          "n": 14980, "ir": 1.2,  "n_cls": 2},
    {"id": "mstz/ringnorm",          "subset": None,    "split": "train", "target": "class",          "n": 7400,  "ir": 1.0,  "n_cls": 2},
    {"id": "mstz/twonorm",           "subset": None,    "split": "train", "target": "class",          "n": 7400,  "ir": 1.0,  "n_cls": 2},
    {"id": "mstz/optical-digits",    "subset": None,    "split": "train", "target": "class",          "n": 5620,  "ir": 1.1,  "n_cls": 10},
    {"id": "mstz/pendigits",         "subset": None,    "split": "train", "target": "class",          "n": 10992, "ir": 1.1,  "n_cls": 10},
    {"id": "mstz/thyroid-disease",   "subset": None,    "split": "train", "target": "class",          "n": 7200,  "ir": 41.6, "n_cls": 3},
    {"id": "scikit-learn/adult-census-income", "subset": None, "split": "train", "target": "income",  "n": 48842, "ir": 3.2,  "n_cls": 2},
    {"id": "inria-soda/carte-blanche","subset": None,  "split": "train",  "target": "fraud",          "n": 20000, "ir": 50.0, "n_cls": 2},
]


def search_hf() -> pd.DataFrame:
    """Возвращает список HuggingFace датасетов как DataFrame кандидатов."""
    rows = []
    for ds in HF_DATASETS:
        ir = ds.get("ir", float("nan"))
        n  = ds.get("n", 0)
        k  = ds.get("n_cls", 0)
        # Фильтруем по критериям
        c = CRITERIA
        if (n > c["max_samples"] or n < c["min_samples"] or
                k > c["max_classes"] or k < c["min_classes"] or
                (not np.isnan(ir) and ir > c["max_ir"])):
            continue
        rows.append({
            "name":       ds["id"].replace("/", "_"),
            "source":     "huggingface",
            "n_samples":  n,
            "n_features": 0,
            "n_classes":  k,
            "ir_approx":  ir,
            "openml_id":  np.nan,
            "hf_id":      ds["id"],
            "hf_subset":  ds.get("subset"),
            "hf_split":   ds.get("split", "train"),
            "hf_target":  ds.get("target"),
        })
    df = pd.DataFrame(rows)
    print(f"HuggingFace: {len(df)} датасетов в каталоге")
    return df


def download_hf(row: dict, output_dir: Path, cache_dir: Path) -> dict | None:
    """Скачивает один датасет с HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"__error": "pip install datasets"}

    hf_id     = row["hf_id"]
    hf_subset = row.get("hf_subset")
    hf_split  = row.get("hf_split", "train")
    target    = row.get("hf_target")

    try:
        load_kwargs = dict(
            path=hf_id,
            split=hf_split,
            cache_dir=str(cache_dir / "hf"),
        )
        if hf_subset:
            load_kwargs["name"] = hf_subset

        ds = load_dataset(**load_kwargs)
        df = ds.to_pandas()

        # Определяем целевую колонку
        if target and target in df.columns:
            target_col = target
        else:
            # Пробуем угадать: последняя колонка или колонка с "class"/"label"/"target"
            guess = next(
                (c for c in df.columns if c.lower() in ("class", "label", "target", "y")),
                df.columns[-1]
            )
            target_col = guess

        y_raw = df[target_col].values
        X_df  = df.drop(columns=[target_col])

        # Кодируем категориальные признаки
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.fillna(X_df.median())

        # Проверка размера
        c = CRITERIA
        if X_df.shape[1] > c["max_features"] or X_df.shape[1] < c["min_features"]:
            return None

        X = X_df.values.astype(np.float64)
        y = LabelEncoder().fit_transform(y_raw).astype(np.int32)

        ds_name = f"hf_{hf_id.replace('/', '_')[:45]}"
        return _finalize(
            X, y,
            name=ds_name,
            source="huggingface",
            source_id=hf_id,
            original_name=hf_id,
            output_dir=output_dir,
        )

    except Exception as e:
        return {"__error": str(e)[:200]}




def _finalize(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    source: str,
    source_id: str,
    original_name: str,
    output_dir: Path,
) -> dict | None:
    """Проверяет IR, делает train/test split, нормализует, сохраняет."""

    classes, counts = np.unique(y, return_counts=True)
    n_cls = len(classes)

    if n_cls < CRITERIA["min_classes"]:
        return None

    ir = float(counts.max() / counts.min())
    if ir > CRITERIA["max_ir"]:
        return None

    # Стратифицированный split (80/20)
    # Если какой-то класс слишком маленький — fallback без стратификации
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Сохранение
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "X_train.npy", X_train)
    np.save(ds_dir / "X_test.npy",  X_test)
    np.save(ds_dir / "y_train.npy", y_train)
    np.save(ds_dir / "y_test.npy",  y_test)

    # IR по IR-bins
    ir_zone = "unknown"
    for lo, hi, zone in IR_BINS:
        if lo <= ir < hi:
            ir_zone = zone
            break

    # class counts
    tr_classes, tr_counts = np.unique(y_train, return_counts=True)
    actual_weights = (tr_counts / tr_counts.sum()).tolist()

    meta = {
        "name":             name,
        "group":            f"real_{source}",
        "source":           source,
        "source_id":        source_id,
        "original_name":    original_name,
        "base_type":        f"real_{source}",
        "n_samples_total":  int(len(y)),
        "n_samples_train":  int(len(y_train)),
        "n_samples_test":   int(len(y_test)),
        "n_features":       int(X.shape[1]),
        "n_classes":        int(n_cls),
        "imbalance_ratio":  round(ir, 4),
        "ir_zone":          ir_zone,
        "actual_weights":   actual_weights,
        "class_counts_train": tr_counts.tolist(),
        # Поля-заглушки для совместимости с meta.json синтетических датасетов
        "noise":             0.0,
        "overlap":           0.0,
        "noise_type":        None,
        "noise_scale":       0.0,
        "outlier_frac":      0.0,
        "spatial_distortion": False,
        "target_weights":    None,
        "random_state":      42,
    }

    with open(ds_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ---------------------------------------------------------------------------
# Стратифицированный отбор кандидатов по IR-зонам
# ---------------------------------------------------------------------------

def stratified_select(
    candidates: pd.DataFrame,
    target: int,
    done_names: set[str],
) -> pd.DataFrame:
    """
    Отбирает датасеты равномерно по IR-зонам.
    Если ir_approx неизвестен — распределяем равномерно.
    """
    if candidates.empty:
        return candidates

    # Убираем уже скачанные
    if "name" in candidates.columns:
        candidates = candidates[~candidates["name"].isin(done_names)].copy()

    per_bin = max(1, target // len(IR_BINS))
    selected = []

    for lo, hi, zone in IR_BINS:
        if "ir_approx" in candidates.columns and candidates["ir_approx"].notna().any():
            bin_df = candidates[
                candidates["ir_approx"].between(lo, hi, inclusive="left")
            ]
        else:
            bin_df = candidates  # нет IR — берём всё равномерно

        # Shuffle для случайного порядка внутри бина
        bin_df = bin_df.sample(frac=1, random_state=42).head(per_bin)
        selected.append(bin_df)

    result = pd.concat(selected).drop_duplicates(
        subset=["name"] if "name" in candidates.columns else None
    )

    # Добираем до target если не хватает
    remaining = candidates[~candidates.index.isin(result.index)]
    n_more = target - len(result)
    if n_more > 0 and not remaining.empty:
        extra = remaining.sample(min(n_more, len(remaining)), random_state=42)
        result = pd.concat([result, extra])

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run(
    sources: list[str],
    output_dir: Path,
    cache_dir: Path,
    target: int,
    resume: bool,
    dry_run: bool,
    openml_key: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Уже скачанные
    done_names: set[str] = set()
    all_meta: list[dict] = []
    summary_path = output_dir / "all_meta.json"
    if resume and summary_path.exists():
        with open(summary_path) as f:
            all_meta = json.load(f)
        done_names = {m["name"] for m in all_meta}
        print(f"Resume: уже скачано {len(done_names)} датасетов")

    # Настройка OpenML API ключа (опционально)
    if "openml" in sources and openml_key:
        try:
            import openml
            openml.config.apikey = openml_key
        except ImportError:
            pass

    # ── Поиск кандидатов ──────────────────────────────────────────────
    all_candidates = []

    per_source = target // len(sources) + 10  # +10 запас на ошибки

    if "openml" in sources:
        oml_candidates = search_openml()
        if not oml_candidates.empty:
            oml_selected = stratified_select(oml_candidates, per_source, done_names)
            all_candidates.append(oml_selected)
            print(f"  OpenML кандидатов отобрано: {len(oml_selected)}")

    if "pmlb" in sources:
        pmlb_candidates = search_pmlb()
        if not pmlb_candidates.empty:
            pmlb_selected = stratified_select(pmlb_candidates, per_source, done_names)
            all_candidates.append(pmlb_selected)
            print(f"  PMLB кандидатов отобрано: {len(pmlb_selected)}")

    if "keel" in sources:
        keel_candidates = search_keel()
        if not keel_candidates.empty:
            keel_selected = stratified_select(keel_candidates, per_source, done_names)
            all_candidates.append(keel_selected)
            print(f"  KEEL кандидатов отобрано: {len(keel_selected)}")

    if "huggingface" in sources:
        hf_candidates = search_hf()
        if not hf_candidates.empty:
            hf_selected = stratified_select(hf_candidates, per_source, done_names)
            all_candidates.append(hf_selected)
            print(f"  HuggingFace кандидатов отобрано: {len(hf_selected)}")

    if not all_candidates:
        print("Нет кандидатов для скачивания.")
        return

    candidates = pd.concat(all_candidates, ignore_index=True)
    print(f"\nИтого кандидатов: {len(candidates)}")

    if dry_run:
        print("\n[DRY RUN] Список кандидатов (не скачиваем):")
        print(candidates[["name", "source", "n_samples", "n_features",
                           "n_classes", "ir_approx"]].to_string(index=False))
        return

    # ── Скачивание ────────────────────────────────────────────────────
    ok = err = skip = 0
    total = len(candidates)

    for i, row in candidates.iterrows():
        source = row.get("source", "unknown")
        name   = str(row.get("name", f"ds_{i}"))
        n      = int(row.get("n_samples", 0))
        f      = int(row.get("n_features", 0))
        k      = int(row.get("n_classes", 0))

        # Проверяем уже скачанные
        ds_name_prefix = f"{source}_{name[:40]}"
        already = any(
            m["name"].startswith(f"openml_") and str(row.get("openml_id","")) in m["name"]
            or m["name"] == f"pmlb_{name[:40]}"
            for m in all_meta
        )
        if already:
            skip += 1
            continue

        print(f"[{ok+err+skip+1:>4}/{total}] [{source}] {name[:45]:<45} "
              f"n={n:<7} f={f:<5} k={k:<3}", end=" ", flush=True)

        meta = None
        try:
            if source == "openml":
                openml_id = int(row.get("openml_id", 0))
                meta = download_openml(openml_id, name, output_dir)
            elif source == "pmlb":
                meta = download_pmlb(name, output_dir, cache_dir)
            elif source == "keel":
                meta = download_keel(name, output_dir, cache_dir)
            elif source == "huggingface":
                meta = download_hf(row.to_dict(), output_dir, cache_dir)

            if meta and "__error" not in meta:
                all_meta.append(meta)
                ir = meta.get("imbalance_ratio", float("nan"))
                print(f"OK  IR={ir:.1f}  zone={meta.get('ir_zone','?')}")
                ok += 1
            elif meta and "__error" in meta:
                print(f"ERR {meta['__error'][:50]}")
                err += 1
            else:
                print("SKIP (фильтр)")
                skip += 1

        except Exception as e:
            print(f"ERR {str(e)[:60]}")
            err += 1

        # Промежуточное сохранение каждые 10 датасетов
        if (ok + err) % 10 == 0:
            _save_summary(all_meta, summary_path)

        # Небольшая пауза чтобы не перегружать API
        time.sleep(0.3)

        if ok >= target:
            print(f"\nДостигнуто целевое число: {target}")
            break

    # Финальное сохранение
    _save_summary(all_meta, summary_path)

    # ── Итог ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Готово!")
    print(f"  Скачано успешно:  {ok}")
    print(f"  Ошибок:           {err}")
    print(f"  Пропущено:        {skip}")
    print(f"  Итого в каталоге: {len(all_meta)}")
    print(f"  Папка:            {output_dir}")

    if all_meta:
        _print_summary(all_meta)


def _save_summary(all_meta: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)


def _print_summary(all_meta: list[dict]):
    """Печатает распределение по IR-зонам и источникам."""
    from collections import Counter
    zones   = Counter(m.get("ir_zone", "?") for m in all_meta)
    sources = Counter(m.get("source", "?") for m in all_meta)
    irs     = [m["imbalance_ratio"] for m in all_meta if "imbalance_ratio" in m]

    print(f"\n  По источникам:  {dict(sources)}")
    print(f"  По IR-зонам:    {dict(zones)}")
    if irs:
        print(f"  IR: min={min(irs):.1f}  median={sorted(irs)[len(irs)//2]:.1f}  max={max(irs):.1f}")

    n_cls_vals = [m["n_classes"] for m in all_meta if "n_classes" in m]
    if n_cls_vals:
        print(f"  n_classes: {sorted(set(n_cls_vals))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Скачивание реальных датасетов с OpenML, PMLB, KEEL, HuggingFace",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--output",    default="datasets",
                        help="Папка для сохранения (default: datasets)")
    parser.add_argument("--cache",     default=".dataset_cache",
                        help="Папка кэша (default: .dataset_cache)")
    parser.add_argument("--target",    type=int, default=150,
                        help="Целевое число датасетов (default: 150)")
    parser.add_argument("--sources",   default="openml,pmlb,keel,huggingface",
                        help="Источники: openml, pmlb, keel, huggingface\n"
                             "(default: все четыре)")
    parser.add_argument("--resume",    action="store_true",
                        help="Пропустить уже скачанные")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Только показать кандидатов, не скачивать")
    parser.add_argument("--openml-key", default="",
                        help="API ключ OpenML (необязательно)")
    parser.add_argument("--keel-dir",  default=None,
                        help="Папка с вручную скачанными KEEL .dat/.zip файлами.\n"
                             "Если указана — используется как кэш для KEEL.\n"
                             "Скачать датасеты: https://sci2s.ugr.es/keel/imbalanced.php")
    parser.add_argument("--hf-check",  action="store_true",
                        help="Проверить список HuggingFace датасетов и показать рабочие.\n"
                             "Используй чтобы найти новые датасеты для добавления в HF_DATASETS.")
    args = parser.parse_args()

    # ── --hf-check ────────────────────────────────────────────────────
    if args.hf_check:
        check_ids = [
            "scikit-learn/adult-census-income",
            "inria-soda/carte-blanche",
            "mstz/ecoli", "mstz/yeast", "mstz/mammography",
            "mstz/diabetes", "mstz/satimage", "mstz/phoneme",
            "mstz/bank-marketing", "mstz/thyroid-disease",
            "openml/wine-quality-red", "openml/credit-g",
            "openml/diabetes", "openml/adult",
            "jlh/uci_wine", "imodels/credit-card",
        ]
        check_hf_datasets(check_ids)
        return

    sources = [s.strip().lower() for s in args.sources.split(",")]
    valid   = {"openml", "pmlb", "keel", "huggingface"}
    invalid = set(sources) - valid
    if invalid:
        print(f"Неизвестные источники: {invalid}. Допустимые: {valid}")
        return

    # Если указана папка с KEEL файлами — копируем в кэш
    cache_dir = Path(args.cache)
    if args.keel_dir:
        keel_src = Path(args.keel_dir)
        keel_dst = cache_dir / "keel"
        keel_dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in keel_src.glob("*.dat"):
            dst = keel_dst / f.name
            if not dst.exists():
                import shutil
                shutil.copy2(f, dst)
                copied += 1
        for f in keel_src.glob("*.zip"):
            dst = keel_dst / f.name
            if not dst.exists():
                import shutil
                shutil.copy2(f, dst)
                copied += 1
        if copied:
            print(f"Скопировано {copied} KEEL файлов из {keel_src} → {keel_dst}")

    print(f"Источники:  {sources}")
    print(f"Цель:       {args.target} датасетов")
    print(f"Папка:      {args.output}")
    print(f"\nКритерии отбора:")
    for k, v in CRITERIA.items():
        print(f"  {k}: {v}")
    print()

    run(
        sources=sources,
        output_dir=Path(args.output),
        cache_dir=cache_dir,
        target=args.target,
        resume=args.resume,
        dry_run=args.dry_run,
        openml_key=args.openml_key,
    )


if __name__ == "__main__":
    main()