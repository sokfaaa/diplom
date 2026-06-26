"""
Сборщик МНОГОКЛАССОВЫХ реальных датасетов (3–10 классов).
Источники: OpenML, PMLB.
Никакой синтетики — только реальные данные.

Обработка признаков:
  - Числовые пропуски → медиана
  - Категориальные пропуски → мода
  - Выбросы → clip по 1-му и 99-му перцентилю
  - Категориальные признаки:
      ≤ 10 уникальных  → OneHotEncoder (редкие < 1% → 'other')
      10–100 уникальных → OneHotEncoder с объединением редких в 'other'
      > 100 уникальных  → TargetEncoder

Запуск:
    python fetch_multiclass_real.py                   # все источники
    python fetch_multiclass_real.py --sources pmlb    # только PMLB
    python fetch_multiclass_real.py --sources openml  # только OpenML
    python fetch_multiclass_real.py --resume          # продолжить
    python fetch_multiclass_real.py --dry-run         # список без скачивания
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, TargetEncoder,
)

warnings.filterwarnings("ignore")

# ── Жёсткие критерии — ТОЛЬКО многоклассовые ─────────────────────────────
MIN_CLASSES  = 3
MAX_CLASSES  = 10
MIN_SAMPLES  = 200
MAX_SAMPLES  = 200_000
MIN_FEATURES = 2
MAX_FEATURES = 500
MAX_MISSING  = 0.35

# ── Хардкодированный список PMLB многоклассовых ───────────────────────────
# Взят напрямую из all_summary_stats.tsv — только n_classes 3-10
PMLB_MULTICLASS = [
    # (name, n_classes)
    ("allbp", 3), ("allhypo", 3), ("ann_thyroid", 3), ("balance_scale", 3),
    ("cars", 3), ("contraceptive", 3), ("connect_4", 3), ("cmc", 3),
    ("dna", 3), ("splice", 3), ("waveform_21", 3), ("waveform_40", 3),
    ("schizo", 3), ("new_thyroid", 3), ("penguins", 3),
    ("analcatdata_germangss", 4), ("vehicle", 4), ("nursery", 4),
    ("car", 4), ("allhyper", 4), ("car_evaluation", 4),
    ("analcatdata_authorship", 4), ("allrep", 4),
    ("solar_flare_1", 5), ("glass", 5), ("cleveland_nominal", 5),
    ("auto", 5), ("cleveland", 5), ("ecoli", 5), ("prnn_fglass", 5),
    ("sleep", 5), ("calendarDOW", 5), ("page_blocks", 5),
    ("wine_quality_red", 6), ("analcatdata_dmft", 6), ("dermatology", 6),
    ("solar_flare_2", 6), ("satimage", 6),
    ("wine_quality_white", 7), ("shuttle", 7), ("segmentation", 7),
    ("fars", 8),
    ("yeast", 9),
    ("mfeat_karhunen", 10), ("mfeat_fourier", 10), ("mfeat_factors", 10),
    ("led7", 10), ("led24", 10), ("mfeat_zernike", 10), ("mfeat_pixel", 10),
    ("mfeat_morphological", 10), ("pendigits", 10), ("optdigits", 10),
    ("mnist", 10),
]


# ── Финализация ───────────────────────────────────────────────────────────

def _finalize(X, y, name, source, source_id, original_name, output_dir):
    classes, counts = np.unique(y, return_counts=True)
    n_cls = len(classes)

    # СТРОГАЯ проверка числа классов
    if not (MIN_CLASSES <= n_cls <= MAX_CLASSES):
        return None
    if not (MIN_SAMPLES <= len(y) <= MAX_SAMPLES):
        return None

    ir = float(counts.max() / counts.min())

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "X_train.npy", X_train)
    np.save(ds_dir / "X_test.npy",  X_test)
    np.save(ds_dir / "y_train.npy", y_train)
    np.save(ds_dir / "y_test.npy",  y_test)

    tr_cls, tr_cnt = np.unique(y_train, return_counts=True)
    meta = {
        "name":             name,
        "group":            f"real_{source}_multiclass",
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
        "actual_weights":   (tr_cnt / tr_cnt.sum()).tolist(),
        "class_counts_train": tr_cnt.tolist(),
        "noise": 0.0, "overlap": 0.0, "noise_type": None,
        "noise_scale": 0.0, "outlier_frac": 0.0,
        "spatial_distortion": False, "target_weights": None, "random_state": 42,
    }
    with open(ds_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _clean_X(df_X: pd.DataFrame, y_series: pd.Series | None = None) -> pd.DataFrame:
    """
    Полный препроцессинг признаков:

    1. Числовые пропуски  → медиана
    2. Категориальные пропуски → мода
    3. Выбросы числовых  → clip по 1-му и 99-му перцентилю
    4. Кодирование категориальных признаков:
         ≤ 100 уникальных → OneHotEncoder
                            (редкие категории < 1% объектов → 'other')
         > 100 уникальных → TargetEncoder (требует y_series)
                            при отсутствии y_series — LabelEncoder как fallback
    5. Дроп колонок с >MAX_MISSING пропусков (до заполнения)
    """
    df = df_X.copy()
    y  = y_series  # алиас для читаемости

    # Разделяем на числовые и категориальные
    # Учитываем pandas StringDtype ('string'), object и category
    def _is_categorical(series: pd.Series) -> bool:
        dtype_str = str(series.dtype).lower()
        return (series.dtype == object or
                dtype_str == "category" or
                "string" in dtype_str or
                dtype_str == "str")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if _is_categorical(df[c])]

    # ── 1. Дроп колонок с >MAX_MISSING пропусков ─────────────────────
    miss_frac = df.isna().mean()
    drop_cols = miss_frac[miss_frac > MAX_MISSING].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        num_cols = [c for c in num_cols if c not in drop_cols]
        cat_cols = [c for c in cat_cols if c not in drop_cols]

    # ── 2. Заполнение пропусков ───────────────────────────────────────
    # Числовые → медиана
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Категориальные → мода
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            df[col]  = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "unknown")

    # ── 3. Clip выбросов числовых (1-й и 99-й перцентиль) ────────────
    for col in num_cols:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        if p01 < p99:   # защита от константных колонок
            df[col] = df[col].clip(lower=p01, upper=p99)

    # ── 4. Кодирование категориальных признаков ───────────────────────
    ohe_cols     = []   # OneHotEncoder
    target_cols  = []   # TargetEncoder (> 100 уникальных)

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 100:
            target_cols.append(col)
        else:
            ohe_cols.append(col)

    # OneHotEncoder — редкие категории < 1% → 'other'
    if ohe_cols:
        new_frames = []
        for col in ohe_cols:
            ser = df[col].astype(str)
            # Объединяем редкие категории
            freq = ser.value_counts(normalize=True)
            rare = freq[freq < 0.01].index
            if len(rare) > 0:
                ser = ser.replace(rare, "other")
            df[col] = ser

        try:
            ohe = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float64,
            )
            ohe_arr   = ohe.fit_transform(df[ohe_cols])
            ohe_names = ohe.get_feature_names_out(ohe_cols)
            df_ohe    = pd.DataFrame(ohe_arr, columns=ohe_names, index=df.index)
            df = df.drop(columns=ohe_cols)
            df = pd.concat([df, df_ohe], axis=1)
        except Exception:
            # Fallback: LabelEncoder
            for col in ohe_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # TargetEncoder — для признаков с > 100 уникальных значений
    if target_cols:
        if y is not None:
            try:
                te = TargetEncoder(random_state=42)
                te_arr = te.fit_transform(
                    df[target_cols].astype(str), y
                )
                df_te = pd.DataFrame(
                    te_arr, columns=target_cols, index=df.index
                )
                df = df.drop(columns=target_cols)
                df = pd.concat([df, df_te], axis=1)
            except Exception:
                # Fallback: LabelEncoder
                for col in target_cols:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            # Без y — LabelEncoder
            for col in target_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Финальный привод к float
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    return df


# ── PMLB ─────────────────────────────────────────────────────────────────

def fetch_pmlb_all(output_dir, cache_dir, done_names):
    try:
        import pmlb
    except ImportError:
        print("pip install pmlb")
        return []

    results = []
    total = len(PMLB_MULTICLASS)
    for i, (ds_name, n_cls_expected) in enumerate(PMLB_MULTICLASS, 1):
        name = f"pmlb_{ds_name}"
        if name in done_names:
            print(f"  [{i:>2}/{total}] {ds_name:<30} ПРОПУСК (уже скачан)")
            continue
        print(f"  [{i:>2}/{total}] {ds_name:<30} cls≈{n_cls_expected}", end=" ", flush=True)
        try:
            X_raw, y_raw = pmlb.fetch_data(
                ds_name, return_X_y=True,
                local_cache_dir=str(cache_dir / "pmlb"),
            )
            X = np.array(X_raw, dtype=np.float64)

            # Числовые пропуски → медиана
            if np.isnan(X).any():
                med = np.nanmedian(X, axis=0)
                nm  = np.isnan(X)
                X[nm] = np.take(med, np.where(nm)[1])

            # Clip выбросов: 1-й и 99-й перцентиль по каждому признаку
            p01 = np.nanpercentile(X, 1, axis=0)
            p99 = np.nanpercentile(X, 99, axis=0)
            for j in range(X.shape[1]):
                if p01[j] < p99[j]:
                    X[:, j] = np.clip(X[:, j], p01[j], p99[j])

            y = LabelEncoder().fit_transform(y_raw).astype(np.int32)

            meta = _finalize(X, y, name=name, source="pmlb",
                             source_id=ds_name, original_name=ds_name,
                             output_dir=output_dir)
            if meta:
                results.append(meta)
                print(f"OK  cls={meta['n_classes']}  n={meta['n_samples_total']}  "
                      f"IR={meta['imbalance_ratio']:.1f}")
            else:
                print(f"SKIP (cls={len(np.unique(y))} не в [{MIN_CLASSES},{MAX_CLASSES}])")
        except Exception as e:
            print(f"ERR {str(e)[:60]}")
        time.sleep(0.5)
    return results


# ── OpenML ────────────────────────────────────────────────────────────────

def fetch_openml_all(output_dir, done_names, max_datasets=400):
    try:
        import openml
    except ImportError:
        print("pip install openml")
        return []

    print("  Запрашиваю список OpenML...")
    try:
        df = openml.datasets.list_datasets(
            output_format="dataframe", status="active"
        )
    except Exception as e:
        print(f"  OpenML ошибка: {e}")
        return []

    # Фильтрация — ТОЛЬКО многоклассовые
    needed = {"NumberOfClasses", "NumberOfInstances", "NumberOfFeatures"}
    if not needed.issubset(df.columns):
        print("  OpenML: нет нужных колонок в списке")
        return []

    mask = (
        df["NumberOfClasses"].between(MIN_CLASSES, MAX_CLASSES) &
        df["NumberOfInstances"].between(MIN_SAMPLES, MAX_SAMPLES) &
        df["NumberOfFeatures"].between(MIN_FEATURES, MAX_FEATURES)
    )
    candidates = df[mask].drop_duplicates("name").copy()
    # Сортируем: больше классов — приоритет (нам нужны многоклассовые)
    candidates = candidates.sort_values("NumberOfClasses", ascending=False)
    print(f"  Найдено многоклассовых 3-10: {len(candidates)}")

    results = []
    total = min(len(candidates), max_datasets)
    for i, (_, row) in enumerate(candidates.head(max_datasets).iterrows(), 1):
        did  = int(row["did"])
        dname = str(row["name"])
        ncls = int(row["NumberOfClasses"])
        ns   = int(row["NumberOfInstances"])
        name = f"openml_{did}_{dname[:25]}"

        if name in done_names:
            print(f"  [{i:>3}/{total}] {dname[:35]:<35} ПРОПУСК")
            continue

        print(f"  [{i:>3}/{total}] {dname[:35]:<35} cls={ncls} n={ns}", end=" ", flush=True)
        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=True
            )
            X_raw, y_raw, _, _ = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute,
            )
            if X_raw is None or y_raw is None:
                print("SKIP (нет данных)")
                continue

            # Кодируем y до _clean_X чтобы передать для TargetEncoder
            if y_raw.dtype == object or str(y_raw.dtype) == "category":
                y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int32)
            else:
                y = LabelEncoder().fit_transform(y_raw).astype(np.int32)

            X_df = _clean_X(X_raw.copy(), y_series=pd.Series(y))
            if X_df.shape[1] < MIN_FEATURES or X_df.shape[1] > MAX_FEATURES:
                print(f"SKIP (f={X_df.shape[1]})")
                continue

            X = X_df.values.astype(np.float64)

            meta = _finalize(X, y, name=name, source="openml",
                             source_id=str(did), original_name=dname,
                             output_dir=output_dir)
            if meta:
                results.append(meta)
                print(f"OK  cls={meta['n_classes']}  "
                      f"n={meta['n_samples_total']}  "
                      f"IR={meta['imbalance_ratio']:.1f}")
            else:
                actual_cls = len(np.unique(y))
                print(f"SKIP (реальных классов={actual_cls})")
        except Exception as e:
            print(f"ERR {str(e)[:70]}")

        time.sleep(0.4)

    return results


# ── Kaggle ────────────────────────────────────────────────────────────────

def _kaggle_auth():
    """Аутентификация Kaggle с поддержкой access_token."""
    import os, sys
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    kaggle_dir = Path.home() / ".kaggle"
    token_path = kaggle_dir / "access_token"
    json_path  = kaggle_dir / "kaggle.json"

    def read_token(p):
        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
            try:
                t = p.read_bytes().decode(enc).strip().strip('"').strip("'")
                if t:
                    return t
            except Exception:
                pass
        return "".join(chr(b) for b in p.read_bytes() if 32 <= b < 128).strip()

    if token_path.exists():
        token = read_token(token_path)
        os.environ["KAGGLE_KEY"]      = token
        os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "user")
        if not json_path.exists():
            json_path.write_text(
                json.dumps({"username": "user", "key": token}),
                encoding="utf-8"
            )

    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def _kaggle_find_target(df, hint):
    if hint and hint in df.columns:
        return hint
    for col in reversed(df.columns.tolist()):
        if col.lower() in ("class", "label", "target", "y", "type",
                           "species", "variety", "category"):
            return col
    return df.columns[-1]


def fetch_kaggle_all(output_dir, cache_dir, done_names):
    try:
        api = _kaggle_auth()
        print("  Kaggle: аутентификация OK")
    except Exception as e:
        print(f"  Kaggle: ошибка аутентификации — {e}")
        return []

    results = []
    total = len(KAGGLE_MULTICLASS)

    for i, (ds_id, target_hint, desc, ncls_approx) in enumerate(KAGGLE_MULTICLASS, 1):
        name = f"kaggle_{ds_id.replace('/', '_')[:45]}"
        if name in done_names:
            print(f"  [{i:>2}/{total}] {desc:<35} ПРОПУСК")
            continue

        print(f"  [{i:>2}/{total}] {desc:<35} cls≈{ncls_approx}", end=" ", flush=True)

        cache_path = cache_dir / "kaggle" / ds_id.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)

        csv_files = list(cache_path.glob("**/*.csv"))
        if not csv_files:
            try:
                api.dataset_download_files(
                    ds_id, path=str(cache_path), unzip=True, quiet=True
                )
                csv_files = list(cache_path.glob("**/*.csv"))
            except Exception as e:
                err = str(e)
                if "403" in err:
                    print(f"ERR 403 Forbidden (прими правила на kaggle.com/datasets/{ds_id})")
                else:
                    print(f"ERR {err[:60]}")
                continue

        if not csv_files:
            print("ERR нет CSV")
            continue

        csv_file = max(csv_files, key=lambda f: f.stat().st_size)
        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception as e:
            print(f"ERR read: {e}")
            continue

        target_col = _kaggle_find_target(df, target_hint)
        y_raw = df[target_col]

        # Биннинг если целевая непрерывная или слишком много классов
        if y_raw.dtype in (float, "float64") or y_raw.nunique() > MAX_CLASSES:
            try:
                df[target_col] = pd.qcut(y_raw, q=5, labels=False, duplicates="drop")
            except Exception:
                pass
            y_raw = df[target_col]

        X_df = _clean_X(df.drop(columns=[target_col]))
        if X_df.shape[1] < MIN_FEATURES or X_df.shape[1] > MAX_FEATURES:
            print(f"SKIP f={X_df.shape[1]}")
            continue

        X = X_df.values.astype(np.float64)
        y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int32)

        meta = _finalize(X, y, name=name, source="kaggle",
                         source_id=ds_id, original_name=ds_id,
                         output_dir=output_dir)
        if meta:
            results.append(meta)
            print(f"OK  cls={meta['n_classes']}  n={meta['n_samples_total']}  "
                  f"IR={meta['imbalance_ratio']:.1f}")
        else:
            print(f"SKIP cls={len(np.unique(y))} не в [{MIN_CLASSES},{MAX_CLASSES}]")

        time.sleep(0.3)
    return results


# ── Главная функция ────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description=f"Сборщик реальных многоклассовых датасетов ({MIN_CLASSES}–{MAX_CLASSES} классов)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--output",  default="datasets",
                    help="Папка для сохранения (default: datasets)")
    pa.add_argument("--cache",   default=".dataset_cache")
    pa.add_argument("--sources", default="pmlb,openml",
                    help="pmlb, openml (default: оба)")
    pa.add_argument("--resume",  action="store_true",
                    help="Пропустить уже скачанные")
    pa.add_argument("--dry-run", action="store_true",
                    help="Показать список без скачивания")
    args = pa.parse_args()

    sources   = [s.strip().lower() for s in args.sources.split(",")]
    output_dir = Path(args.output)
    cache_dir  = Path(args.cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "all_meta.json"
    all_meta: list[dict] = []
    done_names: set[str] = set()

    if args.resume and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            all_meta = json.load(f)
        done_names = {m["name"] for m in all_meta}
        print(f"Resume: уже скачано {len(done_names)}")

    print(f"\n{'='*60}")
    print(f"  Источники:     {sources}")
    print(f"  Классов:       {MIN_CLASSES}–{MAX_CLASSES}  (ТОЛЬКО многоклассовые)")
    print(f"  Объектов:      {MIN_SAMPLES}–{MAX_SAMPLES:,}")
    print(f"  Признаков:     {MIN_FEATURES}–{MAX_FEATURES}")
    print(f"  Папка:         {args.output}")
    print(f"{'='*60}\n")

    if args.dry_run:
        if "pmlb" in sources:
            print(f"PMLB ({len(PMLB_MULTICLASS)} датасетов):")
            for name, ncls in PMLB_MULTICLASS:
                print(f"  {name:<35} {ncls} классов")
        if "openml" in sources:
            print(f"\nOpenML: запрос при реальном запуске (300+ многоклассовых)")
        return

    new_results = []

    if "pmlb" in sources:
        print("=" * 40)
        print("PMLB — многоклассовые датасеты")
        print("=" * 40)
        new_results += fetch_pmlb_all(output_dir, cache_dir, done_names)

    if "openml" in sources:
        print("\n" + "=" * 40)
        print("OpenML — многоклассовые датасеты")
        print("=" * 40)
        new_results += fetch_openml_all(output_dir, done_names)

    all_meta += new_results

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    # Итог
    mc = [m for m in all_meta if MIN_CLASSES <= m.get("n_classes", 0) <= MAX_CLASSES]
    cls_dist = {}
    for m in mc:
        k = m["n_classes"]
        cls_dist[k] = cls_dist.get(k, 0) + 1

    print(f"\n{'='*60}")
    print(f"Готово!")
    print(f"  Новых скачано:         {len(new_results)}")
    print(f"  Всего многоклассовых:  {len(mc)}")
    print(f"\n  Распределение по числу классов:")
    for k in sorted(cls_dist):
        bar = "█" * cls_dist[k]
        print(f"    {k:>2} классов: {cls_dist[k]:>4}  {bar}")

    if mc:
        irs = [m["imbalance_ratio"] for m in mc]
        srcs = {}
        for m in mc:
            s = m["source"]
            srcs[s] = srcs.get(s, 0) + 1
        print(f"\n  По источникам: {srcs}")
        print(f"  IR: min={min(irs):.1f}  "
              f"median={sorted(irs)[len(irs)//2]:.1f}  "
              f"max={max(irs):.1f}")
    print(f"\n  Файлы: {output_dir}/")


if __name__ == "__main__":
    main()