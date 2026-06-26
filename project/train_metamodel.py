"""
Метамодель: мета-признаки датасета → лучший сэмплер.

Задача: многоклассовая классификация.
Целевая переменная: best_method — сэмплер с лучшим средним рангом
по метрикам balanced_accuracy, f1_macro, g_mean.

Модели (сравниваются через CV, лучшая обучается финально):
  - RandomForest
  - LightGBM
  - XGBoost
  - LogisticRegression
  - KNN

Подбор гиперпараметров: Optuna (TPE sampler) для лучшей модели.

Выходные файлы (в --output папке):
  cv_report.csv          — CV-метрики всех моделей до тюнинга
  best_model.pkl         — финальная модель + препроцессор
  optuna_study.pkl       — объект study для анализа
  predictions.csv        — предсказания с вероятностями
  importance.csv         — важность мета-признаков
  plots/
    cv_comparison.png    — сравнение моделей
    confusion_matrix.png
    feature_importance.png
    optuna_history.png   — история оптимизации Optuna
    optuna_params.png    — важность гиперпараметров

Запуск:
  python train_metamodel.py
  python train_metamodel.py --metafeatures mf.csv --results results_full.csv
  python train_metamodel.py --trials 100 --cv 10
  python train_metamodel.py --no-plots
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

METRICS_BENCH = ["balanced_accuracy", "f1_macro", "g_mean"]

# ── Визуальная тема ────────────────────────────────────────────────────────
BG      = "#0d0d0f"
SURFACE = "#141418"
BORDER  = "#2a2a35"
TEXT    = "#e8e8f0"
MUTED   = "#6b6b80"
ACCENT  = "#7c6aff"
ACCENT2 = "#ff6a9b"
ACCENT3 = "#6affd4"
ACCENT4 = "#fbbf24"
COLORS  = [ACCENT, ACCENT2, ACCENT3, ACCENT4,
           "#60a5fa", "#f472b6", "#34d399", "#fb923c"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#1e1e28",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
})


# ══════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════

def compute_targets(results_path: Path) -> pd.DataFrame:
    """
    Из results_full.csv вычисляет best_method для каждого датасета:
    avg_rank по трём метрикам → сэмплер с минимальным рангом.
    """
    df = pd.read_csv(results_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    ds_col  = next(c for c in ["dataset", "dataset_name"] if c in df.columns)
    smp_col = "sampler"
    avail   = [m for m in METRICS_BENCH if m in df.columns]

    if not avail:
        raise ValueError(f"Не найдено метрик {METRICS_BENCH}")

    print(f"  Метрики для ранжирования: {avail}")

    for m in avail:
        df[f"rank_{m}"] = df.groupby(ds_col)[m].rank(
            ascending=False, method="average"
        )
    df["avg_rank"]  = df[[f"rank_{m}" for m in avail]].mean(axis=1)
    df["avg_score"] = df[avail].mean(axis=1)

    rows = []
    for ds, grp in df.groupby(ds_col):
        base = grp[grp[smp_col] == "Baseline"]["avg_score"]
        baseline_score = float(base.iloc[0]) if not base.empty else np.nan

        non_base = grp[grp[smp_col] != "Baseline"]
        if non_base.empty:
            continue
        best_idx  = non_base["avg_rank"].idxmin()
        best      = non_base.loc[best_idx]
        rows.append({
            "dataset":        ds,
            "best_method":    best[smp_col],
            "gain":           float(best["avg_score"] - baseline_score)
                              if not np.isnan(baseline_score) else np.nan,
            "baseline_score": baseline_score,
            "best_score":     float(best["avg_score"]),
        })

    targets = pd.DataFrame(rows)
    counts = targets["best_method"].value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        print(f"Редкие сэмплеры ({len(rare)}): {list(rare)}")
        targets = targets[~targets["best_method"].isin(rare)].copy()
        print(f"Осталось датасетов: {len(targets)}")
        if targets.empty:
            raise ValueError("Нет данных после удаления редких классов")

    print(f"  Датасетов с целевой переменной: {len(targets)}")
    print(f"\n  Распределение best_method:")
    for m, c in targets["best_method"].value_counts().items():
        bar = "█" * c
        print(f"    {m:<26} {c:>4}  {bar}")

    return targets


def load_and_merge(mf_path: Path, targets: pd.DataFrame):
    """
    Объединяет metafeatures.csv с целевыми переменными.
    Очищает данные и масштабирует.
    """
    mf = pd.read_csv(mf_path)
    ds_col = next(
        (c for c in ["dataset_name", "name", "dataset"] if c in mf.columns),
        mf.columns[0]
    )
    mf[ds_col] = mf[ds_col].astype(str).str.strip()
    targets["dataset"] = targets["dataset"].astype(str).str.strip()

    df = mf.merge(targets, left_on=ds_col, right_on="dataset", how="inner")
    print(f"\n  После объединения: {len(df)} датасетов "
          f"(потеряно {len(mf) - len(df)} из metafeatures)")

    if len(df) == 0:
        raise ValueError(
            "После объединения 0 строк.\n"
            f"  mf names (5): {mf[ds_col].head().tolist()}\n"
            f"  targets (5):  {targets['dataset'].head().tolist()}"
        )

    # Отбираем мета-признаки
    EXCL_PFXS = ("gen_", "__", "dataset_")
    EXCL_EXACT = {
        ds_col, "dataset", "group", "base_type", "name",
        "random_state", "noise_type", "spatial_distortion",
        "n_samples_total", "n_samples_train", "n_samples_test",
        "target_weights", "actual_weights", "class_counts_train",
        "circles_factor", "cluster_std", "n_informative", "lhs_index",
        "source", "source_id", "original_name", "ir_zone",
        "best_method", "gain", "baseline_score", "best_score",
    }
    mf_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in EXCL_PFXS)
        and c not in EXCL_EXACT
        and df[c].dtype != object
    ]

    X_raw = df[mf_cols].copy()

    # Очистка
    n_inf = np.isinf(X_raw.values).sum()
    if n_inf:
        print(f"  inf → NaN: {n_inf}")
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    drop_nan = X_raw.columns[X_raw.isna().mean() > 0.50].tolist()
    if drop_nan:
        print(f"  Дроп >50% NaN: {len(drop_nan)} колонок")
        X_raw = X_raw.drop(columns=drop_nan)
        mf_cols = [c for c in mf_cols if c not in drop_nan]

    X_raw = X_raw.fillna(X_raw.median())

    q1, q3 = X_raw.quantile(0.25), X_raw.quantile(0.75)
    X_raw  = X_raw.clip(lower=q1 - 10*(q3-q1), upper=q3 + 10*(q3-q1), axis=1)

    zero_var = X_raw.columns[X_raw.std() < 1e-10].tolist()
    if zero_var:
        X_raw = X_raw.drop(columns=zero_var)
        mf_cols = [c for c in mf_cols if c not in zero_var]

    print(f"  Мета-признаков: {len(mf_cols)}")

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw).astype(np.float64)

    le = LabelEncoder()
    y  = le.fit_transform(df["best_method"])

    print(f"  X shape: {X.shape}")
    print(f"  Классов: {len(le.classes_)}: {list(le.classes_)}")

    return df, X, y, mf_cols, le, scaler


# ══════════════════════════════════════════════════════════════════════════
# 2. БАЗОВЫЕ МОДЕЛИ И CV
# ══════════════════════════════════════════════════════════════════════════

def get_base_models(n_cls: int, seed: int = 42) -> dict:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, random_state=seed,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=min(7, n_cls + 2),
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=seed, verbose=-1,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=seed, eval_metric="mlogloss", verbosity=0,
        )
    except ImportError:
        pass
    return models


def run_cv(models: dict, X, y, n_splits: int, seed: int) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = ["accuracy", "balanced_accuracy", "f1_macro"]
    rows = []

    print(f"\nБазовая CV ({n_splits} фолдов):")
    for name, model in models.items():
        print(f"  {name:<22}", end=" ", flush=True)
        try:
            sc = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            row = {
                "model":                    name,
                "accuracy_mean":            sc["test_accuracy"].mean(),
                "accuracy_std":             sc["test_accuracy"].std(),
                "balanced_accuracy_mean":   sc["test_balanced_accuracy"].mean(),
                "balanced_accuracy_std":    sc["test_balanced_accuracy"].std(),
                "f1_macro_mean":            sc["test_f1_macro"].mean(),
                "f1_macro_std":             sc["test_f1_macro"].std(),
            }
            print(f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
                  f"f1={row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}  "
                  f"ba={row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f}")
        except Exception as e:
            print(f"ERR: {e}")
            row = {"model": name, "f1_macro_mean": -1.0,
                   "accuracy_mean": -1.0, "balanced_accuracy_mean": -1.0}
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
# 3. OPTUNA — ПОДБОР ГИПЕРПАРАМЕТРОВ
# ══════════════════════════════════════════════════════════════════════════

def build_objective(model_name: str, X, y, cv, seed: int):
    """
    Возвращает objective-функцию для Optuna.
    Каждая модель имеет свё пространство гиперпараметров.
    """
    def objective(trial: optuna.Trial) -> float:

        if model_name == "RandomForest":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
                "max_depth":         trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":      trial.suggest_categorical(
                                         "max_features", ["sqrt", "log2", None]),
                "class_weight":      "balanced",
                "random_state":      seed,
                "n_jobs":            1,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "LightGBM":
            from lightgbm import LGBMClassifier
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
                "max_depth":       trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "class_weight":    "balanced",
                "random_state":    seed,
                "verbose":        -1,
            }
            model = LGBMClassifier(**params)

        elif model_name == "XGBoost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":         trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "eval_metric":   "mlogloss",
                "verbosity":     0,
                "random_state":  seed,
            }
            model = XGBClassifier(**params)

        elif model_name == "LogisticRegression":
            params = {
                "C":            trial.suggest_float("C", 1e-4, 100.0, log=True),
                "max_iter":     trial.suggest_int("max_iter", 500, 3000),
                "solver":       trial.suggest_categorical(
                                    "solver", ["lbfgs", "saga"]),
                "class_weight": "balanced",
                "random_state": seed,
            }
            model = LogisticRegression(**params)

        elif model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric":      trial.suggest_categorical(
                                   "metric", ["euclidean", "manhattan", "minkowski"]),
            }
            model = KNeighborsClassifier(**params)

        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        scores = cross_val_score(
            model, X, y, cv=cv, scoring="f1_macro", n_jobs=1
        )
        return float(scores.mean())

    return objective


def tune_model(
    model_name: str,
    X, y,
    cv,
    n_trials: int,
    seed: int,
    n_jobs_optuna: int = 1,
) -> optuna.Study:
    """Запускает Optuna для поиска лучших гиперпараметров."""
    sampler   = TPESampler(seed=seed)
    study     = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"metamodel_{model_name}",
    )
    objective = build_objective(model_name, X, y, cv, seed)

    print(f"\nOptuna: тюнинг {model_name} ({n_trials} trials)...")
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs_optuna,
        show_progress_bar=True,
    )

    print(f"  Лучший F1-macro: {study.best_value:.4f}")
    print(f"  Лучшие параметры:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study


def build_tuned_model(model_name: str, best_params: dict, seed: int):
    """Создаёт модель с лучшими найденными гиперпараметрами."""
    if model_name == "RandomForest":
        return RandomForestClassifier(**best_params, random_state=seed, n_jobs=-1)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**best_params, random_state=seed, verbose=-1)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**best_params, random_state=seed, verbosity=0)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**best_params, random_state=seed)
    elif model_name == "KNN":
        return KNeighborsClassifier(**best_params)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


# ══════════════════════════════════════════════════════════════════════════
# 4. ВАЖНОСТЬ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, mf_cols: list) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature":    mf_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 \
              else np.abs(model.coef_)
        return pd.DataFrame({
            "feature":    mf_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)
    return None


# ══════════════════════════════════════════════════════════════════════════
# 5. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════

def plot_cv_comparison(cv_df: pd.DataFrame, out: Path):
    valid = cv_df[cv_df["f1_macro_mean"] >= 0].copy()
    if valid.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    metrics = [
        ("f1_macro_mean",          "f1_macro_std",          "F1-macro"),
        ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy"),
        ("accuracy_mean",          "accuracy_std",          "Accuracy"),
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        if mean_col not in valid.columns:
            ax.set_visible(False)
            continue
        vals   = valid[mean_col].values
        stds   = valid.get(std_col, pd.Series([0]*len(valid))).values
        models = valid["model"].values
        cols   = [ACCENT if i == 0 else "#3a3a50" for i in range(len(models))]

        bars = ax.barh(models, vals, xerr=stds, color=cols,
                       capsize=4, height=0.5, edgecolor=BORDER)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9, color=TEXT)

        ax.set_xlabel(title, fontsize=10)
        ax.set_title(f"{title}\n(Stratified CV)", fontsize=11, color=TEXT)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Сравнение метамоделей (до тюнинга)", fontsize=13,
                 color=TEXT, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


def plot_confusion(y_true, y_pred, le, out: Path, title: str):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(le.classes_)

    fig, ax = plt.subplots(figsize=(max(8, n*0.9), max(7, n*0.8)),
                           facecolor=BG)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Purples",
        xticklabels=le.classes_, yticklabels=le.classes_,
        ax=ax, linewidths=0.5, vmin=0, vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {title}\n(нормировано по строкам)",
                 fontsize=12, color=TEXT)
    ax.set_xlabel("Предсказано", color=TEXT)
    ax.set_ylabel("Истина", color=TEXT)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


def plot_importance(imp_df: pd.DataFrame, out: Path, title: str):
    if imp_df is None or imp_df.empty:
        return
    top = imp_df.head(min(25, len(imp_df)))
    fig, ax = plt.subplots(figsize=(11, max(5, len(top)*0.38)), facecolor=BG)

    # Цветовой градиент по важности
    norm_imp = (top["importance"] - top["importance"].min()) / \
               (top["importance"].max() - top["importance"].min() + 1e-9)
    cols = plt.cm.RdYlGn(0.3 + 0.6 * norm_imp.values[::-1])

    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color=cols, edgecolor=BORDER, height=0.6)
    ax.set_title(title, fontsize=12, color=TEXT)
    ax.set_xlabel("Важность признака", color=TEXT)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


def plot_optuna_history(study: optuna.Study, out_history: Path, out_params: Path):
    """Два графика Optuna: история оптимизации и важность гиперпараметров."""

    # ── История ───────────────────────────────────────────────────────
    trials_df = study.trials_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)

    # Значение каждого trial
    ax = axes[0]
    vals = [t.value for t in study.trials if t.value is not None]
    best = [max(vals[:i+1]) for i in range(len(vals))]
    ax.scatter(range(len(vals)), vals, color=ACCENT, s=20, alpha=0.5,
               label="Trial F1-macro")
    ax.plot(range(len(best)), best, color=ACCENT2, lw=2,
            label=f"Лучшее: {max(best):.4f}")
    ax.set_title("История оптимизации Optuna", fontsize=12, color=TEXT)
    ax.set_xlabel("Номер trial")
    ax.set_ylabel("F1-macro (CV)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Распределение лучших значений
    ax = axes[1]
    top_vals = sorted(vals, reverse=True)[:max(10, len(vals)//5)]
    ax.hist(vals, bins=20, color=ACCENT, alpha=0.7, edgecolor=BORDER,
            label="Все trials")
    ax.axvline(study.best_value, color=ACCENT2, lw=2, ls="--",
               label=f"Лучшее: {study.best_value:.4f}")
    ax.set_title("Распределение F1-macro по trials", fontsize=12, color=TEXT)
    ax.set_xlabel("F1-macro")
    ax.set_ylabel("Число trials")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Optuna — {study.study_name}", fontsize=13,
                 color=TEXT, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_history, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out_history}")

    # ── Важность гиперпараметров ──────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            params = list(importances.keys())[:15]
            imps   = [importances[p] for p in params]

            fig, ax = plt.subplots(figsize=(10, max(4, len(params)*0.4)),
                                   facecolor=BG)
            ax.barh(params[::-1], imps[::-1], color=ACCENT3,
                    edgecolor=BORDER, height=0.6)
            ax.set_title("Важность гиперпараметров (Optuna FAnova)",
                         fontsize=12, color=TEXT)
            ax.set_xlabel("Важность")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            fig.savefig(out_params, dpi=150, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
            print(f"  Сохранён: {out_params}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="Метамодель с подбором гиперпараметров через Optuna",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--metafeatures", default="metafeatures.csv",
                    help="CSV с мета-признаками (default: metafeatures.csv)")
    pa.add_argument("--results",      default="results_full.csv",
                    help="CSV с результатами бенчмарка (default: results_full.csv)")
    pa.add_argument("--output",       default="metamodel_output",
                    help="Папка для результатов (default: metamodel_output)")
    pa.add_argument("--cv",           type=int, default=5,
                    help="Число фолдов CV (default: 5)")
    pa.add_argument("--trials",       type=int, default=50,
                    help="Число trials Optuna (default: 50)")
    pa.add_argument("--seed",         type=int, default=42)
    pa.add_argument("--no-plots",     action="store_true")
    pa.add_argument("--tune-all",     action="store_true",
                    help="Тюнить все модели, а не только лучшую")
    args = pa.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    plots = out / "plots"
    if not args.no_plots:
        plots.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Мета-признаки:  {args.metafeatures}")
    print(f"  Бенчмарк:       {args.results}")
    print(f"  CV:             {args.cv} фолдов")
    print(f"  Optuna trials:  {args.trials}")
    print(f"  Seed:           {args.seed}")
    print(f"{'='*60}\n")

    # ── 1. Данные ─────────────────────────────────────────────────────
    print("Загружаю целевые переменные...")
    targets = compute_targets(Path(args.results))

    print("\nЗагружаю мета-признаки...")
    df, X, y, mf_cols, le, scaler = load_and_merge(
        Path(args.metafeatures), targets
    )

    n_cls   = len(le.classes_)
    n_samp  = len(y)
    min_cnt = pd.Series(y).value_counts().min()
    n_splits = min(args.cv, min_cnt)
    if n_splits < args.cv:
        print(f"\n  ⚠️  CV уменьшен до {n_splits} (мин. класс: {min_cnt})")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # ── 2. Базовая CV ─────────────────────────────────────────────────
    models = get_base_models(n_cls, args.seed)
    cv_report = run_cv(models, X, y, n_splits, args.seed)
    cv_report.to_csv(out / "cv_report.csv", index=False)

    best_model_name = cv_report.iloc[0]["model"]
    best_model_f1   = cv_report.iloc[0]["f1_macro_mean"]
    print(f"\n  Лучшая модель до тюнинга: {best_model_name} "
          f"(F1={best_model_f1:.4f})")

    if not args.no_plots:
        plot_cv_comparison(cv_report, plots / "cv_comparison.png")

    # ── 3. Optuna тюнинг ──────────────────────────────────────────────
    models_to_tune = (
        [m for m in cv_report["model"] if cv_report.loc[
            cv_report["model"]==m, "f1_macro_mean"].values[0] >= 0]
        if args.tune_all
        else [best_model_name]
    )

    studies = {}
    tuned_scores = {}

    for model_name in models_to_tune:
        if model_name not in models:
            continue
        study = tune_model(
            model_name, X, y, cv,
            n_trials=args.trials,
            seed=args.seed,
        )
        studies[model_name] = study
        tuned_scores[model_name] = study.best_value

    # Выбираем лучшую после тюнинга
    if tuned_scores:
        best_tuned_name  = max(tuned_scores, key=tuned_scores.get)
        best_tuned_score = tuned_scores[best_tuned_name]
    else:
        best_tuned_name  = best_model_name
        best_tuned_score = best_model_f1

    print(f"\n  Лучшая после тюнинга: {best_tuned_name} "
          f"(F1={best_tuned_score:.4f})")

    # Сравнение до/после
    improvement = best_tuned_score - best_model_f1
    print(f"  Улучшение от Optuna: {improvement:+.4f}")

    # ── 4. Финальная модель ───────────────────────────────────────────
    if best_tuned_name in studies:
        best_params = studies[best_tuned_name].best_params
        final_model = build_tuned_model(best_tuned_name, best_params, args.seed)
        print(f"\nОбучаю финальную модель ({best_tuned_name}) на всех данных...")
    else:
        final_model = models[best_tuned_name]
        best_params = {}
        print(f"\nОбучаю {best_tuned_name} на всех данных...")

    final_model.fit(X, y)
    y_pred = final_model.predict(X)

    acc = accuracy_score(y, y_pred)
    ba  = balanced_accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"  Train Accuracy:          {acc:.4f}")
    print(f"  Train Balanced Accuracy: {ba:.4f}")
    print(f"  Train F1-macro:          {f1:.4f}")
    print(f"\n{classification_report(y, y_pred, target_names=le.classes_, zero_division=0)}")

    # ── 5. Сохранение модели ─────────────────────────────────────────
    artifact = {
        "model":         final_model,
        "model_name":    best_tuned_name,
        "best_params":   best_params,
        "scaler":        scaler,
        "label_encoder": le,
        "mf_cols":       mf_cols,
        "cv_f1_before":  best_model_f1,
        "cv_f1_after":   best_tuned_score,
        "optuna_trials": args.trials,
    }
    joblib.dump(artifact, out / "best_model.pkl")

    # Сохраняем study
    if studies:
        joblib.dump(studies, out / "optuna_study.pkl")

    # ── 6. Предсказания ───────────────────────────────────────────────
    ds_col = next((c for c in ["dataset_name","name"] if c in df.columns),
                  df.columns[0])
    pred_df = pd.DataFrame({
        "dataset":            df[ds_col].values,
        "true_best_method":   le.inverse_transform(y),
        "pred_best_method":   le.inverse_transform(y_pred),
        "correct":            y == y_pred,
    })
    if hasattr(final_model, "predict_proba"):
        proba = final_model.predict_proba(X)
        for i, cls in enumerate(le.classes_):
            pred_df[f"proba_{cls}"] = proba[:, i]
    pred_df.to_csv(out / "predictions.csv", index=False)

    # ── 7. Важность признаков ─────────────────────────────────────────
    imp = get_feature_importance(final_model, mf_cols)
    if imp is not None:
        imp.to_csv(out / "importance.csv", index=False)
        print(f"\n  Топ-10 мета-признаков:")
        for _, row in imp.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")

    # ── 8. Графики ────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nСтроим графики...")
        plot_confusion(
            y, y_pred, le,
            plots / "confusion_matrix.png",
            best_tuned_name,
        )
        if imp is not None:
            plot_importance(
                imp,
                plots / "feature_importance.png",
                f"Важность мета-признаков — {best_tuned_name}",
            )
        if best_tuned_name in studies:
            plot_optuna_history(
                studies[best_tuned_name],
                plots / "optuna_history.png",
                plots / "optuna_params.png",
            )

    # ── 9. Итог ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Готово!")
    print(f"  Датасетов:           {n_samp}")
    print(f"  Мета-признаков:      {len(mf_cols)}")
    print(f"  Классов:             {n_cls}: {list(le.classes_)}")
    print(f"\n  Лучшая модель:       {best_tuned_name}")
    print(f"  F1-macro (CV):       {best_model_f1:.4f} → {best_tuned_score:.4f} "
          f"({improvement:+.4f} от Optuna)")
    print(f"  Train accuracy:      {acc:.4f}")
    print(f"\n  Файлы в {out}/:")
    print(f"    best_model.pkl     ← модель + препроцессор")
    print(f"    cv_report.csv      ← CV до тюнинга")
    print(f"    predictions.csv    ← предсказания с вероятностями")
    print(f"    importance.csv     ← важность признаков")
    if studies:
        print(f"    optuna_study.pkl   ← study объект")
    if not args.no_plots:
        print(f"    plots/             ← графики")
    print(f"\n  Инференс на новом датасете:")
    print(f"    import joblib, numpy as np")
    print(f"    art = joblib.load('{out}/best_model.pkl')")
    print(f"    X_new = art['scaler'].transform([mf_vector])")
    print(f"    method = art['label_encoder'].inverse_transform(")
    print(f"               art['model'].predict(X_new))[0]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()