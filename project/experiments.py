from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from imblearn.metrics import geometric_mean_score

import sampling
import metrics


# =========================================================
# Models
# =========================================================

def build_model(model_name: str, random_state: int = 42):
    """
    Создаёт модель по имени.

    Supported:
    - LogisticRegression
    - RandomForest
    """
    name = model_name.lower()

    if name in {"logisticregression", "logistic_regression", "lr"}:
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state
        )

    if name in {"randomforest", "random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        )

    raise ValueError(f"Неизвестная модель: {model_name}")


def get_default_model_names() -> list[str]:
    return ["LogisticRegression", "RandomForest"]


def model_needs_scaling(model_name: str) -> bool:
    """
    Масштабирование нужно для LogisticRegression.
    Для RandomForest обычно не нужно.
    """
    return model_name.lower() in {"logisticregression", "logistic_regression", "lr"}


# =========================================================
# Model metrics
# =========================================================

def compute_model_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Считает метрики качества модели для multiclass classification.
    """
    result = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "gmean_macro": geometric_mean_score(y_true, y_pred, average="macro"),
        "auc_roc_ovr_macro": np.nan,
    }

    if y_proba is not None:
        try:
            result["auc_roc_ovr_macro"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro"
            )
        except Exception:
            result["auc_roc_ovr_macro"] = np.nan

    return result


# =========================================================
# Data complexity metrics
# =========================================================

def compute_data_metrics(X, y) -> Dict[str, float]:
    """
    Считает характеристики сложности данных.
    """
    result = metrics.data_complexity_summary(X, y)

    return {
        "ir": result["ir"],
        "n3": result["n3"],
        "f1_fisher_mean": result["f1_fisher_mean"],
        "f1_fisher_max": result["f1_fisher_max"],
        "f1_fisher_min": result["f1_fisher_min"],
        "f2_overlap_mean": result["f2_overlap_mean"],
        "f2_overlap_max": result["f2_overlap_max"],
        "f2_overlap_min": result["f2_overlap_min"],
    }


# =========================================================
# Single experiment
# =========================================================

def run_single_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    sampler_name: str,
    model_name: str,
    random_state: int = 42,
    sampler_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Запускает один эксперимент:
    1) считает сложность исходных данных
    2) делает resampling
    3) считает сложность новых данных
    4) обучает модель
    5) считает метрики качества
    """
    sampler_params = sampler_params or {}

    # ----------------------------
    # Data metrics BEFORE resampling
    # ----------------------------
    before_metrics = compute_data_metrics(X_train, y_train)

    # ----------------------------
    # Resampling
    # ----------------------------
    X_res, y_res = sampling.apply_sampler(
        X=X_train,
        y=y_train,
        name=sampler_name,
        random_state=random_state,
        verbose=verbose,
        **sampler_params,
    )

    # ----------------------------
    # Data metrics AFTER resampling
    # ----------------------------
    after_metrics = compute_data_metrics(X_res, y_res)

    # ----------------------------
    # Build model
    # ----------------------------
    model = build_model(model_name, random_state=random_state)

    if model_needs_scaling(model_name):
        pipeline = SklearnPipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    else:
        pipeline = SklearnPipeline([
            ("model", model),
        ])

    # ----------------------------
    # Train
    # ----------------------------
    pipeline.fit(X_res, y_res)

    # ----------------------------
    # Predict
    # ----------------------------
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    # ----------------------------
    # Model metrics
    # ----------------------------
    model_metrics = compute_model_metrics(y_test, y_pred, y_proba)

    # ----------------------------
    # Result row
    # ----------------------------
    result = {
        "sampler": sampler_name,
        "model": model_name,

        "n_train_before": len(y_train),
        "n_train_after": len(y_res),

        "ir_before": before_metrics["ir"],
        "ir_after": after_metrics["ir"],

        "n3_before": before_metrics["n3"],
        "n3_after": after_metrics["n3"],

        "f1_fisher_mean_before": before_metrics["f1_fisher_mean"],
        "f1_fisher_mean_after": after_metrics["f1_fisher_mean"],

        "f1_fisher_max_before": before_metrics["f1_fisher_max"],
        "f1_fisher_max_after": after_metrics["f1_fisher_max"],

        "f1_fisher_min_before": before_metrics["f1_fisher_min"],
        "f1_fisher_min_after": after_metrics["f1_fisher_min"],

        "f2_overlap_mean_before": before_metrics["f2_overlap_mean"],
        "f2_overlap_mean_after": after_metrics["f2_overlap_mean"],

        "f2_overlap_max_before": before_metrics["f2_overlap_max"],
        "f2_overlap_max_after": after_metrics["f2_overlap_max"],

        "f2_overlap_min_before": before_metrics["f2_overlap_min"],
        "f2_overlap_min_after": after_metrics["f2_overlap_min"],
    }

    result.update(model_metrics)

    return result


# =========================================================
# Multiple experiments
# =========================================================

def run_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    sampler_names: Optional[Iterable[str]] = None,
    model_names: Optional[Iterable[str]] = None,
    sampler_param_grid: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = 42,
    verbose: bool = False,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """
    Запускает серию экспериментов по всем sampler/model combinations.

    Parameters
    ----------
    sampler_names : iterable[str], optional
        Если None, берётся sampling.get_default_sampler_names()
    model_names : iterable[str], optional
        Если None, берётся get_default_model_names()
    sampler_param_grid : dict, optional
        Словарь вида:
        {
            "SMOTE": {"k_neighbors": 5},
            "ADASYN": {"n_neighbors": 3},
            "DBSMOTE": {},
        }
    continue_on_error : bool
        Если True, ошибки не ломают весь запуск, а записываются в таблицу.

    Returns
    -------
    pd.DataFrame
    """
    if sampler_names is None:
        sampler_names = sampling.get_default_sampler_names()

    if model_names is None:
        model_names = get_default_model_names()

    sampler_param_grid = sampler_param_grid or {}

    results: List[Dict[str, Any]] = []

    for sampler_name in sampler_names:
        for model_name in model_names:
            params = sampler_param_grid.get(sampler_name, {})

            try:
                row = run_single_experiment(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    sampler_name=sampler_name,
                    model_name=model_name,
                    random_state=random_state,
                    sampler_params=params,
                    verbose=verbose,
                )
                row["status"] = "ok"
                row["error"] = None

            except Exception as e:
                if not continue_on_error:
                    raise

                row = {
                    "sampler": sampler_name,
                    "model": model_name,

                    "n_train_before": np.nan,
                    "n_train_after": np.nan,

                    "ir_before": np.nan,
                    "ir_after": np.nan,

                    "n3_before": np.nan,
                    "n3_after": np.nan,

                    "f1_fisher_mean_before": np.nan,
                    "f1_fisher_mean_after": np.nan,
                    "f1_fisher_max_before": np.nan,
                    "f1_fisher_max_after": np.nan,
                    "f1_fisher_min_before": np.nan,
                    "f1_fisher_min_after": np.nan,

                    "f2_overlap_mean_before": np.nan,
                    "f2_overlap_mean_after": np.nan,
                    "f2_overlap_max_before": np.nan,
                    "f2_overlap_max_after": np.nan,
                    "f2_overlap_min_before": np.nan,
                    "f2_overlap_min_after": np.nan,

                    "balanced_accuracy": np.nan,
                    "f1_macro": np.nan,
                    "gmean_macro": np.nan,
                    "auc_roc_ovr_macro": np.nan,

                    "status": "error",
                    "error": str(e),
                }

            results.append(row)

    return pd.DataFrame(results)



# =========================================================
# Ready-to-use default experiment
# =========================================================

def run_default_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Запускает стандартный набор sampler'ов и моделей.
    """
    sampler_names = sampling.get_default_sampler_names()
    model_names = get_default_model_names()

    sampler_param_grid = {
        "SMOTE": {"k_neighbors": 5},
        "BorderlineSMOTE": {"k_neighbors": 5, "m_neighbors": 10, "kind": "borderline-1"},
        "SVMSMOTE": {"k_neighbors": 5, "m_neighbors": 10},
        "ADASYN": {"n_neighbors": 5},
        "KMeansSMOTE": {"k_neighbors": 2},
        "DBSMOTE": {},
        "MWMOTE": {},
    }

    return run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sampler_names=sampler_names,
        model_names=model_names,
        sampler_param_grid=sampler_param_grid,
        random_state=random_state,
        verbose=verbose,
        continue_on_error=True,
    )


# =========================================================
# Helpers
# =========================================================

def make_pivot_table(
    results_df: pd.DataFrame,
    metric: str = "f1_macro"
) -> pd.DataFrame:
    """
    Делает pivot-таблицу:
    строки = sampler
    столбцы = model
    значения = выбранная метрика
    """
    return results_df.pivot(
        index="sampler",
        columns="model",
        values=metric
    )


def sort_results(
    results_df: pd.DataFrame,
    by: str = "f1_macro",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Сортировка итоговой таблицы.
    """
    return results_df.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def keep_only_successful(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляет только успешно выполненные эксперименты.
    """
    return results_df[results_df["status"] == "ok"].reset_index(drop=True)


def split_results_tables(results_df):
    data_table = results_df[
        [
            "sampler",
            "n_train_before",
            "n_train_after",
            "ir_before",
            "ir_after",
            "n3_before",
            "n3_after",
            "f1_fisher_mean_before",
            "f1_fisher_mean_after",
            "f2_overlap_mean_before",
            "f2_overlap_mean_after",
            "status"
        ]
    ].drop_duplicates(subset=["sampler"]).reset_index(drop=True)

    model_table = results_df[
        [
            "sampler",
            "model",
            "balanced_accuracy",
            "f1_macro",
            "gmean_macro",
            "auc_roc_ovr_macro",
            "status"
        ]
    ].reset_index(drop=True)

    return data_table.round(3), model_table.round(3)


def run_baseline_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    model_names=None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Запускает модели БЕЗ sampler'ов.
    """
    if model_names is None:
        model_names = get_default_model_names()

    results = []

    # характеристики данных до/после одинаковые
    data_metrics = compute_data_metrics(X_train, y_train)

    for model_name in model_names:
        model = build_model(model_name, random_state=random_state)

        if model_needs_scaling(model_name):
            pipeline = SklearnPipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
        else:
            pipeline = SklearnPipeline([
                ("model", model),
            ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception:
                y_proba = None

        model_metrics = compute_model_metrics(y_test, y_pred, y_proba)

        row = {
            "sampler": "NoSampler",
            "model": model_name,

            "n_train_before": len(y_train),
            "n_train_after": len(y_train),

            "ir_before": data_metrics["ir"],
            "ir_after": data_metrics["ir"],

            "n3_before": data_metrics["n3"],
            "n3_after": data_metrics["n3"],

            "f1_fisher_mean_before": data_metrics["f1_fisher_mean"],
            "f1_fisher_mean_after": data_metrics["f1_fisher_mean"],

            "f1_fisher_max_before": data_metrics["f1_fisher_max"],
            "f1_fisher_max_after": data_metrics["f1_fisher_max"],

            "f1_fisher_min_before": data_metrics["f1_fisher_min"],
            "f1_fisher_min_after": data_metrics["f1_fisher_min"],

            "f2_overlap_mean_before": data_metrics["f2_overlap_mean"],
            "f2_overlap_mean_after": data_metrics["f2_overlap_mean"],

            "f2_overlap_max_before": data_metrics["f2_overlap_max"],
            "f2_overlap_max_after": data_metrics["f2_overlap_max"],

            "f2_overlap_min_before": data_metrics["f2_overlap_min"],
            "f2_overlap_min_after": data_metrics["f2_overlap_min"],

            "status": "ok",
            "error": None,
        }

        row.update(model_metrics)
        results.append(row)

    return pd.DataFrame(results)