"""
Алгоритм рекомендации сэмплера на основе k ближайших датасетов.

Идея: для нового датасета найти k наиболее похожих из базы знаний
(по мета-признакам) и рекомендовать тот сэмплер который чаще всего
был лучшим у этих соседей.

Почему это лучше метамодели при малом числе датасетов:
  - Нет предположений о форме решающей границы
  - Работает при любом числе классов (сэмплеров)
  - Легко интерпретируется: "датасет похож на X, Y, Z где лучше SMOTE"
  - При k=1 — чистый nearest-neighbor поиск

Оценка качества:
  - Leave-One-Out CV (LOO) — самая честная при малом числе датасетов
  - Accuracy, F1-macro, Top-2 accuracy (рекомендация попадает в топ-2)

Выходные файлы:
  knn_recommender.pkl    — обученный рекомендатор (сохраняет всю базу)
  knn_report.csv         — метрики при разных k
  knn_predictions.csv    — LOO предсказания для каждого датасета
  plots/
    knn_k_selection.png  — как меняется accuracy при разных k
    knn_neighbors.png    — для каждого датасета: кто его ближайшие соседи
    knn_confusion.png    — confusion matrix

Запуск:
  python knn_recommender.py
  python knn_recommender.py --metafeatures mf.csv --results results_full.csv
  python knn_recommender.py --k 5              # фиксированный k
  python knn_recommender.py --kmax 15          # подобрать k от 1 до 15
  python knn_recommender.py --new dataset.npy  # рекомендация для нового датасета
"""

import argparse
import json
import warnings
from pathlib import Path
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

METRICS_BENCH = ["balanced_accuracy", "f1_macro", "g_mean"]

BG      = "#0d0d0f"
SURFACE = "#141418"
BORDER  = "#2a2a35"
TEXT    = "#e8e8f0"
MUTED   = "#6b6b80"
ACCENT  = "#7c6aff"
ACCENT2 = "#ff6a9b"
ACCENT3 = "#6affd4"
ACCENT4 = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#1e1e28",
    "grid.linewidth": 0.5, "font.family": "monospace",
    "legend.facecolor": SURFACE, "legend.edgecolor": BORDER,
})


# ══════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════

def compute_targets(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    ds_col  = next(c for c in ["dataset", "dataset_name"] if c in df.columns)
    smp_col = "sampler"
    avail   = [m for m in METRICS_BENCH if m in df.columns]

    if not avail:
        raise ValueError(f"Нет метрик {METRICS_BENCH}")

    for m in avail:
        df[f"rank_{m}"] = df.groupby(ds_col)[m].rank(
            ascending=False, method="average"
        )
    df["avg_rank"]  = df[[f"rank_{m}" for m in avail]].mean(axis=1)
    df["avg_score"] = df[avail].mean(axis=1)

    rows = []
    for ds, grp in df.groupby(ds_col):
        base = grp[grp[smp_col] == "Baseline"]["avg_score"]
        base_score = float(base.iloc[0]) if not base.empty else np.nan

        if grp.empty:
            continue

        # Все сэмплеры с их рангами (включая Baseline) — для анализа Top-2
        sampler_ranks = grp.set_index(smp_col)["avg_rank"].to_dict()
        best = grp.loc[grp["avg_rank"].idxmin()]

        rows.append({
            "dataset":        ds,
            "best_method":    best[smp_col],
            "gain":           float(best["avg_score"] - base_score)
                              if not np.isnan(base_score) else np.nan,
            "sampler_ranks":  sampler_ranks,   # все ранги — для Top-2 accuracy
        })

    targets = pd.DataFrame(rows)
    print(f"  Датасетов: {len(targets)}")
    print(f"\n  Распределение best_method:")
    for m, c in targets["best_method"].value_counts().items():
        bar = "█" * c
        print(f"    {m:<26} {c:>4}  {bar}")
    return targets


def load_metafeatures(mf_path: Path, targets: pd.DataFrame):
    mf = pd.read_csv(mf_path)
    ds_col = next(
        (c for c in ["dataset_name", "name", "dataset"] if c in mf.columns),
        mf.columns[0]
    )
    mf[ds_col] = mf[ds_col].astype(str).str.strip()
    targets["dataset"] = targets["dataset"].astype(str).str.strip()

    df = mf.merge(targets, left_on=ds_col, right_on="dataset", how="inner")
    print(f"\n  После объединения: {len(df)} датасетов")

    EXCL_PFXS = ("gen_", "__", "dataset_")
    EXCL_EXACT = {
        ds_col, "dataset", "group", "base_type", "name",
        "random_state", "noise_type", "spatial_distortion",
        "n_samples_total", "n_samples_train", "n_samples_test",
        "target_weights", "actual_weights", "class_counts_train",
        "source", "source_id", "original_name", "ir_zone",
        "best_method", "gain", "sampler_ranks",
    }
    mf_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in EXCL_PFXS)
        and c not in EXCL_EXACT
        and df[c].dtype != object
    ]

    X_raw = df[mf_cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    drop_nan = X_raw.columns[X_raw.isna().mean() > 0.50].tolist()
    if drop_nan:
        X_raw = X_raw.drop(columns=drop_nan)
        mf_cols = [c for c in mf_cols if c not in drop_nan]

    X_raw = X_raw.fillna(X_raw.median())
    q1, q3 = X_raw.quantile(0.25), X_raw.quantile(0.75)
    X_raw  = X_raw.clip(lower=q1 - 10*(q3-q1), upper=q3 + 10*(q3-q1), axis=1)
    X_raw  = X_raw.loc[:, X_raw.std() > 1e-10]
    mf_cols = list(X_raw.columns)

    print(f"  Мета-признаков: {len(mf_cols)}")

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw).astype(np.float64)

    le = LabelEncoder()
    y  = le.fit_transform(df["best_method"])

    # Сохраняем sampler_ranks для Top-2 расчёта
    sampler_ranks_list = df["sampler_ranks"].tolist()
    dataset_names = df[ds_col].tolist()

    return df, X, y, mf_cols, le, scaler, sampler_ranks_list, dataset_names


# ══════════════════════════════════════════════════════════════════════════
# 2. kNN РЕКОМЕНДАТОР
# ══════════════════════════════════════════════════════════════════════════

class KNNSamplerRecommender:
    """
    Рекомендатор сэмплеров на основе k ближайших датасетов.

    Для нового датасета:
      1. Находим k ближайших по евклидовому расстоянию в пространстве мета-признаков
      2. Смотрим какие сэмплеры были лучшими у этих соседей
      3. Рекомендуем тот что встречается чаще всего (мажоритарное голосование)
      4. Если ничья — берём сэмплер соседа с наименьшим расстоянием

    Дополнительно: взвешенное голосование — более близкие соседи имеют больший вес.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", weighted: bool = True):
        self.k        = k
        self.metric   = metric
        self.weighted = weighted
        self.nn_      = None
        self.X_train_ = None
        self.y_train_ = None   # закодированные метки
        self.le_      = None
        self.names_   = None   # имена датасетов

    def fit(self, X: np.ndarray, y: np.ndarray,
            le: LabelEncoder, names: list):
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.le_      = le
        self.names_   = names

        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(X)),  # +1 т.к. сам датасет тоже сосед при LOO
            metric=self.metric,
            n_jobs=1,
        )
        self.nn_.fit(X)
        return self

    def predict(self, X_new: np.ndarray,
                exclude_self: bool = False) -> np.ndarray:
        """
        Предсказывает best_method для каждой строки X_new.
        exclude_self=True используется при LOO (исключаем сам датасет).
        """
        distances, indices = self.nn_.kneighbors(X_new)
        predictions = []

        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            # При LOO исключаем самого себя (расстояние ≈ 0)
            if exclude_self:
                mask = dists > 1e-10
                dists = dists[mask]
                idxs  = idxs[mask]

            # Берём k соседей
            k_actual = min(self.k, len(idxs))
            dists = dists[:k_actual]
            idxs  = idxs[:k_actual]

            if len(idxs) == 0:
                predictions.append(0)
                continue

            labels = self.y_train_[idxs]

            if self.weighted and len(dists) > 0:
                # Взвешенное голосование: вес = 1 / (d + ε)
                weights = 1.0 / (dists + 1e-9)
                vote_weights = {}
                for label, w in zip(labels, weights):
                    vote_weights[label] = vote_weights.get(label, 0) + w
                best_label = max(vote_weights, key=vote_weights.get)
            else:
                # Простое большинство
                best_label = Counter(labels).most_common(1)[0][0]

            predictions.append(best_label)

        return np.array(predictions)

    def predict_with_explanation(
        self, x_new: np.ndarray, exclude_self: bool = False
    ) -> dict:
        """
        Возвращает предсказание + объяснение (какие соседи повлияли).
        x_new — вектор мета-признаков одного датасета (1D).
        """
        distances, indices = self.nn_.kneighbors(x_new.reshape(1, -1))
        dists = distances[0]
        idxs  = indices[0]

        if exclude_self:
            mask  = dists > 1e-10
            dists = dists[mask]
            idxs  = idxs[mask]

        k_actual = min(self.k, len(idxs))
        dists = dists[:k_actual]
        idxs  = idxs[:k_actual]

        neighbors = []
        vote_weights = {}
        for rank, (idx, dist) in enumerate(zip(idxs, dists)):
            label     = self.y_train_[idx]
            sampler   = self.le_.inverse_transform([label])[0]
            weight    = 1.0 / (dist + 1e-9) if self.weighted else 1.0
            vote_weights[sampler] = vote_weights.get(sampler, 0) + weight
            neighbors.append({
                "rank":       rank + 1,
                "dataset":    self.names_[idx] if self.names_ else str(idx),
                "distance":   round(float(dist), 4),
                "best_method": sampler,
                "weight":     round(float(weight / (sum(vote_weights.values()) + 1e-9)), 3),
            })

        # Сортируем по весу голоса
        sorted_votes = sorted(vote_weights.items(), key=lambda x: -x[1])
        total_weight = sum(vote_weights.values())
        recommendation = sorted_votes[0][0] if sorted_votes else "Unknown"

        return {
            "recommendation": recommendation,
            "confidence":     round(sorted_votes[0][1] / (total_weight + 1e-9), 3)
                              if sorted_votes else 0.0,
            "vote_breakdown": [
                {"sampler": s, "vote_share": round(w / total_weight, 3)}
                for s, w in sorted_votes
            ],
            "neighbors": neighbors,
        }


# ══════════════════════════════════════════════════════════════════════════
# 3. ОЦЕНКА КАЧЕСТВА — Leave-One-Out
# ══════════════════════════════════════════════════════════════════════════

def evaluate_loo(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    names: list,
    sampler_ranks_list: list,
    k: int,
    weighted: bool = True,
) -> dict:
    """
    Leave-One-Out оценка:
    для каждого датасета исключаем его из базы и делаем предсказание.
    """
    n = len(y)
    y_pred = np.zeros(n, dtype=int)

    recommender = KNNSamplerRecommender(k=k, weighted=weighted)
    recommender.fit(X, y, le, names)

    # LOO: каждый раз исключаем i-й элемент
    for i in range(n):
        # Маска всех кроме i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        temp_rec = KNNSamplerRecommender(k=min(k, n-1), weighted=weighted)
        temp_rec.fit(X[mask], y[mask], le,
                     [names[j] for j in range(n) if j != i])
        y_pred[i] = temp_rec.predict(X[i:i+1])[0]

    # Метрики
    acc    = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_wei = f1_score(y, y_pred, average="weighted", zero_division=0)
    ba     = accuracy_score(y, y_pred)  # balanced ниже

    from sklearn.metrics import balanced_accuracy_score
    ba = balanced_accuracy_score(y, y_pred)

    # Top-2 accuracy: правильно если true_best входит в топ-2 по рангу у соседей
    top2_correct = 0
    for i in range(n):
        true_sampler = le.inverse_transform([y[i]])[0]
        pred_sampler = le.inverse_transform([y_pred[i]])[0]
        if pred_sampler == true_sampler:
            top2_correct += 1
        elif isinstance(sampler_ranks_list[i], dict):
            # Проверяем входит ли pred_sampler в топ-2 реальных рангов
            ranks = sampler_ranks_list[i]
            if ranks:
                sorted_samplers = sorted(ranks, key=ranks.get)
                if pred_sampler in sorted_samplers[:2]:
                    top2_correct += 1

    top2_acc = top2_correct / n

    return {
        "k":           k,
        "accuracy":    acc,
        "f1_macro":    f1_mac,
        "f1_weighted": f1_wei,
        "balanced_accuracy": ba,
        "top2_accuracy":     top2_acc,
        "y_pred":      y_pred,
    }


def select_best_k(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    names: list,
    sampler_ranks_list: list,
    k_max: int,
    weighted: bool = True,
) -> pd.DataFrame:
    """Подбирает лучший k через LOO по accuracy."""
    print(f"\nПодбор k (LOO, k от 1 до {k_max}):")
    print(f"  {'k':>4}  {'accuracy':>10}  {'f1_macro':>10}  {'top2_acc':>10}  {'bal_acc':>10}")
    print("  " + "-"*50)

    rows = []
    for k in range(1, k_max + 1):
        res = evaluate_loo(X, y, le, names, sampler_ranks_list, k, weighted)
        rows.append({
            "k":                k,
            "accuracy":         res["accuracy"],
            "f1_macro":         res["f1_macro"],
            "f1_weighted":      res["f1_weighted"],
            "balanced_accuracy": res["balanced_accuracy"],
            "top2_accuracy":    res["top2_accuracy"],
        })
        print(f"  {k:>4}  {res['accuracy']:>10.4f}  {res['f1_macro']:>10.4f}  "
              f"{res['top2_accuracy']:>10.4f}  {res['balanced_accuracy']:>10.4f}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 4. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════

def plot_k_selection(report_df: pd.DataFrame, best_k: int, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)

    metrics = {
        "accuracy":          (ACCENT,  "Accuracy"),
        "f1_macro":          (ACCENT2, "F1-macro"),
        "top2_accuracy":     (ACCENT3, "Top-2 Accuracy"),
        "balanced_accuracy": (ACCENT4, "Balanced Accuracy"),
    }
    for col, (color, label) in metrics.items():
        if col in report_df.columns:
            ax.plot(report_df["k"], report_df[col],
                    color=color, lw=2, marker="o", markersize=5,
                    markerfacecolor=color, label=label)

    ax.axvline(best_k, color="white", lw=1.5, ls="--", alpha=0.7,
               label=f"Лучший k={best_k}")

    # Отметим лучшее значение accuracy
    best_row = report_df.loc[report_df["accuracy"].idxmax()]
    ax.scatter([best_row["k"]], [best_row["accuracy"]],
               color="white", s=120, zorder=5)

    ax.set_title(f"Подбор k — Leave-One-Out CV\n"
                 f"(Лучший k={best_k}, acc={best_row['accuracy']:.4f})",
                 fontsize=13, color=TEXT)
    ax.set_xlabel("Число соседей k")
    ax.set_ylabel("Метрика")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(report_df["k"].values)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


def plot_confusion(y_true, y_pred, le, out: Path, title: str):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    n       = len(le.classes_)

    fig, ax = plt.subplots(figsize=(max(8, n*0.9), max(7, n*0.8)), facecolor=BG)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Purples",
        xticklabels=le.classes_, yticklabels=le.classes_,
        ax=ax, linewidths=0.5, vmin=0, vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{title}\n(нормировано по строкам, LOO)", fontsize=12, color=TEXT)
    ax.set_xlabel("Предсказано", color=TEXT)
    ax.set_ylabel("Истина", color=TEXT)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


def plot_neighbors_map(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    names: list,
    best_k: int,
    out: Path,
):
    """
    PCA 2D карта датасетов с соединениями между соседями.
    Показывает структуру пространства мета-признаков.
    """
    # PCA для визуализации
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    expl = pca.explained_variance_ratio_.sum()

    classes  = le.classes_
    n_cls    = len(classes)
    palette  = plt.cm.tab20(np.linspace(0, 1, n_cls))
    cls2col  = {cls: palette[i] for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)

    # Рисуем рёбра к ближайшим соседям
    nn_vis = NearestNeighbors(n_neighbors=min(best_k+1, len(X)), metric="euclidean")
    nn_vis.fit(X)
    distances, indices = nn_vis.kneighbors(X)

    for i in range(len(X)):
        for j, dist in zip(indices[i][1:best_k+1], distances[i][1:best_k+1]):
            ax.plot([X2d[i,0], X2d[j,0]], [X2d[i,1], X2d[j,1]],
                    color="#2a2a35", lw=0.8, alpha=0.5, zorder=1)

    # Рисуем точки
    for cls_idx, cls in enumerate(classes):
        mask = y == cls_idx
        ax.scatter(
            X2d[mask, 0], X2d[mask, 1],
            color=palette[cls_idx], s=60, alpha=0.85,
            edgecolors="#0d0d0f", linewidths=0.5,
            label=cls, zorder=2,
        )

    ax.set_title(
        f"Карта датасетов в пространстве мета-признаков\n"
        f"(PCA 2D, {expl:.1%} дисперсии, соединения при k={best_k})",
        fontsize=12, color=TEXT,
    )
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.legend(fontsize=8, loc="upper right", ncol=2,
              framealpha=0.3, markerscale=1.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Сохранён: {out}")


# ══════════════════════════════════════════════════════════════════════════
# 5. РЕКОМЕНДАЦИЯ ДЛЯ НОВОГО ДАТАСЕТА
# ══════════════════════════════════════════════════════════════════════════

def recommend_for_new(
    recommender: KNNSamplerRecommender,
    x_new: np.ndarray,
    scaler: RobustScaler,
    dataset_name: str = "новый датасет",
):
    """Выдаёт рекомендацию с объяснением для нового датасета."""
    x_scaled = scaler.transform(x_new.reshape(1, -1))[0]
    result   = recommender.predict_with_explanation(x_scaled)

    print(f"\n{'='*55}")
    print(f"  Рекомендация для: {dataset_name}")
    print(f"{'='*55}")
    print(f"  ★ Рекомендуемый сэмплер: {result['recommendation']}")
    print(f"  Уверенность: {result['confidence']:.1%}")

    print(f"\n  Голосование соседей:")
    for v in result["vote_breakdown"]:
        bar = "█" * int(v["vote_share"] * 20)
        print(f"    {v['sampler']:<26} {v['vote_share']:.1%}  {bar}")

    print(f"\n  Ближайшие {recommender.k} датасетов:")
    for nb in result["neighbors"]:
        print(f"    #{nb['rank']}  {nb['dataset'][:40]:<40}  "
              f"dist={nb['distance']:.3f}  → {nb['best_method']}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="kNN рекомендатор сэмплеров по мета-признакам",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--metafeatures", default="metafeatures.csv")
    pa.add_argument("--results",      default="results_full.csv")
    pa.add_argument("--output",       default="knn_output")
    pa.add_argument("--k",            type=int, default=None,
                    help="Фиксированный k (если не задан — подбирается автоматически)")
    pa.add_argument("--kmax",         type=int, default=20,
                    help="Максимальный k при автоподборе (default: 20)")
    pa.add_argument("--unweighted",   action="store_true",
                    help="Простое голосование (без весов по расстоянию)")
    pa.add_argument("--no-plots",     action="store_true")
    pa.add_argument("--new",          default=None,
                    help="Путь к .npy файлу с вектором мета-признаков нового датасета")
    pa.add_argument("--new-name",     default="новый датасет",
                    help="Имя нового датасета для вывода")
    args = pa.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    if not args.no_plots:
        (out / "plots").mkdir(exist_ok=True)

    weighted = not args.unweighted

    print(f"\n{'='*55}")
    print(f"  Алгоритм: kNN рекомендатор сэмплеров")
    print(f"  Голосование: {'взвешенное (1/dist)' if weighted else 'простое'}")
    print(f"  Метаfeatures: {args.metafeatures}")
    print(f"  Результаты:   {args.results}")
    print(f"{'='*55}\n")

    # ── 1. Загрузка ───────────────────────────────────────────────────
    print("Загружаю целевые переменные...")
    targets = compute_targets(Path(args.results))

    print("\nЗагружаю мета-признаки...")
    (df, X, y, mf_cols, le, scaler,
     sampler_ranks_list, dataset_names) = load_metafeatures(
        Path(args.metafeatures), targets
    )

    n = len(y)
    print(f"\n  Датасетов в базе: {n}")
    print(f"  Сэмплеров:        {len(le.classes_)}: {list(le.classes_)}")
    print(f"  Мета-признаков:   {len(mf_cols)}")

    # ── 2. Подбор k ───────────────────────────────────────────────────
    kmax_safe = min(args.kmax, n - 1)

    if args.k is not None:
        best_k = args.k
        # Всё равно оцениваем качество
        print(f"\nОцениваю качество при k={best_k} (LOO)...")
        loo_res  = evaluate_loo(X, y, le, dataset_names,
                                sampler_ranks_list, best_k, weighted)
        report_df = pd.DataFrame([{
            "k": best_k,
            "accuracy":          loo_res["accuracy"],
            "f1_macro":          loo_res["f1_macro"],
            "balanced_accuracy": loo_res["balanced_accuracy"],
            "top2_accuracy":     loo_res["top2_accuracy"],
        }])
        y_pred_loo = loo_res["y_pred"]
    else:
        report_df  = select_best_k(
            X, y, le, dataset_names, sampler_ranks_list, kmax_safe, weighted
        )
        # Лучший k — по accuracy (Top-2 если ничья)
        best_k_acc  = int(report_df.loc[report_df["accuracy"].idxmax(), "k"])
        best_k_top2 = int(report_df.loc[report_df["top2_accuracy"].idxmax(), "k"])
        best_k = best_k_acc   # главный критерий — accuracy
        print(f"\n  Лучший k по accuracy:      {best_k_acc}")
        print(f"  Лучший k по Top-2 accuracy: {best_k_top2}")
        print(f"  Выбран k = {best_k}")

        # LOO предсказания для лучшего k
        final_loo  = evaluate_loo(X, y, le, dataset_names,
                                  sampler_ranks_list, best_k, weighted)
        y_pred_loo = final_loo["y_pred"]

    report_df.to_csv(out / "knn_report.csv", index=False)

    # ── 3. Итоговые метрики ───────────────────────────────────────────
    acc_final  = accuracy_score(y, y_pred_loo)
    f1_final   = f1_score(y, y_pred_loo, average="macro", zero_division=0)
    f1w_final  = f1_score(y, y_pred_loo, average="weighted", zero_division=0)

    from sklearn.metrics import balanced_accuracy_score
    ba_final = balanced_accuracy_score(y, y_pred_loo)

    top2_correct = sum(
        1 for i in range(n)
        if (le.inverse_transform([y_pred_loo[i]])[0] ==
            le.inverse_transform([y[i]])[0]) or
        (isinstance(sampler_ranks_list[i], dict) and
         le.inverse_transform([y_pred_loo[i]])[0] in
         sorted(sampler_ranks_list[i], key=sampler_ranks_list[i].get)[:2])
    )
    top2_final = top2_correct / n

    print(f"\n{'='*55}")
    print(f"  ИТОГОВЫЕ МЕТРИКИ (k={best_k}, LOO CV)")
    print(f"{'='*55}")
    print(f"  Accuracy:          {acc_final:.4f}")
    print(f"  Balanced Accuracy: {ba_final:.4f}")
    print(f"  F1-macro:          {f1_final:.4f}")
    print(f"  F1-weighted:       {f1w_final:.4f}")
    print(f"  Top-2 Accuracy:    {top2_final:.4f}  ← рекомендация в топ-2")

    print(f"\n{classification_report(y, y_pred_loo, target_names=le.classes_, zero_division=0)}")

    # ── 4. Сохранение предсказаний ────────────────────────────────────
    ds_col = next((c for c in ["dataset_name","name"] if c in df.columns),
                  df.columns[0])
    pred_df = pd.DataFrame({
        "dataset":          df[ds_col].values,
        "true_best_method": le.inverse_transform(y),
        "pred_best_method": le.inverse_transform(y_pred_loo),
        "correct":          y == y_pred_loo,
    })
    pred_df.to_csv(out / "knn_predictions.csv", index=False)

    # ── 5. Обучаем финальный рекомендатор на всех данных ─────────────
    final_recommender = KNNSamplerRecommender(
        k=best_k, weighted=weighted
    )
    final_recommender.fit(X, y, le, dataset_names)

    joblib.dump({
        "recommender":  final_recommender,
        "scaler":       scaler,
        "mf_cols":      mf_cols,
        "le":           le,
        "best_k":       best_k,
        "metrics": {
            "accuracy":          acc_final,
            "balanced_accuracy": ba_final,
            "f1_macro":          f1_final,
            "top2_accuracy":     top2_final,
        },
    }, out / "knn_recommender.pkl")
    print(f"\n  Рекомендатор сохранён: {out}/knn_recommender.pkl")

    # ── 6. Графики ────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nСтроим графики...")
        if len(report_df) > 1:
            plot_k_selection(
                report_df, best_k,
                out / "plots" / "knn_k_selection.png",
            )
        plot_confusion(
            y, y_pred_loo, le,
            out / "plots" / "knn_confusion.png",
            f"kNN Рекомендатор (k={best_k})",
        )
        plot_neighbors_map(
            X, y, le, dataset_names, best_k,
            out / "plots" / "knn_neighbors_map.png",
        )

    # ── 7. Рекомендация для нового датасета ──────────────────────────
    if args.new:
        print(f"\nРекомендация для нового датасета: {args.new}")
        try:
            x_new = np.load(args.new)
            recommend_for_new(final_recommender, x_new, scaler, args.new_name)
        except Exception as e:
            print(f"  Ошибка: {e}")

    # ── 8. Итог ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Готово!")
    print(f"  База датасетов:    {n}")
    print(f"  Лучший k:          {best_k}")
    print(f"  Accuracy (LOO):    {acc_final:.4f}")
    print(f"  Top-2 Acc (LOO):   {top2_final:.4f}")
    print(f"\n  Файлы в {out}/:")
    print(f"    knn_recommender.pkl   ← готовый рекомендатор")
    print(f"    knn_report.csv        ← метрики при разных k")
    print(f"    knn_predictions.csv   ← LOO предсказания")
    print(f"\n  Инференс:")
    print(f"    import joblib, numpy as np")
    print(f"    art = joblib.load('{out}/knn_recommender.pkl')")
    print(f"    x = art['scaler'].transform([mf_vector])")
    print(f"    rec = art['recommender']")
    print(f"    method = art['le'].inverse_transform(rec.predict(x))[0]")
    print(f"    # С объяснением:")
    print(f"    detail = rec.predict_with_explanation(x[0])")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()