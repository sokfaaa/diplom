"""
Подбор оптимального числа кластеров для select_datasets.py.

Считает три метрики для диапазона k:
  - Inertia (метод локтя)      — ищем "колено" на графике
  - Silhouette score           — чем выше, тем лучше (макс = 1)
  - Calinski-Harabasz index    — чем выше, тем лучше
  - Davies-Bouldin index       — чем ниже, тем лучше

Запуск:
    python find_optimal_k.py                        # metafeatures.csv, k от 10 до 200
    python find_optimal_k.py --input mf.csv         # другой файл
    python find_optimal_k.py --kmin 20 --kmax 150   # другой диапазон
    python find_optimal_k.py --step 5               # шаг 5 вместо 10
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ── Загрузка и очистка (та же логика что в select_datasets.py) ────────────

def load_and_clean(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "name" in df.columns and "dataset_name" not in df.columns:
            df = df.rename(columns={"name": "dataset_name"})
    else:
        df = pd.read_csv(path)

    EXCL_PREFIXES = ("gen_", "__", "dataset_")
    EXCL_EXACT = {
        "dataset_name", "name", "dataset", "group", "base_type",
        "random_state", "noise_type", "spatial_distortion",
        "n_samples_total", "n_samples_train", "n_samples_test",
        "target_weights", "actual_weights", "class_counts_train",
        "source", "source_id", "original_name", "ir_zone",
    }

    mf_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in EXCL_PREFIXES)
        and c not in EXCL_EXACT
        and df[c].dtype != object
    ]

    X_raw = df[mf_cols].copy()

    # Очистка
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    drop_nan = X_raw.columns[X_raw.isna().mean() > 0.40].tolist()
    if drop_nan:
        X_raw = X_raw.drop(columns=drop_nan)
    X_raw = X_raw.fillna(X_raw.median())

    # Clip ±10 IQR
    q1, q3 = X_raw.quantile(0.25), X_raw.quantile(0.75)
    iqr = q3 - q1
    X_raw = X_raw.clip(lower=q1 - 10*iqr, upper=q3 + 10*iqr, axis=1)

    # Дроп константных
    X_raw = X_raw.loc[:, X_raw.std() > 1e-10]

    # Log-transform тяжёлых хвостов
    LOG_PATS = ["nr_inst", "nr_attr", "eq_num_attr", "IR", "HDB_mean", "HDB_std",
                "eigenvalues", "cov"]
    for col in X_raw.columns:
        if any(p in col for p in LOG_PATS) and (X_raw[col] >= 0).all():
            X_raw[col] = np.log1p(X_raw[col])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA для кластеризации (95% дисперсии)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Признаков:    {X_raw.shape[1]}")
    print(f"  PCA компонент: {pca.n_components_} ({pca.explained_variance_ratio_.sum():.1%} дисперсии)")

    return X_pca, df


# ── Вычисление метрик ─────────────────────────────────────────────────────

def compute_metrics(X: np.ndarray, k_values: list[int], seed: int = 42) -> pd.DataFrame:
    rows = []
    total = len(k_values)

    for i, k in enumerate(k_values, 1):
        print(f"  [{i:>3}/{total}]  k={k:<5}", end=" ", flush=True)

        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
        labels = km.fit_predict(X)

        inertia = km.inertia_

        # Silhouette на подвыборке (дорогая метрика)
        n_sub = min(len(X), 2000)
        idx = np.random.default_rng(seed).choice(len(X), n_sub, replace=False)
        sil = silhouette_score(X[idx], labels[idx], metric="euclidean")

        ch  = calinski_harabasz_score(X, labels)
        db  = davies_bouldin_score(X, labels)

        rows.append({
            "k":         k,
            "inertia":   inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin":    db,
        })
        print(f"inertia={inertia:>9.1f}  sil={sil:.4f}  CH={ch:>9.1f}  DB={db:.4f}")

    return pd.DataFrame(rows)


# ── Нахождение "локтя" ────────────────────────────────────────────────────

def find_elbow(k_values: list[int], inertias: list[float]) -> int:
    """Метод локтя — максимальная кривизна."""
    k_arr = np.array(k_values, dtype=float)
    y_arr = np.array(inertias, dtype=float)

    # Нормируем для нахождения точки максимального удаления от прямой
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    y_norm = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min())

    # Вектор от первой до последней точки
    vec = np.array([k_norm[-1] - k_norm[0], y_norm[-1] - y_norm[0]])
    vec = vec / np.linalg.norm(vec)

    # Расстояние каждой точки от этой прямой
    dists = []
    for kn, yn in zip(k_norm, y_norm):
        point = np.array([kn - k_norm[0], yn - y_norm[0]])
        dist  = abs(np.cross(vec, point))
        dists.append(dist)

    elbow_idx = int(np.argmax(dists))
    return k_values[elbow_idx]


# ── Визуализация ──────────────────────────────────────────────────────────

THEME = dict(
    facecolor="#0d0d0f",
    edgecolor="#2a2a35",
)
TEXT_COLOR  = "#e8e8f0"
MUTED_COLOR = "#6b6b80"
ACCENT      = "#7c6aff"
ACCENT2     = "#ff6a9b"
ACCENT3     = "#6affd4"
ACCENT4     = "#fbbf24"
GRID_COLOR  = "#1e1e28"


def make_plot(metrics_df: pd.DataFrame, elbow_k: int, best_sil_k: int,
              best_ch_k: int, best_db_k: int, n_datasets: int,
              output_path: Path):

    plt.rcParams.update({
        "figure.facecolor":  "#0d0d0f",
        "axes.facecolor":    "#141418",
        "axes.edgecolor":    "#2a2a35",
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       MUTED_COLOR,
        "ytick.color":       MUTED_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Подбор оптимального числа кластеров  |  {n_datasets} датасетов",
        fontsize=15, color=TEXT_COLOR, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    k = metrics_df["k"].values

    # ── 1. Inertia (метод локтя) ──────────────────────────────────────
    ax = axes[0]
    ax.plot(k, metrics_df["inertia"], color=ACCENT, lw=2, marker="o",
            markersize=4, markerfacecolor=ACCENT, label="Inertia")
    ax.axvline(elbow_k, color=ACCENT2, lw=1.5, ls="--",
               label=f"Локоть: k={elbow_k}")
    ax.set_title("① Метод локтя (Inertia)", color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Inertia (сумма квадратов)")
    ax.legend(fontsize=10, framealpha=0.3, edgecolor="#2a2a35")
    ax.grid(True, alpha=0.4)
    _annotate_optimum(ax, elbow_k,
                      metrics_df.loc[metrics_df["k"]==elbow_k, "inertia"].values[0],
                      ACCENT2)

    # ── 2. Silhouette ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(k, metrics_df["silhouette"], color=ACCENT3, lw=2, marker="o",
            markersize=4, markerfacecolor=ACCENT3)
    ax.axvline(best_sil_k, color=ACCENT2, lw=1.5, ls="--",
               label=f"Лучший: k={best_sil_k}")
    ax.set_title("② Silhouette Score (↑ лучше)", color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Silhouette Score")
    ax.legend(fontsize=10, framealpha=0.3, edgecolor="#2a2a35")
    ax.grid(True, alpha=0.4)
    best_sil = metrics_df.loc[metrics_df["k"]==best_sil_k, "silhouette"].values[0]
    _annotate_optimum(ax, best_sil_k, best_sil, ACCENT2)

    # ── 3. Calinski-Harabasz ─────────────────────────────────────────
    ax = axes[2]
    ax.plot(k, metrics_df["calinski_harabasz"], color=ACCENT4, lw=2, marker="o",
            markersize=4, markerfacecolor=ACCENT4)
    ax.axvline(best_ch_k, color=ACCENT2, lw=1.5, ls="--",
               label=f"Лучший: k={best_ch_k}")
    ax.set_title("③ Calinski-Harabasz Index (↑ лучше)", color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Calinski-Harabasz")
    ax.legend(fontsize=10, framealpha=0.3, edgecolor="#2a2a35")
    ax.grid(True, alpha=0.4)
    best_ch = metrics_df.loc[metrics_df["k"]==best_ch_k, "calinski_harabasz"].values[0]
    _annotate_optimum(ax, best_ch_k, best_ch, ACCENT2)

    # ── 4. Davies-Bouldin ─────────────────────────────────────────────
    ax = axes[3]
    ax.plot(k, metrics_df["davies_bouldin"], color="#60a5fa", lw=2, marker="o",
            markersize=4, markerfacecolor="#60a5fa")
    ax.axvline(best_db_k, color=ACCENT2, lw=1.5, ls="--",
               label=f"Лучший: k={best_db_k}")
    ax.set_title("④ Davies-Bouldin Index (↓ лучше)", color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Davies-Bouldin")
    ax.legend(fontsize=10, framealpha=0.3, edgecolor="#2a2a35")
    ax.grid(True, alpha=0.4)
    best_db = metrics_df.loc[metrics_df["k"]==best_db_k, "davies_bouldin"].values[0]
    _annotate_optimum(ax, best_db_k, best_db, ACCENT2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0f")
    plt.close(fig)
    print(f"\n  График сохранён: {output_path}")


def _annotate_optimum(ax, x, y, color):
    ax.annotate(
        f" k={x}",
        xy=(x, y), xytext=(x + 1, y),
        color=color, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1),
    )


# ── Итоговая рекомендация ─────────────────────────────────────────────────

def recommend_k(metrics_df: pd.DataFrame, elbow_k: int,
                n_datasets: int) -> int:
    """
    Взвешенная рекомендация:
      - Локоть (Inertia):          вес 3
      - Silhouette (максимум):     вес 4  ← самая надёжная метрика
      - Calinski-Harabasz (макс):  вес 2
      - Davies-Bouldin (мин):      вес 2
    """
    best_sil = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    best_ch  = int(metrics_df.loc[metrics_df["calinski_harabasz"].idxmax(), "k"])
    best_db  = int(metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"])

    # Взвешенное среднее
    k_weighted = (elbow_k * 3 + best_sil * 4 + best_ch * 2 + best_db * 2) / 11

    # Округляем до ближайшего значения из проверенных
    k_values = metrics_df["k"].values
    k_rec = int(k_values[np.argmin(np.abs(k_values - k_weighted))])

    # Ограничение: не больше n_datasets / 2
    k_rec = min(k_rec, n_datasets // 2)

    return k_rec, best_sil, best_ch, best_db


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Подбор оптимального числа кластеров",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--input",  default="metafeatures.csv",
                    help="CSV/JSON с мета-признаками (default: metafeatures.csv)")
    pa.add_argument("--kmin",   type=int, default=10,
                    help="Минимальное k (default: 10)")
    pa.add_argument("--kmax",   type=int, default=200,
                    help="Максимальное k (default: 200)")
    pa.add_argument("--step",   type=int, default=10,
                    help="Шаг (default: 10)")
    pa.add_argument("--seed",   type=int, default=42)
    pa.add_argument("--output", default="optimal_k_plot.png",
                    help="Путь для сохранения графика")
    args = pa.parse_args()

    print(f"\n{'='*55}")
    print(f"  Файл:    {args.input}")
    print(f"  k range: {args.kmin} – {args.kmax}  шаг {args.step}")
    print(f"{'='*55}\n")

    # Загружаем данные
    print("Загружаю и очищаю мета-признаки...")
    X_pca, df = load_and_clean(Path(args.input))
    n = len(df)
    print(f"  Датасетов:     {n}")
    print(f"  PCA shape:     {X_pca.shape}\n")

    # Диапазон k
    kmax_safe = min(args.kmax, n - 1)
    k_values  = list(range(args.kmin, kmax_safe + 1, args.step))
    if kmax_safe not in k_values:
        k_values.append(kmax_safe)
    k_values = sorted(set(k_values))

    print(f"Считаю метрики для {len(k_values)} значений k...\n")
    np.random.seed(args.seed)
    metrics_df = compute_metrics(X_pca, k_values, seed=args.seed)

    # Сохраняем метрики
    csv_path = Path(args.output).with_suffix(".csv")
    metrics_df.to_csv(csv_path, index=False)

    # Рекомендации
    elbow_k = find_elbow(k_values, metrics_df["inertia"].tolist())
    k_rec, best_sil_k, best_ch_k, best_db_k = recommend_k(metrics_df, elbow_k, n)

    # График
    make_plot(
        metrics_df, elbow_k, best_sil_k, best_ch_k, best_db_k,
        n_datasets=n,
        output_path=Path(args.output),
    )

    # Итог
    print(f"\n{'='*55}")
    print(f"  Метод локтя (Inertia):         k = {elbow_k}")
    print(f"  Silhouette (максимум):         k = {best_sil_k}")
    print(f"  Calinski-Harabasz (максимум):  k = {best_ch_k}")
    print(f"  Davies-Bouldin (минимум):      k = {best_db_k}")
    print(f"\n  ★ РЕКОМЕНДУЕМОЕ k = {k_rec}")
    print(f"    (взвешенное среднее всех метрик)")
    print(f"\n  Запусти select_datasets.py с этим k:")
    print(f"    python select_datasets.py --input {args.input} --clusters {k_rec}")
    print(f"\n  Метрики сохранены: {csv_path}")
    print(f"  График сохранён:  {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()