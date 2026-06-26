"""
Предварительная оценка целесообразности метамодели.

Отвечает на три вопроса:
  1. Есть ли вообще "лучший" сэмплер или все одинаковые? (NFL-тест)
  2. Зависит ли лучший сэмплер от мета-признаков? (корреляционный анализ)
  3. Что лучше: метамодель, k-NN, правила или просто "всегда SMOTE"?

Запуск:
    python evaluate_metamodel.py
    python evaluate_metamodel.py --results results_full.csv --metafeatures metafeatures.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

METRICS = ["balanced_accuracy", "f1_macro", "g_mean"]

THEME = "#0d0d0f"
TEXT  = "#e8e8f0"
MUTED = "#6b6b80"
ACCENT  = "#7c6aff"
ACCENT2 = "#ff6a9b"
ACCENT3 = "#6affd4"
ACCENT4 = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor": THEME, "axes.facecolor": "#141418",
    "axes.edgecolor": "#2a2a35", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#1e1e28",
    "grid.linewidth": 0.5, "font.family": "monospace",
})


# ── 1. Загрузка целевых переменных ───────────────────────────────────────

def load_targets(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    ds_col  = next(c for c in ["dataset", "dataset_name"] if c in df.columns)
    smp_col = "sampler"

    avail = [m for m in METRICS if m in df.columns]
    for m in avail:
        df[f"rank_{m}"] = df.groupby(ds_col)[m].rank(ascending=False, method="average")
    df["avg_rank"]  = df[[f"rank_{m}" for m in avail]].mean(axis=1)
    df["avg_score"] = df[avail].mean(axis=1)

    rows = []
    for ds, grp in df.groupby(ds_col):
        base = grp[grp[smp_col] == "Baseline"]["avg_score"]
        base_score = float(base.iloc[0]) if not base.empty else grp["avg_score"].min()

        non_base = grp[grp[smp_col] != "Baseline"]
        if non_base.empty:
            continue
        best = non_base.loc[non_base["avg_rank"].idxmin()]
        rows.append({
            "dataset":        ds,
            "best_method":    best[smp_col],
            "gain":           float(best["avg_score"] - base_score),
            "baseline_score": base_score,
            "best_score":     float(best["avg_score"]),
            # Все сэмплеры для анализа вариабельности
            "score_std":      float(non_base["avg_score"].std()),
            "score_range":    float(non_base["avg_score"].max() - non_base["avg_score"].min()),
            "n_samplers":     len(non_base),
        })

    targets = pd.DataFrame(rows)
    print(f"Датасетов с целевыми переменными: {len(targets)}")
    return targets


# ── 2. Загрузка мета-признаков ────────────────────────────────────────────

def load_metafeatures(mf_path: Path, targets: pd.DataFrame):
    mf = pd.read_csv(mf_path)
    ds_col = next((c for c in ["dataset_name","name","dataset"] if c in mf.columns),
                  mf.columns[0])
    mf[ds_col] = mf[ds_col].astype(str).str.strip()
    targets["dataset"] = targets["dataset"].astype(str).str.strip()

    df = mf.merge(targets, left_on=ds_col, right_on="dataset", how="inner")
    print(f"После объединения: {len(df)} датасетов")

    EXCL = {"dataset_name","name","dataset","group","base_type","random_state",
            "noise_type","spatial_distortion","n_samples_total","n_samples_train",
            "n_samples_test","target_weights","actual_weights","class_counts_train",
            "source","source_id","original_name","ir_zone",
            "best_method","gain","baseline_score","best_score","score_std",
            "score_range","n_samplers"}

    mf_cols = [c for c in df.columns
               if not any(c.startswith(p) for p in ("gen_","__","dataset_"))
               and c not in EXCL and df[c].dtype != object]

    X = df[mf_cols].copy()
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median())
    q1,q3 = X.quantile(0.25), X.quantile(0.75)
    X = X.clip(lower=q1-10*(q3-q1), upper=q3+10*(q3-q1), axis=1)
    X = X.loc[:, X.std() > 1e-10]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y  = le.fit_transform(df["best_method"])

    print(f"Мета-признаков: {X_scaled.shape[1]}")
    print(f"Классов (сэмплеров): {len(le.classes_)}: {list(le.classes_)}")

    return df, X_scaled, y, list(X.columns), le


# ── 3. NFL-тест: есть ли вообще разница между сэмплерами? ────────────────

def nfl_analysis(targets: pd.DataFrame) -> dict:
    """
    No Free Lunch тест:
    - Если score_range мал → все сэмплеры одинаковы → метамодель бесполезна
    - Если gain мал → сэмплинг вообще не помогает → зачем метамодель?
    - Если best_method всегда одинаковый → метамодель тривиальна
    """
    results = {}

    # Средний прирост над Baseline
    results["mean_gain"]      = targets["gain"].mean()
    results["median_gain"]    = targets["gain"].median()
    results["pct_gain_positive"] = (targets["gain"] > 0).mean()
    results["pct_gain_gt005"] = (targets["gain"] > 0.05).mean()

    # Вариабельность между сэмплерами
    results["mean_score_range"] = targets["score_range"].mean()
    results["median_score_range"] = targets["score_range"].median()

    # Концентрация best_method (если один доминирует — тривиально)
    freq = targets["best_method"].value_counts(normalize=True)
    results["top1_freq"]   = float(freq.iloc[0])
    results["top1_method"] = freq.index[0]
    results["entropy"]     = float(-(freq * np.log(freq + 1e-9)).sum())

    # Вывод
    print("\n" + "="*55)
    print("  NFL АНАЛИЗ (есть ли смысл в метамодели?)")
    print("="*55)
    print(f"  Средний прирост gain над Baseline:   {results['mean_gain']:+.4f}")
    print(f"  Медианный gain:                      {results['median_gain']:+.4f}")
    print(f"  Датасетов где gain > 0:              {results['pct_gain_positive']:.1%}")
    print(f"  Датасетов где gain > 0.05:           {results['pct_gain_gt005']:.1%}")
    print(f"  Средний разброс между сэмплерами:    {results['mean_score_range']:.4f}")
    print(f"  Самый частый сэмплер:                {results['top1_method']} ({results['top1_freq']:.1%})")
    print(f"  Энтропия best_method:                {results['entropy']:.3f}")
    print()

    # Интерпретация
    if results["mean_gain"] < 0.01:
        verdict = "⚠️  Сэмплинг почти не помогает — gain ≈ 0. Метамодель бессмысленна."
    elif results["top1_freq"] > 0.5:
        verdict = (f"⚠️  {results['top1_method']} выигрывает в {results['top1_freq']:.0%} случаев. "
                   f"Проще всегда рекомендовать его.")
    elif results["mean_score_range"] < 0.02:
        verdict = "⚠️  Разброс между сэмплерами мал — выбор не критичен."
    elif results["entropy"] > 1.5 and results["mean_gain"] > 0.02:
        verdict = "✅  Метамодель имеет смысл: gain значителен и best_method разнообразен."
    else:
        verdict = "🔶  Пограничный случай. Метамодель может дать небольшое улучшение."

    results["verdict"] = verdict
    print(f"  ВЫВОД: {verdict}")
    return results


# ── 4. Сравнение стратегий ────────────────────────────────────────────────

def compare_strategies(X, y, le, n_splits=5, seed=42) -> pd.DataFrame:
    """
    Сравниваем 5 стратегий предсказания best_method:
      1. Baseline: всегда предсказывать самый частый сэмплер
      2. Random:   случайный выбор
      3. k-NN:     k=5 по мета-признакам (твоя идея)
      4. Decision Tree: читаемые правила
      5. Random Forest: полная метамодель
    """
    min_cls = pd.Series(y).value_counts().min()
    n_splits = min(n_splits, min_cls)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    strategies = {
        "Baseline (всегда топ-1)": DummyClassifier(strategy="most_frequent"),
        "Random":                  DummyClassifier(strategy="stratified", random_state=seed),
        "k-NN (k=5)":              KNeighborsClassifier(n_neighbors=5),
        "k-NN (k=3)":              KNeighborsClassifier(n_neighbors=3),
        "Decision Tree":           DecisionTreeClassifier(max_depth=5, random_state=seed,
                                                          class_weight="balanced"),
        "Random Forest":           RandomForestClassifier(n_estimators=200, random_state=seed,
                                                          class_weight="balanced", n_jobs=-1),
    }

    print("\n" + "="*55)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ (Stratified CV)")
    print("="*55)

    rows = []
    for name, model in strategies.items():
        scores_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        scores_f1  = cross_val_score(model, X, y, cv=cv, scoring="f1_macro",  n_jobs=1)
        scores_ba  = cross_val_score(model, X, y, cv=cv,
                                     scoring="balanced_accuracy", n_jobs=1)
        row = {
            "Стратегия":  name,
            "Accuracy":   scores_acc.mean(),
            "F1-macro":   scores_f1.mean(),
            "Bal.Acc.":   scores_ba.mean(),
            "Acc±":       scores_acc.std(),
        }
        rows.append(row)
        print(f"  {name:<28} acc={row['Accuracy']:.3f}±{row['Acc±']:.3f}  "
              f"f1={row['F1-macro']:.3f}  ba={row['Bal.Acc.']:.3f}")

    df_comp = pd.DataFrame(rows).sort_values("F1-macro", ascending=False)

    # Выводим лучшую
    best = df_comp.iloc[0]
    baseline_f1 = df_comp[df_comp["Стратегия"].str.startswith("Baseline")]["F1-macro"].values[0]
    improvement = best["F1-macro"] - baseline_f1

    print(f"\n  Лучшая стратегия: {best['Стратегия']}")
    print(f"  F1-macro: {best['F1-macro']:.3f} (против Baseline {baseline_f1:.3f}, "
          f"улучшение: {improvement:+.3f})")

    if improvement < 0.03:
        print("  ⚠️  Улучшение над Baseline < 0.03 — метамодель не даёт выигрыша")
    elif improvement < 0.08:
        print("  🔶  Умеренное улучшение — метамодель работает, но слабо")
    else:
        print("  ✅  Существенное улучшение — метамодель оправдана")

    return df_comp


# ── 5. Читаемые правила Decision Tree ────────────────────────────────────

def extract_rules(X, y, mf_cols, le, max_depth=4, seed=42):
    dt = DecisionTreeClassifier(
        max_depth=max_depth, random_state=seed, class_weight="balanced"
    )
    dt.fit(X, y)
    rules = export_text(dt, feature_names=mf_cols, max_depth=max_depth)

    # Заменяем числовые метки на имена сэмплеров
    for i, cls in enumerate(le.classes_):
        rules = rules.replace(f"class: {i}", f"→ {cls}")

    print("\n" + "="*55)
    print("  ЧИТАЕМЫЕ ПРАВИЛА (Decision Tree, depth=4)")
    print("="*55)
    print(rules[:3000])  # Первые 3000 символов
    if len(rules) > 3000:
        print("  ... (правила обрезаны, полные в файле rules.txt)")
    return rules, dt


# ── 6. Визуализация ───────────────────────────────────────────────────────

def make_plots(targets, nfl, comparison_df, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Анализ целесообразности метамодели", fontsize=15,
                 color=TEXT, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Распределение gain ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    gains = targets["gain"].values
    ax.hist(gains, bins=30, color=ACCENT, alpha=0.8, edgecolor="#0d0d0f")
    ax.axvline(0, color=ACCENT2, lw=1.5, ls="--", label="gain=0")
    ax.axvline(gains.mean(), color=ACCENT3, lw=1.5, ls="-",
               label=f"mean={gains.mean():.3f}")
    ax.set_title("Прирост gain над Baseline", color=TEXT, fontsize=11)
    ax.set_xlabel("gain (avg_score(best) − avg_score(baseline))")
    ax.set_ylabel("Число датасетов")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # ── 2. Частота best_method ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    freq = targets["best_method"].value_counts()
    colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#60a5fa",
              "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#e879f9"]
    bars = ax.barh(freq.index, freq.values,
                   color=colors[:len(freq)], edgecolor="#0d0d0f")
    for bar, v in zip(bars, freq.values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=9, color=TEXT)
    ax.set_title("Лучший сэмплер по датасетам", color=TEXT, fontsize=11)
    ax.set_xlabel("Число датасетов")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 3. Разброс между сэмплерами ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(targets["score_range"].values, bins=25,
            color=ACCENT3, alpha=0.8, edgecolor="#0d0d0f")
    ax.axvline(targets["score_range"].median(), color=ACCENT2, lw=1.5,
               label=f"median={targets['score_range'].median():.3f}")
    ax.set_title("Разброс между сэмплерами\n(max−min avg_score)", color=TEXT, fontsize=11)
    ax.set_xlabel("score_range")
    ax.set_ylabel("Число датасетов")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # ── 4. Сравнение стратегий ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    strategies = comparison_df["Стратегия"].values
    f1_vals    = comparison_df["F1-macro"].values
    acc_vals   = comparison_df["Accuracy"].values
    ba_vals    = comparison_df["Bal.Acc."].values

    x = np.arange(len(strategies))
    w = 0.25
    ax.bar(x - w, f1_vals, w, label="F1-macro",   color=ACCENT,  edgecolor="#0d0d0f")
    ax.bar(x,     acc_vals, w, label="Accuracy",   color=ACCENT3, edgecolor="#0d0d0f")
    ax.bar(x + w, ba_vals, w, label="Bal.Acc.",    color=ACCENT2, edgecolor="#0d0d0f")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=9)
    ax.set_title("Сравнение стратегий рекомендации", color=TEXT, fontsize=11)
    ax.set_ylabel("Метрика")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    # ── 5. Вердикт ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    verdict_text = (
        f"ВЫВОД:\n\n"
        f"{nfl['verdict']}\n\n"
        f"Статистика:\n"
        f"• mean gain: {nfl['mean_gain']:+.4f}\n"
        f"• gain > 0: {nfl['pct_gain_positive']:.1%} датасетов\n"
        f"• score range: {nfl['mean_score_range']:.4f}\n"
        f"• top-1 метод: {nfl['top1_method']}\n"
        f"  ({nfl['top1_freq']:.1%} датасетов)\n"
        f"• энтропия: {nfl['entropy']:.3f}\n\n"
        f"Лучшая стратегия:\n"
        f"{comparison_df.iloc[0]['Стратегия']}\n"
        f"F1={comparison_df.iloc[0]['F1-macro']:.3f}"
    )
    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=10, va="top", color=TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#141418",
                      edgecolor=ACCENT, alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = output_dir / "metamodel_feasibility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME)
    plt.close(fig)
    print(f"\n  График сохранён: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Оценка целесообразности метамодели",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument("--results",      default="results_full.csv")
    pa.add_argument("--metafeatures", default="metafeatures.csv")
    pa.add_argument("--output",       default="analysis_output")
    pa.add_argument("--cv",           type=int, default=5)
    pa.add_argument("--seed",         type=int, default=42)
    pa.add_argument("--rules-depth",  type=int, default=4,
                    help="Глубина дерева для читаемых правил (default: 4)")
    args = pa.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Бенчмарк:      {args.results}")
    print(f"  Мета-признаки: {args.metafeatures}")
    print(f"{'='*55}\n")

    # 1. Загрузка
    targets = load_targets(Path(args.results))
    df, X, y, mf_cols, le = load_metafeatures(Path(args.metafeatures), targets)

    # 2. NFL-анализ
    nfl = nfl_analysis(targets)

    # 3. Сравнение стратегий
    comparison = compare_strategies(X, y, le, n_splits=args.cv, seed=args.seed)
    comparison.to_csv(out_dir / "strategy_comparison.csv", index=False)

    # 4. Правила Decision Tree
    rules, dt = extract_rules(X, y, mf_cols, le,
                               max_depth=args.rules_depth, seed=args.seed)
    (out_dir / "decision_rules.txt").write_text(rules, encoding="utf-8")
    print(f"\n  Правила сохранены: {out_dir}/decision_rules.txt")

    # 5. График
    make_plots(targets, nfl, comparison, out_dir)

    # 6. Итог
    best_strategy = comparison.iloc[0]["Стратегия"]
    baseline_f1   = comparison[comparison["Стратегия"].str.startswith("Baseline")]["F1-macro"].values[0]
    knn_f1        = comparison[comparison["Стратегия"].str.startswith("k-NN (k=5)")]["F1-macro"].values[0]
    rf_f1         = comparison[comparison["Стратегия"].str.startswith("Random Forest")]["F1-macro"].values[0]

    print(f"\n{'='*55}")
    print(f"  ИТОГОВАЯ РЕКОМЕНДАЦИЯ")
    print(f"{'='*55}")
    print(f"  {nfl['verdict']}\n")

    if knn_f1 >= rf_f1 - 0.03:
        print(f"  → k-NN (F1={knn_f1:.3f}) сопоставим с RF (F1={rf_f1:.3f})")
        print(f"    Используй k-NN — проще и интерпретируемее")
    else:
        print(f"  → Random Forest значительно лучше (F1={rf_f1:.3f} vs k-NN {knn_f1:.3f})")
        print(f"    Метамодель (RF/LightGBM) оправдана")

    print(f"\n  Файлы в {out_dir}/:")
    print(f"    metamodel_feasibility.png  ← главный график")
    print(f"    strategy_comparison.csv   ← числа по стратегиям")
    print(f"    decision_rules.txt        ← читаемые правила")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()