"""
Метамодель: мета-признаки датасета → лучший сэмплер + прирост над Baseline.

Две задачи одновременно:
  [Классификация] best_method  — какой сэмплер выбрать
  [Регрессия]     gain         — насколько он лучше Baseline по среднему рангу метрик

Входные данные:
  --metafeatures  metafeatures.csv   — мета-признаки (из compute_metafeatures.py)
  --results       results_full.csv   — полный бенчмарк (из benchmark_samplers.py)

Схема вычисления целевых переменных из results_full.csv:
  1. Для каждого датасета × сэмплер считаем avg_rank по трём метрикам
  2. best_method = сэмплер с лучшим avg_rank
  3. gain = avg_score(best) - avg_score(Baseline)  (>0 = лучше Baseline)

Выходные файлы (в --output папке):
  best_clf.pkl / best_reg.pkl    — модели + препроцессор
  clf_report.csv / reg_report.csv
  predictions.csv                — всё вместе с вероятностями
  plots/                         — 6 графиков

Запуск:
  python train_metamodel.py
  python train_metamodel.py --metafeatures mf.csv --results results_full.csv
  python train_metamodel.py --cv 10 --no-plots
"""

import argparse, warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
    mean_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")
METRICS = ["balanced_accuracy", "f1_macro", "g_mean"]


# ── 1. Целевые переменные ────────────────────────────────────────────────────

def compute_targets(results_path: Path) -> pd.DataFrame:
    print(f"Загружаю бенчмарк: {results_path}")
    df = pd.read_csv(results_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    ds_col  = next((c for c in ["dataset", "dataset_name"] if c in df.columns), df.columns[0])
    smp_col = next((c for c in ["sampler"] if c in df.columns), df.columns[1])

    avail = [m for m in METRICS if m in df.columns]
    if not avail:
        raise ValueError(f"Не найдено метрик {METRICS} в {list(df.columns)}")
    print(f"  Метрики: {avail},  датасетов: {df[ds_col].nunique()},  сэмплеров: {df[smp_col].nunique()}")

    for m in avail:
        df[f"rank_{m}"] = df.groupby(ds_col)[m].rank(ascending=False, method="average")
    df["avg_rank"]  = df[[f"rank_{m}" for m in avail]].mean(axis=1)
    df["avg_score"] = df[avail].mean(axis=1)

    rows = []
    for ds_name, grp in df.groupby(ds_col):
        base = grp[grp[smp_col] == "Baseline"]
        baseline_score = base["avg_score"].iloc[0] if not base.empty else grp["avg_score"].min()

        non_base = grp[grp[smp_col] != "Baseline"]
        if non_base.empty:
            continue
        best = non_base.loc[non_base["avg_rank"].idxmin()]
        rows.append({
            "dataset":        ds_name,
            "best_method":    best[smp_col],
            "gain":           float(best["avg_score"] - baseline_score),
            "baseline_score": float(baseline_score),
            "best_score":     float(best["avg_score"]),
        })

    targets = pd.DataFrame(rows)
    print(f"  Целевых переменных: {len(targets)}")
    print(f"  Распределение best_method:")
    for m, c in targets["best_method"].value_counts().items():
        print(f"    {m:<26} {c:>4}  ({c/len(targets):.1%})")
    print(f"  gain: min={targets['gain'].min():.4f}  mean={targets['gain'].mean():.4f}  max={targets['gain'].max():.4f}")
    print(f"  gain > 0: {(targets['gain']>0).sum()} ({(targets['gain']>0).mean():.1%})")
    return targets


# ── 2. Мерж с мета-признаками ────────────────────────────────────────────────

def load_and_merge(mf_path: Path, targets: pd.DataFrame):
    print(f"\nЗагружаю мета-признаки: {mf_path}")
    mf = pd.read_csv(mf_path)
    ds_col = next((c for c in ["dataset_name","name","dataset"] if c in mf.columns), mf.columns[0])
    mf[ds_col] = mf[ds_col].astype(str).str.strip()
    targets["dataset"] = targets["dataset"].astype(str).str.strip()

    df = mf.merge(targets, left_on=ds_col, right_on="dataset", how="inner")
    print(f"  После мержа: {len(df)}  (потеряно из mf: {len(mf)-len(df)})")
    if len(df) == 0:
        raise ValueError(
            f"0 строк после мержа.\nmf names (5): {mf[ds_col].head().tolist()}\n"
            f"targets (5):  {targets['dataset'].head().tolist()}"
        )

    EXCL_PFXS  = ("gen_", "__", "dataset_")
    EXCL_EXACT = {
        ds_col,"dataset","group","base_type","name","random_state","noise_type",
        "spatial_distortion","n_samples_total","n_samples_train","n_samples_test",
        "target_weights","actual_weights","class_counts_train","circles_factor",
        "cluster_std","n_informative","lhs_index","source","source_id",
        "original_name","ir_zone",
        "best_method","gain","baseline_score","best_score",
    }
    mf_cols = [c for c in df.columns
               if not any(c.startswith(p) for p in EXCL_PFXS)
               and c not in EXCL_EXACT
               and df[c].dtype != object]

    X_raw = df[mf_cols].copy()
    n_inf = np.isinf(X_raw.values).sum()
    if n_inf:
        print(f"  inf → NaN: {n_inf}")
        X_raw = X_raw.replace([np.inf,-np.inf], np.nan)

    drop_nan = X_raw.columns[X_raw.isna().mean() > 0.50].tolist()
    if drop_nan:
        print(f"  Дроп >50% NaN: {len(drop_nan)}")
        X_raw = X_raw.drop(columns=drop_nan)
        mf_cols = [c for c in mf_cols if c not in drop_nan]

    X_raw = X_raw.fillna(X_raw.median())
    q1,q3 = X_raw.quantile(0.25), X_raw.quantile(0.75)
    X_raw = X_raw.clip(lower=q1-10*(q3-q1), upper=q3+10*(q3-q1), axis=1)

    zero_var = X_raw.columns[X_raw.std() < 1e-10].tolist()
    if zero_var:
        X_raw = X_raw.drop(columns=zero_var)
        mf_cols = [c for c in mf_cols if c not in zero_var]

    print(f"  Мета-признаков: {len(mf_cols)}")

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw).astype(np.float64)

    le    = LabelEncoder()
    y_cls = le.fit_transform(df["best_method"])
    y_reg = df["gain"].values.astype(np.float64)

    print(f"  X={X.shape}   Классов: {len(le.classes_)}: {list(le.classes_)}")
    print(f"  gain: mean={y_reg.mean():.4f}  std={y_reg.std():.4f}")
    return df, X, y_cls, y_reg, mf_cols, le, scaler


# ── 3. Модели ────────────────────────────────────────────────────────────────

def get_classifiers(n_cls, seed=42):
    ms = {
        "RandomForest":      RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
        "LogisticRegression":LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed),
        "KNN":               KNeighborsClassifier(n_neighbors=min(7, n_cls+2)),
    }
    try:
        from lightgbm import LGBMClassifier
        ms["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, verbose=-1)
    except ImportError: pass
    try:
        from xgboost import XGBClassifier
        ms["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, eval_metric="mlogloss", verbosity=0)
    except ImportError: pass
    return ms

def get_regressors(seed=42):
    ms = {
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1),
        "Ridge":        Ridge(alpha=1.0),
        "KNN":          KNeighborsRegressor(n_neighbors=7),
    }
    try:
        from lightgbm import LGBMRegressor
        ms["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    except ImportError: pass
    try:
        from xgboost import XGBRegressor
        ms["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, verbosity=0)
    except ImportError: pass
    return ms


# ── 4. CV ────────────────────────────────────────────────────────────────────

def cv_clf(models, X, y, n_splits, seed):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    print(f"\nКлассификация — CV ({n_splits} фолдов):")
    for name, model in models.items():
        print(f"  {name:<22}", end=" ", flush=True)
        try:
            sc = cross_validate(model, X, y, cv=cv, n_jobs=1,
                                scoring=["accuracy","balanced_accuracy","f1_macro"])
            row = {"model": name,
                   "accuracy_mean": sc["test_accuracy"].mean(),
                   "accuracy_std":  sc["test_accuracy"].std(),
                   "balanced_accuracy_mean": sc["test_balanced_accuracy"].mean(),
                   "balanced_accuracy_std":  sc["test_balanced_accuracy"].std(),
                   "f1_macro_mean": sc["test_f1_macro"].mean(),
                   "f1_macro_std":  sc["test_f1_macro"].std()}
            print(f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
                  f"f1={row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}  "
                  f"ba={row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f}")
        except Exception as e:
            print(f"ОШИБКА: {e}")
            row = {"model": name, "f1_macro_mean": -1.0, "accuracy_mean": -1.0, "balanced_accuracy_mean": -1.0}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)

def cv_reg(models, X, y, n_splits, seed):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    print(f"\nРегрессия — CV ({n_splits} фолдов):")
    for name, model in models.items():
        print(f"  {name:<22}", end=" ", flush=True)
        try:
            sc = cross_validate(model, X, y, cv=cv, n_jobs=1,
                                scoring=["neg_mean_absolute_error","neg_root_mean_squared_error","r2"])
            row = {"model": name,
                   "mae_mean":  -sc["test_neg_mean_absolute_error"].mean(),
                   "mae_std":    sc["test_neg_mean_absolute_error"].std(),
                   "rmse_mean": -sc["test_neg_root_mean_squared_error"].mean(),
                   "rmse_std":   sc["test_neg_root_mean_squared_error"].std(),
                   "r2_mean":    sc["test_r2"].mean(),
                   "r2_std":     sc["test_r2"].std()}
            print(f"MAE={row['mae_mean']:.4f}±{row['mae_std']:.4f}  "
                  f"RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f}  "
                  f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}")
        except Exception as e:
            print(f"ОШИБКА: {e}")
            row = {"model": name, "mae_mean": 999.0, "r2_mean": -999.0}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


# ── 5. Финальное обучение ────────────────────────────────────────────────────

def train_clf(model, X, y, le):
    model.fit(X, y)
    yp = model.predict(X)
    print(f"\n  [Классификатор] train:")
    print(f"    Accuracy: {accuracy_score(y,yp):.4f}  "
          f"BA: {balanced_accuracy_score(y,yp):.4f}  "
          f"F1: {f1_score(y,yp,average='macro',zero_division=0):.4f}")
    print(classification_report(y, yp, target_names=le.classes_, zero_division=0))
    return model, yp

def train_reg(model, X, y):
    model.fit(X, y)
    yp = model.predict(X)
    print(f"\n  [Регрессор] train:")
    print(f"    MAE: {mean_absolute_error(y,yp):.4f}  "
          f"RMSE: {root_mean_squared_error(y,yp):.4f}  "
          f"R²: {r2_score(y,yp):.4f}")
    return model, yp


# ── 6. Важность признаков ────────────────────────────────────────────────────

def feat_imp(model, cols):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        return pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
    return None


# ── 7. Графики ───────────────────────────────────────────────────────────────

def plot_cv(cv_df, title, metric, metric_std, xlabel, out):
    valid = cv_df[cv_df.get(metric, pd.Series([-1])) > -999] if metric in cv_df.columns else pd.DataFrame()
    if valid.empty: return
    fig, ax = plt.subplots(figsize=(10, max(4, len(valid)*0.65)))
    cols = ["#2ecc71" if i==0 else "#3498db" for i in range(len(valid))]
    bars = ax.barh(valid["model"], valid[metric], xerr=valid.get(metric_std,0),
                   color=cols, capsize=5, height=0.5)
    for bar, v in zip(bars, valid[metric]):
        ax.text(v+0.002, bar.get_y()+bar.get_height()/2, f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Сохранён: {out}")

def plot_cm(y, yp, le, out, name):
    cm = confusion_matrix(y, yp)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(le.classes_)
    fig, ax = plt.subplots(figsize=(max(8,n*0.8), max(7,n*0.7)))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix — {name}\n(нормировано по строкам)", fontsize=12)
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Истина")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Сохранён: {out}")

def plot_imp(imp_df, title, out):
    if imp_df is None or imp_df.empty: return
    top = imp_df.head(min(25, len(imp_df)))
    fig, ax = plt.subplots(figsize=(10, max(5, len(top)*0.38)))
    cols = plt.cm.RdYlGn(np.linspace(0.25, 0.9, len(top)))[::-1]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=cols[::-1])
    ax.set_title(title, fontsize=12); ax.set_xlabel("Важность")
    plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Сохранён: {out}")

def plot_scatter(y_true, y_pred, out, name):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=55, edgecolors="none", color="#3498db")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Идеал (y=x)")
    ax.axhline(0, color="gray", lw=0.8, ls=":"); ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Реальный gain"); ax.set_ylabel("Предсказанный gain")
    ax.set_title(f"Предсказанный vs реальный прирост — {name}\n"
                 f"MAE={mean_absolute_error(y_true,y_pred):.4f}  R²={r2_score(y_true,y_pred):.3f}", fontsize=11)
    ax.legend(); plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Сохранён: {out}")


# ── 8. CLI ───────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Метамодель: мета-признаки → лучший сэмплер + прирост над Baseline",
        formatter_class=argparse.RawTextHelpFormatter)
    pa.add_argument("--metafeatures", default="metafeatures.csv")
    pa.add_argument("--results",      default="results_full.csv")
    pa.add_argument("--output",       default="metamodel_output")
    pa.add_argument("--cv",           type=int, default=5)
    pa.add_argument("--seed",         type=int, default=42)
    pa.add_argument("--no-plots",     action="store_true")
    args = pa.parse_args()

    out = Path(args.output); out.mkdir(exist_ok=True)
    plots = out / "plots"; plots.mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Мета-признаки:  {args.metafeatures}")
    print(f"  Бенчмарк:       {args.results}")
    print(f"  CV: {args.cv} фолдов   Seed: {args.seed}")
    print(f"{'='*65}\n")

    # 1. Целевые переменные
    targets = compute_targets(Path(args.results))

    # 2. Мерж
    df, X, y_cls, y_reg, mf_cols, le, scaler = load_and_merge(Path(args.metafeatures), targets)

    n_cls = len(le.classes_)
    min_cnt = pd.Series(y_cls).value_counts().min()
    n_splits = min(args.cv, min_cnt)
    if n_splits < args.cv:
        print(f"\n  ⚠️  CV: {args.cv}→{n_splits} (мин. класс: {min_cnt})")

    # 3. Классификация
    clfs = get_classifiers(n_cls, args.seed)
    clf_rep = cv_clf(clfs, X, y_cls, n_splits, args.seed)
    clf_rep.to_csv(out / "clf_report.csv", index=False)
    best_clf_name = clf_rep.iloc[0]["model"]
    best_clf = clfs[best_clf_name]
    print(f"\n  Лучший классификатор: {best_clf_name}  F1={clf_rep.iloc[0]['f1_macro_mean']:.4f}")
    best_clf, yp_cls = train_clf(best_clf, X, y_cls, le)

    # 4. Регрессия
    regs = get_regressors(args.seed)
    reg_rep = cv_reg(regs, X, y_reg, n_splits, args.seed)
    reg_rep.to_csv(out / "reg_report.csv", index=False)
    best_reg_name = reg_rep.iloc[0]["model"]
    best_reg = regs[best_reg_name]
    print(f"\n  Лучший регрессор: {best_reg_name}  R²={reg_rep.iloc[0]['r2_mean']:.4f}")
    best_reg, yp_reg = train_reg(best_reg, X, y_reg)

    # 5. Сохранение моделей
    artifact = {"scaler": scaler, "mf_cols": mf_cols, "label_encoder": le}
    joblib.dump({**artifact, "model": best_clf, "model_name": best_clf_name, "task": "classification"}, out/"best_clf.pkl")
    joblib.dump({**artifact, "model": best_reg, "model_name": best_reg_name, "task": "regression"},    out/"best_reg.pkl")

    # 6. Предсказания
    ds_col = next((c for c in ["dataset_name","name"] if c in df.columns), df.columns[0])
    pred_df = pd.DataFrame({
        "dataset":            df[ds_col].values,
        "true_best_method":   le.inverse_transform(y_cls),
        "pred_best_method":   le.inverse_transform(yp_cls),
        "clf_correct":        y_cls == yp_cls,
        "true_gain":          y_reg,
        "pred_gain":          yp_reg,
        "gain_error":         yp_reg - y_reg,
        "true_baseline_score":df["baseline_score"].values,
        "true_best_score":    df["best_score"].values,
    })
    if hasattr(best_clf, "predict_proba"):
        proba = best_clf.predict_proba(X)
        for i, cls in enumerate(le.classes_):
            pred_df[f"proba_{cls}"] = proba[:, i]
    pred_df.to_csv(out / "predictions.csv", index=False)

    # 7. Важность признаков
    imp_clf = feat_imp(best_clf, mf_cols)
    imp_reg = feat_imp(best_reg, mf_cols)
    if imp_clf is not None: imp_clf.to_csv(out/"importance_clf.csv", index=False)
    if imp_reg is not None: imp_reg.to_csv(out/"importance_reg.csv", index=False)

    # 8. Графики
    if not args.no_plots:
        print("\nСтроим графики...")
        plot_cv(clf_rep, "Классификация — сравнение моделей (F1-macro, CV)",
                "f1_macro_mean","f1_macro_std","F1-macro", plots/"clf_cv_comparison.png")
        plot_cv(reg_rep, "Регрессия — сравнение моделей (R², CV)",
                "r2_mean","r2_std","R²", plots/"reg_cv_comparison.png")
        plot_cm(y_cls, yp_cls, le, plots/"confusion_matrix.png", best_clf_name)
        plot_imp(imp_clf, f"Мета-признаки — {best_clf_name} (классификация)",
                 plots/"feature_importance_clf.png")
        plot_imp(imp_reg, f"Мета-признаки — {best_reg_name} (регрессия)",
                 plots/"feature_importance_reg.png")
        plot_scatter(y_reg, yp_reg, plots/"gain_scatter.png", best_reg_name)

    # 9. Итог
    print(f"\n{'='*65}")
    print(f"Готово!  Датасетов: {len(y_cls)}  Мета-признаков: {len(mf_cols)}")
    print(f"\n  [Классификация] {best_clf_name}")
    print(f"    F1-macro (CV):  {clf_rep.iloc[0]['f1_macro_mean']:.4f} ±{clf_rep.iloc[0].get('f1_macro_std',0):.4f}")
    print(f"    Train accuracy: {(y_cls==yp_cls).mean():.4f}")
    print(f"\n  [Регрессия] {best_reg_name}")
    print(f"    R² (CV):  {reg_rep.iloc[0]['r2_mean']:.4f} ±{reg_rep.iloc[0].get('r2_std',0):.4f}")
    print(f"    MAE (CV): {reg_rep.iloc[0]['mae_mean']:.4f}")
    print(f"\n  Файлы в {out}/:")
    print(f"    best_clf.pkl / best_reg.pkl  ← модели")
    print(f"    clf_report.csv / reg_report.csv")
    print(f"    predictions.csv              ← всё с вероятностями")
    if not args.no_plots:
        print(f"    plots/                       ← 6 графиков")
    print(f"\n  Инференс:")
    print(f"    clf = joblib.load('{out}/best_clf.pkl')")
    print(f"    X_new = clf['scaler'].transform([mf_vector])")
    print(f"    method = clf['label_encoder'].inverse_transform(clf['model'].predict(X_new))[0]")
    print(f"    reg = joblib.load('{out}/best_reg.pkl')")
    print(f"    gain  = reg['model'].predict(X_new)[0]")


if __name__ == "__main__":
    main()