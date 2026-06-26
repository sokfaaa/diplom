"""
Расширение анализа для Главы 2: корреляции мета-признаков,
матрица IR x n_classes, мета-модель, rule-based алгоритм.

Запуск: python analysis_ch2_extension.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import scikit_posthocs as sp

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

PALETTE = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0",
           "#00BCD4","#FF5722","#607D8B","#8BC34A","#FFC107",
           "#3F51B5","#009688","#795548","#F44336"]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ── загрузка ─────────────────────────────────────────────────────────────────
full    = pd.read_csv("results_full.csv")
meta    = pd.read_csv("metafeatures.csv")
summary = pd.read_csv("results_summary.csv")

SAMPLERS_ORDER = summary.sort_values("rank_avg")["sampler"].tolist()
KEY_SAMPLERS   = ["ADASYN","SMOTE","SVMSMOTE","BorderlineSMOTE",
                  "KMeansSMOTE","SMOTETomek","RandomOverSampler",
                  "SMOTEENN","MWMOTE","AHC"]

# ── дельты ───────────────────────────────────────────────────────────────────
baseline  = full[full["sampler"]=="Baseline"][["dataset","f1_macro","g_mean","balanced_accuracy"]]
baseline  = baseline.rename(columns={"f1_macro":"f1_base",
                                     "g_mean":"gm_base",
                                     "balanced_accuracy":"ba_base"})
samplers  = [s for s in full["sampler"].unique() if s != "Baseline"]

delta_df  = baseline.copy()
for s in samplers:
    tmp = full[full["sampler"]==s][["dataset","f1_macro"]].rename(columns={"f1_macro":f"delta_{s}"})
    delta_df = delta_df.merge(tmp, on="dataset", how="left")
    delta_df[f"delta_{s}"] = delta_df[f"delta_{s}"] - delta_df["f1_base"]

delta_cols = [f"delta_{s}" for s in samplers]
delta_df["best_sampler"] = delta_df[delta_cols].idxmax(axis=1).str.replace("delta_","")

META_R = meta.rename(columns={"dataset_name":"dataset"})
df     = delta_df.merge(META_R, on="dataset", how="left")

FEATS  = ["IR","nr_class","nr_attr","nr_inst","CV","pIR_cv",
          "n1","n2.mean","n3.mean","f1.mean","f2.mean",
          "mut_inf.mean","class_ent","lsc"]
FEATS  = [f for f in FEATS if f in df.columns]

# ═══════════════════════════════════════════════════════════════════════════
# FIG 10 — Spearman-корреляции мета-признаков с ΔF1
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 10: корреляции мета-признаков …")

FEAT_LABELS = {
    "IR": "IR\n(дисбаланс)",
    "CV": "CV\n(вариация дисб.)",
    "nr_class": "Число классов",
    "nr_attr": "Число признаков",
    "nr_inst": "Число объектов",
    "pIR_cv": "pIR_cv",
    "n1": "N1\n(граница классов)",
    "n2.mean": "N2\n(расст. классов)",
    "n3.mean": "N3\n(ошибка 1-NN)",
    "f1.mean": "F1\n(наложение признак.)",
    "f2.mean": "F2\n(объём наложения)",
    "mut_inf.mean": "Взаимная\nинформация",
    "class_ent": "Энтропия\nклассов",
    "lsc": "LSC\n(лин. разделимость)",
}

corr_records = []
for feat in FEATS:
    row = {"feature": feat}
    for s in KEY_SAMPLERS:
        col = f"delta_{s}"
        valid = df[[feat, col]].dropna()
        if len(valid) > 15:
            r, _ = stats.spearmanr(valid[feat], valid[col])
            row[s] = r
        else:
            row[s] = np.nan
    corr_records.append(row)

corr_df = pd.DataFrame(corr_records).set_index("feature")

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(corr_df[KEY_SAMPLERS].values.T, aspect="auto",
               cmap="RdBu_r", vmin=-0.5, vmax=0.5)

ax.set_xticks(range(len(FEATS)))
ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in FEATS], fontsize=8)
ax.set_yticks(range(len(KEY_SAMPLERS)))
ax.set_yticklabels(KEY_SAMPLERS, fontsize=9)

# значения в ячейках
for i, feat in enumerate(FEATS):
    for j, s in enumerate(KEY_SAMPLERS):
        val = corr_df.loc[feat, s]
        if pd.notna(val):
            color = "white" if abs(val) > 0.28 else "black"
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Коэффициент корреляции Спирмена", fontsize=9)
ax.set_title("Корреляция мета-признаков датасетов с приростом ΔF1-macro\n(красный = положительная связь, синий = отрицательная)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_spearman_heatmap.png")
plt.close()
print(f"  [OK] {OUT}/fig10_spearman_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 11 — Топ-4 корреляции: scatter plots IR / class_ent vs ΔF1
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 11: scatter IR и class_ent …")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Зависимость прироста ΔF1-macro от ключевых мета-признаков", fontsize=12, fontweight="bold")

pairs = [
    ("IR",        "ADASYN",  "Imbalance Ratio",   PALETTE[0]),
    ("IR",        "SMOTE",   "Imbalance Ratio",   PALETTE[1]),
    ("class_ent", "SMOTE",   "Энтропия классов",  PALETTE[2]),
    ("n3.mean",   "ADASYN",  "N3 (ошибка 1-NN)",  PALETTE[3]),
]
for ax, (feat, samp, xlabel, color) in zip(axes.flat, pairs):
    col = f"delta_{samp}"
    sub = df[[feat, col]].dropna()
    ax.scatter(sub[feat], sub[col], alpha=0.55, color=color,
               edgecolors="white", s=45)
    # линия тренда
    z = np.polyfit(sub[feat], sub[col], 1)
    xline = np.linspace(sub[feat].min(), sub[feat].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color="black", linewidth=1.5,
            linestyle="--", alpha=0.7)
    r, p = stats.spearmanr(sub[feat], sub[col])
    ax.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(f"ΔF1 ({samp})", fontsize=9)
    ax.set_title(f"ρ = {r:.3f}, p = {p:.4f}", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig11_scatter_meta_vs_delta.png")
plt.close()
print(f"  [OK] {OUT}/fig11_scatter_meta_vs_delta.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 12 — Матрица IR × n_classes: средний ΔF1 и лучший метод
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 12: матрица IR × n_classes …")

IR_BINS   = [0, 2, 5, 10, float("inf")]
IR_LABELS = ["IR ≤ 2", "2 < IR ≤ 5", "5 < IR ≤ 10", "IR > 10"]
CLS_BINS  = [0, 2, 5, float("inf")]
CLS_LABELS= ["2 класса", "3–5 классов", "6+ классов"]

df["ir_group"]  = pd.cut(df["IR"],       bins=IR_BINS,  labels=IR_LABELS)
df["cls_group"] = pd.cut(df["nr_class"], bins=CLS_BINS, labels=CLS_LABELS)

# Для каждой ячейки — топ-3 метода (по среднему ΔF1 среди KEY_SAMPLERS)
key_cols = [f"delta_{s}" for s in KEY_SAMPLERS]

cell_data = {}
for ir in IR_LABELS:
    for cls in CLS_LABELS:
        sub = df[(df["ir_group"]==ir) & (df["cls_group"]==cls)]
        if len(sub) < 3:
            cell_data[(ir,cls)] = {"n": len(sub), "top": [], "mean_delta": np.nan}
            continue
        means = sub[key_cols].mean()
        means.index = means.index.str.replace("delta_","")
        top3 = means.nlargest(3)
        mean_best = sub[key_cols].max(axis=1).mean()
        cell_data[(ir,cls)] = {"n": len(sub), "top": list(zip(top3.index, top3.values)),
                               "mean_delta": mean_best}

# Матрица среднего ΔF1
delta_matrix = np.array([[cell_data[(ir,cls)]["mean_delta"]
                           for cls in CLS_LABELS] for ir in IR_LABELS],
                         dtype=float)
count_matrix = np.array([[cell_data[(ir,cls)]["n"]
                           for cls in CLS_LABELS] for ir in IR_LABELS])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Анализ в разрезе IR × Число классов", fontsize=13, fontweight="bold")

# Левая часть — heatmap среднего ΔF1
ax = axes[0]
im = ax.imshow(delta_matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=0.08)
ax.set_xticks(range(3)); ax.set_xticklabels(CLS_LABELS, fontsize=10)
ax.set_yticks(range(4)); ax.set_yticklabels(IR_LABELS, fontsize=10)
ax.set_title("Средний прирост ΔF1-macro\n(лучший метод в ячейке)", fontsize=11)
for i in range(4):
    for j in range(3):
        val = delta_matrix[i,j]
        n   = count_matrix[i,j]
        top = cell_data[(IR_LABELS[i], CLS_LABELS[j])]["top"]
        text = f"+{val:.3f}\n(n={n})" if not np.isnan(val) else f"n={n}"
        ax.text(j, i, text, ha="center", va="center",
                fontsize=9, color="black" if val < 0.05 else "white",
                fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8, label="Средний ΔF1")

# Правая часть — топ-3 метода в каждой ячейке
ax = axes[1]
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 3.5)
ax.set_xticks(range(3)); ax.set_xticklabels(CLS_LABELS, fontsize=10)
ax.set_yticks(range(4)); ax.set_yticklabels(IR_LABELS, fontsize=10)
ax.set_title("Топ-3 рекомендуемых метода в ячейке\n(по среднему ΔF1)", fontsize=11)
ax.grid(True, alpha=0.3)
for i, ir in enumerate(IR_LABELS):
    for j, cls in enumerate(CLS_LABELS):
        top = cell_data[(ir,cls)]["top"]
        if not top:
            ax.text(j, 3-i, "—", ha="center", va="center", fontsize=9)
            continue
        lines = "\n".join([f"{k+1}. {name} (+{v:.3f})"
                           for k, (name,v) in enumerate(top)])
        ax.text(j, 3-i, lines, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF",
                          edgecolor="#AAAAAA", alpha=0.9))

plt.tight_layout()
plt.savefig(f"{OUT}/fig12_ir_nclass_matrix.png")
plt.close()
print(f"  [OK] {OUT}/fig12_ir_nclass_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 13 — ΔF1 по группам IR (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 13: прирост ΔF1 по группам IR …")

plot_samplers = ["SMOTE","ADASYN","SVMSMOTE","BorderlineSMOTE",
                 "KMeansSMOTE","SMOTETomek","RandomOverSampler"]
rows = []
for ir in IR_LABELS:
    sub = df[df["ir_group"]==ir]
    for s in plot_samplers:
        col = f"delta_{s}"
        rows.append({"ir_group": ir, "sampler": s,
                     "mean": sub[col].mean(), "se": sub[col].sem()})
ir_plot = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(IR_LABELS))
n_s    = len(plot_samplers)
width  = 0.11
offset = np.linspace(-(n_s-1)/2*width, (n_s-1)/2*width, n_s)

for k, (s, off) in enumerate(zip(plot_samplers, offset)):
    sub  = ir_plot[ir_plot["sampler"]==s]
    bars = ax.bar(x + off, sub["mean"].values, width,
                  color=PALETTE[k], alpha=0.82, label=s, edgecolor="white")
    ax.errorbar(x + off, sub["mean"].values, yerr=sub["se"].values,
                fmt="none", color="black", capsize=2, linewidth=1)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(IR_LABELS, fontsize=10)
ax.set_ylabel("Средний прирост ΔF1-macro (vs Baseline)")
ax.set_title("Прирост F1-macro относительно Baseline\nв зависимости от уровня дисбаланса (IR)", fontsize=12)
ax.legend(fontsize=8, ncol=4, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig13_delta_f1_by_ir.png")
plt.close()
print(f"  [OK] {OUT}/fig13_delta_f1_by_ir.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 14 — CD-диаграмма (критические различия, тест Неменьи)
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 14: CD-диаграмма …")

pivot = full.pivot_table(index="dataset", columns="sampler", values="f1_macro")
pivot_clean = pivot.dropna()   # только датасеты без NaN

# Ранги
ranks_df = pivot_clean.rank(axis=1, ascending=False)
avg_ranks = ranks_df.mean().sort_values()

# Post-hoc Nemenyi
ph = sp.posthoc_nemenyi_friedman(pivot_clean.values)
ph.index   = pivot_clean.columns
ph.columns = pivot_clean.columns

# Количество датасетов и методов
n_ds = len(pivot_clean)
n_alg = len(pivot_clean.columns)
# Критическое расстояние (α=0.05)
q_alpha = 2.569   # для k=15 α=0.05 (таблица)
import math
cd = q_alpha * math.sqrt(n_alg * (n_alg + 1) / (6 * n_ds))
print(f"  n_datasets={n_ds}, n_alg={n_alg}, CD={cd:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
methods = avg_ranks.index.tolist()
ranks   = avg_ranks.values

# Горизонтальная ось рангов
ax.set_xlim(0.5, n_alg + 0.5)
ax.set_ylim(-1, len(methods) + 1)
ax.set_xlabel("Средний ранг (меньше = лучше)", fontsize=11)
ax.set_title(f"CD-диаграмма (тест Неменьи, α=0.05, CD={cd:.2f})\nn={n_ds} датасетов", fontsize=11)

# Горизонтальная линия со значениями
ax.axhline(len(methods), color="black", linewidth=1.5)
for r in np.arange(1, n_alg+1, 1):
    ax.axvline(r, color="grey", linewidth=0.3, linestyle=":")
    ax.text(r, len(methods)+0.3, str(int(r)), ha="center", fontsize=8, color="grey")

# Методы
colors_cd = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]
for i, (m, r) in enumerate(zip(methods, ranks)):
    ax.plot([r, r], [len(methods), i], color=colors_cd[i],
            linewidth=1.5, linestyle="--", alpha=0.5)
    ax.scatter([r], [i], color=colors_cd[i], s=80, zorder=5)
    ax.text(r + 0.05, i, f"{m} ({r:.2f})", va="center", fontsize=8.5,
            color=colors_cd[i], fontweight="bold" if m in ["ADASYN","SMOTE"] else "normal")

# CD-скобка
cd_y = -0.6
ax.annotate("", xy=(ranks[0]+cd, cd_y), xytext=(ranks[0], cd_y),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2))
ax.text(ranks[0]+cd/2, cd_y-0.35, f"CD={cd:.2f}", ha="center",
        fontsize=9, color="red")

# Группы незначимых различий (p > 0.05)
drawn = set()
for i, m1 in enumerate(methods):
    group = [m1]
    for m2 in methods[i+1:]:
        if ph.loc[m1, m2] > 0.05:
            group.append(m2)
    if len(group) > 1:
        key = tuple(sorted(group))
        if key not in drawn:
            drawn.add(key)
            idxs = [methods.index(m) for m in group]
            rmin = min(ranks[methods.index(m)] for m in group)
            rmax = max(ranks[methods.index(m)] for m in group)
            y_line = max(idxs) + 0.4
            ax.plot([rmin, rmax], [y_line, y_line], color="navy",
                    linewidth=3, alpha=0.35, solid_capstyle="round")

ax.axis("off")
ax.set_xlim(0, n_alg + 1)
plt.tight_layout()
plt.savefig(f"{OUT}/fig14_cd_diagram.png")
plt.close()
print(f"  [OK] {OUT}/fig14_cd_diagram.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 15 — Мета-модель: важность признаков + LOO accuracy
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 15: мета-модель …")

df_model = df[FEATS + ["best_sampler"]].dropna()
print(f"  Датасетов для мета-модели: {len(df_model)}")

X  = df_model[FEATS].values
le = LabelEncoder()
y  = le.fit_transform(df_model["best_sampler"])

# LOO
rf   = RandomForestClassifier(n_estimators=300, random_state=42)
loo  = LeaveOneOut()
pred_loo, true_loo = [], []
for tr, te in loo.split(X):
    rf.fit(X[tr], y[tr])
    pred_loo.append(rf.predict(X[te])[0])
    true_loo.append(y[te][0])
acc_loo = accuracy_score(true_loo, pred_loo)
print(f"  LOO точность (14 классов): {acc_loo:.3f}")

# Обучаем на всём для важностей
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
FEAT_SHORT = {
    "IR":"IR","CV":"CV","nr_class":"Число классов","nr_attr":"Число признаков",
    "nr_inst":"Число объектов","pIR_cv":"pIR_cv","n1":"N1","n2.mean":"N2",
    "n3.mean":"N3 (1-NN ошибка)","f1.mean":"F1 (наложение)",
    "f2.mean":"F2 (объём перекр.)","mut_inf.mean":"Взаимная инф.",
    "class_ent":"Энтропия классов","lsc":"LSC",
}

# Regression LOO для ключевых методов
reg_results = []
for s in ["SMOTE","ADASYN","SVMSMOTE","BorderlineSMOTE","SMOTETomek"]:
    col = f"delta_{s}"
    df_r = df[FEATS + [col]].dropna()
    Xr, yr = df_r[FEATS].values, df_r[col].values
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    preds_r = []
    for tr, te in LeaveOneOut().split(Xr):
        reg.fit(Xr[tr], yr[tr])
        preds_r.append(reg.predict(Xr[te])[0])
    mae = mean_absolute_error(yr, preds_r)
    base_mae = mean_absolute_error(yr, np.full_like(yr, yr.mean()))
    r_skill = 1 - mae / base_mae
    reg_results.append({"method": s, "MAE": mae, "Baseline MAE": base_mae, "Skill": r_skill})

reg_df = pd.DataFrame(reg_results)
print("\n  Регрессионная мета-модель (GBR, LOO):")
print(reg_df.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Мета-модель для рекомендации метода оверсэмплинга", fontsize=12, fontweight="bold")

# Важность признаков
ax = axes[0]
colors_imp = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
ax.barh([FEAT_SHORT.get(f,f) for f in imp.index], imp.values,
        color=colors_imp, alpha=0.82, edgecolor="white")
ax.set_xlabel("Важность (Gini)")
ax.set_title(f"Важность мета-признаков (RF)\nLOO accuracy = {acc_loo:.1%}", fontsize=10)

# Skill регрессионных моделей
ax = axes[1]
colors_r = [PALETTE[i] for i in range(len(reg_df))]
bars = ax.bar(reg_df["method"], reg_df["Skill"], color=colors_r, alpha=0.82, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
for bar, val in zip(bars, reg_df["Skill"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Skill score (1 − MAE/MAE_baseline)")
ax.set_title("Качество прогноза ΔF1 для каждого метода\n(GBR регрессия, LOO валидация)", fontsize=10)
ax.set_xticklabels(reg_df["method"], rotation=25, ha="right")

plt.tight_layout()
plt.savefig(f"{OUT}/fig15_metamodel.png")
plt.close()
print(f"  [OK] {OUT}/fig15_metamodel.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 16 — Rule-based алгоритм: визуализация + валидация
# ═══════════════════════════════════════════════════════════════════════════
print(">>> Fig 16: rule-based визуализация …")

def rule_recommend(row):
    ir  = row["IR"]
    nc  = row["nr_class"] if pd.notna(row["nr_class"]) else 2
    n3  = row["n3.mean"]  if pd.notna(row["n3.mean"])  else 0.20
    if ir <= 2:
        return "RandomOverSampler / SMOTE"
    elif ir <= 5:
        if n3 > 0.30:
            return "ADASYN"
        else:
            return "SMOTE"
    elif ir <= 10:
        return "ADASYN"
    else:
        if nc <= 2:
            return "SVMSMOTE"
        else:
            return "KMeansSMOTE"

df["rule_rec"] = df.apply(rule_recommend, axis=1)

# Валидация: % случаев, где рекомендованный метод бьёт Baseline
RULE_SCORE_MAP = {
    "RandomOverSampler / SMOTE": ["RandomOverSampler","SMOTE"],
    "SMOTE":         ["SMOTE"],
    "ADASYN":        ["ADASYN"],
    "SVMSMOTE":      ["SVMSMOTE"],
    "KMeansSMOTE":   ["KMeansSMOTE"],
}
def beats_baseline(row):
    rec = row["rule_rec"]
    methods = RULE_SCORE_MAP.get(rec, [rec])
    return any(row.get(f"delta_{m}", -999) > 0.001 for m in methods)

df["rule_beats"] = df.apply(beats_baseline, axis=1)
validation = df.groupby("rule_rec")["rule_beats"].agg(["mean","count"]).reset_index()
validation.columns = ["Рекомендация","% побед","n"]
validation["% побед"] = (validation["% побед"]*100).round(1)
print("\n  Валидация rule-based алгоритма:")
print(validation.to_string(index=False))

overall = df["rule_beats"].mean()
print(f"\n  Общий % случаев, где рекомендованный метод > Baseline: {overall:.1%}")

# Дерево решений — схема
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Алгоритм рекомендации метода оверсэмплинга\n(rule-based подход)", fontsize=12, fontweight="bold")

def box(ax, x, y, text, color="#E3F2FD", w=1.8, h=0.7, fontsize=8.5):
    ax.add_patch(mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="#555", linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)

def arrow(ax, x1, y1, x2, y2, label="", color="black"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my, label, fontsize=7.5, color="grey")

# Узлы
box(ax, 5, 6.3, "СТАРТ\nВычислить IR, n_classes, n3.mean", color="#FFF9C4", w=3.2, h=0.8, fontsize=9)
box(ax, 5, 5.0, "IR ≤ 2?", color="#E1F5FE", w=2.0)
box(ax, 2, 3.8, "ВЫВОД:\nRandomOverSampler\nили SMOTE", color="#C8E6C9", h=0.9, fontsize=8)
box(ax, 5, 3.6, "IR ≤ 5?", color="#E1F5FE", w=2.0)
box(ax, 5, 2.3, "n3.mean > 0.3?", color="#E1F5FE", w=2.2)
box(ax, 3.2, 1.1, "ВЫВОД:\nADASYN", color="#C8E6C9")
box(ax, 6.8, 1.1, "ВЫВОД:\nSMOTE", color="#C8E6C9")
box(ax, 8, 3.6, "IR ≤ 10?", color="#E1F5FE", w=2.0)
box(ax, 8, 2.3, "ВЫВОД:\nADASYN", color="#C8E6C9")
box(ax, 8, 1.1, "n_classes ≤ 2?", color="#E1F5FE", w=2.2)
box(ax, 6.6, 0.2, "ВЫВОД:\nSVMSMOTE", color="#C8E6C9", h=0.6)
box(ax, 9.4, 0.2, "ВЫВОД:\nKMeansSMOTE", color="#C8E6C9", h=0.6)

# Стрелки
arrow(ax, 5, 5.9, 5, 5.35)
arrow(ax, 5, 4.65, 2, 4.15, "Да")
arrow(ax, 5, 4.65, 5, 3.95, "Нет")
arrow(ax, 5, 3.25, 5, 2.65)
arrow(ax, 5, 1.95, 3.2, 1.45, "Да")
arrow(ax, 5, 1.95, 6.8, 1.45, "Нет")
arrow(ax, 5, 3.25, 8, 3.95, "")
arrow(ax, 8, 3.25, 8, 2.65, "Да")
arrow(ax, 8, 3.25, 8, 1.45, "Нет")
arrow(ax, 8, 0.8, 6.6, 0.5, "Да")
arrow(ax, 8, 0.8, 9.4, 0.5, "Нет")

plt.tight_layout()
plt.savefig(f"{OUT}/fig16_rule_based_tree.png")
plt.close()
print(f"  [OK] {OUT}/fig16_rule_based_tree.png")

# ═══════════════════════════════════════════════════════════════════════════
# ТАБЛИЦЫ для диплома
# ═══════════════════════════════════════════════════════════════════════════
print("\n>>> Сохранение вспомогательных таблиц …")

# Spearman
corr_df.round(3).to_csv(f"{OUT}/table5_spearman_correlations.csv")

# IR x n_classes (ΔF1 и топ-метод)
rows_heat = []
for ir in IR_LABELS:
    for cls in CLS_LABELS:
        d = cell_data[(ir,cls)]
        top1 = d["top"][0][0] if d["top"] else "—"
        rows_heat.append({"IR группа": ir, "Классы": cls,
                          "n": d["n"],
                          "Средний ΔF1 (лучш.)": round(d["mean_delta"],4) if not np.isnan(d["mean_delta"]) else "—",
                          "Топ-1 метод": top1,
                          "Топ-2": d["top"][1][0] if len(d["top"])>1 else "—",
                          "Топ-3": d["top"][2][0] if len(d["top"])>2 else "—"})
pd.DataFrame(rows_heat).to_csv(f"{OUT}/table6_ir_nclass_matrix.csv", index=False)

# Мета-модель
reg_df.round(4).to_csv(f"{OUT}/table7_metamodel_regression.csv", index=False)
pd.DataFrame({"feature":imp.index,"importance":imp.values}).round(4).to_csv(f"{OUT}/table8_feature_importance.csv", index=False)

# Rule-based валидация
validation.to_csv(f"{OUT}/table9_rule_validation.csv", index=False)

print("\n" + "="*55)
print("ИТОГО: все файлы сохранены в", OUT)
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")