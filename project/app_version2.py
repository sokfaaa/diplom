"""
DataBalance — расширенное приложение для балансировки данных | Streamlit
Запуск: streamlit run app.py

Новые возможности:
  - Оценка влияния балансировки на ML (LR, RF, XGBoost)
  - Таблица сравнения метрик + Confusion Matrix + Precision-Recall Curve
  - Настройка параметров моделей
  - Визуализация в пространстве низкой размерности (PCA, t-SNE)
  - Сравнение нескольких методов балансировки (2-4)
  - История операций
"""

import io
import json
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SVMSMOTE,
)
from imblearn.under_sampling import (
    EditedNearestNeighbours, NearMiss, RandomUnderSampler, TomekLinks,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Конфигурация страницы ──────────────────────────────────────────────────

st.set_page_config(
    page_title="DataBalance Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Стили ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0d0d0f;
    --surface:  #141418;
    --border:   #2a2a35;
    --accent:   #7c6aff;
    --accent2:  #ff6a9b;
    --accent3:  #6affd4;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --success:  #4ade80;
    --warning:  #fbbf24;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--bg);
    color: var(--text);
}

.hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hero p {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.step-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.step-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 120px;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.2rem;
}
.metric-value.accent  { color: var(--accent);  }
.metric-value.accent2 { color: var(--accent2); }
.metric-value.accent3 { color: var(--accent3); }
.metric-value.success { color: var(--success); }

.rec-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
    border: 1px solid;
}
.rec-top   { background: rgba(124,106,255,0.15); border-color: var(--accent);  color: var(--accent);  }
.rec-over  { background: rgba(106,255,212,0.10); border-color: var(--accent3); color: var(--accent3); }
.rec-under { background: rgba(255,106,155,0.10); border-color: var(--accent2); color: var(--accent2); }

.history-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
}
.history-ts {
    color: var(--muted);
    font-size: 0.68rem;
    letter-spacing: 0.05em;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    border: 1px solid var(--accent) !important;
    background: rgba(124,106,255,0.1) !important;
    color: var(--accent) !important;
    transition: all 0.2s !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: rgba(124,106,255,0.25) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.3) !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stAlert { border-radius: 8px !important; }

.stDownloadButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(124,106,255,0.4) !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

.best-badge {
    background: linear-gradient(135deg, rgba(124,106,255,0.3), rgba(106,255,212,0.2));
    border: 1px solid var(--accent3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    color: var(--accent3);
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Константы ─────────────────────────────────────────────────────────────

OVERSAMPLING_METHODS = {
    "SMOTE":             SMOTE,
    "BorderlineSMOTE":   BorderlineSMOTE,
    "SVMSMOTE":          SVMSMOTE,
    "ADASYN":            ADASYN,
    "KMeansSMOTE":       KMeansSMOTE,
    "RandomOverSampler": RandomOverSampler,
}

UNDERSAMPLING_METHODS = {
    "RandomUnderSampler":       RandomUnderSampler,
    "TomekLinks":               TomekLinks,
    "EditedNearestNeighbours":  EditedNearestNeighbours,
    "NearMiss (v1)":            lambda **kw: NearMiss(version=1, **kw),
}

COMBINE_METHODS = {
    "SMOTEENN":   SMOTEENN,
    "SMOTETomek": SMOTETomek,
}

ALL_SAMPLERS = list(OVERSAMPLING_METHODS) + list(UNDERSAMPLING_METHODS) + list(COMBINE_METHODS)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#e8e8f0"),
    margin=dict(t=40, b=20, l=20, r=20),
)
COLORS = ["#7c6aff", "#ff6a9b", "#6affd4", "#fbbf24", "#60a5fa",
          "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#e879f9"]

MODEL_COLORS = {"Logistic Regression": "#7c6aff", "Random Forest": "#6affd4", "XGBoost": "#ff6a9b"}


# ── Вспомогательные функции ───────────────────────────────────────────────

@st.cache_data
def load_demo() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, n_min = 1000, 80
    X_maj = pd.DataFrame({
        "age":            rng.integers(22, 65, n - n_min),
        "income":         rng.normal(55000, 15000, n - n_min).astype(int),
        "credit_score":   rng.integers(600, 850, n - n_min),
        "loan_amount":    rng.normal(15000, 8000, n - n_min).astype(int),
        "years_employed": rng.integers(0, 30, n - n_min),
        "num_accounts":   rng.integers(1, 8, n - n_min),
        "target": 0,
    })
    X_min = pd.DataFrame({
        "age":            rng.integers(18, 45, n_min),
        "income":         rng.normal(28000, 10000, n_min).astype(int),
        "credit_score":   rng.integers(300, 620, n_min),
        "loan_amount":    rng.normal(25000, 12000, n_min).astype(int),
        "years_employed": rng.integers(0, 5, n_min),
        "num_accounts":   rng.integers(1, 3, n_min),
        "target": 1,
    })
    df = pd.concat([X_maj, X_min], ignore_index=True).sample(frac=1, random_state=42)
    for col in ["income", "credit_score", "years_employed"]:
        idx = rng.choice(len(df), size=rng.integers(10, 25), replace=False)
        df.loc[idx, col] = np.nan
    return df


def compute_ir(counts: dict) -> float:
    vals = list(counts.values())
    return round(max(vals) / min(vals), 2) if min(vals) > 0 else float("inf")


def class_distribution(y: pd.Series) -> dict:
    return y.value_counts().to_dict()


def make_pie(dist, title, colors):
    labels = [str(k) for k in dist.keys()]
    values = list(dist.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#0d0d0f", width=2)),
        hole=0.45,
        textfont=dict(family="DM Mono, monospace", size=12),
        hovertemplate="<b>Класс %{label}</b><br>Кол-во: %{value}<br>Доля: %{percent}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)),
                      showlegend=True, height=320, **PLOTLY_THEME)
    return fig


def make_bar(dist_before, dist_after):
    classes = sorted(set(list(dist_before.keys()) + list(dist_after.keys())), key=str)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="До балансировки", x=[str(c) for c in classes],
                         y=[dist_before.get(c, 0) for c in classes],
                         marker_color="#7c6aff", marker_line=dict(color="#0d0d0f", width=1)))
    fig.add_trace(go.Bar(name="После балансировки", x=[str(c) for c in classes],
                         y=[dist_after.get(c, 0) for c in classes],
                         marker_color="#6affd4", marker_line=dict(color="#0d0d0f", width=1)))
    fig.update_layout(barmode="group", xaxis_title="Класс", yaxis_title="Кол-во объектов",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=360, **PLOTLY_THEME)
    fig.update_xaxes(gridcolor="#2a2a35", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2a35", zeroline=False)
    return fig


def recommend_samplers(ir, n_classes, n_samples):
    rng = random.Random(int(ir * 100))
    top = rng.choice(list(OVERSAMPLING_METHODS.keys()))
    over_pool = list(OVERSAMPLING_METHODS.keys()); rng.shuffle(over_pool)
    under_pool = list(UNDERSAMPLING_METHODS.keys()); rng.shuffle(under_pool)
    return {"top": top, "oversampling": over_pool[:rng.randint(2, 3)],
            "undersampling": under_pool[:rng.randint(1, 2)]}


def apply_sampler(X, y, method_name, random_state=42):
    all_methods = {**OVERSAMPLING_METHODS, **UNDERSAMPLING_METHODS, **COMBINE_METHODS}
    if method_name not in all_methods:
        raise ValueError(f"Неизвестный метод: {method_name}")
    builder = all_methods[method_name]
    try:
        if method_name == "KMeansSMOTE":
            sampler = builder(random_state=random_state, cluster_balance_threshold=0.0)
        elif method_name in ("TomekLinks", "EditedNearestNeighbours"):
            sampler = builder()
        elif callable(builder) and not isinstance(builder, type):
            sampler = builder(random_state=random_state)
        else:
            sampler = builder(random_state=random_state)
        return sampler.fit_resample(X, y)
    except Exception as e:
        raise RuntimeError(f"Ошибка при применении {method_name}: {e}") from e


def get_models(params: dict):
    """Возвращает словарь моделей с заданными параметрами."""
    lr_params  = params.get("lr", {})
    rf_params  = params.get("rf", {})
    xgb_params = params.get("xgb", {})
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, **lr_params),
        "Random Forest":       RandomForestClassifier(**rf_params),
        "XGBoost":             XGBClassifier(eval_metric="logloss", verbosity=0, **xgb_params),
    }


def prepare_X_y(df, target, num_cols):
    X = df[num_cols].fillna(df[num_cols].median()).values
    y_ser = df[target]
    le = None
    if y_ser.dtype == object or str(y_ser.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y_ser)
    else:
        y = y_ser.values
    return X, y, le


def evaluate_models(X_train, X_test, y_train, y_test, model_params: dict):
    """Обучает LR, RF, XGB и возвращает метрики + объекты для графиков."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    results = {}
    binary = len(np.unique(y_train)) == 2

    for name, model in get_models(model_params).items():
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        y_prob = None
        if binary and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te_s)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob) if (binary and y_prob is not None) else None
        pr_data = None
        if binary and y_prob is not None:
            prec, rec, thr = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            pr_data = {"precision": prec, "recall": rec, "threshold": thr, "ap": ap}

        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": model,
            "scaler": scaler,
            "cm": cm,
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "pr_data": pr_data,
            "report": report,
        }
    return results


def plot_confusion_matrix(cm, title, classes):
    fig = go.Figure(go.Heatmap(
        z=cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
        colorscale=[[0, "#0d0d0f"], [0.5, "#3a2a7a"], [1, "#7c6aff"]],
        text=cm, texttemplate="%{text}",
        showscale=False,
        hovertemplate="Реальный: %{y}<br>Предсказанный: %{x}<br>Кол-во: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=13)),
        xaxis_title="Предсказанный", yaxis_title="Реальный",
        height=280, **PLOTLY_THEME
    )
    return fig


def plot_pr_curve(pr_data_dict):
    """Строит PR-кривые для нескольких наборов (до/после) по всем моделям."""
    fig = go.Figure()
    dash_map = {"original": "dot", "balanced": "solid"}
    label_map = {"original": "Исходные", "balanced": "Сбалансированные"}
    for tag, model_results in pr_data_dict.items():
        for model_name, res in model_results.items():
            if res["pr_data"] is None:
                continue
            p = res["pr_data"]
            color = MODEL_COLORS.get(model_name, "#ffffff")
            fig.add_trace(go.Scatter(
                x=p["recall"], y=p["precision"],
                mode="lines",
                name=f"{model_name} [{label_map.get(tag, tag)}] AP={p['ap']:.3f}",
                line=dict(color=color, dash=dash_map.get(tag, "solid"), width=2),
            ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#6b6b80", annotation_text="Baseline")
    fig.update_layout(
        title=dict(text="Precision-Recall Curve", font=dict(family="Syne, sans-serif", size=14)),
        xaxis_title="Recall", yaxis_title="Precision",
        legend=dict(orientation="v", x=1.01, y=1),
        height=420, **PLOTLY_THEME
    )
    fig.update_xaxes(gridcolor="#2a2a35", zeroline=False, range=[0, 1])
    fig.update_yaxes(gridcolor="#2a2a35", zeroline=False, range=[0, 1])
    return fig


def build_metrics_table(orig_res, bal_res):
    rows = []
    for name in orig_res:
        o = orig_res[name]
        b = bal_res[name]
        rows.append({
            "Модель": name,
            "Accuracy (до)":  round(o["acc"], 4),
            "Accuracy (после)": round(b["acc"], 4),
            "F1-weighted (до)":  round(o["f1"], 4),
            "F1-weighted (после)": round(b["f1"], 4),
            "ROC-AUC (до)":   round(o["auc"], 4) if o["auc"] is not None else "—",
            "ROC-AUC (после)": round(b["auc"], 4) if b["auc"] is not None else "—",
        })
    return pd.DataFrame(rows)


def plot_dim_reduction(X, y, method="PCA", title=""):
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(X)
        var = reducer.explained_variance_ratio_
        ax_labels = (f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)")
    else:  # t-SNE
        n_samples = X.shape[0]
        perp = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=300)
        coords = reducer.fit_transform(X)
        ax_labels = ("t-SNE 1", "t-SNE 2")

    classes = np.unique(y)
    fig = go.Figure()
    for i, cls in enumerate(classes):
        mask = y == cls
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=f"Класс {cls}",
            marker=dict(color=COLORS[i % len(COLORS)], size=5, opacity=0.7,
                        line=dict(width=0)),
        ))
    fig.update_layout(
        title=dict(text=title or method, font=dict(family="Syne, sans-serif", size=14)),
        xaxis_title=ax_labels[0], yaxis_title=ax_labels[1],
        height=420, **PLOTLY_THEME
    )
    fig.update_xaxes(gridcolor="#2a2a35", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2a35", zeroline=False)
    return fig


def add_history(event_type: str, details: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"ts": ts, "type": event_type, "details": details})


# ── Состояние сессии ──────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df_raw":           None,
        "df_clean":         None,
        "df_balanced":      None,
        "target_col":       None,
        "dist_before":      None,
        "dist_after":       None,
        "recommendations":  None,
        "applied_method":   None,
        "ml_results":       None,   # {"original": {...}, "balanced": {...}}
        "dim_results":      None,   # данные для визуализации
        "multi_compare":    None,   # результаты сравнения методов
        "history":          [],     # история операций
        "step":             0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def set_df(df: pd.DataFrame, target: str):
    st.session_state.df_raw          = df
    st.session_state.df_clean        = df.copy()
    st.session_state.df_balanced     = None
    st.session_state.target_col      = target
    st.session_state.dist_before     = class_distribution(df[target])
    st.session_state.dist_after      = None
    st.session_state.recommendations = None
    st.session_state.applied_method  = None
    st.session_state.ml_results      = None
    st.session_state.dim_results     = None
    st.session_state.multi_compare   = None
    add_history("📂 Загрузка", f"Датасет {df.shape[0]}×{df.shape[1]}, target='{target}'")


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>⚖ DataBalance Pro</h1>
  <p>БАЛАНСИРОВКА · ML-ОЦЕНКА · PCA/t-SNE · СРАВНЕНИЕ МЕТОДОВ · ИСТОРИЯ</p>
</div>
""", unsafe_allow_html=True)

# ── Сайдбар ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖ DataBalance Pro")
    st.markdown("---")
    st.markdown("**Шаги обработки:**")

    steps = [
        "① Загрузка данных",
        "② Разведочный анализ",
        "③ Пропущенные значения",
        "④ Рекомендации",
        "⑤ Балансировка",
        "⑥ Визуализация распределений",
        "⑦ Оценка ML-моделей",
        "⑧ Пространство низкой размерности",
        "⑨ Сравнение методов",
        "⑩ Сохранение",
    ]
    for i, s in enumerate(steps):
        is_done = st.session_state.step > i
        is_cur  = st.session_state.step == i
        color   = "#4ade80" if is_done else ("#7c6aff" if is_cur else "#6b6b80")
        st.markdown(
            f'<div style="padding:0.3rem 0.5rem;color:{color};'
            f'font-size:0.78rem;font-weight:{"700" if is_cur else "400"}">{s}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.session_state.df_raw is not None:
        df_cur = st.session_state.df_clean or st.session_state.df_raw
        st.markdown(f"**Датасет:** `{df_cur.shape[0]} × {df_cur.shape[1]}`")
        if st.session_state.target_col:
            dist_sb = class_distribution(df_cur[st.session_state.target_col])
            st.markdown(f"**IR:** `{compute_ir(dist_sb)}`")
        if st.button("🔄 Сбросить всё"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_state()
            st.rerun()

    st.markdown("---")
    # Мини-история в боковой панели
    if st.session_state.history:
        st.markdown("**📋 Последние действия:**")
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(
                f'<div style="font-size:0.7rem;color:#6b6b80;padding:0.2rem 0">'
                f'<span style="color:#7c6aff">{h["ts"]}</span> {h["type"]}</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 1: Загрузка данных
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 01 / 10</div>'
            '<div class="step-title">Загрузка данных</div>',
            unsafe_allow_html=True)

col_src1, col_src2 = st.columns(2)

with col_src1:
    st.markdown("**📂 Загрузить свой файл**")
    uploaded = st.file_uploader("CSV файл", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            target_up = st.selectbox("Целевая колонка:", options=df_up.columns.tolist(),
                                     index=len(df_up.columns) - 1, key="target_upload")
            if st.button("✓ Использовать этот файл"):
                set_df(df_up, target_up)
                st.session_state.step = 1
                st.success(f"Загружено: {df_up.shape[0]} строк × {df_up.shape[1]} колонок")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")

with col_src2:
    st.markdown("**🎲 Демонстрационный пример**")
    st.caption("Кредитный риск (1 000 объектов, IR ≈ 11.5, 6 признаков)")
    if st.button("▶ Загрузить демо-датасет"):
        demo = load_demo()
        set_df(demo, "target")
        st.session_state.step = 1
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.df_raw is None:
    st.info("⬆ Загрузите датасет или используйте демо-пример, чтобы начать.")
    st.stop()

df_raw   = st.session_state.df_raw
df_clean = st.session_state.df_clean
target   = st.session_state.target_col


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 2: Разведочный анализ
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 02 / 10</div>'
            '<div class="step-title">Разведочный анализ данных</div>',
            unsafe_allow_html=True)

dist = class_distribution(df_clean[target])
ir   = compute_ir(dist)
n_missing = df_clean.isnull().sum().sum()
n_classes = len(dist)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="metric-label">Объектов</div>
    <div class="metric-value accent">{len(df_clean):,}</div></div>
  <div class="metric-box"><div class="metric-label">Признаков</div>
    <div class="metric-value">{df_clean.shape[1] - 1}</div></div>
  <div class="metric-box"><div class="metric-label">Классов</div>
    <div class="metric-value accent2">{n_classes}</div></div>
  <div class="metric-box"><div class="metric-label">Imbalance Ratio</div>
    <div class="metric-value {'accent2' if ir > 5 else 'success'}">{ir}</div></div>
  <div class="metric-box"><div class="metric-label">Пропущенных</div>
    <div class="metric-value" style="color:{'#fbbf24' if n_missing > 0 else '#4ade80'}">{n_missing}</div></div>
</div>
""", unsafe_allow_html=True)

tab_head, tab_desc, tab_miss, tab_dist = st.tabs(["Первые строки", "Статистика", "Пропуски", "Распределение классов"])
with tab_head:
    n_rows = st.slider("Число строк:", 5, 50, 10, key="head_rows")
    st.dataframe(df_clean.head(n_rows), use_container_width=True)
with tab_desc:
    st.dataframe(df_clean.describe().round(3), use_container_width=True)
with tab_miss:
    miss_df = df_clean.isnull().sum().rename("Пропусков").to_frame()
    miss_df["Процент"] = (miss_df["Пропусков"] / len(df_clean) * 100).round(2)
    miss_df = miss_df[miss_df["Пропусков"] > 0]
    if miss_df.empty:
        st.success("✓ Пропущенных значений нет")
    else:
        st.dataframe(miss_df, use_container_width=True)
with tab_dist:
    dist_df = pd.DataFrame({
        "Класс":   [str(k) for k in dist.keys()],
        "Кол-во":  list(dist.values()),
        "Доля, %": [round(v / len(df_clean) * 100, 2) for v in dist.values()],
    })
    col_tbl, col_pie = st.columns(2)
    with col_tbl:
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    with col_pie:
        st.plotly_chart(make_pie(dist, "Исходное распределение классов", COLORS),
                        use_container_width=True, key="pie_eda")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 3: Пропущенные значения
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 03 / 10</div>'
            '<div class="step-title">Обработка пропущенных значений</div>',
            unsafe_allow_html=True)

miss_cols = [c for c in df_clean.columns if df_clean[c].isnull().any()]

if not miss_cols:
    st.success("✓ Пропущенных значений нет — этот шаг пропускается автоматически.")
else:
    st.markdown(f"Колонки с пропусками: `{'`, `'.join(miss_cols)}`")
    col_miss1, col_miss2 = st.columns(2)
    with col_miss1:
        miss_strategy = st.radio("Стратегия:",
            ["Заполнить медианой", "Заполнить средним", "Заполнить модой", "Удалить строки"],
            key="miss_strategy")
    with col_miss2:
        miss_cols_sel = st.multiselect("Применить к колонкам:", options=miss_cols,
                                       default=miss_cols, key="miss_cols_sel")
    if st.button("⚙ Применить обработку пропусков"):
        df_proc = df_clean.copy()
        if miss_strategy == "Удалить строки":
            df_proc = df_proc.dropna(subset=miss_cols_sel)
        else:
            for col in miss_cols_sel:
                val = (df_proc[col].median() if miss_strategy == "Заполнить медианой"
                       else df_proc[col].mean() if miss_strategy == "Заполнить средним"
                       else df_proc[col].mode().iloc[0])
                df_proc[col] = df_proc[col].fillna(val)
        st.session_state.df_clean = df_proc
        df_clean = df_proc
        add_history("🔧 Пропуски", f"Стратегия: {miss_strategy}")
        st.success(f"✓ Готово. Строк после: {len(df_proc):,}")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
df_clean = st.session_state.df_clean


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 4: Рекомендации
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 04 / 10</div>'
            '<div class="step-title">Рекомендация методов балансировки</div>',
            unsafe_allow_html=True)

dist_cur = class_distribution(df_clean[target])
ir_cur   = compute_ir(dist_cur)

if st.button("🎲 Получить рекомендации"):
    recs = recommend_samplers(ir_cur, len(dist_cur), len(df_clean))
    st.session_state.recommendations = recs
    st.session_state.step = max(st.session_state.step, 4)
    add_history("💡 Рекомендации", f"Топ: {recs['top']}, IR={ir_cur}")

recs = st.session_state.recommendations

if recs:
    st.markdown(f"""
    <div style="margin:1rem 0 0.5rem">
      <span class="metric-label">Характеристики датасета</span><br>
      <span>IR = <b style="color:#ff6a9b">{ir_cur}</b>
      · Классов: <b style="color:#7c6aff">{len(dist_cur)}</b>
      · Объектов: <b>{len(df_clean):,}</b></span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**🏆 Рекомендованный метод:**")
    st.markdown(f'<span class="rec-badge rec-top">⭐ {recs["top"]}</span>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Оверсэмплинг:**")
        for m in recs["oversampling"]:
            st.markdown(f'<span class="rec-badge rec-over">↑ {m}</span>', unsafe_allow_html=True)
    with col_r2:
        st.markdown("**Андерсэмплинг:**")
        for m in recs["undersampling"]:
            st.markdown(f'<span class="rec-badge rec-under">↓ {m}</span>', unsafe_allow_html=True)
    st.caption("💡 Рекомендации ориентировочные. Вы можете выбрать любой метод независимо от них.")
else:
    st.info("Нажмите кнопку выше, чтобы получить рекомендации.")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 5: Балансировка
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 05 / 10</div>'
            '<div class="step-title">Балансировка данных</div>',
            unsafe_allow_html=True)

num_cols = [c for c in df_clean.columns if c != target and pd.api.types.is_numeric_dtype(df_clean[c])]

if not num_cols:
    st.error("Нет числовых признаков для балансировки.")
else:
    col_bal1, col_bal2 = st.columns(2)
    with col_bal1:
        bal_type = st.radio("Тип балансировки:",
            ["Увеличение меньшего класса (Oversampling)",
             "Уменьшение большего класса (Undersampling)",
             "Комбинированный метод"], key="bal_type")
    with col_bal2:
        if "Oversampling" in bal_type:
            method_options = list(OVERSAMPLING_METHODS.keys())
        elif "Undersampling" in bal_type:
            method_options = list(UNDERSAMPLING_METHODS.keys())
        else:
            method_options = list(COMBINE_METHODS.keys())
        chosen_method = st.selectbox("Метод:", options=method_options, key="chosen_method")
        if recs and chosen_method == recs.get("top"):
            st.markdown('<span style="color:#7c6aff;font-size:0.75rem">⭐ Этот метод рекомендован</span>',
                        unsafe_allow_html=True)

    rs = st.slider("Random state:", 0, 100, 42, key="bal_rs")

    if st.button("▶ Применить балансировку"):
        with st.spinner(f"Применяю {chosen_method}..."):
            try:
                X_arr, y_arr, le = prepare_X_y(df_clean, target, num_cols)
                X_res, y_res = apply_sampler(X_arr, y_arr, chosen_method, rs)
                if le is not None:
                    y_res = le.inverse_transform(y_res)
                df_bal = pd.DataFrame(X_res, columns=num_cols)
                df_bal[target] = y_res
                st.session_state.df_balanced  = df_bal
                st.session_state.dist_after   = class_distribution(df_bal[target])
                st.session_state.applied_method = chosen_method
                st.session_state.step = max(st.session_state.step, 5)
                st.session_state.ml_results = None  # сбрасываем старые ML-результаты

                ir_new = compute_ir(st.session_state.dist_after)
                delta  = len(df_bal) - len(df_clean)
                add_history("⚖️ Балансировка", f"{chosen_method}: {len(df_clean)}→{len(df_bal)} строк, IR {ir_cur}→{ir_new}")
                st.success(f"✓ Балансировка завершена! Строк: {len(df_bal):,} (Δ {delta:+,}) · IR: {ir_cur} → {ir_new}")
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Неожиданная ошибка: {e}")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 6: Визуализация распределений
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 06 / 10</div>'
            '<div class="step-title">Визуализация распределения классов</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    dist_before = st.session_state.dist_before
    dist_after  = st.session_state.dist_after
    method_used = st.session_state.applied_method
    ir_before   = compute_ir(dist_before)
    ir_after    = compute_ir(dist_after)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="metric-label">Метод</div>
        <div class="metric-value" style="font-size:1rem;color:#7c6aff">{method_used}</div></div>
      <div class="metric-box"><div class="metric-label">IR до</div>
        <div class="metric-value accent2">{ir_before}</div></div>
      <div class="metric-box"><div class="metric-label">IR после</div>
        <div class="metric-value success">{ir_after}</div></div>
      <div class="metric-box"><div class="metric-label">Строк до</div>
        <div class="metric-value">{sum(dist_before.values()):,}</div></div>
      <div class="metric-box"><div class="metric-label">Строк после</div>
        <div class="metric-value accent3">{sum(dist_after.values()):,}</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        st.plotly_chart(make_pie(dist_before, "До балансировки", COLORS),
                        use_container_width=True, key="pie_before")
    with col_pie2:
        st.plotly_chart(make_pie(dist_after, "После балансировки",
                                 ["#6affd4", "#7c6aff", "#ff6a9b", "#fbbf24", "#60a5fa"]),
                        use_container_width=True, key="pie_after")

    st.plotly_chart(make_bar(dist_before, dist_after), use_container_width=True, key="bar_compare")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 7: Оценка ML-моделей
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 07 / 10</div>'
            '<div class="step-title">Оценка влияния балансировки на ML-модели</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    df_balanced = st.session_state.df_balanced

    # Настройка параметров моделей
    with st.expander("⚙️ Настройка параметров моделей", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("**Logistic Regression**")
            lr_C   = st.select_slider("Regularization C:", [0.01, 0.1, 1.0, 10.0, 100.0],
                                      value=1.0, key="lr_C")
            lr_sol = st.selectbox("Solver:", ["lbfgs", "liblinear", "saga"], key="lr_sol")
        with col_m2:
            st.markdown("**Random Forest**")
            rf_n   = st.slider("n_estimators:", 10, 300, 100, step=10, key="rf_n")
            rf_dep = st.slider("max_depth:", 2, 20, 10, key="rf_dep")
            rf_mss = st.slider("min_samples_split:", 2, 20, 2, key="rf_mss")
        with col_m3:
            st.markdown("**XGBoost**")
            xgb_n  = st.slider("n_estimators:", 10, 300, 100, step=10, key="xgb_n")
            xgb_lr = st.select_slider("learning_rate:", [0.01, 0.05, 0.1, 0.2, 0.3],
                                      value=0.1, key="xgb_lr")
            xgb_d  = st.slider("max_depth:", 2, 12, 6, key="xgb_d")

    model_params = {
        "lr":  {"C": lr_C, "solver": lr_sol},
        "rf":  {"n_estimators": rf_n, "max_depth": rf_dep, "min_samples_split": rf_mss, "random_state": 42},
        "xgb": {"n_estimators": xgb_n, "learning_rate": xgb_lr, "max_depth": xgb_d, "random_state": 42},
    }

    test_size = st.slider("Доля тестовой выборки:", 0.1, 0.4, 0.2, step=0.05, key="ml_test_size")

    if st.button("🚀 Запустить обучение и оценку моделей"):
        with st.spinner("Обучаю модели на исходных и сбалансированных данных..."):
            try:
                X_orig, y_orig, le_orig = prepare_X_y(df_clean, target, num_cols)
                X_bal,  y_bal,  le_bal  = prepare_X_y(df_balanced, target, num_cols)

                # Единое тестовое множество — из исходных данных
                X_tr_o, X_te, y_tr_o, y_te = train_test_split(
                    X_orig, y_orig, test_size=test_size, random_state=42, stratify=y_orig)

                # На сбалансированных обучаем на всём, тест — тот же
                X_tr_b = X_bal
                y_tr_b = y_bal

                orig_res = evaluate_models(X_tr_o, X_te, y_tr_o, y_te, model_params)
                bal_res  = evaluate_models(X_tr_b, X_te, y_tr_b, y_te, model_params)

                st.session_state.ml_results = {
                    "original": orig_res,
                    "balanced": bal_res,
                    "y_test":   y_te,
                    "classes":  np.unique(y_orig),
                }
                st.session_state.step = max(st.session_state.step, 7)
                add_history("🤖 ML-оценка", f"LR, RF, XGBoost · test_size={test_size}")
                st.success("✓ Обучение завершено!")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

    ml = st.session_state.ml_results
    if ml:
        orig_res = ml["original"]
        bal_res  = ml["balanced"]
        y_te     = ml["y_test"]
        classes  = ml["classes"]

        # ── Таблица сравнения метрик ──
        st.markdown("#### 📊 Таблица сравнения метрик")
        metrics_df = build_metrics_table(orig_res, bal_res)
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=[c for c in metrics_df.columns if "после" in c.lower()],
                color="#1a2a1a"
            ),
            use_container_width=True, hide_index=True
        )

        # Скачать таблицу
        csv_m = metrics_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ Скачать таблицу метрик", csv_m, "ml_metrics.csv", "text/csv")

        # ── Confusion Matrices ──
        st.markdown("#### 🗂 Confusion Matrix")
        cm_tab_orig, cm_tab_bal = st.tabs(["Исходные данные", "Сбалансированные данные"])
        with cm_tab_orig:
            cols_cm = st.columns(3)
            for i, (name, res) in enumerate(orig_res.items()):
                with cols_cm[i]:
                    fig_cm = plot_confusion_matrix(res["cm"], name, classes)
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_orig_{i}")
        with cm_tab_bal:
            cols_cm2 = st.columns(3)
            for i, (name, res) in enumerate(bal_res.items()):
                with cols_cm2[i]:
                    fig_cm = plot_confusion_matrix(res["cm"], name, classes)
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_bal_{i}")

        # ── Precision-Recall Curve ──
        binary = len(classes) == 2
        if binary:
            st.markdown("#### 📈 Precision-Recall Curve")
            pr_data = {"original": orig_res, "balanced": bal_res}
            fig_pr = plot_pr_curve(pr_data)
            st.plotly_chart(fig_pr, use_container_width=True, key="pr_curve")
        else:
            st.info("PR-кривая доступна только для бинарной классификации.")

        # ── Гистограмма F1 ──
        st.markdown("#### 📊 Сравнение F1-weighted по моделям")
        fig_f1 = go.Figure()
        model_names = list(orig_res.keys())
        f1_orig = [orig_res[n]["f1"] for n in model_names]
        f1_bal  = [bal_res[n]["f1"]  for n in model_names]

        fig_f1.add_trace(go.Bar(name="Исходные данные", x=model_names, y=f1_orig,
                                marker_color="#ff6a9b", marker_line=dict(color="#0d0d0f", width=1)))
        fig_f1.add_trace(go.Bar(name="Сбалансированные", x=model_names, y=f1_bal,
                                marker_color="#6affd4", marker_line=dict(color="#0d0d0f", width=1)))
        fig_f1.update_layout(barmode="group", yaxis_title="F1-weighted", height=340,
                             legend=dict(orientation="h", y=1.1), **PLOTLY_THEME)
        fig_f1.update_yaxes(gridcolor="#2a2a35", range=[0, 1])
        st.plotly_chart(fig_f1, use_container_width=True, key="f1_bar")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 8: Пространство низкой размерности
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 08 / 10</div>'
            '<div class="step-title">Визуализация в пространстве низкой размерности</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    col_dim1, col_dim2, col_dim3 = st.columns(3)
    with col_dim1:
        dim_method = st.selectbox("Метод:", ["PCA", "t-SNE"], key="dim_method")
    with col_dim2:
        dim_show = st.radio("Показывать:", ["До и после", "Только до", "Только после"], key="dim_show")
    with col_dim3:
        max_pts = st.slider("Макс. точек (для скорости):", 100, 2000, 500, step=100, key="dim_max_pts")

    if st.button("🔍 Построить визуализацию"):
        with st.spinner(f"Вычисляю {dim_method}..."):
            try:
                X_o, y_o, _ = prepare_X_y(df_clean, target, num_cols)
                X_b, y_b, _ = prepare_X_y(st.session_state.df_balanced, target, num_cols)

                # subsample
                def subsample(X, y, n):
                    if len(X) <= n:
                        return X, y
                    idx = np.random.choice(len(X), n, replace=False)
                    return X[idx], y[idx]

                X_o_s, y_o_s = subsample(X_o, y_o, max_pts)
                X_b_s, y_b_s = subsample(X_b, y_b, max_pts)

                scaler = StandardScaler()
                X_o_sc = scaler.fit_transform(X_o_s)
                X_b_sc = scaler.fit_transform(X_b_s)

                fig_orig = plot_dim_reduction(X_o_sc, y_o_s, dim_method,
                                              f"{dim_method} — До балансировки")
                fig_bal  = plot_dim_reduction(X_b_sc, y_b_s, dim_method,
                                              f"{dim_method} — После балансировки")

                st.session_state.dim_results = {
                    "fig_orig": fig_orig,
                    "fig_bal":  fig_bal,
                    "method":   dim_method,
                    "show":     dim_show,
                }
                st.session_state.step = max(st.session_state.step, 8)
                add_history(f"📉 {dim_method}", f"max_pts={max_pts}, показ: {dim_show}")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

    dr = st.session_state.dim_results
    if dr:
        show = dr["show"]
        if show in ("До и после", "Только до"):
            st.plotly_chart(dr["fig_orig"], use_container_width=True, key="dim_orig")
        if show in ("До и после", "Только после"):
            st.plotly_chart(dr["fig_bal"],  use_container_width=True, key="dim_bal")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 9: Сравнение нескольких методов балансировки
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 09 / 10</div>'
            '<div class="step-title">Сравнение нескольких методов балансировки</div>',
            unsafe_allow_html=True)

if len(num_cols) == 0:
    st.error("Нет числовых признаков.")
else:
    st.markdown("Выберите **2–4 метода** для сравнения. Будет обучена модель (Random Forest) на каждом варианте и показаны метрики.")

    all_method_names = list(OVERSAMPLING_METHODS) + list(UNDERSAMPLING_METHODS) + list(COMBINE_METHODS)
    selected_methods = st.multiselect(
        "Методы для сравнения:",
        options=all_method_names,
        default=["SMOTE", "ADASYN", "RandomOverSampler"],
        max_selections=4,
        key="multi_methods"
    )

    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        cmp_rs   = st.slider("Random state:", 0, 100, 42, key="cmp_rs")
    with col_cmp2:
        cmp_test = st.slider("Доля теста:", 0.1, 0.4, 0.2, step=0.05, key="cmp_test")

    if st.button("⚡ Сравнить методы") and len(selected_methods) >= 2:
        with st.spinner("Сравниваю методы..."):
            try:
                X_orig, y_orig, le = prepare_X_y(df_clean, target, num_cols)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_orig, y_orig, test_size=cmp_test, random_state=42, stratify=y_orig)

                scaler_cmp = StandardScaler()
                X_te_sc = scaler_cmp.fit_transform(X_te)

                compare_rows = []
                # Добавляем исходные данные как baseline
                rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
                X_tr_sc_base = scaler_cmp.fit_transform(X_tr)
                X_te_sc_base = scaler_cmp.transform(X_te)
                rf_base.fit(X_tr_sc_base, y_tr)
                y_pred_base = rf_base.predict(X_te_sc_base)
                binary_cmp = len(np.unique(y_orig)) == 2
                auc_base = None
                if binary_cmp and hasattr(rf_base, "predict_proba"):
                    yp = rf_base.predict_proba(X_te_sc_base)[:, 1]
                    auc_base = round(roc_auc_score(y_te, yp), 4)

                compare_rows.append({
                    "Метод":            "— Исходные данные —",
                    "Строк обучения":   len(X_tr),
                    "IR после":         compute_ir(class_distribution(pd.Series(y_tr))),
                    "Accuracy":         round(accuracy_score(y_te, y_pred_base), 4),
                    "F1-weighted":      round(f1_score(y_te, y_pred_base, average="weighted"), 4),
                    "ROC-AUC":          auc_base if auc_base else "—",
                })

                for mname in selected_methods:
                    try:
                        X_res, y_res = apply_sampler(X_tr, y_tr, mname, cmp_rs)
                        sc = StandardScaler()
                        X_r_sc = sc.fit_transform(X_res)
                        X_t_sc = sc.transform(X_te)
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_r_sc, y_res)
                        y_pred = rf.predict(X_t_sc)
                        auc_val = None
                        if binary_cmp and hasattr(rf, "predict_proba"):
                            yp2 = rf.predict_proba(X_t_sc)[:, 1]
                            auc_val = round(roc_auc_score(y_te, yp2), 4)
                        ir_res = compute_ir(class_distribution(pd.Series(y_res)))
                        compare_rows.append({
                            "Метод":            mname,
                            "Строк обучения":   len(X_res),
                            "IR после":         ir_res,
                            "Accuracy":         round(accuracy_score(y_te, y_pred), 4),
                            "F1-weighted":      round(f1_score(y_te, y_pred, average="weighted"), 4),
                            "ROC-AUC":          auc_val if auc_val else "—",
                        })
                    except Exception as ex:
                        compare_rows.append({
                            "Метод": mname, "Строк обучения": "—", "IR после": "—",
                            "Accuracy": "Ошибка", "F1-weighted": "Ошибка", "ROC-AUC": str(ex)[:40],
                        })

                st.session_state.multi_compare = {
                    "rows": compare_rows,
                    "methods": selected_methods,
                    "binary": binary_cmp,
                }
                st.session_state.step = max(st.session_state.step, 9)
                add_history("🔬 Сравнение", f"{', '.join(selected_methods)}")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
    elif len(selected_methods) < 2:
        st.warning("Выберите минимум 2 метода.")

    mc = st.session_state.multi_compare
    if mc:
        rows = mc["rows"]
        df_cmp = pd.DataFrame(rows)
        st.markdown("#### 📋 Таблица сравнения методов")
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # Скачать
        csv_cmp = df_cmp.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ Скачать таблицу", csv_cmp, "methods_comparison.csv", "text/csv")

        # Гистограмма F1
        numeric_rows = [r for r in rows if isinstance(r["F1-weighted"], float)]
        if numeric_rows:
            st.markdown("#### 📊 F1-weighted по методам")
            fig_cmp = go.Figure()
            methods_x = [r["Метод"] for r in numeric_rows]
            f1_y      = [r["F1-weighted"] for r in numeric_rows]
            colors_cmp = ["#6b6b80" if "Исходные" in m else "#7c6aff" for m in methods_x]
            fig_cmp.add_trace(go.Bar(x=methods_x, y=f1_y,
                                     marker_color=colors_cmp,
                                     marker_line=dict(color="#0d0d0f", width=1),
                                     text=[f"{v:.4f}" for v in f1_y], textposition="outside"))
            fig_cmp.update_layout(yaxis_title="F1-weighted", yaxis_range=[0, 1.1],
                                  height=360, **PLOTLY_THEME)
            fig_cmp.update_yaxes(gridcolor="#2a2a35")
            st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_f1_bar")

        # IR до/после
        numeric_ir = [r for r in rows if isinstance(r.get("IR после"), (int, float))]
        if len(numeric_ir) > 1:
            st.markdown("#### ⚖️ Imbalance Ratio после балансировки")
            fig_ir = go.Figure()
            fig_ir.add_trace(go.Bar(
                x=[r["Метод"] for r in numeric_ir],
                y=[r["IR после"] for r in numeric_ir],
                marker_color="#ff6a9b",
                marker_line=dict(color="#0d0d0f", width=1),
                text=[str(r["IR после"]) for r in numeric_ir], textposition="outside"
            ))
            fig_ir.update_layout(yaxis_title="Imbalance Ratio", height=320, **PLOTLY_THEME)
            fig_ir.update_yaxes(gridcolor="#2a2a35")
            st.plotly_chart(fig_ir, use_container_width=True, key="cmp_ir_bar")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 10: Сохранение
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 10 / 10</div>'
            '<div class="step-title">Сохранение итогового датасета</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    df_final = st.session_state.df_balanced
    method_used = st.session_state.applied_method or "balanced"

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        st.markdown("**Параметры сохранения:**")
        sep = st.radio("Разделитель:", [",", ";", "\\t"], horizontal=True, key="csv_sep")
        include_index = st.checkbox("Включить индекс", value=False, key="csv_index")
        filename = st.text_input("Имя файла:", value=f"balanced_{method_used}.csv", key="csv_filename")
    with col_save2:
        st.markdown("**Предпросмотр (5 строк):**")
        st.dataframe(df_final.head(5), use_container_width=True)

    csv_sep = "\t" if sep == "\\t" else sep
    csv_buf = io.BytesIO()
    df_final.to_csv(csv_buf, index=include_index, sep=csv_sep, encoding="utf-8-sig")
    csv_bytes = csv_buf.getvalue()

    st.markdown(f"""
    <div class="metric-row" style="margin:1rem 0">
      <div class="metric-box"><div class="metric-label">Размер файла</div>
        <div class="metric-value" style="font-size:1.1rem">{len(csv_bytes)/1024:.1f} KB</div></div>
      <div class="metric-box"><div class="metric-label">Строк</div>
        <div class="metric-value accent">{len(df_final):,}</div></div>
      <div class="metric-box"><div class="metric-label">Колонок</div>
        <div class="metric-value">{df_final.shape[1]}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label=f"⬇ Скачать {filename}",
        data=csv_bytes, file_name=filename, mime="text/csv",
        use_container_width=True,
    )
    st.session_state.step = 10

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# История операций
# ══════════════════════════════════════════════════════════════════════════

if st.session_state.history:
    st.markdown('<div class="step-card">'
                '<div class="step-label">📋 История</div>'
                '<div class="step-title">Журнал операций сессии</div>',
                unsafe_allow_html=True)

    for i, h in enumerate(reversed(st.session_state.history)):
        st.markdown(f"""
        <div class="history-item">
          <div class="history-ts">#{len(st.session_state.history) - i} · {h["ts"]}</div>
          <div style="color:#e8e8f0;margin-top:0.2rem">
            <b style="color:#7c6aff">{h["type"]}</b> — {h["details"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Скачать историю как JSON
    history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button("⬇ Скачать историю (JSON)", history_json.encode("utf-8"),
                       "session_history.json", "application/json")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#6b6b80;font-size:0.72rem;
letter-spacing:0.08em;border-top:1px solid #2a2a35;margin-top:2rem">
  DATABALANCE PRO · SMOTE · ADASYN · BorderlineSMOTE · RandomForest · XGBoost · PCA · t-SNE
</div>
""", unsafe_allow_html=True)