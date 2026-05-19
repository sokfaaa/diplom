"""
Приложение для балансировки данных | Streamlit
Запуск: streamlit run app.py
"""

import io
import random
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SVMSMOTE,
)
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Конфигурация страницы ──────────────────────────────────────────────────

st.set_page_config(
    page_title="DataBalance",
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

/* Заголовок */
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

/* Карточки шагов */
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

/* Метрики */
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

/* Теги рекомендаций */
.rec-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
    border: 1px solid;
}
.rec-top {
    background: rgba(124,106,255,0.15);
    border-color: var(--accent);
    color: var(--accent);
}
.rec-over {
    background: rgba(106,255,212,0.1);
    border-color: var(--accent3);
    color: var(--accent3);
}
.rec-under {
    background: rgba(255,106,155,0.1);
    border-color: var(--accent2);
    color: var(--accent2);
}

/* Таблица */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Кнопки */
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

/* Прогресс и разделители */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Info / Warning boxes */
.stAlert { border-radius: 8px !important; }

/* Download button */
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
</style>
""", unsafe_allow_html=True)


# ── Константы ─────────────────────────────────────────────────────────────

OVERSAMPLING_METHODS = {
    "SMOTE":              SMOTE,
    "BorderlineSMOTE":    BorderlineSMOTE,
    "SVMSMOTE":           SVMSMOTE,
    "ADASYN":             ADASYN,
    "KMeansSMOTE":        KMeansSMOTE,
    "RandomOverSampler":  RandomOverSampler,
}

UNDERSAMPLING_METHODS = {
    "RandomUnderSampler":        RandomUnderSampler,
    "TomekLinks":                TomekLinks,
    "EditedNearestNeighbours":   EditedNearestNeighbours,
    "NearMiss (v1)":             lambda **kw: NearMiss(version=1, **kw),
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


# ── Вспомогательные функции ───────────────────────────────────────────────

@st.cache_data
def load_demo() -> pd.DataFrame:
    """Демонстрационный датасет: кредитный риск с выраженным дисбалансом."""
    rng = np.random.default_rng(42)
    n = 1000
    n_min = 80   # minority class

    # Majority class (0 = нет дефолта)
    X_maj = pd.DataFrame({
        "age":          rng.integers(22, 65, n - n_min),
        "income":       rng.normal(55000, 15000, n - n_min).astype(int),
        "credit_score": rng.integers(600, 850, n - n_min),
        "loan_amount":  rng.normal(15000, 8000, n - n_min).astype(int),
        "years_employed": rng.integers(0, 30, n - n_min),
        "num_accounts": rng.integers(1, 8, n - n_min),
        "target":       0,
    })
    # Minority class (1 = дефолт)
    X_min = pd.DataFrame({
        "age":          rng.integers(18, 45, n_min),
        "income":       rng.normal(28000, 10000, n_min).astype(int),
        "credit_score": rng.integers(300, 620, n_min),
        "loan_amount":  rng.normal(25000, 12000, n_min).astype(int),
        "years_employed": rng.integers(0, 5, n_min),
        "num_accounts": rng.integers(1, 3, n_min),
        "target":       1,
    })
    df = pd.concat([X_maj, X_min], ignore_index=True).sample(frac=1, random_state=42)
    # Добавляем несколько пропусков
    for col in ["income", "credit_score", "years_employed"]:
        idx = rng.choice(len(df), size=rng.integers(10, 25), replace=False)
        df.loc[idx, col] = np.nan
    return df


def compute_ir(counts: dict) -> float:
    vals = list(counts.values())
    return round(max(vals) / min(vals), 2) if min(vals) > 0 else float("inf")


def class_distribution(y: pd.Series) -> dict:
    return y.value_counts().to_dict()


def make_pie(dist: dict, title: str, colors: list) -> go.Figure:
    labels = [str(k) for k in dist.keys()]
    values = list(dist.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors[:len(labels)],
                    line=dict(color="#0d0d0f", width=2)),
        hole=0.45,
        textfont=dict(family="DM Mono, monospace", size=12),
        hovertemplate="<b>Класс %{label}</b><br>Кол-во: %{value}<br>Доля: %{percent}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)),
                      showlegend=True, height=320, **PLOTLY_THEME)
    return fig


def make_bar(dist_before: dict, dist_after: dict) -> go.Figure:
    classes = sorted(set(list(dist_before.keys()) + list(dist_after.keys())), key=str)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="До балансировки",
        x=[str(c) for c in classes],
        y=[dist_before.get(c, 0) for c in classes],
        marker_color="#7c6aff",
        marker_line=dict(color="#0d0d0f", width=1),
    ))
    fig.add_trace(go.Bar(
        name="После балансировки",
        x=[str(c) for c in classes],
        y=[dist_after.get(c, 0) for c in classes],
        marker_color="#6affd4",
        marker_line=dict(color="#0d0d0f", width=1),
    ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Класс",
        yaxis_title="Кол-во объектов",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        **PLOTLY_THEME,
    )
    fig.update_xaxes(gridcolor="#2a2a35", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2a35", zeroline=False)
    return fig


def recommend_samplers(ir: float, n_classes: int, n_samples: int) -> dict:
    """
    Выдаёт случайные рекомендации из всего списка (с небольшой логикой).
    По заданию — рекомендация выбирается случайно из всего списка.
    """
    rng = random.Random(int(ir * 100))

    # Случайный топ-1 из oversampling
    top = rng.choice(list(OVERSAMPLING_METHODS.keys()))

    # Случайные 2-3 oversampling
    over_pool = list(OVERSAMPLING_METHODS.keys())
    rng.shuffle(over_pool)
    over_recs = over_pool[:rng.randint(2, 3)]

    # Случайные 1-2 undersampling
    under_pool = list(UNDERSAMPLING_METHODS.keys())
    rng.shuffle(under_pool)
    under_recs = under_pool[:rng.randint(1, 2)]

    return {"top": top, "oversampling": over_recs, "undersampling": under_recs}


def apply_sampler(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет выбранный сэмплер."""
    all_methods = {**OVERSAMPLING_METHODS, **UNDERSAMPLING_METHODS, **COMBINE_METHODS}

    if method_name not in all_methods:
        raise ValueError(f"Неизвестный метод: {method_name}")

    builder = all_methods[method_name]
    try:
        if method_name == "KMeansSMOTE":
            sampler = builder(random_state=random_state,
                              cluster_balance_threshold=0.0)
        elif method_name in ("TomekLinks", "EditedNearestNeighbours"):
            sampler = builder()
        elif callable(builder) and not isinstance(builder, type):
            # lambda (NearMiss)
            sampler = builder(random_state=random_state)
        else:
            sampler = builder(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        raise RuntimeError(f"Ошибка при применении {method_name}: {e}") from e


# ── Состояние сессии ──────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df_raw":       None,   # загруженный датасет
        "df_clean":     None,   # после обработки пропусков
        "df_balanced":  None,   # после балансировки
        "target_col":   None,
        "dist_before":  None,
        "dist_after":   None,
        "recommendations": None,
        "applied_method":  None,
        "step":         0,      # текущий шаг (0-based)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Хелперы состояния ─────────────────────────────────────────────────────

def set_df(df: pd.DataFrame, target: str):
    st.session_state.df_raw      = df
    st.session_state.df_clean    = df.copy()
    st.session_state.df_balanced = None
    st.session_state.target_col  = target
    st.session_state.dist_before = class_distribution(df[target])
    st.session_state.dist_after  = None
    st.session_state.recommendations = None
    st.session_state.applied_method  = None


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

# ── Заголовок ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>⚖ DataBalance</h1>
  <p>ИНСТРУМЕНТ БАЛАНСИРОВКИ ДАТАСЕТОВ · РАЗВЕДОЧНЫЙ АНАЛИЗ · РЕКОМЕНДАЦИИ</p>
</div>
""", unsafe_allow_html=True)


# ── Сайдбар — навигация ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖ DataBalance")
    st.markdown("---")
    st.markdown("**Шаги обработки:**")

    steps = [
        "① Загрузка данных",
        "② Разведочный анализ",
        "③ Пропущенные значения",
        "④ Рекомендации",
        "⑤ Балансировка",
        "⑥ Визуализация",
        "⑦ Сохранение",
    ]
    for i, s in enumerate(steps):
        is_done = st.session_state.step > i
        is_cur  = st.session_state.step == i
        color   = "#4ade80" if is_done else ("#7c6aff" if is_cur else "#6b6b80")
        st.markdown(
            f'<div style="padding:0.3rem 0.5rem;color:{color};'
            f'font-size:0.8rem;font-weight:{"700" if is_cur else "400"}">{s}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.session_state.df_raw is not None:
        df_cur = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        st.markdown(f"**Датасет:** `{df_cur.shape[0]} × {df_cur.shape[1]}`")
        if st.session_state.target_col:
            dist = class_distribution(df_cur[st.session_state.target_col])
            st.markdown(f"**IR:** `{compute_ir(dist)}`")
        if st.button("🔄 Сбросить всё"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_state()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 1: Загрузка данных
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 01 / 07</div>'
            '<div class="step-title">Загрузка данных</div>',
            unsafe_allow_html=True)

col_src1, col_src2 = st.columns([1, 1])

with col_src1:
    st.markdown("**📂 Загрузить свой файл**")
    uploaded = st.file_uploader(
        "CSV файл",
        type=["csv"],
        label_visibility="collapsed",
    )
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            target_up = st.selectbox(
                "Целевая колонка (метки классов):",
                options=df_up.columns.tolist(),
                index=len(df_up.columns) - 1,
                key="target_upload",
            )
            if st.button("✓ Использовать этот файл"):
                set_df(df_up, target_up)
                st.session_state.step = 1
                st.success(f"Загружено: {df_up.shape[0]} строк × {df_up.shape[1]} колонок")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")

with col_src2:
    st.markdown("**🎲 Демонстрационный пример**")
    st.caption("Кредитный риск (1000 объектов, IR ≈ 11.5, 6 признаков)")
    if st.button("▶ Загрузить демо-датасет"):
        demo = load_demo()
        set_df(demo, "target")
        st.session_state.step = 1
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Если данные не загружены — дальше не идём
if st.session_state.df_raw is None:
    st.info("⬆ Загрузите датасет или используйте демо-пример, чтобы начать.")
    st.stop()


# Удобные алиасы
df_raw    = st.session_state.df_raw
df_clean  = st.session_state.df_clean
target    = st.session_state.target_col


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 2: Разведочный анализ
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 02 / 07</div>'
            '<div class="step-title">Разведочный анализ данных</div>',
            unsafe_allow_html=True)

dist = class_distribution(df_clean[target])
ir   = compute_ir(dist)
n_missing = df_clean.isnull().sum().sum()
n_classes = len(dist)

# Метрики верхнего ряда
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-label">Объектов</div>
    <div class="metric-value accent">{len(df_clean):,}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Признаков</div>
    <div class="metric-value">{df_clean.shape[1] - 1}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Классов</div>
    <div class="metric-value accent2">{n_classes}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Imbalance Ratio</div>
    <div class="metric-value {'accent2' if ir > 5 else 'success'}">{ir}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Пропущенных</div>
    <div class="metric-value {'warning' if n_missing > 0 else 'success'}"
         style="color:{'#fbbf24' if n_missing > 0 else '#4ade80'}">{n_missing}</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_head, tab_desc, tab_miss, tab_dist = st.tabs([
    "Первые строки", "Статистика", "Пропуски", "Распределение классов"
])

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
        "Класс":      [str(k) for k in dist.keys()],
        "Кол-во":     list(dist.values()),
        "Доля, %":    [round(v / len(df_clean) * 100, 2) for v in dist.values()],
    })
    col_tbl, col_pie = st.columns([1, 1])
    with col_tbl:
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    with col_pie:
        fig_pie0 = make_pie(dist, "Исходное распределение классов", COLORS)
        st.plotly_chart(fig_pie0, use_container_width=True, key="pie_eda")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 3: Обработка пропущенных значений
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 03 / 07</div>'
            '<div class="step-title">Обработка пропущенных значений</div>',
            unsafe_allow_html=True)

miss_cols = [c for c in df_clean.columns if df_clean[c].isnull().any()]

if not miss_cols:
    st.success("✓ Пропущенных значений нет — этот шаг пропускается автоматически.")
else:
    st.markdown(f"Колонки с пропусками: `{'`, `'.join(miss_cols)}`")

    col_miss1, col_miss2 = st.columns([1, 1])
    with col_miss1:
        miss_strategy = st.radio(
            "Стратегия обработки:",
            options=["Заполнить медианой", "Заполнить средним", "Заполнить модой", "Удалить строки"],
            key="miss_strategy",
        )
    with col_miss2:
        miss_cols_sel = st.multiselect(
            "Применить к колонкам:",
            options=miss_cols,
            default=miss_cols,
            key="miss_cols_sel",
        )

    if st.button("⚙ Применить обработку пропусков"):
        df_proc = df_clean.copy()
        if miss_strategy == "Удалить строки":
            df_proc = df_proc.dropna(subset=miss_cols_sel)
        else:
            for col in miss_cols_sel:
                if miss_strategy == "Заполнить медианой":
                    val = df_proc[col].median()
                elif miss_strategy == "Заполнить средним":
                    val = df_proc[col].mean()
                else:  # мода
                    val = df_proc[col].mode().iloc[0]
                df_proc[col] = df_proc[col].fillna(val)

        st.session_state.df_clean = df_proc
        df_clean = df_proc
        n_removed = len(df_clean) - len(df_proc)
        st.success(
            f"✓ Готово. Строк после: {len(df_proc):,}"
            + (f"  (удалено: {n_removed})" if n_removed else "")
        )
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Обновляем алиас после возможного изменения
df_clean = st.session_state.df_clean


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 4: Рекомендации
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 04 / 07</div>'
            '<div class="step-title">Рекомендация методов балансировки</div>',
            unsafe_allow_html=True)

dist_cur = class_distribution(df_clean[target])
ir_cur   = compute_ir(dist_cur)

if st.button("🎲 Получить рекомендации"):
    recs = recommend_samplers(ir_cur, len(dist_cur), len(df_clean))
    st.session_state.recommendations = recs
    st.session_state.step = max(st.session_state.step, 4)

recs = st.session_state.recommendations

if recs:
    st.markdown(f"""
    <div style="margin:1rem 0 0.5rem">
      <span class="metric-label">Характеристики датасета</span><br>
      <span style="color:#e8e8f0">IR = <b style="color:#ff6a9b">{ir_cur}</b>
      · Классов: <b style="color:#7c6aff">{len(dist_cur)}</b>
      · Объектов: <b>{len(df_clean):,}</b></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🏆 Рекомендованный метод:**")
    st.markdown(f'<span class="rec-badge rec-top">⭐ {recs["top"]}</span>',
                unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Оверсэмплинг:**")
        for m in recs["oversampling"]:
            st.markdown(f'<span class="rec-badge rec-over">↑ {m}</span>',
                        unsafe_allow_html=True)
    with col_r2:
        st.markdown("**Андерсэмплинг:**")
        for m in recs["undersampling"]:
            st.markdown(f'<span class="rec-badge rec-under">↓ {m}</span>',
                        unsafe_allow_html=True)

    st.caption("💡 Рекомендации носят ориентировочный характер. "
               "На следующем шаге вы можете выбрать любой метод независимо от них.")
else:
    st.info("Нажмите кнопку выше, чтобы получить рекомендации.")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 5: Балансировка
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 05 / 07</div>'
            '<div class="step-title">Балансировка данных</div>',
            unsafe_allow_html=True)

# Разделяем X и y
num_cols = [c for c in df_clean.columns
            if c != target and pd.api.types.is_numeric_dtype(df_clean[c])]

if not num_cols:
    st.error("Нет числовых признаков для балансировки.")
else:
    col_bal1, col_bal2 = st.columns([1, 1])

    with col_bal1:
        bal_type = st.radio(
            "Тип балансировки:",
            ["Увеличение меньшего класса (Oversampling)",
             "Уменьшение большего класса (Undersampling)",
             "Комбинированный метод"],
            key="bal_type",
        )

    with col_bal2:
        if "Oversampling" in bal_type:
            method_options = list(OVERSAMPLING_METHODS.keys())
        elif "Undersampling" in bal_type:
            method_options = list(UNDERSAMPLING_METHODS.keys())
        else:
            method_options = list(COMBINE_METHODS.keys())

        chosen_method = st.selectbox(
            "Метод:",
            options=method_options,
            key="chosen_method",
        )

        if recs and chosen_method == recs.get("top"):
            st.markdown(
                '<span style="color:#7c6aff;font-size:0.75rem">⭐ Этот метод рекомендован</span>',
                unsafe_allow_html=True
            )

    rs = st.slider("Random state:", 0, 100, 42, key="bal_rs")

    if st.button("▶ Применить балансировку"):
        with st.spinner(f"Применяю {chosen_method}..."):
            try:
                X_arr = df_clean[num_cols].fillna(df_clean[num_cols].median()).values
                y_ser = df_clean[target]

                # Кодируем строковые метки если нужно
                le = None
                if y_ser.dtype == object or str(y_ser.dtype) == "category":
                    le = LabelEncoder()
                    y_arr = le.fit_transform(y_ser)
                else:
                    y_arr = y_ser.values

                X_res, y_res = apply_sampler(X_arr, y_arr, chosen_method, rs)

                # Декодируем метки обратно
                if le is not None:
                    y_res = le.inverse_transform(y_res)

                df_bal = pd.DataFrame(X_res, columns=num_cols)
                df_bal[target] = y_res

                st.session_state.df_balanced  = df_bal
                st.session_state.dist_after   = class_distribution(df_bal[target])
                st.session_state.applied_method = chosen_method
                st.session_state.step = max(st.session_state.step, 5)

                dist_a = st.session_state.dist_after
                ir_new = compute_ir(dist_a)
                delta  = len(df_bal) - len(df_clean)

                st.success(
                    f"✓ Балансировка завершена!  "
                    f"Строк: {len(df_bal):,}  (Δ {delta:+,})  ·  IR: {ir_cur} → {ir_new}"
                )
                st.rerun()

            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Неожиданная ошибка: {e}")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 6: Визуализация
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 06 / 07</div>'
            '<div class="step-title">Визуализация распределения классов</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    dist_before = st.session_state.dist_before
    dist_after  = st.session_state.dist_after
    method_used = st.session_state.applied_method

    ir_before = compute_ir(dist_before)
    ir_after  = compute_ir(dist_after)

    # Сводные метрики
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Метод</div>
        <div class="metric-value" style="font-size:1rem;color:#7c6aff">{method_used}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">IR до</div>
        <div class="metric-value accent2">{ir_before}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">IR после</div>
        <div class="metric-value success">{ir_after}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Строк до</div>
        <div class="metric-value">{sum(dist_before.values()):,}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Строк после</div>
        <div class="metric-value accent3">{sum(dist_after.values()):,}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Круговые диаграммы
    st.markdown("#### Круговые диаграммы")
    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        fig1 = make_pie(dist_before, "До балансировки", COLORS)
        st.plotly_chart(fig1, use_container_width=True, key="pie_before")
    with col_pie2:
        fig2 = make_pie(dist_after, "После балансировки", ["#6affd4", "#7c6aff", "#ff6a9b",
                                                            "#fbbf24", "#60a5fa"])
        st.plotly_chart(fig2, use_container_width=True, key="pie_after")

    # Столбчатая диаграмма сравнения
    st.markdown("#### Сравнение распределений")
    fig_bar = make_bar(dist_before, dist_after)
    st.plotly_chart(fig_bar, use_container_width=True, key="bar_compare")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 7: Сохранение
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-card">'
            '<div class="step-label">Шаг 07 / 07</div>'
            '<div class="step-title">Сохранение итогового датасета</div>',
            unsafe_allow_html=True)

if st.session_state.df_balanced is None:
    st.info("Сначала примените балансировку на шаге 05.")
else:
    df_final = st.session_state.df_balanced
    method_used = st.session_state.applied_method or "balanced"

    col_save1, col_save2 = st.columns([1, 1])

    with col_save1:
        st.markdown("**Параметры сохранения:**")
        sep = st.radio("Разделитель:", [",", ";", "\\t"], horizontal=True, key="csv_sep")
        include_index = st.checkbox("Включить индекс", value=False, key="csv_index")
        filename = st.text_input(
            "Имя файла:",
            value=f"balanced_{method_used}.csv",
            key="csv_filename",
        )

    with col_save2:
        st.markdown("**Предпросмотр (5 строк):**")
        st.dataframe(df_final.head(5), use_container_width=True)

    csv_sep = "\t" if sep == "\\t" else sep
    csv_buf = io.BytesIO()
    df_final.to_csv(csv_buf, index=include_index, sep=csv_sep, encoding="utf-8-sig")
    csv_bytes = csv_buf.getvalue()

    st.markdown(f"""
    <div class="metric-row" style="margin:1rem 0">
      <div class="metric-box">
        <div class="metric-label">Размер файла</div>
        <div class="metric-value" style="font-size:1.1rem">{len(csv_bytes)/1024:.1f} KB</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Строк</div>
        <div class="metric-value accent">{len(df_final):,}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Колонок</div>
        <div class="metric-value">{df_final.shape[1]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label=f"⬇ Скачать {filename}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

    st.session_state.step = 7

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#6b6b80;font-size:0.72rem;
letter-spacing:0.08em;border-top:1px solid #2a2a35;margin-top:2rem">
  DATABALANCE · ИНСТРУМЕНТ БАЛАНСИРОВКИ ДАННЫХ ·
  SMOTE · ADASYN · BorderlineSMOTE · RandomOverSampler · и другие
</div>
""", unsafe_allow_html=True)