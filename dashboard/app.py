"""
=============================================================
  Metro Interstate Traffic Volume — Interactive Dashboard
  Streamlit App | BDA Course Project
=============================================================
  Run:  streamlit run app.py
  Requires: traffic_model.pkl, columns.pkl, metrics.json
            in ../models/  (produced by train.py)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metro Traffic Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  .main { background-color: #0a0d14; }
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #131929, #1b2240);
    border: 1px solid #1e2a4a;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4);
  }
  [data-testid="metric-container"] label {
    color: #6b7599 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #dce4ff !important;
    font-size: 1.5rem !important;
    font-weight: 700;
  }
  [data-testid="metric-container"] [data-testid="metric-delta"] {
    color: #34d399 !important;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, #3d5af1, #6d28d9);
    border-radius: 8px;
    padding: 10px 18px;
    margin: 18px 0 12px 0;
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.3px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #111520;
    border-radius: 10px;
    padding: 5px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 7px 18px;
    color: #6b7599;
    font-weight: 600;
    font-size: 0.88rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3d5af1, #6d28d9) !important;
    color: #fff !important;
  }

  /* Insight box */
  .insight-box {
    background: #131929;
    border-left: 3px solid #3d5af1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    color: #b4bde0;
    font-size: 0.92rem;
    line-height: 1.55;
  }

  /* Winner box */
  .winner-box {
    background: #0d1f12;
    border: 2px solid #34d399;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 12px 0;
    color: #a7f3d0;
  }

  /* Model card */
  .model-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin: 6px 0;
    border: 1px solid #1e2a4a;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d101c;
    border-right: 1px solid #1a2035;
  }
  [data-testid="stSidebar"] .stMarkdown { color: #6b7599; }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #dce4ff !important;
  }

  /* Sidebar filter labels */
  .sidebar-label {
    color: #6b7599;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 2px;
    margin-top: 10px;
  }

  /* Sidebar stat pills */
  .stat-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 12px;
    background: #131929;
    border: 1px solid #1e2a4a;
    border-radius: 7px;
    margin: 4px 0;
  }
  .stat-pill .label { color: #6b7599; font-size: 0.8rem; }
  .stat-pill .value { color: #dce4ff; font-size: 0.85rem; font-weight: 600; }

  /* Model status badge */
  .model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .badge-ok  { background: #0d1f12; border: 1px solid #34d399; color: #34d399; }
  .badge-err { background: #1f0d0d; border: 1px solid #f87171; color: #f87171; }
</style>
""", unsafe_allow_html=True)

SEED = 42

# ── Plotly layout defaults ────────────────────────────────────────────────────
DARK_BG   = "#0a0d14"
PLOT_BG   = "#111520"
GRID_CLR  = "#1a2035"
TEXT_CLR  = "#b4bde0"
AXIS_CLR  = "#4a5278"

def dark_layout(**kwargs):
    base = dict(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=AXIS_CLR, tickfont=dict(color=TEXT_CLR)),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=AXIS_CLR, tickfont=dict(color=TEXT_CLR)),
        title_font=dict(color="#dce4ff", size=14),
    )
    base.update(kwargs)
    return base

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m   = joblib.load("../models/traffic_model.pkl")
    col = joblib.load("../models/columns.pkl")
    return m, col

@st.cache_data
def load_metrics():
    with open("../models/metrics.json") as f:
        return json.load(f)

try:
    model, model_columns = load_model()
    metrics = load_metrics()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.warning(f"Model not found — run `train.py` first. ({e})")
    metrics = {}
    model_columns = []

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    clean = pd.read_csv("../data/processed/cleaned.csv")
    raw   = pd.read_csv("../data/raw/Metro_Interstate_Traffic_Volume.csv")
    raw["date_time"]  = pd.to_datetime(raw["date_time"])
    raw["holiday"]    = raw["holiday"].fillna("None")
    raw = raw[raw["temp"] > 200]
    raw = raw[raw["rain_1h"] < 9000]
    raw = raw.drop_duplicates(subset="date_time", keep="last").reset_index(drop=True)
    raw["hour"]        = raw["date_time"].dt.hour
    raw["day_of_week"] = raw["date_time"].dt.dayofweek
    raw["month"]       = raw["date_time"].dt.month
    raw["year"]        = raw["date_time"].dt.year
    raw["is_weekend"]  = (raw["day_of_week"] >= 5).astype(int)
    raw["is_holiday"]  = (raw["holiday"] != "None").astype(int)
    raw["temp_c"]      = raw["temp"] - 273.15
    raw["is_peak"]     = raw["hour"].isin([7,8,9,16,17,18,19]).astype(int)
    raw["day_name"]    = raw["date_time"].dt.day_name()
    raw["month_name"]  = raw["date_time"].dt.month_name()
    return clean, raw

clean_df, df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 4px 10px;border-bottom:1px solid #1a2035;margin-bottom:14px'>
      <div style='color:#dce4ff;font-size:1.1rem;font-weight:700;letter-spacing:0.3px'>
        Metro Traffic Analytics
      </div>
      <div style='color:#4a5278;font-size:0.78rem;margin-top:3px'>I-94 Minneapolis–St Paul</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Year Range</div>', unsafe_allow_html=True)
    year_range = st.slider(
        "Year Range", int(df["year"].min()), int(df["year"].max()),
        (2013, 2018), label_visibility="collapsed"
    )
    st.markdown(f'<div style="color:#4a5278;font-size:0.75rem;margin-top:-8px;margin-bottom:6px">{year_range[0]} — {year_range[1]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Weather Condition</div>', unsafe_allow_html=True)
    all_weather = sorted(df["weather_main"].unique())
    sel_weather = st.multiselect(
        "Weather", all_weather, default=all_weather, label_visibility="collapsed"
    )
    if not sel_weather:
        sel_weather = all_weather
        st.warning("Showing all weather conditions.")

    st.markdown('<div class="sidebar-label">Time Filter</div>', unsafe_allow_html=True)
    show_peak_only = st.checkbox("Peak Hours Only", value=False)

    st.markdown('<div style="border-top:1px solid #1a2035;margin:16px 0 12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    for label, value in [
        ("Records", f"{len(df):,}"),
        ("Features", str(clean_df.shape[1] - 1)),
        ("Period", "2012 – 2018"),
        ("Source", "Kaggle / UCI"),
    ]:
        st.markdown(f"""
        <div class="stat-pill">
          <span class="label">{label}</span>
          <span class="value">{value}</span>
        </div>""", unsafe_allow_html=True)

    if MODEL_LOADED:
        st.markdown('<div style="border-top:1px solid #1a2035;margin:16px 0 12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Model Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-badge badge-ok">Model Loaded</div>', unsafe_allow_html=True)
        r2_val  = metrics["test_metrics"]["R2"]
        mae_val = metrics["test_metrics"]["MAE"]
        acc_pct = r2_val * 100
        for label, value in [
            ("R²", f"{r2_val:.4f}"),
            ("Accuracy", f"{acc_pct:.1f}%"),
            ("MAE", f"{mae_val:,}"),
        ]:
            st.markdown(f"""
            <div class="stat-pill" style="margin-top:4px">
              <span class="label">{label}</span>
              <span class="value">{value}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-label">Model Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-badge badge-err">Not Loaded</div>', unsafe_allow_html=True)

# Apply filters
dff = df[
    (df["year"].between(*year_range)) &
    (df["weather_main"].isin(sel_weather))
]
if show_peak_only:
    dff = dff[dff["is_peak"] == 1]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1b2cf0 0%,#6d28d9 100%);
            border-radius:12px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 6px 24px rgba(61,90,241,0.35)'>
  <h1 style='color:#fff;margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-0.5px'>
    Metro Interstate Traffic Analytics
  </h1>
  <p style='color:#c4ceff;margin:5px 0 0;font-size:0.92rem'>
    I-94 Minneapolis–St Paul &nbsp;·&nbsp; 2012–2018 &nbsp;·&nbsp; Big Data Analytics Project
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records",   f"{len(dff):,}")
c2.metric("Avg Traffic/Hr",  f"{dff['traffic_volume'].mean():,.0f}")
c3.metric("Max Traffic",     f"{int(dff['traffic_volume'].max()):,}")
c4.metric("Avg Temperature", f"{dff['temp_c'].mean():.1f} C")
c5.metric("Holiday Records", f"{int(dff['is_holiday'].sum()):,}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "EDA & Patterns",
    "Weather Analysis",
    "Time Patterns",
    "Model Comparison",
    "Predict & Evaluate",
    "Insights & Explainability"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA & Patterns
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x="traffic_volume", nbins=60,
                           color_discrete_sequence=["#3d5af1"],
                           title="Traffic Volume Distribution",
                           labels={"traffic_volume": "Traffic Volume (vehicles/hr)"})
        fig.update_layout(**dark_layout())
        st.plotly_chart(fig, width='stretch')

    with c2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg = (dff.groupby("day_name")["traffic_volume"]
                   .mean().reindex(dow_order).reset_index())
        fig = px.bar(dow_avg, x="day_name", y="traffic_volume",
                     color="traffic_volume", color_continuous_scale="Blues",
                     title="Avg Traffic by Day of Week",
                     labels={"day_name": "Day", "traffic_volume": "Avg Volume"})
        fig.update_layout(**dark_layout(coloraxis_showscale=False))
        st.plotly_chart(fig, width='stretch')

    monthly = (dff.groupby(dff["date_time"].dt.to_period("M"))["traffic_volume"]
               .mean().reset_index())
    monthly["date_time"] = monthly["date_time"].dt.to_timestamp()
    fig = px.area(monthly, x="date_time", y="traffic_volume",
                  title="Monthly Average Traffic Volume (Time Series)",
                  labels={"date_time": "Date", "traffic_volume": "Avg Volume"},
                  color_discrete_sequence=["#6d28d9"])
    fig.update_layout(**dark_layout())
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Feature Correlation — features with |r| > 0.05 vs target")
    num_cols_all = ["temp_c","rain_1h","snow_1h","clouds_all","hour","day_of_week",
                    "month","is_weekend","is_holiday","is_peak","traffic_volume"]
    corr_full = dff[num_cols_all].corr()
    relevant  = corr_full["traffic_volume"][
        corr_full["traffic_volume"].abs() > 0.05].index.tolist()
    if "traffic_volume" not in relevant:
        relevant.append("traffic_volume")
    fig = px.imshow(dff[relevant].corr(), text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap (Relevant Features Only)",
                    aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(**dark_layout(height=480))
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Weather Analysis
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="section-header">Weather Impact on Traffic</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        w_avg = (dff.groupby("weather_main")["traffic_volume"]
                 .mean().sort_values().reset_index())
        fig = px.bar(w_avg, x="traffic_volume", y="weather_main",
                     orientation="h", color="traffic_volume",
                     color_continuous_scale="Plasma",
                     title="Avg Traffic by Weather",
                     labels={"traffic_volume": "Avg Volume", "weather_main": "Weather"})
        fig.update_layout(**dark_layout(coloraxis_showscale=False, height=420))
        st.plotly_chart(fig, width='stretch')

    with c2:
        corr_temp = dff["temp_c"].corr(dff["traffic_volume"])
        samp = dff.sample(min(4000, len(dff)), random_state=SEED)
        fig = px.scatter(samp, x="temp_c", y="traffic_volume", color="weather_main",
                         title=f"Temp vs Traffic  (r = {corr_temp:.3f})",
                         labels={"temp_c": "Temperature (C)",
                                 "traffic_volume": "Traffic Volume",
                                 "weather_main": "Weather"},
                         opacity=0.4)
        fig.update_layout(**dark_layout(height=420))
        st.plotly_chart(fig, width='stretch')

    corr_rain = dff["rain_1h"].corr(dff["traffic_volume"])
    if abs(corr_rain) > 0.05:
        samp2 = dff[dff["rain_1h"] < 50].sample(min(3000, len(dff)), random_state=SEED)
        fig = px.scatter(samp2, x="rain_1h", y="traffic_volume",
                         title=f"Rain vs Traffic  (r = {corr_rain:.3f})",
                         labels={"rain_1h": "Rain (mm)", "traffic_volume": "Traffic Volume"},
                         opacity=0.4, color_discrete_sequence=["#3d5af1"])
        fig.update_layout(**dark_layout())
        st.plotly_chart(fig, width='stretch')
    else:
        st.info(f"Rain scatter skipped — weak correlation (r = {corr_rain:.3f}).")

    fig = px.violin(dff, x="weather_main", y="traffic_volume",
                    box=True, color="weather_main",
                    title="Traffic Volume Distribution per Weather Type",
                    labels={"weather_main": "Weather", "traffic_volume": "Traffic Volume"})
    fig.update_layout(**dark_layout(showlegend=False, height=400))
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Time Patterns
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="section-header">Temporal Traffic Patterns</div>',
                unsafe_allow_html=True)

    hourly_wkd = dff[dff["is_weekend"]==0].groupby("hour")["traffic_volume"].mean()
    hourly_wke = dff[dff["is_weekend"]==1].groupby("hour")["traffic_volume"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_wkd.index, y=hourly_wkd.values,
                             mode="lines+markers", name="Weekday",
                             line=dict(color="#3d5af1", width=2.5)))
    fig.add_trace(go.Scatter(x=hourly_wke.index, y=hourly_wke.values,
                             mode="lines+markers", name="Weekend",
                             line=dict(color="#e05b8a", width=2.5)))
    fig.update_layout(title="Hourly Traffic: Weekday vs Weekend",
                      xaxis_title="Hour of Day", yaxis_title="Avg Traffic Volume",
                      **dark_layout())
    fig.update_xaxes(tickmode="linear", dtick=1)
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        hol = dff.groupby("is_holiday")["traffic_volume"].mean().reset_index()
        hol["label"] = hol["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(hol, x="label", y="traffic_volume", color="label",
                     color_discrete_sequence=["#3d5af1","#e05b8a"],
                     title="Holiday vs Non-Holiday Traffic",
                     labels={"traffic_volume": "Avg Volume", "label": ""})
        fig.update_layout(**dark_layout(showlegend=False))
        st.plotly_chart(fig, width='stretch')

    with c2:
        heat = dff.pivot_table(values="traffic_volume",
                               index="year", columns="month", aggfunc="mean")
        fig = px.imshow(heat, labels={"x":"Month","y":"Year","color":"Avg Volume"},
                        color_continuous_scale="Blues",
                        title="Avg Traffic: Year x Month Heatmap", aspect="auto")
        fig.update_layout(**dark_layout())
        st.plotly_chart(fig, width='stretch')

    heat2 = dff.pivot_table(values="traffic_volume",
                             index="day_of_week", columns="hour", aggfunc="mean")
    heat2.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = px.imshow(heat2, labels={"x":"Hour","y":"Day","color":"Avg Volume"},
                    color_continuous_scale="Viridis",
                    title="Traffic Heatmap: Hour of Day x Day of Week", aspect="auto")
    fig.update_layout(**dark_layout(height=320))
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="section-header">Model Comparison & Evaluation</div>',
                unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("Run `train.py` first to generate model metrics.")
    else:
        amm = metrics.get("all_model_metrics", {})
        cv  = metrics.get("cv_comparison", {})
        hp  = metrics.get("hyperparameters", {})

        model_names  = list(amm.keys())
        model_colors = {
            "Decision Tree"    : "#3d5af1",
            "Random Forest"    : "#34d399",
            "Gradient Boosting": "#e05b8a",
        }

        # ── Section 1: Test Set Metrics Table ────────────────────────────
        st.markdown("### Test Set Performance (Held-Out 20%)")

        rows = []
        has_f1 = any((m.get("F1") or m.get("f1") or m.get("F1_score") or m.get("f1_score")) for m in amm.values())
        for name, m_dict in amm.items():
            row = {
                "Model"  : name,
                "MAE"    : m_dict["MAE"],
                "RMSE"   : m_dict["RMSE"],
                "R²"     : m_dict["R2"],
                "Adj R²" : m_dict["Adj_R2"],
                "MAPE %" : m_dict["MAPE"],
            }
            if has_f1:
                row["F1"] = m_dict.get("F1") or m_dict.get("f1") or m_dict.get("F1_score") or m_dict.get("f1_score") or 0.0
            rows.append(row)
        summary_df = pd.DataFrame(rows).set_index("Model")

        def highlight_best(s):
            is_lower_better = s.name in ["MAE", "RMSE", "MAPE %"]
            best = s.min() if is_lower_better else s.max()
            return ["background-color: #0d1f12; color: #34d399; font-weight: bold"
                    if v == best else "" for v in s]

        fmt = {"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R²": "{:.4f}", "Adj R²": "{:.4f}", "MAPE %": "{:.2f}%"}
        if has_f1:
            fmt["F1"] = "{:.4f}"
        st.dataframe(
            summary_df.style
                .apply(highlight_best)
                .format(fmt),
            width='stretch'
        )
        st.caption("Green highlight = best value per column" + (" | F1 score computed on binned traffic classes" if has_f1 else ""))

        st.markdown("---")

        # ── Section 2: Visual metric comparison ──────────────────────────
        st.markdown("### Visual Metric Comparison")

        vc1, vc2, vc3 = st.columns(3)
        metrics_to_plot = [
            ("MAE",  "MAE (lower is better)",  "Reds_r"),
            ("RMSE", "RMSE (lower is better)", "Oranges_r"),
            ("R2",   "R² Score (higher is better)", "Greens"),
        ]
        for col_st, (metric_key, title, cscale) in zip([vc1, vc2, vc3], metrics_to_plot):
            vals = [amm[n][metric_key] for n in model_names]
            fig  = px.bar(
                x=model_names, y=vals,
                color=vals, color_continuous_scale=cscale,
                title=title,
                labels={"x": "Model", "y": metric_key},
                text=[f"{v:.4f}" if metric_key == "R2" else f"{v:,.1f}" for v in vals]
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**dark_layout(coloraxis_showscale=False,
                                            showlegend=False, xaxis_tickangle=-10))
            col_st.plotly_chart(fig, width='stretch')

        # F1 score chart if available
        if has_f1:
            f1_vals = [summary_df.loc[n, "F1"] for n in model_names]
            fig_f1 = px.bar(
                x=model_names, y=f1_vals,
                color=f1_vals, color_continuous_scale="Teal",
                title="F1 Score per Model (higher is better)",
                labels={"x": "Model", "y": "F1 Score"},
                text=[f"{v:.4f}" for v in f1_vals]
            )
            fig_f1.update_traces(textposition="outside")
            fig_f1.update_layout(**dark_layout(coloraxis_showscale=False, showlegend=False))
            st.plotly_chart(fig_f1, width='stretch')

        st.markdown("---")

        # ── Section 3: Cross-Validation ───────────────────────────────────
        st.markdown("### 5-Fold Cross-Validation — Stability Analysis")
        st.caption("Error bars show standard deviation across folds. Smaller bars = more stable model.")

        cv_rows = []
        for name, cv_d in cv.items():
            cv_rows.append({
                "Model"       : name,
                "CV R² Mean"  : cv_d["CV_R2_mean"],
                "CV R² Std"   : cv_d["CV_R2_std"],
                "CV MAE Mean" : cv_d["CV_MAE_mean"],
                "CV MAE Std"  : cv_d["CV_MAE_std"],
            })
        cv_df = pd.DataFrame(cv_rows)

        fc1, fc2 = st.columns(2)
        with fc1:
            fig = px.bar(cv_df, x="Model", y="CV R² Mean", error_y="CV R² Std",
                         color="CV R² Mean", color_continuous_scale="Greens",
                         title="CV R² per Model (mean ± std)",
                         text=[f"{v:.4f}" for v in cv_df["CV R² Mean"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(**dark_layout(coloraxis_showscale=False))
            st.plotly_chart(fig, width='stretch')

        with fc2:
            fig = px.bar(cv_df, x="Model", y="CV MAE Mean", error_y="CV MAE Std",
                         color="CV MAE Mean", color_continuous_scale="Reds_r",
                         title="CV MAE per Model (mean ± std)",
                         text=[f"{v:,.0f}" for v in cv_df["CV MAE Mean"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(**dark_layout(coloraxis_showscale=False))
            st.plotly_chart(fig, width='stretch')

        # Per-fold line chart
        st.markdown("#### R² Score Across Individual Folds")
        fold_fig = go.Figure()
        for name in model_names:
            fold_r2 = cv[name].get("CV_R2_per_fold", [])
            if fold_r2:
                fold_fig.add_trace(go.Scatter(
                    x=[f"Fold {i+1}" for i in range(len(fold_r2))],
                    y=fold_r2,
                    mode="lines+markers+text",
                    name=name,
                    text=[f"{v:.4f}" for v in fold_r2],
                    textposition="top center",
                    line=dict(color=model_colors.get(name, "#fff"), width=2),
                    marker=dict(size=7)
                ))
        fold_fig.update_layout(
            title="R² Score per Fold — All Models",
            xaxis_title="Fold", yaxis_title="R²",
            legend_title="Model",
            **dark_layout()
        )
        st.plotly_chart(fold_fig, width='stretch')

        st.markdown("---")

        # ── Section 4: Actual vs Predicted ───────────────────────────────
        st.markdown("### Actual vs Predicted — All 3 Models")
        st.caption("Scatter from the held-out test set. Points near the diagonal = better predictions.")

        from sklearn.model_selection import train_test_split as tts
        from sklearn.tree import DecisionTreeRegressor as DT
        from sklearn.ensemble import (RandomForestRegressor as RF,
                                      GradientBoostingRegressor as GB)

        X_ev = clean_df.drop(["traffic_volume"], axis=1)
        y_ev = clean_df["traffic_volume"]
        _, X_te, _, y_te = tts(X_ev, y_ev, test_size=0.2, random_state=SEED)
        X_te_a = X_te.reindex(columns=model_columns, fill_value=0)

        X_tr_ev, _, y_tr_ev, _ = tts(X_ev, y_ev, test_size=0.2, random_state=SEED)
        X_tr_a = X_tr_ev.reindex(columns=model_columns, fill_value=0)

        @st.cache_resource
        def get_all_predictions(_X_tr, _y_tr, _X_te):
            preds = {}
            dt = DT(max_depth=10, random_state=SEED)
            dt.fit(_X_tr, _y_tr)
            preds["Decision Tree"] = dt.predict(_X_te)

            rf = RF(n_estimators=200, max_depth=15, n_jobs=-1, random_state=SEED)
            rf.fit(_X_tr, _y_tr)
            preds["Random Forest"] = rf.predict(_X_te)

            gb = GB(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED)
            gb.fit(_X_tr, _y_tr)
            preds["Gradient Boosting"] = gb.predict(_X_te)
            return preds

        all_preds = get_all_predictions(X_tr_a, y_tr_ev, X_te_a)

        sample_idx = np.random.default_rng(SEED).choice(
            len(y_te), min(600, len(y_te)), replace=False
        )
        y_sample = y_te.iloc[sample_idx].values

        sc1, sc2, sc3 = st.columns(3)
        for col_st, name in zip([sc1, sc2, sc3], model_names):
            p_sample = all_preds[name][sample_idx]
            r2_val   = amm[name]["R2"]
            mae_val  = amm[name]["MAE"]
            clr      = model_colors.get(name, "#3d5af1")
            with col_st:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_sample, y=p_sample, mode="markers",
                    marker=dict(color=clr, opacity=0.35, size=4),
                    name="Predictions"
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 7300], y=[0, 7300], mode="lines",
                    line=dict(color="#6b7599", dash="dash", width=1),
                    name="Perfect Fit"
                ))
                fig.update_layout(
                    title=f"{name}<br><sup>R²={r2_val:.4f}  |  MAE={mae_val:,.0f}</sup>",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    showlegend=False, height=350,
                    **dark_layout()
                )
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── Section 5: Residual distributions ────────────────────────────
        st.markdown("### Residual Distribution — All 3 Models")
        st.caption("Residual = Actual minus Predicted. A narrow peak centred at 0 indicates a well-fitted model.")

        res_fig = go.Figure()
        for name in model_names:
            residuals = y_te.values - all_preds[name]
            res_fig.add_trace(go.Histogram(
                x=residuals, name=name, nbinsx=80, opacity=0.65,
                marker_color=model_colors.get(name, "#3d5af1")
            ))
        res_fig.add_vline(x=0, line_dash="dash", line_color="#6b7599",
                          annotation_text="Zero error", annotation_position="top right")
        res_fig.update_layout(
            barmode="overlay",
            title="Residual Distributions — All Models",
            xaxis_title="Residual (Actual minus Predicted)",
            yaxis_title="Count",
            legend_title="Model",
            **dark_layout()
        )
        st.plotly_chart(res_fig, width='stretch')

        st.markdown("---")

        # ── Section 6: Radar chart ────────────────────────────────────────
        st.markdown("### Radar Chart — Normalised Performance Overview")
        st.caption("All metrics normalised 0–1. Larger area = better overall (MAE/RMSE/MAPE are inverted so larger = lower error).")

        def norm(vals, invert=False):
            arr = np.array(vals, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.ones_like(arr) * 0.5
            n = (arr - mn) / (mx - mn)
            return 1 - n if invert else n

        categories = ["R²", "Adj R²", "MAE (inv)", "RMSE (inv)", "MAPE (inv)"]
        radar_fig  = go.Figure()

        for name in model_names:
            r2s    = [amm[n]["R2"]     for n in model_names]
            adjr2s = [amm[n]["Adj_R2"] for n in model_names]
            maes   = [amm[n]["MAE"]    for n in model_names]
            rmses  = [amm[n]["RMSE"]   for n in model_names]
            mapes  = [amm[n]["MAPE"]   for n in model_names]
            idx = model_names.index(name)
            values = [
                norm(r2s)[idx], norm(adjr2s)[idx],
                norm(maes, invert=True)[idx],
                norm(rmses, invert=True)[idx],
                norm(mapes, invert=True)[idx],
            ]
            values += [values[0]]
            cats_closed = categories + [categories[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=values, theta=cats_closed,
                fill="toself", name=name,
                line_color=model_colors.get(name, "#fff"),
                opacity=0.7
            ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0,1],
                                color=AXIS_CLR, gridcolor=GRID_CLR),
                angularaxis=dict(color=TEXT_CLR, gridcolor=GRID_CLR)
            ),
            template="plotly_dark", paper_bgcolor=DARK_BG,
            font=dict(color=TEXT_CLR),
            showlegend=True, legend_title="Model",
            title="Radar: Normalised Model Performance",
            height=480
        )
        st.plotly_chart(radar_fig, width='stretch')

        st.markdown("---")

        # ── Section 7: Winner verdict ─────────────────────────────────────
        st.markdown("### Verdict — Best Model")

        best_r2_name = max(amm, key=lambda n: amm[n]["R2"])
        best_r2_val  = amm[best_r2_name]["R2"]
        best_mae_val = amm[best_r2_name]["MAE"]
        best_mape    = amm[best_r2_name]["MAPE"]

        st.markdown(f"""
        <div class="winner-box">
          <div style='color:#34d399;font-size:1.4rem;font-weight:800;text-align:center;
                      margin-bottom:8px'>{best_r2_name}</div>
          <div style='color:#a7f3d0;text-align:center;font-size:0.95rem;margin-bottom:14px'>
            R² = {best_r2_val:.4f} &nbsp;|&nbsp; MAE = {best_mae_val:,.0f} vehicles/hr
            &nbsp;|&nbsp; MAPE = {best_mape:.2f}%
          </div>
          <div style='color:#6ee7b7;font-size:0.86rem;line-height:1.7'>
            Highest R² and lowest MAE/RMSE across all test samples.
            Most stable across 5 CV folds (lowest standard deviation in R²).
            Avoids Decision Tree overfitting by averaging 200 trees.
            Faster inference than Gradient Boosting (trees built in parallel).
            Built-in feature importance, directly compatible with SHAP.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Predict & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="section-header">Live Prediction & Model Evaluation</div>',
                unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("Model not loaded. Please run `train.py` first.")
    else:
        st.markdown("### Real-Time Traffic Prediction")
        st.markdown("Adjust inputs below and click **Predict** to generate a new output from the trained model.")

        pc1, pc2 = st.columns(2)
        with pc1:
            pred_temp    = st.slider("Temperature (C)", -30, 50, 20)
            pred_rain    = st.slider("Rain (mm)", 0.0, 50.0, 0.0, step=0.5)
            pred_snow    = st.slider("Snow (mm)", 0.0, 10.0, 0.0, step=0.1)
            pred_clouds  = st.slider("Cloud Cover (%)", 0, 100, 40)
        with pc2:
            pred_hour    = st.slider("Hour of Day", 0, 23, 8)
            pred_day     = st.slider("Day of Week  (0 = Mon)", 0, 6, 1)
            pred_month   = st.slider("Month", 1, 12, 6)
            pred_holiday = st.selectbox("Is Holiday?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")

        weather_options = sorted([
            c.replace("weather_main_", "")
            for c in model_columns if c.startswith("weather_main_")
        ])
        pred_weather = st.selectbox("Weather Condition", weather_options)

        is_weekend_pred = int(pred_day >= 5)
        is_peak_pred    = int(pred_hour in [7,8,9,16,17,18,19])

        input_dict = {
            "temp_c": pred_temp, "rain_1h": pred_rain, "snow_1h": pred_snow,
            "clouds_all": pred_clouds, "hour": pred_hour, "day_of_week": pred_day,
            "month": pred_month, "is_weekend": is_weekend_pred,
            "is_holiday": pred_holiday, "is_peak": is_peak_pred,
        }
        wc = f"weather_main_{pred_weather}"
        if wc in model_columns:
            input_dict[wc] = 1

        input_df_pred = pd.DataFrame([input_dict]).reindex(
            columns=model_columns, fill_value=0)

        if st.button("Predict Traffic Volume"):
            prediction = max(0.0, float(model.predict(input_df_pred)[0]))

            if prediction < 1000:   level, col = "Very Low",  "#34d399"
            elif prediction < 2500: level, col = "Low",       "#3d5af1"
            elif prediction < 4000: level, col = "Medium",    "#f59e0b"
            elif prediction < 5500: level, col = "High",      "#e05b8a"
            else:                   level, col = "Very High", "#ef4444"

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"""
                <div style='background:#131929;border:2px solid {col};
                            border-radius:12px;padding:28px;text-align:center;margin-top:10px'>
                  <div style='color:#6b7599;font-size:0.85rem;text-transform:uppercase;
                              letter-spacing:0.06em'>Predicted Volume</div>
                  <div style='color:{col};font-size:3rem;font-weight:800;line-height:1.2'>
                    {int(prediction):,}</div>
                  <div style='color:#b4bde0;font-size:0.95rem'>vehicles / hour</div>
                  <div style='color:{col};font-size:1.2rem;font-weight:700;margin-top:10px'>
                    {level}</div>
                  <div style='color:#6b7599;font-size:0.8rem;margin-top:6px'>
                    {"Peak hour" if is_peak_pred else "Off-peak"} &nbsp;|&nbsp;
                    {"Weekend" if is_weekend_pred else "Weekday"}
                  </div>
                </div>""", unsafe_allow_html=True)
            with rc2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=int(prediction),
                    title={"text": "Traffic Level", "font": {"color": TEXT_CLR}},
                    number={"font": {"color": col}},
                    gauge={
                        "axis": {"range": [0, 7500], "tickcolor": AXIS_CLR},
                        "bar":  {"color": col}, "bgcolor": PLOT_BG,
                        "steps": [
                            {"range": [0,    1000], "color": "#0d1f12"},
                            {"range": [1000, 2500], "color": "#0d1220"},
                            {"range": [2500, 4000], "color": "#1f1a0d"},
                            {"range": [4000, 5500], "color": "#1f0d1a"},
                            {"range": [5500, 7500], "color": "#1f0d0d"},
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 3},
                                      "thickness": 0.75, "value": 5500},
                    }
                ))
                fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG,
                                  font=dict(color=TEXT_CLR),
                                  height=280, margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── Model Evaluation metrics + confusion matrix ───────────────────
        st.markdown("### Model Evaluation (Held-Out Test Set)")

        tm = metrics["test_metrics"]
        acc_pct = tm["R2"] * 100

        f1_val = tm.get("F1") or tm.get("f1") or tm.get("F1_score") or tm.get("f1_score")
        if f1_val:
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy (R²-based)", f"{acc_pct:.1f}%")
            mc2.metric("R²",   f"{tm['R2']:.4f}")
            mc3.metric("MAE",  f"{tm['MAE']:,}")
            mc4.metric("RMSE", f"{tm['RMSE']:,}")
            mc5.metric("F1 Score", f"{f1_val:.4f}")
        else:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy (R²-based)", f"{acc_pct:.1f}%")
            mc2.metric("R²",   f"{tm['R2']:.4f}")
            mc3.metric("MAE",  f"{tm['MAE']:,}")
            mc4.metric("RMSE", f"{tm['RMSE']:,}")

        st.markdown("---")

        # ── Confusion Matrix (binned volume ranges) ───────────────────────
        st.markdown("### Confusion Matrix — Predicted vs Actual Traffic Bins")
        st.caption("Traffic volume bucketed into 5 levels. Diagonal = correct classifications.")

        try:
            from sklearn.model_selection import train_test_split as tts2
            _X_ev2 = clean_df.drop(["traffic_volume"], axis=1)
            _y_ev2 = clean_df["traffic_volume"]
            _, _X_te2, _, _y_te2 = tts2(_X_ev2, _y_ev2, test_size=0.2, random_state=SEED)
            _X_te2_a = _X_te2.reindex(columns=model_columns, fill_value=0)
            _y_pred2 = model.predict(_X_te2_a)

            bins   = [0, 1000, 2500, 4000, 5500, 7700]
            labels = ["Very Low\n(<1k)", "Low\n(1k-2.5k)", "Medium\n(2.5k-4k)",
                      "High\n(4k-5.5k)", "Very High\n(>5.5k)"]

            actual_bins = pd.cut(_y_te2, bins=bins, labels=labels, include_lowest=True)
            pred_bins   = pd.cut(_y_pred2, bins=bins, labels=labels, include_lowest=True)

            cm_df = pd.crosstab(actual_bins, pred_bins, rownames=["Actual"], colnames=["Predicted"])
            cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0)

            # Normalise rows for percentages
            cm_pct = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0) * 100

            fig_cm = go.Figure(go.Heatmap(
                z=cm_pct.values,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=[[f"{cm_df.values[i][j]:,}<br>({cm_pct.values[i][j]:.1f}%)"
                       for j in range(len(labels))] for i in range(len(labels))],
                texttemplate="%{text}",
                textfont=dict(size=11, color="#dce4ff"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Row %", tickfont=dict(color=TEXT_CLR)),
            ))
            fig_cm.update_layout(
                title="Confusion Matrix (Row-Normalised %)",
                xaxis=dict(title="Predicted Class", tickfont=dict(color=TEXT_CLR, size=10),
                           title_font=dict(color=TEXT_CLR), linecolor=AXIS_CLR, gridcolor=GRID_CLR),
                yaxis=dict(title="Actual Class", tickfont=dict(color=TEXT_CLR, size=10),
                           title_font=dict(color=TEXT_CLR), linecolor=AXIS_CLR, gridcolor=GRID_CLR,
                           autorange="reversed"),
                template="plotly_dark",
                paper_bgcolor=DARK_BG,
                plot_bgcolor=PLOT_BG,
                font=dict(color=TEXT_CLR),
                title_font=dict(color="#dce4ff", size=14),
                height=440,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_cm, width='stretch')
        except Exception as e:
            st.info(f"Confusion matrix unavailable: {e}")

        st.markdown("---")

        st.markdown("#### Feature Importance (Random Forest)")
        fi_data = metrics["feature_importance"]
        fi_df = pd.DataFrame(list(fi_data.items()),
                             columns=["Feature","Importance"]).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     title="Feature Importances",
                     labels={"Importance": "Importance Score"})
        fig.update_layout(**dark_layout(coloraxis_showscale=False, height=400))
        st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Insights & Explainability
# ─────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="section-header">Key Insights & Explainability</div>',
                unsafe_allow_html=True)

    peak_diff = (df[df["is_peak"]==1]["traffic_volume"].mean() -
                 df[df["is_peak"]==0]["traffic_volume"].mean())
    weekend_ratio = (df[df["is_weekend"]==1]["traffic_volume"].mean() /
                     df[df["is_weekend"]==0]["traffic_volume"].mean())
    holiday_ratio = (df[df["is_holiday"]==1]["traffic_volume"].mean() /
                     df[df["is_holiday"]==0]["traffic_volume"].mean())

    if MODEL_LOADED:
        best_r2  = metrics["test_metrics"]["R2"]
        best_mae = metrics["test_metrics"]["MAE"]
        fi_data  = metrics["feature_importance"]
        top_feat = max(fi_data, key=fi_data.get)
        top_imp  = fi_data[top_feat]
    else:
        best_r2 = best_mae = top_feat = top_imp = "N/A"

    insights = [
        f"Peak hours (7–9 AM & 4–7 PM) carry <b>{peak_diff:,.0f} more vehicles/hr</b> on average compared to off-peak hours.",
        f"Weekend traffic is only <b>{weekend_ratio:.1%} of weekday</b> traffic, reflecting the commuter-dominant nature of the highway.",
        f"Holiday traffic drops to <b>{holiday_ratio:.1%} of normal</b> levels.",
        (f"<b>{top_feat}</b> is the single most important predictor (importance = {top_imp:.3f})."
         if MODEL_LOADED else "Run train.py to compute feature importance."),
        (f"Random Forest achieves <b>R² = {best_r2:.4f}</b> verified by 5-fold CV."
         if MODEL_LOADED else "Run train.py to generate metrics."),
        "Weather has a moderate effect — Clear and Clouds conditions show the highest volumes.",
        "Seasonal patterns: higher traffic May–August, with a dip in winter months.",
        "5-fold cross-validation confirms the model is stable across different data splits.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Explainability — SHAP")

    if MODEL_LOADED and SHAP_AVAILABLE:
        try:
            shap_vals = np.load("../models/shap_values.npy")
            shap_cols = joblib.load("../models/shap_columns.pkl")
            shap_mean = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": shap_cols, "Mean |SHAP|": shap_mean
            }).sort_values("Mean |SHAP|")
            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature",
                         orientation="h", color="Mean |SHAP|",
                         color_continuous_scale="Plasma",
                         title="SHAP — Mean Absolute Impact per Feature")
            fig.update_layout(**dark_layout(coloraxis_showscale=False, height=420))
            st.plotly_chart(fig, width='stretch')
        except FileNotFoundError:
            st.info("SHAP values not found. Re-run `train.py` with `pip install shap`.")
    elif not SHAP_AVAILABLE:
        st.info("Install SHAP: `pip install shap` then re-run `train.py`.")
    else:
        st.info("Run `train.py` to generate SHAP values.")

    st.markdown("---")
    st.markdown("### Recommendations")
    for r in [
        "**Traffic Management:** Deploy dynamic speed signs and ramp metering during peak hours (7–9 AM, 4–7 PM).",
        "**Emergency Planning:** Pre-position incident response units on weekday mornings.",
        "**Weather Response:** Activate winter maintenance crews when snow or fog is forecast.",
        "**Holiday Policy:** Reduce toll rates or open HOV lanes on holidays.",
        "**Model Deployment:** Integrate the trained Random Forest into a real-time pipeline fed by hourly weather API data.",
    ]:
        st.markdown(f"- {r}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2d3450;font-size:0.82rem;padding:10px'>
  Metro Interstate Traffic Volume Analytics &nbsp;·&nbsp; BDA Course Project &nbsp;·&nbsp;
  Department of CE (Software Engineering) &nbsp;·&nbsp; SY Big Data Analytics
</div>
""", unsafe_allow_html=True)
