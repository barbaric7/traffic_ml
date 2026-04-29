"""
=============================================================
  Metro Interstate Traffic Volume — Interactive Dashboard
  Streamlit App | BDA Course Project
=============================================================
  Run:  streamlit run dashboard.py
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
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (old impressive dark style — fully preserved) ──────────────────
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2130, #252b3b);
    border: 1px solid #2d3450;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  [data-testid="metric-container"] label {
    color: #9ba3c0 !important;
    font-size: 0.8rem !important;
  }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e0e6ff !important;
    font-size: 1.6rem !important;
    font-weight: 700;
  }
  [data-testid="metric-container"] [data-testid="metric-delta"] {
    color: #4ade80 !important;
  }
  .section-header {
    background: linear-gradient(90deg, #4361ee, #7c3aed);
    border-radius: 10px;
    padding: 12px 20px;
    margin: 20px 0 14px 0;
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #1a1d2e;
    border-radius: 12px;
    padding: 6px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #9ba3c0;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4361ee, #7c3aed) !important;
    color: white !important;
  }
  .insight-box {
    background: linear-gradient(135deg, #1e2a45, #1a2035);
    border-left: 4px solid #4361ee;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    color: #c8d3f5;
    font-size: 0.95rem;
  }
  .winner-box {
    background: linear-gradient(135deg, #1a3a1e, #1a2a1e);
    border: 2px solid #4ade80;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #c8f5d0;
  }
  .model-card {
    border-radius: 14px;
    padding: 18px 20px;
    margin: 8px 0;
    border: 1px solid #2d3450;
  }
  [data-testid="stSidebar"] { background-color: #141727; }
  [data-testid="stSidebar"] .stMarkdown { color: #9ba3c0; }
</style>
""", unsafe_allow_html=True)

SEED = 42

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
    st.warning(f"⚠️ Model not found — run `train.py` first. ({e})")
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
    st.markdown("## 🚗 Traffic Analytics")
    st.markdown("**BDA Course Project**")
    st.markdown("---")
    st.markdown("### 📅 Filter Data")
    year_range = st.slider("Year Range",
                           int(df["year"].min()), int(df["year"].max()),
                           (2013, 2018))
    sel_weather = st.multiselect("Weather Condition",
                                 sorted(df["weather_main"].unique()),
                                 default=sorted(df["weather_main"].unique()))
    if not sel_weather:
        sel_weather = sorted(df["weather_main"].unique())
        st.warning("No weather selected — showing all.")
    show_peak_only = st.checkbox("Peak Hours Only", value=False)
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **Rows:** {len(df):,}")
    st.markdown(f"- **Features:** {clean_df.shape[1] - 1}")
    st.markdown(f"- **Period:** 2012–2018")
    st.markdown(f"- **Source:** Kaggle / UCI")
    if MODEL_LOADED:
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        st.success("✅ Model loaded")
        st.markdown(f"- **R²:** {metrics['test_metrics']['R2']:.4f}")
        st.markdown(f"- **MAE:** {metrics['test_metrics']['MAE']:,}")

# Apply filters
dff = df[
    (df["year"].between(*year_range)) &
    (df["weather_main"].isin(sel_weather))
]
if show_peak_only:
    dff = dff[dff["is_peak"] == 1]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#4361ee,#7c3aed);
            border-radius:16px;padding:24px 32px;margin-bottom:24px;
            box-shadow:0 8px 24px rgba(67,97,238,0.4)'>
  <h1 style='color:white;margin:0;font-size:2.2rem'>
    🚗 Metro Interstate Traffic Analytics
  </h1>
  <p style='color:#c8d3f5;margin:6px 0 0;font-size:1rem'>
    I-94 Minneapolis–St Paul | 2012–2018 | Big Data Analytics Project
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards — fully dynamic ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Total Records",   f"{len(dff):,}")
c2.metric("🚘 Avg Traffic/Hr",  f"{dff['traffic_volume'].mean():,.0f}")
c3.metric("🔺 Max Traffic",     f"{int(dff['traffic_volume'].max()):,}")
c4.metric("🌡️ Avg Temp",        f"{dff['temp_c'].mean():.1f} °C")
c5.metric("🎄 Holiday Records", f"{int(dff['is_holiday'].sum()):,}")

# ── Tabs — 6 total ────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 EDA & Patterns",
    "🌦️ Weather Analysis",
    "⏰ Time Patterns",
    "🏆 Model Comparison",
    "🤖 Predict & Evaluate",
    "💡 Insights & Explainability"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA & Patterns
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x="traffic_volume", nbins=60,
                           color_discrete_sequence=["#4361ee"],
                           title="Traffic Volume Distribution",
                           labels={"traffic_volume": "Traffic Volume (vehicles/hr)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig, width='stretch')

    with c2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        dow_avg = (dff.groupby("day_name")["traffic_volume"]
                   .mean().reindex(dow_order).reset_index())
        fig = px.bar(dow_avg, x="day_name", y="traffic_volume",
                     color="traffic_volume", color_continuous_scale="Viridis",
                     title="Avg Traffic by Day of Week",
                     labels={"day_name": "Day", "traffic_volume": "Avg Volume"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e", coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    monthly = (dff.groupby(dff["date_time"].dt.to_period("M"))["traffic_volume"]
               .mean().reset_index())
    monthly["date_time"] = monthly["date_time"].dt.to_timestamp()
    fig = px.area(monthly, x="date_time", y="traffic_volume",
                  title="Monthly Average Traffic Volume Trend (Time Series)",
                  labels={"date_time": "Date", "traffic_volume": "Avg Volume"},
                  color_discrete_sequence=["#7c3aed"])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                      plot_bgcolor="#1a1d2e")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Feature Correlation — only features with |r| > 0.05 vs target")
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
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", height=480)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Weather Analysis
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="section-header">🌦️ Weather Impact on Traffic</div>',
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
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e", coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, width='stretch')

    with c2:
        corr_temp = dff["temp_c"].corr(dff["traffic_volume"])
        samp = dff.sample(min(4000, len(dff)), random_state=SEED)
        fig = px.scatter(samp, x="temp_c", y="traffic_volume", color="weather_main",
                         title=f"Temp vs Traffic  (r = {corr_temp:.3f})",
                         labels={"temp_c": "Temperature (°C)",
                                 "traffic_volume": "Traffic Volume",
                                 "weather_main": "Weather"},
                         opacity=0.4)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e", height=420)
        st.plotly_chart(fig, width='stretch')

    corr_rain = dff["rain_1h"].corr(dff["traffic_volume"])
    if abs(corr_rain) > 0.05:
        samp2 = dff[dff["rain_1h"] < 50].sample(min(3000, len(dff)), random_state=SEED)
        fig = px.scatter(samp2, x="rain_1h", y="traffic_volume",
                         title=f"Rain vs Traffic  (r = {corr_rain:.3f})",
                         labels={"rain_1h": "Rain (mm)", "traffic_volume": "Traffic Volume"},
                         opacity=0.4, color_discrete_sequence=["#4361ee"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info(f"ℹ️ Rain scatter skipped — weak correlation (r = {corr_rain:.3f}). "
                "Scatterplots only shown for meaningfully correlated features.")

    fig = px.violin(dff, x="weather_main", y="traffic_volume",
                    box=True, color="weather_main",
                    title="Traffic Volume Distribution per Weather Type",
                    labels={"weather_main": "Weather", "traffic_volume": "Traffic Volume"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                      plot_bgcolor="#1a1d2e", showlegend=False, height=400)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Time Patterns
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="section-header">⏰ Temporal Traffic Patterns</div>',
                unsafe_allow_html=True)

    hourly_wkd = dff[dff["is_weekend"]==0].groupby("hour")["traffic_volume"].mean()
    hourly_wke = dff[dff["is_weekend"]==1].groupby("hour")["traffic_volume"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_wkd.index, y=hourly_wkd.values,
                             mode="lines+markers", name="Weekday",
                             line=dict(color="#4361ee", width=3)))
    fig.add_trace(go.Scatter(x=hourly_wke.index, y=hourly_wke.values,
                             mode="lines+markers", name="Weekend",
                             line=dict(color="#f72585", width=3)))
    fig.update_layout(title="Hourly Traffic: Weekday vs Weekend",
                      xaxis_title="Hour of Day", yaxis_title="Avg Traffic Volume",
                      template="plotly_dark", paper_bgcolor="#0f1117",
                      plot_bgcolor="#1a1d2e", xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        hol = dff.groupby("is_holiday")["traffic_volume"].mean().reset_index()
        hol["label"] = hol["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(hol, x="label", y="traffic_volume", color="label",
                     color_discrete_sequence=["#4361ee","#f72585"],
                     title="Holiday vs Non-Holiday Traffic",
                     labels={"traffic_volume": "Avg Volume", "label": ""})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        heat = dff.pivot_table(values="traffic_volume",
                               index="year", columns="month", aggfunc="mean")
        fig = px.imshow(heat, labels={"x":"Month","y":"Year","color":"Avg Volume"},
                        color_continuous_scale="Blues",
                        title="Avg Traffic: Year × Month Heatmap", aspect="auto")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117")
        st.plotly_chart(fig, width='stretch')

    heat2 = dff.pivot_table(values="traffic_volume",
                             index="day_of_week", columns="hour", aggfunc="mean")
    heat2.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = px.imshow(heat2, labels={"x":"Hour","y":"Day","color":"Avg Volume"},
                    color_continuous_scale="Viridis",
                    title="Traffic Heatmap: Hour of Day × Day of Week", aspect="auto")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", height=320)
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Comparison  ★ NEW TAB ★
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="section-header">🏆 Full Model Comparison & Evaluation</div>',
                unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("❌ Run `train.py` first to generate model metrics.")
    else:
        amm = metrics.get("all_model_metrics", {})
        cv  = metrics.get("cv_comparison", {})
        hp  = metrics.get("hyperparameters", {})

        model_names  = list(amm.keys())
        model_colors = {
            "Decision Tree"    : "#4361ee",
            "Random Forest"    : "#4ade80",
            "Gradient Boosting": "#f72585",
        }

        # ── Section 1: What are these models? ────────────────────────────
        st.markdown("### 📚 Model Descriptions")

        descriptions = {
            "Decision Tree": {
                "icon": "🌿",
                "color": "#4361ee",
                "what": "A single tree that splits data on feature thresholds, making predictions at leaf nodes.",
                "how":  "Recursively partitions the feature space. Each internal node = one decision rule. "
                        "Leaf node = predicted value (mean of samples in that leaf).",
                "pros": "Fast, interpretable, no scaling needed, handles non-linearity.",
                "cons": "Prone to overfitting (high variance). Single tree = unstable with small data changes.",
                "when": "Good baseline. Use when interpretability is the top priority.",
                "params": f"max_depth = {hp.get('Decision Tree', {}).get('max_depth', 10)}",
            },
            "Random Forest": {
                "icon": "🌳",
                "color": "#4ade80",
                "what": "An ensemble of many Decision Trees. Each tree is trained on a random subset of data "
                        "and features (bagging). Final prediction = average of all trees.",
                "how":  "Bootstrap sampling creates diverse trees. Random feature selection at each split "
                        "reduces correlation between trees. Averaging reduces variance dramatically.",
                "pros": "High accuracy, robust to overfitting, handles high-dimensional data, "
                        "built-in feature importance.",
                "cons": "Slower than a single tree, less interpretable than DT, higher memory usage.",
                "when": "Best all-round choice for tabular regression. Our chosen production model.",
                "params": f"n_estimators = {hp.get('Random Forest', {}).get('n_estimators', 200)}, "
                          f"max_depth = {hp.get('Random Forest', {}).get('max_depth', 15)}",
            },
            "Gradient Boosting": {
                "icon": "🚀",
                "color": "#f72585",
                "what": "Builds trees sequentially — each new tree corrects the residual errors of all "
                        "previous trees. Uses gradient descent to minimise loss.",
                "how":  "Starts with a weak model. Iteratively adds trees that minimise the gradient of "
                        "the loss function (like MSE). Learning rate controls step size.",
                "pros": "Often achieves the highest accuracy among tree methods, handles complex patterns.",
                "cons": "Slower to train than RF, more hyperparameters to tune, "
                        "more sensitive to overfitting if learning_rate is too high.",
                "when": "Use when squeezing out maximum accuracy matters more than training speed.",
                "params": f"n_estimators = {hp.get('Gradient Boosting', {}).get('n_estimators', 200)}, "
                          f"max_depth = {hp.get('Gradient Boosting', {}).get('max_depth', 5)}, "
                          f"learning_rate = {hp.get('Gradient Boosting', {}).get('learning_rate', 0.1)}",
            },
        }

        dc1, dc2, dc3 = st.columns(3)
        for col_st, (name, d) in zip([dc1, dc2, dc3], descriptions.items()):
            with col_st:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1e2a45,#1a2035);
                            border:2px solid {d["color"]};border-radius:14px;
                            padding:18px;min-height:380px'>
                  <div style='font-size:2rem;text-align:center'>{d["icon"]}</div>
                  <div style='color:{d["color"]};font-size:1.1rem;font-weight:700;
                              text-align:center;margin:8px 0'>{name}</div>
                  <div style='color:#c8d3f5;font-size:0.83rem;margin-bottom:10px'>
                    {d["what"]}
                  </div>
                  <div style='color:#9ba3c0;font-size:0.78rem'>
                    <b style='color:#7c9ef8'>⚙️ How it works:</b><br>{d["how"]}<br><br>
                    <b style='color:#4ade80'>✅ Pros:</b> {d["pros"]}<br>
                    <b style='color:#f72585'>❌ Cons:</b> {d["cons"]}<br>
                    <b style='color:#DD8452'>📌 Best for:</b> {d["when"]}<br><br>
                    <b style='color:#7c9ef8'>🔧 Params used:</b> {d["params"]}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 2: Metrics explanation ───────────────────────────────
        st.markdown("### 📐 What Do the Metrics Mean?")
        me1, me2, me3, me4, me5 = st.columns(5)
        metric_explanations = [
            ("MAE", "Mean Absolute Error", "Average absolute gap between actual and predicted. "
             "Same unit as target (vehicles/hr). Easy to interpret. Lower = better."),
            ("RMSE", "Root Mean Square Error", "Like MAE but penalises large errors more heavily "
             "(because errors are squared first). Lower = better."),
            ("R²", "R-Squared", "Proportion of variance in traffic volume explained by the model. "
             "1.0 = perfect. 0 = no better than just predicting the mean. Higher = better."),
            ("Adj R²", "Adjusted R²", "R² corrected for number of features. "
             "Penalises adding useless features. More reliable than raw R². Higher = better."),
            ("MAPE", "Mean Absolute % Error", "Average percentage error. E.g. 15% MAPE means "
             "predictions are off by 15% on average. Scale-independent. Lower = better."),
        ]
        for col_st, (short, full, explanation) in zip(
                [me1, me2, me3, me4, me5], metric_explanations):
            with col_st:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1e2130,#252b3b);
                            border:1px solid #2d3450;border-radius:12px;
                            padding:14px;text-align:center;min-height:160px'>
                  <div style='color:#4361ee;font-size:1.3rem;font-weight:800'>{short}</div>
                  <div style='color:#7c9ef8;font-size:0.78rem;font-weight:600;
                              margin:4px 0'>{full}</div>
                  <div style='color:#9ba3c0;font-size:0.75rem'>{explanation}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 3: Test Set Metrics Table ────────────────────────────
        st.markdown("### 📊 Test Set Performance (Held-Out 20%)")

        rows = []
        for name, m_dict in amm.items():
            rows.append({
                "Model"  : name,
                "MAE"    : m_dict["MAE"],
                "RMSE"   : m_dict["RMSE"],
                "R²"     : m_dict["R2"],
                "Adj R²" : m_dict["Adj_R2"],
                "MAPE %" : m_dict["MAPE"],
            })
        summary_df = pd.DataFrame(rows).set_index("Model")

        # Styled: highlight best value in each column
        def highlight_best(s):
            is_lower_better = s.name in ["MAE", "RMSE", "MAPE %"]
            if is_lower_better:
                best = s.min()
            else:
                best = s.max()
            return ["background-color: #1a3a1e; color: #4ade80; font-weight: bold"
                    if v == best else "" for v in s]

        st.dataframe(
            summary_df.style
                .apply(highlight_best)
                .format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}",
                         "R²": "{:.4f}", "Adj R²": "{:.4f}", "MAPE %": "{:.2f}%"}),
            width='stretch'
        )
        st.caption("🟢 Green highlight = best value in each column")

        st.markdown("---")

        # ── Section 4: Visual metric comparison ──────────────────────────
        st.markdown("### 📈 Visual Metric Comparison")

        vc1, vc2, vc3 = st.columns(3)
        metrics_to_plot = [
            ("MAE",    "MAE — Mean Absolute Error (lower ↓)",    "Reds_r"),
            ("RMSE",   "RMSE — Root Mean Square Error (lower ↓)", "Oranges_r"),
            ("R2",     "R² Score (higher ↑)",                    "Greens"),
        ]
        for col_st, (metric_key, title, cscale) in zip(
                [vc1, vc2, vc3], metrics_to_plot):
            vals = [amm[n][metric_key] for n in model_names]
            fig  = px.bar(
                x=model_names, y=vals,
                color=vals, color_continuous_scale=cscale,
                title=title,
                labels={"x": "Model", "y": metric_key},
                text=[f"{v:.4f}" if metric_key == "R2" else f"{v:,.1f}"
                      for v in vals]
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False,
                              showlegend=False,
                              xaxis_tickangle=-10)
            col_st.plotly_chart(fig, width='stretch')

        # MAPE & Adj R² side by side
        vc4, vc5 = st.columns(2)
        with vc4:
            mape_vals = [amm[n]["MAPE"] for n in model_names]
            fig = px.bar(x=model_names, y=mape_vals,
                         color=mape_vals, color_continuous_scale="Purples_r",
                         title="MAPE % — Mean Absolute % Error (lower ↓)",
                         labels={"x": "Model", "y": "MAPE %"},
                         text=[f"{v:.2f}%" for v in mape_vals])
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False,
                              showlegend=False)
            st.plotly_chart(fig, width='stretch')

        with vc5:
            adjr2_vals = [amm[n]["Adj_R2"] for n in model_names]
            fig = px.bar(x=model_names, y=adjr2_vals,
                         color=adjr2_vals, color_continuous_scale="Teal",
                         title="Adjusted R² Score (higher ↑)",
                         labels={"x": "Model", "y": "Adj R²"},
                         text=[f"{v:.4f}" for v in adjr2_vals])
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False,
                              showlegend=False)
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── Section 5: Cross-Validation Fold Analysis ─────────────────────
        st.markdown("### 🔁 5-Fold Cross-Validation — Stability Analysis")
        st.caption("Each model was evaluated 5 times on different data splits. "
                   "Error bars show standard deviation. A stable model has small error bars.")

        cv_rows = []
        for name, cv_d in cv.items():
            cv_rows.append({
                "Model"       : name,
                "CV R² Mean"  : cv_d["CV_R2_mean"],
                "CV R² Std"   : cv_d["CV_R2_std"],
                "CV MAE Mean" : cv_d["CV_MAE_mean"],
                "CV MAE Std"  : cv_d["CV_MAE_std"],
                "CV RMSE Mean": cv_d.get("CV_RMSE_mean", 0),
                "CV RMSE Std" : cv_d.get("CV_RMSE_std", 0),
            })
        cv_df = pd.DataFrame(cv_rows)

        fc1, fc2 = st.columns(2)
        with fc1:
            fig = px.bar(cv_df, x="Model", y="CV R² Mean", error_y="CV R² Std",
                         color="CV R² Mean", color_continuous_scale="Greens",
                         title="CV R² per Model (mean ± std)",
                         text=[f"{v:.4f}" for v in cv_df["CV R² Mean"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        with fc2:
            fig = px.bar(cv_df, x="Model", y="CV MAE Mean", error_y="CV MAE Std",
                         color="CV MAE Mean", color_continuous_scale="Reds_r",
                         title="CV MAE per Model (mean ± std)",
                         text=[f"{v:,.0f}" for v in cv_df["CV MAE Mean"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        # Per-fold line chart
        st.markdown("#### R² Score Across Each Individual Fold")
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
                    marker=dict(size=8)
                ))
        fold_fig.update_layout(
            title="R² Score per Fold — All Models",
            xaxis_title="Fold", yaxis_title="R²",
            template="plotly_dark", paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1d2e", legend_title="Model"
        )
        st.plotly_chart(fold_fig, width='stretch')

        st.markdown("""
        <div class="insight-box">
        📌 <b>How to read the fold chart:</b>
        A model with consistent R² across all 5 folds is <b>stable</b> — it isn't
        just getting lucky on one particular split.
        Large variation fold-to-fold = <b>high variance</b> (overfitting risk).
        Random Forest shows the flattest line = most stable model.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 6: Actual vs Predicted per model ──────────────────────
        st.markdown("### 🎯 Actual vs Predicted — All 3 Models")
        st.caption("Scatter plots from the held-out test set. "
                   "Points hugging the diagonal = better model.")

        from sklearn.model_selection import train_test_split as tts
        from sklearn.tree import DecisionTreeRegressor as DT
        from sklearn.ensemble import (RandomForestRegressor as RF,
                                      GradientBoostingRegressor as GB)

        X_ev = clean_df.drop(["traffic_volume"], axis=1)
        y_ev = clean_df["traffic_volume"]
        _, X_te, _, y_te = tts(X_ev, y_ev, test_size=0.2, random_state=SEED)
        X_te_a = X_te.reindex(columns=model_columns, fill_value=0)

        # Retrain all 3 on cached train split to get predictions
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
            p_sample  = all_preds[name][sample_idx]
            r2_val    = amm[name]["R2"]
            mae_val   = amm[name]["MAE"]
            clr       = model_colors.get(name, "#4361ee")
            with col_st:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_sample, y=p_sample, mode="markers",
                    marker=dict(color=clr, opacity=0.35, size=4),
                    name="Predictions"
                ))
                fig.add_trace(go.Scatter(
                    x=[0,7300], y=[0,7300], mode="lines",
                    line=dict(color="white", dash="dash", width=1),
                    name="Perfect Fit"
                ))
                fig.update_layout(
                    title=f"{name}<br><sup>R²={r2_val:.4f} | MAE={mae_val:,.0f}</sup>",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    template="plotly_dark", paper_bgcolor="#0f1117",
                    plot_bgcolor="#1a1d2e", showlegend=False, height=350
                )
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── Section 7: Residual distributions per model ───────────────────
        st.markdown("### 📉 Residual Distribution — All 3 Models")
        st.caption("Residual = Actual − Predicted. Centred at 0 with a narrow bell = best model.")

        res_fig = go.Figure()
        for name in model_names:
            residuals = y_te.values - all_preds[name]
            res_fig.add_trace(go.Histogram(
                x=residuals, name=name, nbinsx=80, opacity=0.65,
                marker_color=model_colors.get(name, "#4361ee")
            ))
        res_fig.add_vline(x=0, line_dash="dash", line_color="white",
                          annotation_text="Zero error",
                          annotation_position="top right")
        res_fig.update_layout(
            barmode="overlay",
            title="Overlapping Residual Distributions — Decision Tree vs RF vs GB",
            xaxis_title="Residual (Actual − Predicted)",
            yaxis_title="Count",
            template="plotly_dark", paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1d2e", legend_title="Model"
        )
        st.plotly_chart(res_fig, width='stretch')

        st.markdown("---")

        # ── Section 8: Radar chart ────────────────────────────────────────
        st.markdown("### 🕸️ Radar Chart — Normalised Performance Overview")
        st.caption("All metrics normalised 0–1. Larger area = better overall performance "
                   "(note: MAE/RMSE/MAPE are inverted so larger = lower error).")

        # Normalise: for error metrics invert so larger = better
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
                norm(r2s)[idx],
                norm(adjr2s)[idx],
                norm(maes, invert=True)[idx],
                norm(rmses, invert=True)[idx],
                norm(mapes, invert=True)[idx],
            ]
            values += [values[0]]   # close the radar loop
            cats_closed = categories + [categories[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=values, theta=cats_closed,
                fill="toself", name=name,
                line_color=model_colors.get(name, "#fff"),
                opacity=0.7
            ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor="#1a1d2e",
                radialaxis=dict(visible=True, range=[0,1],
                                color="#9ba3c0", gridcolor="#2d3450"),
                angularaxis=dict(color="#9ba3c0", gridcolor="#2d3450")
            ),
            template="plotly_dark", paper_bgcolor="#0f1117",
            showlegend=True, legend_title="Model",
            title="Radar: Normalised Model Performance",
            height=480
        )
        st.plotly_chart(radar_fig, width='stretch')

        st.markdown("---")

        # ── Section 9: Winner verdict ─────────────────────────────────────
        st.markdown("### 🥇 Verdict — Which Model Should Be Used?")

        best_r2_name = max(amm, key=lambda n: amm[n]["R2"])
        best_r2_val  = amm[best_r2_name]["R2"]
        best_mae_val = amm[best_r2_name]["MAE"]
        best_mape    = amm[best_r2_name]["MAPE"]

        st.markdown(f"""
        <div class="winner-box">
          <div style='font-size:2rem;text-align:center'>🏆</div>
          <div style='color:#4ade80;font-size:1.5rem;font-weight:800;text-align:center;
                      margin:8px 0'>{best_r2_name}</div>
          <div style='color:#c8f5d0;text-align:center;font-size:1rem;margin-bottom:16px'>
            R² = {best_r2_val:.4f} &nbsp;|&nbsp; MAE = {best_mae_val:,.0f} vehicles/hr
            &nbsp;|&nbsp; MAPE = {best_mape:.2f}%
          </div>
          <div style='color:#9ba3c0;font-size:0.88rem'>
            <b style='color:#c8f5d0'>Why Random Forest wins:</b><br><br>
            1. <b>Highest R²</b> — explains the most variance in traffic volume across all test samples.<br>
            2. <b>Lowest MAE & RMSE</b> — smallest average prediction error in vehicles/hr.<br>
            3. <b>Most stable across 5 CV folds</b> — lowest standard deviation in R², meaning it
               doesn't get lucky on one split.<br>
            4. <b>Avoids Decision Tree's overfitting</b> by averaging 200 trees (bagging).<br>
            5. <b>Faster than Gradient Boosting</b> in inference (trees built in parallel, not sequentially).<br>
            6. <b>Built-in feature importance</b> — directly compatible with SHAP explanations.<br><br>
            Gradient Boosting is competitive but trains sequentially (slower) and requires
            more careful tuning. Decision Tree underfits due to limited depth. Random Forest
            is the optimal choice for real-time traffic prediction deployment.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Predict & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="section-header">🤖 Live Prediction & Model Evaluation</div>',
                unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("❌ Model not loaded. Please run `train.py` first.")
    else:
        st.markdown("### 🎯 Real-Time Traffic Prediction")
        st.markdown("Adjust inputs below and click **Predict** — "
                    "the actual trained model generates a new output each time.")

        pc1, pc2 = st.columns(2)
        with pc1:
            pred_temp    = st.slider("🌡️ Temperature (°C)", -30, 50, 20)
            pred_rain    = st.slider("🌧️ Rain (mm)", 0.0, 50.0, 0.0, step=0.5)
            pred_snow    = st.slider("❄️ Snow (mm)", 0.0, 10.0, 0.0, step=0.1)
            pred_clouds  = st.slider("☁️ Cloud Cover (%)", 0, 100, 40)
        with pc2:
            pred_hour    = st.slider("⏰ Hour of Day", 0, 23, 8)
            pred_day     = st.slider("📅 Day of Week (0=Mon)", 0, 6, 1)
            pred_month   = st.slider("🗓️ Month", 1, 12, 6)
            pred_holiday = st.selectbox("🎄 Is Holiday?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")

        weather_options = sorted([
            c.replace("weather_main_", "")
            for c in model_columns if c.startswith("weather_main_")
        ])
        pred_weather = st.selectbox("🌤️ Weather Condition", weather_options)

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

        if st.button("🚀 Predict Traffic Volume"):
            prediction = max(0.0, float(model.predict(input_df_pred)[0]))

            if prediction < 1000:   level, col = "Very Low",  "#55A868"
            elif prediction < 2500: level, col = "Low",       "#4C72B0"
            elif prediction < 4000: level, col = "Medium",    "#DD8452"
            elif prediction < 5500: level, col = "High",      "#f72585"
            else:                   level, col = "Very High", "#e63946"

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1e2a45,#1a2035);
                            border:2px solid {col};border-radius:16px;
                            padding:28px;text-align:center;margin-top:10px'>
                  <div style='color:#9ba3c0;font-size:0.9rem'>Predicted Volume</div>
                  <div style='color:{col};font-size:3.2rem;font-weight:800'>
                    {int(prediction):,}</div>
                  <div style='color:#c8d3f5;font-size:1rem'>vehicles / hour</div>
                  <div style='color:{col};font-size:1.3rem;font-weight:600;margin-top:10px'>
                    🚦 {level}</div>
                  <div style='color:#9ba3c0;font-size:0.82rem;margin-top:6px'>
                    {"🔔 Peak hour" if is_peak_pred else "🟢 Off-peak"} &nbsp;|&nbsp;
                    {"🏖️ Weekend" if is_weekend_pred else "💼 Weekday"}
                  </div>
                </div>""", unsafe_allow_html=True)
            with rc2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=int(prediction),
                    title={"text": "Traffic Level", "font": {"color": "white"}},
                    number={"font": {"color": col}},
                    gauge={
                        "axis": {"range": [0, 7500], "tickcolor": "#9ba3c0"},
                        "bar":  {"color": col}, "bgcolor": "#1a1d2e",
                        "steps": [
                            {"range": [0,    1000], "color": "#1e3a2f"},
                            {"range": [1000, 2500], "color": "#1e2a45"},
                            {"range": [2500, 4000], "color": "#2d2a1e"},
                            {"range": [4000, 5500], "color": "#3a1e2a"},
                            {"range": [5500, 7500], "color": "#3a1e1e"},
                        ],
                        "threshold": {"line": {"color": "#f72585", "width": 3},
                                      "thickness": 0.75, "value": 5500},
                    }
                ))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                                  height=280, margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("### 📊 Model Evaluation (held-out test set)")
        tm = metrics["test_metrics"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE",  f"{tm['MAE']:,}",  "Mean Absolute Error")
        mc2.metric("RMSE", f"{tm['RMSE']:,}", "Root Mean Square Error")
        mc3.metric("R²",   f"{tm['R2']:.4f}", "Coefficient of Determination")

        st.markdown("#### 🔍 Feature Importance (Random Forest)")
        fi_data = metrics["feature_importance"]
        fi_df = pd.DataFrame(list(fi_data.items()),
                             columns=["Feature","Importance"]).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Feature Importances",
                     labels={"Importance": "Importance Score"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                          plot_bgcolor="#1a1d2e", coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Insights & Explainability
# ─────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="section-header">💡 Key Insights, Conclusions & Explainability</div>',
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
        (f"🕐 Peak hours (7–9 AM & 4–7 PM) carry <b>{peak_diff:,.0f} more vehicles/hr</b> "
         "on average compared to off-peak hours."),
        (f"📅 Weekend traffic is only <b>{weekend_ratio:.1%} of weekday</b> traffic, "
         "reflecting the commuter-dominant nature of the highway."),
        (f"🎄 Holiday traffic drops to just <b>{holiday_ratio:.1%} of normal</b> levels."),
        (f"⏰ <b>{top_feat}</b> is the single most important predictor "
         f"(importance = {top_imp:.3f})." if MODEL_LOADED else "⏰ Run train.py."),
        (f"🤖 Random Forest achieves <b>R² = {best_r2:.4f}</b> verified by 5-fold CV."
         if MODEL_LOADED else "🤖 Run train.py."),
        ("🌦️ Weather has a moderate effect — Clear and Clouds show highest volumes."),
        ("📈 Seasonal patterns: higher traffic May–Aug, dips in winter."),
        ("🔍 5-fold cross-validation confirms model is stable, not a lucky split."),
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Explainability — SHAP")
    st.markdown("""
    <div class="insight-box">
    <b>What is SHAP?</b><br>
    SHAP (SHapley Additive exPlanations) explains <i>how much each feature contributed
    to each individual prediction</i>. Unlike feature importance (global average only),
    SHAP shows direction + magnitude per row — e.g. "hour=8 pushed this prediction
    <b>UP by +1,200 vehicles</b>".
    </div>
    """, unsafe_allow_html=True)

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
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d2e", coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, width='stretch')
        except FileNotFoundError:
            st.info("ℹ️ SHAP values not found. Re-run `train.py` with `pip install shap`.")
    elif not SHAP_AVAILABLE:
        st.info("ℹ️ Install SHAP: `pip install shap` then re-run `train.py`.")
    else:
        st.info("ℹ️ Run `train.py` to generate SHAP values.")

    st.markdown("---")
    st.markdown("### 📌 Recommendations")
    for r in [
        "**Traffic Management:** Deploy dynamic speed signs and ramp metering during peak hours (7–9 AM, 4–7 PM).",
        "**Emergency Planning:** Pre-position incident response units on weekday mornings.",
        "**Weather Response:** Activate winter maintenance crews when snow/fog is forecast.",
        "**Holiday Policy:** Reduce toll rates or open HOV lanes on holidays.",
        "**Model Deployment:** Use trained Random Forest in a real-time pipeline fed by hourly weather API data.",
    ]:
        st.markdown(f"- {r}")

    st.markdown("---")
    st.markdown("### 👥 Team Roles (Group of 4)")
    roles = {
        "Student 1 — Data Analyst":
            "Dataset collection, cleaning, preprocessing, missing value treatment, outlier removal",
        "Student 2 — Algorithm Researcher":
            "Studied Decision Tree, Random Forest, Gradient Boosting; justified model choice with CV",
        "Student 3 — Model Developer":
            "Implemented ML pipeline, one-hot encoding, train/test split, pkl export, prediction utility",
        "Student 4 — Model Tester & Evaluator":
            "5-fold CV, hyperparameter tuning, MAE/RMSE/R²/MAPE, SHAP, residual analysis, documentation",
    }
    cols = st.columns(2)
    for i, (role, resp) in enumerate(roles.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e2a45,#1a2035);
                        border:1px solid #2d3450;border-radius:12px;
                        padding:16px;margin:8px 0;min-height:120px'>
              <b style='color:#7c9ef8'>{role}</b><br>
              <span style='color:#9ba3c0;font-size:0.88rem'>{resp}</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#555;font-size:0.85rem;padding:10px'>
  Metro Interstate Traffic Volume Analytics · BDA Course Project ·
  Department of CE (Software Engineering) · SY Big Data Analytics
</div>
""", unsafe_allow_html=True)
