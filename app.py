import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
# ==============================
LOOKBACK = 720  # MUST match training

FEATURE_COLS = [
    "T1_PO4", "po4_rate", "T1_O2", "IN_Q", "TEMPERATURE",
    "IN_METAL_Q", "METAL_Q", "MAX_CF",
    "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET"
]

OPERATIONAL_THRESHOLD = 1.5
REGULATORY_THRESHOLD = 2.0

tf.keras.backend.set_floatx("float32")

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Phosphorus Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# DARK MODE CSS
# ==============================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, system-ui;
    background-color: #0B0F14;
    color: #E5E7EB;
}
.main-header {
    font-size: 2.3rem;
    font-weight: 700;
    color: #F9FAFB;
}
.sub-header {
    font-size: 1.35rem;
    font-weight: 600;
    color: #E5E7EB;
    border-bottom: 1px solid #1F2937;
    margin-top: 2rem;
}
.metric-card {
    background: #111827;
    padding: 1.4rem;
    border-radius: 12px;
    border: 1px solid #1F2937;
}
.risk-low { background: #064E3B; color: #D1FAE5; padding: 1rem; border-radius: 10px; }
.risk-medium { background: #78350F; color: #FEF3C7; padding: 1rem; border-radius: 10px; }
.risk-high { background: #7F1D1D; color: #FEE2E2; padding: 1rem; border-radius: 10px; }
.stCaption { color: #9CA3AF !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "lstm_risk_model.keras",
            compile=False,
            safe_mode=False,
            custom_objects={"InputLayer": tf.keras.layers.InputLayer}
        )
        st.sidebar.success("✓ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.warning("⚠️ Model unavailable — demo mode")
        st.sidebar.caption(str(e)[:100])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        dummy = np.zeros((1, LOOKBACK, len(FEATURE_COLS)), dtype=np.float32)
        model.predict(dummy, verbose=0)
        return model

# ==============================
# LOAD SCALER
# ==============================
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("feature_scaler.save")
        st.sidebar.success("✓ Scaler loaded")
        return scaler
    except:
        from sklearn.preprocessing import StandardScaler
        st.sidebar.warning("⚠️ Using default scaler")
        return StandardScaler()

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("scada_with_risk_labels.csv")
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        st.sidebar.success(f"✓ Data loaded ({len(df)} rows)")
        return df
    except:
        st.sidebar.warning("⚠️ Using synthetic demo data")
        return generate_demo_data()

def generate_demo_data():
    n = 10000
    t = pd.date_range("2024-01-01", periods=n, freq="2min")
    df = pd.DataFrame({
        "date": t,
        "T1_PO4": np.cumsum(np.random.randn(n) * 0.01) + 0.5,
        "po4_rate": np.random.randn(n) * 0.1,
        "T1_O2": np.random.rand(n),
        "IN_Q": 800 + 200 * np.random.randn(n),
        "TEMPERATURE": 20 + 5 * np.random.randn(n),
        "IN_METAL_Q": 40 + 20 * np.random.randn(n),
        "METAL_Q": 5 + 10 * np.random.randn(n),
        "MAX_CF": 70 + 20 * np.random.randn(n),
        "PROCESSPHASE_INLET": np.random.choice([1, 2], n),
        "PROCESSPHASE_OUTLET": np.random.choice([1, 2], n)
    })
    return df

# ==============================
# HELPER
# ==============================
def get_risk_category(p):
    if p < 0.3: return "LOW", "🟢", "risk-low"
    if p < 0.7: return "MODERATE", "🟡", "risk-medium"
    return "HIGH", "🔴", "risk-high"

# ==============================
# INITIALIZE
# ==============================
st.sidebar.markdown("### 📊 System Initialization")
model = load_model()
scaler = load_scaler()
df = load_data()

# ==============================
# SIDEBAR TIME SELECTION
# ==============================
st.sidebar.markdown("### 📅 Analysis Timestamp")

selected_date = st.sidebar.date_input(
    "Date", df["date"].max().date()
)
selected_time = st.sidebar.time_input(
    "Time", df["date"].max().time()
)

selected_datetime = pd.Timestamp(
    datetime.combine(selected_date, selected_time)
)

st.sidebar.markdown("---")
st.sidebar.success(selected_datetime.strftime("%Y-%m-%d %H:%M"))

# ==============================
# MAIN UI
# ==============================
st.markdown('<div class="main-header">🚨 Phosphorus Early Warning System</div>', unsafe_allow_html=True)
st.markdown("Community-centric AI dashboard | **ChemTech 2026**")

tab1, tab2 = st.tabs(["📊 Risk Assessment", "📈 Process Trends"])

# ==============================
# TAB 1
# ==============================
with tab1:
    window_df = df[df["date"] <= selected_datetime].tail(LOOKBACK)

    X = scaler.transform(window_df[FEATURE_COLS])
    X = X.reshape(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)

    risk_prob = float(model.predict(X, verbose=0)[0][0])
    risk_cat, icon, css = get_risk_category(risk_prob)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Probability", f"{risk_prob:.2f}")

    with col2:
        st.markdown(f"<div class='{css}'><h3>{icon} {risk_cat}</h3></div>", unsafe_allow_html=True)

    with col3:
        po4 = window_df["T1_PO4"].iloc[-1]
        st.metric("Current PO₄", f"{po4:.2f} mg/L")

# ==============================
# TAB 2
# ==============================
with tab2:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=window_df["date"], y=window_df["T1_PO4"],
                             name="PO₄ (mg/L)", line=dict(color="#EF4444")), 1, 1)

    fig.add_trace(go.Scatter(x=window_df["date"], y=window_df["T1_O2"],
                             name="O₂ (mg/L)", line=dict(color="#3B82F6")), 2, 1)

    fig.add_trace(go.Scatter(x=window_df["date"], y=window_df["IN_Q"],
                             name="Flow (m³/h)", line=dict(color="#22C55E")), 3, 1)

    fig.update_layout(
        height=650,
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",
        font=dict(color="#E5E7EB"),
        legend=dict(font=dict(color="#E5E7EB"))
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("© ChemTech 2026 | Phosphorus Early Warning System")
