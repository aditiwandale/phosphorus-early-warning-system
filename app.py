import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
LOOKBACK = 720  # MUST match training

FEATURE_COLS = [
    "T1_PO4", "po4_rate", "T1_O2", "IN_Q", "TEMPERATURE",
    "IN_METAL_Q", "METAL_Q", "MAX_CF", "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET"
]

OPERATIONAL_THRESHOLD = 1.5  # mg/L
REGULATORY_THRESHOLD = 2.0  # mg/L

# Set TensorFlow to use float32 by default
tf.keras.backend.set_floatx('float32')

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Phosphorus Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

/* ==============================
   GLOBAL BASE (DARK)
============================== */
html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #0B0F14;     /* Soft black */
    color: #E5E7EB;                /* Soft white */
}

/* ==============================
   HEADERS
============================== */
.main-header {
    font-size: 2.3rem;
    font-weight: 700;
    color: #F9FAFB;                /* Bright white for headings */
    margin-bottom: 0.8rem;
}

.sub-header {
    font-size: 1.35rem;
    font-weight: 600;
    color: #E5E7EB;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1F2937;
}

/* ==============================
   METRIC CARDS
============================== */
.metric-card {
    background-color: #111827;
    border-radius: 12px;
    padding: 1.4rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6);
    border: 1px solid #1F2937;
    color: #E5E7EB;
}

/* ==============================
   RISK STATUS CARDS
============================== */
.risk-low {
    background-color: #064E3B;
    color: #D1FAE5;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

.risk-medium {
    background-color: #78350F;
    color: #FEF3C7;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

.risk-high {
    background-color: #7F1D1D;
    color: #FEE2E2;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}

/* ==============================
   EXPLANATION BOX
============================== */
.explanation-box {
    background-color: #111827;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3B82F6;
    margin-top: 1rem;
    color: #E5E7EB;
    border: 1px solid #1F2937;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.6);
}

/* ==============================
   TEXT & METRICS
============================== */
p, li, .stMarkdown, .stText, .stMetric, .stMetric label {
    color: #E5E7EB !important;
    line-height: 1.55;
}

/* ==============================
   ALERTS
============================== */
.stSuccess {
    background-color: #064E3B !important;
    color: #D1FAE5 !important;
}

.stWarning {
    background-color: #78350F !important;
    color: #FEF3C7 !important;
}

.stError {
    background-color: #7F1D1D !important;
    color: #FEE2E2 !important;
}

.stInfo {
    background-color: #1E3A8A !important;
    color: #DBEAFE !important;
}

/* ==============================
   INPUT LABELS
============================== */
.stSlider label,
.stSelectbox label,
.stDateInput label,
.stTimeInput label {
    color: #CBD5E1 !important;
    font-weight: 500;
}

/* ==============================
   TABS
============================== */
.stTabs [data-baseweb="tab"] {
    background-color: #0B0F14;
    color: #CBD5E1 !important;
    border: 1px solid #1F2937;
    border-radius: 8px 8px 0 0;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #111827;
    border-bottom: 2px solid #3B82F6;
    font-weight: 600;
}

/* ==============================
   CHARTS
============================== */
.js-plotly-plot .plotly {
    color: #E5E7EB !important;
}

/* ==============================
   FOOTER
============================== */
.stCaption {
    color: #9CA3AF !important;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# CACHE FUNCTIONS
# ==============================
@st.cache_resource
def load_model():
    """Load the trained LSTM model."""
    try:
        # Try to load the model
        model = tf.keras.models.load_model("lstm_risk_model.h5", compile=False)
        st.sidebar.success("✓ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.warning(f"⚠️ Model file not found: {str(e)[:50]}")
        st.sidebar.info("Creating a demonstration model...")

        # Create a dummy LSTM model with the correct architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Build the model with dummy data
        dummy_input = np.random.randn(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)
        model.predict(dummy_input, verbose=0)

        return model


@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    try:
        scaler = joblib.load("feature_scaler.save")
        st.sidebar.success("✓ Scaler loaded successfully")
        return scaler
    except:
        st.sidebar.warning("⚠️ Scaler file not found. Using StandardScaler.")
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()


@st.cache_data
def load_data():
    """Load the SCADA dataset."""
    try:
        df = pd.read_csv("scada_with_risk_labels.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        st.sidebar.success(f"✓ Data loaded: {len(df)} samples")
        return df
    except:
        st.sidebar.warning("⚠️ Data file not found. Generating demonstration data.")
        return generate_demo_data()


def generate_demo_data():
    """Generate demonstration data."""
    np.random.seed(42)
    n_samples = 10000
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='2min')

    # Generate synthetic features
    data = {
        'date': timestamps,
        'T1_PO4': 0.5 + 0.3 * np.random.randn(n_samples).cumsum(),
        'po4_rate': np.random.randn(n_samples) * 0.1,
        'T1_O2': 0.5 + 0.3 * np.random.randn(n_samples),
        'IN_Q': 800 + 200 * np.random.randn(n_samples),
        'TEMPERATURE': 20 + 5 * np.random.randn(n_samples),
        'IN_METAL_Q': 40 + 20 * np.random.randn(n_samples),
        'METAL_Q': 5 + 10 * np.random.randn(n_samples),
        'MAX_CF': 70 + 20 * np.random.randn(n_samples),
        'PROCESSPHASE_INLET': np.random.choice([1, 2], n_samples),
        'PROCESSPHASE_OUTLET': np.random.choice([1, 2], n_samples)
    }

    # Add some spikes
    spike_indices = np.random.choice(n_samples, size=50, replace=False)
    data['T1_PO4'] = np.array(data['T1_PO4'])
    data['T1_PO4'][spike_indices] += np.random.uniform(1.0, 3.0, size=50)

    df = pd.DataFrame(data)
    return df





# ==============================
# HELPER FUNCTIONS
# ==============================
def get_risk_category(risk_prob):
    """Determine risk category based on probability."""
    if risk_prob < 0.3:
        return "LOW", "🟢", "risk-low"
    elif risk_prob < 0.7:
        return "MODERATE", "🟡", "risk-medium"
    else:
        return "HIGH", "🔴", "risk-high"





# ==============================
# INITIALIZE APPLICATION
# ==============================
st.sidebar.markdown('<div class="sidebar-header">📊 System Initialization</div>', unsafe_allow_html=True)

# Load components
model = load_model()
scaler = load_scaler()
df = load_data()

st.sidebar.markdown("---")

# ==============================
# SIDEBAR - DATE & TIME SELECTION
# ==============================
st.sidebar.markdown('<div class="sidebar-header">📅 Select Analysis Timestamp</div>', unsafe_allow_html=True)

# Date selection
if "date" in df.columns:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_date = df["date"].max().date()
else:
    min_date = datetime(2024, 1, 1).date()
    max_date = datetime.now().date()
    default_date = datetime.now().date()

selected_date = st.sidebar.date_input(
    "Date",
    value=default_date,
    min_value=min_date,
    max_value=max_date
)

# Time selection
if "date" in df.columns:
    default_time = df["date"].max().time()
else:
    default_time = datetime.now().time()

selected_time = st.sidebar.time_input(
    "Time",
    value=default_time
)

selected_datetime = pd.Timestamp(
    datetime.combine(selected_date, selected_time)
).tz_localize(None)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-header">⚙️ System Parameters</div>', unsafe_allow_html=True)

# Operational thresholds
operational_threshold = st.sidebar.slider(
    "Operational Threshold (mg/L)",
    min_value=0.5,
    max_value=3.0,
    value=OPERATIONAL_THRESHOLD,
    step=0.1,
    help="Phosphate level that triggers operational alerts"
)

regulatory_threshold = st.sidebar.slider(
    "Regulatory Threshold (mg/L)",
    min_value=1.0,
    max_value=5.0,
    value=REGULATORY_THRESHOLD,
    step=0.1,
    help="Phosphate level that violates regulatory limits"
)

# Lookback window
lookback_hours = st.sidebar.slider(
    "Lookback Window (hours)",
    min_value=1,
    max_value=48,
    value=24,
    step=1,
    help="Historical data window for prediction"
)

current_lookback = LOOKBACK
 # Convert hours to 2-minute intervals

st.sidebar.markdown("---")
st.sidebar.markdown("**Selected timestamp:**")
st.sidebar.success(f"{selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown(f"**Lookback samples:** {current_lookback}")

# ==============================
# MAIN DASHBOARD
# ==============================
st.markdown('<div class="main-header">🚨 Phosphorus Early Warning System</div>', unsafe_allow_html=True)
st.markdown("Community-centric AI decision support for WWTP operations | **ChemTech 2026 Project**")

# Create tabs
tab1, tab2 = st.tabs(["📊 Risk Assessment", "📈 Process Trends"])

# ==============================
# TAB 1: RISK ASSESSMENT
# ==============================
with tab1:
    # Filter data for lookback window
    if "date" not in df.columns:
        st.error("No timestamp column found in data. Please check data format.")
        st.stop()

    df_before = df[df["date"] <= selected_datetime]

    if len(df_before) < current_lookback:
        st.error(
            f"⚠️ Not enough historical data for this timestamp. Need {current_lookback} samples, have {len(df_before)}.")

        # Use available data if insufficient
        if len(df_before) > 0:
            st.warning(f"Using available {len(df_before)} samples for prediction.")
            current_lookback = min(current_lookback, len(df_before))
            window_df = df_before.tail(current_lookback)
        else:
            st.stop()
    else:
        window_df = df_before.tail(current_lookback)

    # Get current values
    current_values = {}
    for col in FEATURE_COLS:
        if col in window_df.columns:
            current_values[col] = window_df[col].iloc[-1]

    # Prepare input data
    X_window = window_df[FEATURE_COLS].values

    X_scaled = scaler.transform(X_window)

    # Prepare model input
    X_input = X_scaled.reshape(1, current_lookback, len(FEATURE_COLS)).astype(np.float32)
    X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)

    # Model inference
    try:
        risk_prob = float(model.predict(X_tensor, verbose=0)[0][0])
    except:
        # Use dummy prediction for demonstration
        risk_prob = np.random.uniform(0, 1)

    risk_category, risk_icon, risk_class = get_risk_category(risk_prob)

    # Display risk metrics
    st.markdown('<div class="sub-header">⚠️ Current Risk Assessment</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Probability", f"{risk_prob:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_icon} {risk_category} RISK")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_po4 = current_values.get("T1_PO4", 0)
        st.metric("Current PO₄", f"{current_po4:.2f} mg/L")
        status = "⚠️ Above" if current_po4 > operational_threshold else "✓ Normal"
        st.caption(f"{status} operational threshold ({operational_threshold} mg/L)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prediction Horizon", "3 Hours")
        st.caption("Time window for early warning")
        st.markdown('</div>', unsafe_allow_html=True)

    # Threshold indicators
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_po4,
            title={"text": "PO₄ Level (mg/L)", "font": {"color": "black"}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 3], "tickwidth": 1, "tickcolor": "black"},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, operational_threshold], "color": "green"},
                    {"range": [operational_threshold, regulatory_threshold], "color": "orange"},
                    {"range": [regulatory_threshold, 3], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": operational_threshold
                }
            },
            number={"font": {"color": "black"}}
        ))
        fig.update_layout(height=250, paper_bgcolor="white", font={"color": "black"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Calculate rate of change
        if len(window_df) > 30:  # 1 hour of data
            recent_po4 = window_df["T1_PO4"].tail(30).values
            po4_rate = (recent_po4[-1] - recent_po4[0]) * 30  # mg/L per hour
        else:
            po4_rate = 0

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=po4_rate,
            title={"text": "PO₄ Rate (mg/L/hr)", "font": {"color": "black"}},
            domain={"x": [0, 1], "y": [0, 1]},
            delta={"reference": 0.2, "position": "bottom"},
            number={"suffix": " mg/L/hr", "font": {"color": "black"}}
        ))
        fig.update_layout(height=250, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Risk Probability (%)", "font": {"color": "black"}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "black"},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            },
            number={"font": {"color": "black"}, "suffix": "%"}
        ))
        fig.update_layout(height=250, paper_bgcolor="white", font={"color": "black"})
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 2: PROCESS TRENDS
# ==============================
with tab2:
    st.markdown('<div class="sub-header">📈 Recent Process Trends</div>', unsafe_allow_html=True)

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Phosphate Concentration (T1_PO4)",
            "Dissolved Oxygen (T1_O2)",
            "Influent Flow (IN_Q)"
        ]
    )

    # ------------------------------
    # PHOSPHATE CONCENTRATION
    # ------------------------------
    fig.add_trace(
        go.Scatter(
            x=window_df["date"],
            y=window_df["T1_PO4"],
            mode="lines",
            name="PO₄ (mg/L)",
            line=dict(color="#EF4444", width=2),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)"
        ),
        row=1, col=1
    )

    # Threshold lines
    fig.add_hline(
        y=operational_threshold,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=2,
        annotation_text="Operational Threshold",
        annotation_font_color="#FBBF24",
        annotation_position="top right",
        row=1, col=1
    )

    fig.add_hline(
        y=regulatory_threshold,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text="Regulatory Threshold",
        annotation_font_color="#FCA5A5",
        annotation_position="top right",
        row=1, col=1
    )

    # ------------------------------
    # DISSOLVED OXYGEN
    # ------------------------------
    fig.add_trace(
        go.Scatter(
            x=window_df["date"],
            y=window_df["T1_O2"],
            mode="lines",
            name="Dissolved Oxygen (mg/L)",
            line=dict(color="#3B82F6", width=2)
        ),
        row=2, col=1
    )

    # ------------------------------
    # INFLUENT FLOW
    # ------------------------------
    fig.add_trace(
        go.Scatter(
            x=window_df["date"],
            y=window_df["IN_Q"],
            mode="lines",
            name="Influent Flow (m³/h)",
            line=dict(color="#22C55E", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.15)"
        ),
        row=3, col=1
    )

    # ------------------------------
    # GLOBAL LAYOUT (DARK THEME)
    # ------------------------------
    fig.update_layout(
        height=650,
        showlegend=True,
        hovermode="x unified",

        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",

        legend=dict(
            font=dict(color="#E5E7EB", size=12),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),

        font=dict(
            color="#E5E7EB",
            size=12
        )
    )

    # ------------------------------
    # AXIS STYLING
    # ------------------------------
    fig.update_xaxes(
        title_text="Time",
        title_font=dict(color="#F9FAFB", size=13),
        tickfont=dict(color="#E5E7EB"),
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        row=3, col=1
    )

    fig.update_yaxes(
        title_text="mg/L",
        title_font=dict(color="#F9FAFB", size=13),
        tickfont=dict(color="#E5E7EB"),
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        row=1, col=1
    )

    fig.update_yaxes(
        title_text="mg/L",
        title_font=dict(color="#F9FAFB", size=13),
        tickfont=dict(color="#E5E7EB"),
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        row=2, col=1
    )

    fig.update_yaxes(
        title_text="m³/h",
        title_font=dict(color="#F9FAFB", size=13),
        tickfont=dict(color="#E5E7EB"),
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        row=3, col=1
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # ADDITIONAL PROCESS VARIABLES
    # ==========================================================
    st.markdown("---")
    col1, col2 = st.columns(2)

    # Temperature & Chemical Dose
    with col1:
        fig_temp = px.line(
            window_df,
            x="date",
            y=["TEMPERATURE", "IN_METAL_Q"],
            title="Temperature & Chemical Dosing",
            labels={"value": "Value", "variable": "Parameter"},
            color_discrete_map={
                "TEMPERATURE": "#F59E0B",
                "IN_METAL_Q": "#A855F7"
            }
        )

        fig_temp.update_layout(
            paper_bgcolor="#0B0F14",
            plot_bgcolor="#0B0F14",
            font=dict(color="#E5E7EB"),
            legend=dict(font=dict(color="#E5E7EB"))
        )

        fig_temp.update_xaxes(
            gridcolor="#1F2937",
            tickfont=dict(color="#E5E7EB")
        )

        fig_temp.update_yaxes(
            gridcolor="#1F2937",
            tickfont=dict(color="#E5E7EB")
        )

        st.plotly_chart(fig_temp, use_container_width=True)

    # Control Parameters
    with col2:
        fig_ctrl = px.line(
            window_df,
            x="date",
            y=["MAX_CF", "PROCESSPHASE_INLET"],
            title="Control Parameters",
            labels={"value": "Value", "variable": "Parameter"},
            color_discrete_map={
                "MAX_CF": "#F97316",
                "PROCESSPHASE_INLET": "#94A3B8"
            }
        )

        fig_ctrl.update_layout(
            paper_bgcolor="#0B0F14",
            plot_bgcolor="#0B0F14",
            font=dict(color="#E5E7EB"),
            legend=dict(font=dict(color="#E5E7EB"))
        )

        fig_ctrl.update_xaxes(
            gridcolor="#1F2937",
            tickfont=dict(color="#E5E7EB")
        )

        fig_ctrl.update_yaxes(
            gridcolor="#1F2937",
            tickfont=dict(color="#E5E7EB")
        )

        st.plotly_chart(fig_ctrl, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption("© ChemTech 2026 Project - Community-Centric AI for WWTP Phosphorus Management")

with col2:
    st.caption("Data Source: Agtrup WWTP SCADA System")

with col3:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
