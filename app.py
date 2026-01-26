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
import os

warnings.filterwarnings('ignore')

# ==============================
# FIX 1: Remove Keras 3 patches (not needed for Streamlit Cloud)
# ==============================

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
    background-color: #0B0F14;
    color: #E5E7EB;
}

/* ==============================
   SIDEBAR HEADERS
============================== */
.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F9FAFB;
    margin-bottom: 0.8rem;
    padding: 0.5rem 0;
}

/* ==============================
   HEADERS
============================== */
.main-header {
    font-size: 2.3rem;
    font-weight: 700;
    color: #F9FAFB;
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
   CHARTS
============================== */
.js-plotly-plot .plotly {
    color: #E5E7EB !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CACHE FUNCTIONS - FIXED FOR STREAMLIT CLOUD
# ==============================
@st.cache_resource
def load_model():
    """Load the LSTM model with fallback to dummy model."""
    try:
        # Check if model file exists
        if os.path.exists("lstm_risk_model.keras"):
            model = tf.keras.models.load_model(
                "lstm_risk_model.keras",
                compile=False
            )
            st.sidebar.success("✓ Model loaded successfully")
        else:
            st.sidebar.warning("⚠️ Model file not found. Using demonstration model.")
            model = create_dummy_model()
            
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading model: {str(e)}")
        st.sidebar.info("Using demonstration model instead.")
        model = create_dummy_model()
    
    return model

def create_dummy_model():
    """Create a dummy LSTM model for demonstration."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model (important for prediction)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Build the model with dummy data
    dummy_input = np.random.randn(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)
    model.predict(dummy_input, verbose=0)
    
    return model

@st.cache_resource
def load_scaler():
    """Load the feature scaler with fallback."""
    try:
        if os.path.exists("feature_scaler.save"):
            scaler = joblib.load("feature_scaler.save")
            st.sidebar.success("✓ Scaler loaded successfully")
        elif os.path.exists("feature_scaler (1).save"):
            scaler = joblib.load("feature_scaler (1).save")
            st.sidebar.success("✓ Scaler loaded successfully")
        else:
            st.sidebar.warning("⚠️ Scaler file not found. Creating dummy scaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Fit with dummy data
            dummy_data = np.random.randn(100, len(FEATURE_COLS))
            scaler.fit(dummy_data)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading scaler: {str(e)}. Creating dummy scaler.")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dummy_data = np.random.randn(100, len(FEATURE_COLS))
        scaler.fit(dummy_data)
    
    return scaler

@st.cache_data
def load_data():
    """Load the SCADA dataset with fallback to demo data."""
    try:
        # Try multiple possible file names
        if os.path.exists("scada_with_risk_labels.csv"):
            df = pd.read_csv("scada_with_risk_labels.csv")
        elif os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
        else:
            raise FileNotFoundError("No data file found")
            
        # Ensure date column is properly formatted
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.tz_localize(None)
            # Remove rows with NaT in date
            df = df.dropna(subset=['date'])
        
        st.sidebar.success(f"✓ Data loaded: {len(df)} samples")
        return df
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading data: {str(e)}. Generating demonstration data.")
        return generate_demo_data()

def generate_demo_data():
    """Generate demonstration data."""
    np.random.seed(42)
    n_samples = 10000
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='2min')

    # Generate synthetic features
    data = {
        'date': timestamps,
        'T1_PO4': np.clip(0.5 + 0.01 * np.random.randn(n_samples).cumsum(), 0, 3),
        'po4_rate': np.random.randn(n_samples) * 0.1,
        'T1_O2': np.clip(0.5 + 0.3 * np.random.randn(n_samples), 0, 2),
        'IN_Q': np.clip(800 + 200 * np.random.randn(n_samples), 0, 1500),
        'TEMPERATURE': np.clip(20 + 5 * np.random.randn(n_samples), 10, 30),
        'IN_METAL_Q': np.clip(40 + 20 * np.random.randn(n_samples), 0, 100),
        'METAL_Q': np.clip(5 + 10 * np.random.randn(n_samples), 0, 30),
        'MAX_CF': np.clip(70 + 20 * np.random.randn(n_samples), 40, 100),
        'PROCESSPHASE_INLET': np.random.choice([1, 2], n_samples),
        'PROCESSPHASE_OUTLET': np.random.choice([1, 2], n_samples)
    }

    # Add some spikes
    spike_indices = np.random.choice(n_samples, size=50, replace=False)
    data['T1_PO4'] = np.array(data['T1_PO4'])
    data['T1_PO4'][spike_indices] += np.random.uniform(1.0, 2.0, size=50)
    data['T1_PO4'] = np.clip(data['T1_PO4'], 0, 5)

    df = pd.DataFrame(data)
    
    # Calculate po4_rate as difference
    df['po4_rate'] = df['T1_PO4'].diff().fillna(0)
    
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

# Load components with error handling
try:
    model = load_model()
    scaler = load_scaler()
    df = load_data()
except Exception as e:
    st.sidebar.error(f"Error initializing: {str(e)}")
    st.stop()

st.sidebar.markdown("---")

# ==============================
# SIDEBAR - DATE & TIME SELECTION
# ==============================
st.sidebar.markdown('<div class="sidebar-header">📅 Select Analysis Timestamp</div>', unsafe_allow_html=True)

# Date selection
if "date" in df.columns and len(df) > 0:
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

# Time selection - simplified
selected_time = st.sidebar.time_input(
    "Time",
    value=datetime.now().time()
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

st.sidebar.markdown("---")
st.sidebar.markdown("**Selected timestamp:**")
st.sidebar.success(f"{selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown(f"**Lookback samples:** {LOOKBACK}")
st.sidebar.markdown(f"**Data points:** {len(df):,}")

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
    
    # Check if we have enough data
    if len(df_before) < LOOKBACK:
        st.warning(f"⚠️ Not enough historical data. Need {LOOKBACK} samples, have {len(df_before)}.")
        current_lookback = min(LOOKBACK, len(df_before))
        if current_lookback == 0:
            st.error("No data available for selected timestamp.")
            st.stop()
        window_df = df_before.tail(current_lookback).copy()
    else:
        window_df = df_before.tail(LOOKBACK).copy()
        current_lookback = LOOKBACK
    
    # Ensure all feature columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in window_df.columns]
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}. Using default values.")
        for col in missing_cols:
            window_df[col] = 0

    # Get current values
    current_values = {}
    for col in FEATURE_COLS:
        if col in window_df.columns and len(window_df) > 0:
            current_values[col] = window_df[col].iloc[-1]
        else:
            current_values[col] = 0

    # Prepare input data
    try:
        X_window = window_df[FEATURE_COLS].values
        X_scaled = scaler.transform(X_window)
        
        # Prepare model input
        X_input = X_scaled.reshape(1, current_lookback, len(FEATURE_COLS)).astype(np.float32)
        
        # Model inference
        risk_prob = float(model.predict(X_input, verbose=0)[0][0])
    except Exception as e:
        st.warning(f"⚠️ Prediction error: {str(e)}. Using random probability.")
        risk_prob = np.random.uniform(0, 1)

    risk_category, risk_icon, risk_class = get_risk_category(risk_prob)
    current_po4 = current_values.get("T1_PO4", 0)

    # Display risk metrics
    st.markdown('<div class="sub-header">⚠️ Current Risk Assessment</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Probability", f"{risk_prob:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_icon} {risk_category} RISK")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
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
            title={"text": "PO₄ Level (mg/L)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 3], "tickwidth": 1},
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
            }
        ))
        fig.update_layout(height=250, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Calculate rate of change
        if len(window_df) > 30:
            recent_po4 = window_df["T1_PO4"].tail(30).values
            po4_rate = (recent_po4[-1] - recent_po4[0]) * 30  # mg/L per hour
        else:
            po4_rate = 0

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=po4_rate,
            title={"text": "PO₄ Rate (mg/L/hr)"},
            domain={"x": [0, 1], "y": [0, 1]},
            delta={"reference": 0.2, "position": "bottom"},
            number={"suffix": " mg/L/hr"}
        ))
        fig.update_layout(height=250, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Risk Probability (%)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
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
            number={"suffix": "%"}
        ))
        fig.update_layout(height=250, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 2: PROCESS TRENDS
# ==============================
with tab2:
    st.markdown('<div class="sub-header">📈 Recent Process Trends</div>', unsafe_allow_html=True)
    
    # Ensure we have enough data
    if len(window_df) < 2:
        st.warning("Not enough data for trend visualization.")
    else:
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
            font=dict(color="#E5E7EB", size=12),
            legend=dict(
                font=dict(color="#E5E7EB", size=12),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
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
                labels={"value": "Value", "variable": "Parameter"}
            )

            fig_temp.update_traces(
                line=dict(width=2),
                selector=dict(name="TEMPERATURE")
            )
            fig_temp.data[0].line.color = "#F59E0B"
            if len(fig_temp.data) > 1:
                fig_temp.data[1].line.color = "#A855F7"

            fig_temp.update_layout(
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="#E5E7EB"),
                legend=dict(font=dict(color="#E5E7EB"))
            )

            fig_temp.update_xaxes(gridcolor="#1F2937", tickfont=dict(color="#E5E7EB"))
            fig_temp.update_yaxes(gridcolor="#1F2937", tickfont=dict(color="#E5E7EB"))

            st.plotly_chart(fig_temp, use_container_width=True)

        # Control Parameters
        with col2:
            # Ensure columns exist
            plot_cols = []
            if "MAX_CF" in window_df.columns:
                plot_cols.append("MAX_CF")
            if "PROCESSPHASE_INLET" in window_df.columns:
                plot_cols.append("PROCESSPHASE_INLET")
            
            if plot_cols:
                fig_ctrl = px.line(
                    window_df,
                    x="date",
                    y=plot_cols,
                    title="Control Parameters",
                    labels={"value": "Value", "variable": "Parameter"}
                )
                
                fig_ctrl.update_layout(
                    paper_bgcolor="#0B0F14",
                    plot_bgcolor="#0B0F14",
                    font=dict(color="#E5E7EB"),
                    legend=dict(font=dict(color="#E5E7EB"))
                )
                
                fig_ctrl.update_xaxes(gridcolor="#1F2937", tickfont=dict(color="#E5E7EB"))
                fig_ctrl.update_yaxes(gridcolor="#1F2937", tickfont=dict(color="#E5E7EB"))
                
                st.plotly_chart(fig_ctrl, use_container_width=True)
            else:
                st.info("Control parameters not available in data.")

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
