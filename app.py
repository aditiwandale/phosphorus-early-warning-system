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

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================
# TENSORFLOW CONFIGURATION
# ==============================
# Set environment variables for TensorFlow compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for compatibility

# Set TensorFlow to use float32 by default
tf.keras.backend.set_floatx('float32')

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
# CUSTOM CSS
# ==============================
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
   SIDEBAR
============================== */
.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 1rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #374151;
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
    """Load the trained LSTM model with robust error handling."""
    try:
        # First try to load .keras format
        try:
            model = tf.keras.models.load_model(
                "lstm_risk_model.keras",
                compile=False
            )
            st.sidebar.success("✓ Model loaded successfully from .keras")
            return model
        except:
            # Try .h5 format
            try:
                model = tf.keras.models.load_model(
                    "lstm_risk_model.h5",
                    compile=False
                )
                st.sidebar.success("✓ Model loaded successfully from .h5")
                return model
            except:
                # Try .pb format (SavedModel)
                try:
                    model = tf.keras.models.load_model(
                        "lstm_risk_model.keras",
                        compile=False
                    )
                    st.sidebar.success("✓ Model loaded successfully from SavedModel")
                    return model
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load saved model: {str(e)[:80]}")
                    raise Exception("All model loading attempts failed")
    except Exception as e:
        st.sidebar.info("Creating a demonstration model...")
        
        # Create a demonstration model with the exact architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build with dummy data
        dummy_input = np.random.randn(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)
        model.predict(dummy_input, verbose=0)
        
        st.sidebar.info("✓ Demonstration model created and ready")
        return model

@st.cache_resource
def load_scaler():
    """Load the feature scaler with multiple format support."""
    # Try multiple scaler file formats
    scaler_files = ["feature_scaler.pkl", "feature_scaler.save", "scaler.joblib"]
    
    for scaler_file in scaler_files:
        try:
            scaler = joblib.load(scaler_file)
            st.sidebar.success(f"✓ Scaler loaded from {scaler_file}")
            return scaler
        except FileNotFoundError:
            continue
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading {scaler_file}: {str(e)[:50]}")
            continue
    
    # If no scaler file found, create a new one
    st.sidebar.info("Creating new scaler for demonstration...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Load data to fit scaler
    df_temp = load_data()
    if len(df_temp) > 0 and all(col in df_temp.columns for col in FEATURE_COLS):
        scaler.fit(df_temp[FEATURE_COLS].values)
        st.sidebar.info("✓ Scaler fitted with available data")
    
    return scaler

@st.cache_data
def load_data():
    """Load the SCADA dataset with robust error handling."""
    try:
        # Try multiple possible file names
        data_files = ["scada_with_risk_labels.csv", "scada_data.csv", "wwtp_data.csv"]
        
        for data_file in data_files:
            try:
                df = pd.read_csv(data_file)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.tz_localize(None)
                st.sidebar.success(f"✓ Data loaded from {data_file}: {len(df)} samples")
                return df
            except FileNotFoundError:
                continue
                
        # If no file found, create demonstration data
        raise FileNotFoundError("No data file found")
        
    except Exception as e:
        st.sidebar.warning("⚠️ Data file not found. Generating demonstration data.")
        return generate_demo_data()

def generate_demo_data():
    """Generate realistic demonstration data."""
    np.random.seed(42)
    n_samples = 10000
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='2min')

    # Generate base features with realistic patterns
    base_po4 = 0.5 + 0.1 * np.sin(np.linspace(0, 20*np.pi, n_samples))
    base_po4 += 0.05 * np.random.randn(n_samples).cumsum()
    
    # Add daily patterns
    hour_of_day = (timestamps.hour + timestamps.minute/60).values
    daily_pattern = 0.2 * np.sin(2*np.pi * hour_of_day / 24)
    base_po4 += daily_pattern
    
    data = {
        'date': timestamps,
        'T1_PO4': base_po4,
        'po4_rate': np.random.randn(n_samples) * 0.05,
        'T1_O2': 2.0 + 0.5 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + 0.2*np.random.randn(n_samples),
        'IN_Q': 800 + 100 * np.sin(np.linspace(0, 5*np.pi, n_samples)) + 50*np.random.randn(n_samples),
        'TEMPERATURE': 18 + 4 * np.sin(np.linspace(0, 2*np.pi, n_samples)) + 1*np.random.randn(n_samples),
        'IN_METAL_Q': 40 + 10 * np.sin(np.linspace(0, 3*np.pi, n_samples)) + 5*np.random.randn(n_samples),
        'METAL_Q': 5 + 2 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + 1*np.random.randn(n_samples),
        'MAX_CF': 70 + 10 * np.sin(np.linspace(0, np.pi, n_samples)) + 5*np.random.randn(n_samples),
        'PROCESSPHASE_INLET': np.random.choice([1, 2], n_samples, p=[0.7, 0.3]),
        'PROCESSPHASE_OUTLET': np.random.choice([1, 2], n_samples, p=[0.8, 0.2])
    }

    # Add realistic spikes (5% of data points)
    spike_indices = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    data['T1_PO4'] = np.array(data['T1_PO4'])
    data['T1_PO4'][spike_indices] += np.random.uniform(0.5, 2.5, size=len(spike_indices))
    
    # Add some consecutive spikes for realistic incident patterns
    for i in range(10):
        start_idx = np.random.randint(0, n_samples-10)
        data['T1_PO4'][start_idx:start_idx+5] += np.random.uniform(1.0, 3.0)
    
    df = pd.DataFrame(data)
    df['T1_PO4'] = np.maximum(0.1, df['T1_PO4'])  # Ensure non-negative values
    
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

def prepare_prediction_data(df, selected_datetime, lookback_samples):
    """Prepare data for model prediction."""
    # Filter data up to selected datetime
    df_before = df[df["date"] <= selected_datetime].copy()
    
    if len(df_before) < lookback_samples:
        # If not enough data, pad with zeros or use available data
        available_samples = len(df_before)
        padding_needed = lookback_samples - available_samples
        
        if available_samples > 0:
            # Use available data and pad with zeros
            window_df = df_before.tail(available_samples).copy()
            
            # Create padding dataframe
            if padding_needed > 0:
                padding_data = {}
                for col in FEATURE_COLS:
                    padding_data[col] = [window_df[col].iloc[0]] * padding_needed
                
                padding_df = pd.DataFrame(padding_data)
                window_df = pd.concat([padding_df, window_df], ignore_index=True)
        else:
            # No data available, create zero-filled dataframe
            data = {}
            for col in FEATURE_COLS:
                data[col] = [0.0] * lookback_samples
            window_df = pd.DataFrame(data)
    else:
        window_df = df_before.tail(lookback_samples).copy()
    
    return window_df

# ==============================
# INITIALIZE APPLICATION
# ==============================
st.sidebar.markdown('<div class="sidebar-header">📊 System Initialization</div>', unsafe_allow_html=True)

# Load components with progress indicators
with st.sidebar:
    with st.spinner("Loading model..."):
        model = load_model()
    
    with st.spinner("Loading scaler..."):
        scaler = load_scaler()
    
    with st.spinner("Loading data..."):
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
    default_date = min(max_date, datetime.now().date())
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

current_lookback = lookback_hours * 30  # Convert hours to 2-minute intervals (120 mins/hour / 2 mins = 30 samples/hour)

st.sidebar.markdown("---")
st.sidebar.markdown("**Selected timestamp:**")
st.sidebar.success(f"{selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown(f"**Lookback samples:** {current_lookback}")
st.sidebar.markdown(f"**Lookback hours:** {lookback_hours}")

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
    # Prepare data for prediction
    window_df = prepare_prediction_data(df, selected_datetime, current_lookback)
    
    # Get current values
    current_values = {}
    for col in FEATURE_COLS:
        if col in window_df.columns and len(window_df) > 0:
            current_values[col] = window_df[col].iloc[-1]
        else:
            current_values[col] = 0.0
    
    # Prepare input data for model
    X_window = window_df[FEATURE_COLS].values
    
    # Scale the data
    try:
        X_scaled = scaler.transform(X_window)
    except:
        # If scaler hasn't been fitted, fit and transform
        scaler.fit(X_window)
        X_scaled = scaler.transform(X_window)
    
    # Reshape for LSTM
    X_input = X_scaled.reshape(1, current_lookback, len(FEATURE_COLS)).astype(np.float32)
    
    # Model inference
    with st.spinner("Running risk assessment..."):
        try:
            risk_prob = float(model.predict(X_input, verbose=0)[0][0])
            risk_prob = np.clip(risk_prob, 0.0, 1.0)  # Ensure valid probability
        except Exception as e:
            st.error(f"⚠️ Model prediction error: {str(e)[:100]}")
            # Fallback: Calculate risk based on current phosphate level
            current_po4 = current_values.get("T1_PO4", 0)
            if current_po4 > regulatory_threshold:
                risk_prob = 0.85
            elif current_po4 > operational_threshold:
                risk_prob = 0.55
            else:
                risk_prob = 0.2
    
    risk_category, risk_icon, risk_class = get_risk_category(risk_prob)
    
    # Display risk metrics
    st.markdown('<div class="sub-header">⚠️ Current Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Probability", f"{risk_prob:.2%}")
        st.progress(float(risk_prob))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_icon} {risk_category} RISK")
        if risk_category == "HIGH":
            st.markdown("⚠️ Immediate action required")
        elif risk_category == "MODERATE":
            st.markdown("⚠️ Monitor closely")
        else:
            st.markdown("✓ Operations normal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_po4 = current_values.get("T1_PO4", 0)
        st.metric("Current PO₄", f"{current_po4:.2f} mg/L")
        
        # Determine status
        if current_po4 > regulatory_threshold:
            status = "🔴 Regulatory Violation"
            color = "red"
        elif current_po4 > operational_threshold:
            status = "🟡 Above Operational"
            color = "orange"
        else:
            status = "🟢 Normal"
            color = "green"
        
        st.markdown(f'<span style="color:{color}; font-weight:bold;">{status}</span>', unsafe_allow_html=True)
        st.caption(f"Operational: {operational_threshold} mg/L")
        st.caption(f"Regulatory: {regulatory_threshold} mg/L")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prediction Horizon", "3 Hours")
        st.caption("Time window for early warning")
        st.metric("Data Points Used", f"{len(window_df)}")
        st.caption(f"Last update: {window_df['date'].iloc[-1].strftime('%H:%M') if 'date' in window_df.columns else 'N/A'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Threshold indicators
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Create phosphate gauge
        current_po4 = current_values.get("T1_PO4", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_po4,
            title={"text": "PO₄ Level (mg/L)", "font": {"size": 16}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, max(3, regulatory_threshold*1.2)], "tickwidth": 1},
                "bar": {"color": "#3B82F6"},
                "steps": [
                    {"range": [0, operational_threshold], "color": "#10B981"},
                    {"range": [operational_threshold, regulatory_threshold], "color": "#F59E0B"},
                    {"range": [regulatory_threshold, max(3, regulatory_threshold*1.2)], "color": "#EF4444"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": operational_threshold
                }
            },
            number={"suffix": " mg/L", "font": {"size": 24}}
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor="#0B0F14",
            font={"color": "#E5E7EB"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate rate of change
        if len(window_df) > 30:
            recent_po4 = window_df["T1_PO4"].tail(30).values
            if len(recent_po4) > 1:
                po4_rate = (recent_po4[-1] - recent_po4[0]) / (30 * 2/60)  # mg/L per hour
            else:
                po4_rate = 0
        else:
            po4_rate = 0
        
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=po4_rate,
            title={"text": "PO₄ Rate of Change", "font": {"size": 16}},
            domain={"x": [0, 1], "y": [0, 1]},
            delta={"reference": 0, "position": "bottom", "font": {"size": 14}},
            number={
                "suffix": " mg/L/hr",
                "font": {"size": 24},
                "valueformat": ".3f"
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor="#0B0F14",
            font={"color": "#E5E7EB"}
        )
        
        # Add color based on rate
        if po4_rate > 0.1:
            fig.update_traces(number_font_color="#EF4444")
        elif po4_rate < -0.1:
            fig.update_traces(number_font_color="#10B981")
        else:
            fig.update_traces(number_font_color="#E5E7EB")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Risk probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Risk Probability", "font": {"size": 16}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#8B5CF6"},
                "steps": [
                    {"range": [0, 30], "color": "#10B981"},
                    {"range": [30, 70], "color": "#F59E0B"},
                    {"range": [70, 100], "color": "#EF4444"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 70
                }
            },
            number={"suffix": "%", "font": {"size": 24}}
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor="#0B0F14",
            font={"color": "#E5E7EB"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 2: PROCESS TRENDS
# ==============================
with tab2:
    st.markdown('<div class="sub-header">📈 Recent Process Trends</div>', unsafe_allow_html=True)
    
    # Ensure we have data
    if len(window_df) == 0:
        st.warning("No data available for the selected time period.")
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
            title_font=dict(color="#E5E7EB", size=18),
            legend=dict(
                font=dict(color="#E5E7EB", size=12),
                bgcolor="rgba(0,0,0,0.5)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(color="#E5E7EB", size=12)
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
            showgrid=True,
            row=3, col=1
        )
        
        fig.update_yaxes(
            title_text="mg/L",
            title_font=dict(color="#F9FAFB", size=13),
            tickfont=dict(color="#E5E7EB"),
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            showgrid=True,
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="mg/L",
            title_font=dict(color="#F9FAFB", size=13),
            tickfont=dict(color="#E5E7EB"),
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            showgrid=True,
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="m³/h",
            title_font=dict(color="#F9FAFB", size=13),
            tickfont=dict(color="#E5E7EB"),
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            showgrid=True,
            row=3, col=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==========================================================
        # ADDITIONAL PROCESS VARIABLES
        # ==========================================================
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature & Chemical Dose
            fig_temp = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=["Temperature", "Chemical Dosing"]
            )
            
            fig_temp.add_trace(
                go.Scatter(
                    x=window_df["date"],
                    y=window_df["TEMPERATURE"],
                    mode="lines",
                    name="Temperature (°C)",
                    line=dict(color="#F59E0B", width=2)
                ),
                row=1, col=1
            )
            
            fig_temp.add_trace(
                go.Scatter(
                    x=window_df["date"],
                    y=window_df["IN_METAL_Q"],
                    mode="lines",
                    name="Inlet Metal (mg/L)",
                    line=dict(color="#A855F7", width=2)
                ),
                row=2, col=1
            )
            
            fig_temp.update_layout(
                height=400,
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="#E5E7EB"),
                legend=dict(font=dict(color="#E5E7EB"))
            )
            
            fig_temp.update_xaxes(
                gridcolor="#1F2937",
                tickfont=dict(color="#E5E7EB"),
                row=2, col=1
            )
            
            fig_temp.update_yaxes(
                gridcolor="#1F2937",
                tickfont=dict(color="#E5E7EB")
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Control Parameters
            fig_ctrl = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=["Maximum CF", "Process Phase"]
            )
            
            fig_ctrl.add_trace(
                go.Scatter(
                    x=window_df["date"],
                    y=window_df["MAX_CF"],
                    mode="lines",
                    name="Max CF (%)",
                    line=dict(color="#F97316", width=2)
                ),
                row=1, col=1
            )
            
            fig_ctrl.add_trace(
                go.Scatter(
                    x=window_df["date"],
                    y=window_df["PROCESSPHASE_INLET"],
                    mode="lines",
                    name="Process Phase",
                    line=dict(color="#94A3B8", width=2)
                ),
                row=2, col=1
            )
            
            fig_ctrl.update_layout(
                height=400,
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="#E5E7EB"),
                legend=dict(font=dict(color="#E5E7EB"))
            )
            
            fig_ctrl.update_xaxes(
                gridcolor="#1F2937",
                tickfont=dict(color="#E5E7EB"),
                row=2, col=1
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
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
