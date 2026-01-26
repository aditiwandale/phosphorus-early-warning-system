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

warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
LOOKBACK = 720
FEATURE_COLS = [
    "T1_PO4", "po4_rate", "T1_O2", "IN_Q", "TEMPERATURE",
    "IN_METAL_Q", "METAL_Q", "MAX_CF", "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET"
]

OPERATIONAL_THRESHOLD = 1.5
REGULATORY_THRESHOLD = 2.0

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Phosphorus Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
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
.metric-card {
    background-color: #111827;
    border-radius: 12px;
    padding: 1.4rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6);
    border: 1px solid #1F2937;
    color: #E5E7EB;
}
.risk-low { background-color: #064E3B; color: #D1FAE5; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600; }
.risk-medium { background-color: #78350F; color: #FEF3C7; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600; }
.risk-high { background-color: #7F1D1D; color: #FEE2E2; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================
# ROBUST MODEL LOADING
# ==============================
@st.cache_resource
def load_model():
    """Load model with multiple fallback strategies."""
    try:
        # First try to load from .keras file
        if os.path.exists("lstm_risk_model.keras"):
            st.sidebar.info("🔄 Attempting to load LSTM model...")
            
            # METHOD 1: Try direct load with safe mode
            try:
                model = tf.keras.models.load_model(
                    "lstm_risk_model.keras",
                    compile=False,
                    safe_mode=False  # Disable safe mode for compatibility
                )
                st.sidebar.success("✓ Model loaded successfully")
                return model
            except Exception as e1:
                st.sidebar.warning(f"Method 1 failed: {str(e1)[:80]}")
                
                # METHOD 2: Create model architecture and load weights
                try:
                    st.sidebar.info("🔄 Trying alternative loading method...")
                    # Define the model architecture
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(32),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(16, activation='relu'),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    
                    # Try to load weights only
                    model.load_weights("lstm_risk_model.keras")
                    model.compile(optimizer='adam', loss='binary_crossentropy')
                    st.sidebar.success("✓ Model weights loaded successfully")
                    return model
                except Exception as e2:
                    st.sidebar.warning(f"Method 2 failed: {str(e2)[:80]}")
        
        # If all loading attempts fail, create demonstration model
        st.sidebar.warning("⚠️ Using demonstration model")
        return create_demo_model()
        
    except Exception as e:
        st.sidebar.error(f"⚠️ Model loading error: {str(e)[:80]}")
        return create_demo_model()

def create_demo_model():
    """Create a demonstration model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(LOOKBACK, len(FEATURE_COLS))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Initialize with dummy prediction
    dummy_data = np.random.randn(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)
    _ = model.predict(dummy_data, verbose=0)
    
    return model
@st.cache_resource
def load_model():
    """Load or create model."""
    try:
        # Try to load the model
        if os.path.exists("lstm_risk_model.keras"):
            st.sidebar.info("🔄 Loading LSTM model...")
            try:
                # Try with compile=False
                model = tf.keras.models.load_model(
                    "lstm_risk_model.keras",
                    compile=False
                )
                # Manually compile if needed
                if not hasattr(model, 'optimizer') or model.optimizer is None:
                    model.compile(optimizer='adam', loss='binary_crossentropy')
                st.sidebar.success("✓ Model loaded successfully")
                return model
            except Exception as e:
                st.sidebar.warning(f"⚠️ Model load failed: {str(e)[:80]}")
        
        # If loading fails, create demo model
        st.sidebar.info("📊 Creating demonstration model")
        return create_demo_model()
        
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {str(e)[:80]}")
        return create_demo_model()

@st.cache_data
def load_data():
    """Load or generate data."""
    try:
        if os.path.exists("scada_with_risk_labels.csv"):
            df = pd.read_csv("scada_with_risk_labels.csv")
        elif os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
        else:
            raise FileNotFoundError("No data file found")
            
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df = df.dropna(subset=['date'])
            
        st.sidebar.success(f"✓ Data loaded: {len(df)} rows")
        return df
    except:
        st.sidebar.warning("⚠️ Generating demo data")
        return generate_demo_data()

def generate_demo_data():
    """Generate demonstration data."""
    np.random.seed(42)
    n_samples = 10000
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='2min')
    
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
    
    df = pd.DataFrame(data)
    df['po4_rate'] = df['T1_PO4'].diff().fillna(0)
    return df

# ==============================
# HELPER FUNCTIONS
# ==============================
def get_risk_category(risk_prob):
    if risk_prob < 0.3:
        return "LOW", "🟢", "risk-low"
    elif risk_prob < 0.7:
        return "MODERATE", "🟡", "risk-medium"
    else:
        return "HIGH", "🔴", "risk-high"

# ==============================
# INITIALIZE
# ==============================
st.sidebar.markdown("### 📊 System Initialization")
model = load_model()
scaler = load_scaler()
df = load_data()

# Fit scaler if needed
if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == 0:
    scaler.fit(df[FEATURE_COLS].head(1000))

st.sidebar.markdown("---")

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.markdown("### 📅 Select Analysis Time")

if "date" in df.columns:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_date = max_date
else:
    min_date = datetime(2024, 1, 1).date()
    max_date = datetime.now().date()
    default_date = max_date

selected_date = st.sidebar.date_input("Date", value=default_date)
selected_time = st.sidebar.time_input("Time", value=datetime.now().time())
selected_datetime = datetime.combine(selected_date, selected_time)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Threshold Settings")

operational_threshold = st.sidebar.slider(
    "Operational Threshold (mg/L)", 0.5, 3.0, OPERATIONAL_THRESHOLD, 0.1
)

regulatory_threshold = st.sidebar.slider(
    "Regulatory Threshold (mg/L)", 1.0, 5.0, REGULATORY_THRESHOLD, 0.1
)

st.sidebar.markdown("---")
st.sidebar.success(f"**Selected:** {selected_datetime.strftime('%Y-%m-%d %H:%M')}")

# ==============================
# MAIN DASHBOARD
# ==============================
st.markdown('<div class="main-header">🚨 Phosphorus Early Warning System</div>', unsafe_allow_html=True)
st.markdown("Community-centric AI decision support for WWTP operations")

tab1, tab2 = st.tabs(["📊 Risk Assessment", "📈 Process Trends"])

# ==============================
# TAB 1: RISK ASSESSMENT
# ==============================
with tab1:
    # Prepare data
    # Prepare data - fix datetime comparison
    selected_timestamp = pd.Timestamp(selected_datetime)
    # Ensure both are timezone-naive
    selected_timestamp = selected_timestamp.tz_localize(None) if selected_timestamp.tz else selected_timestamp
    
    if df["date"].dt.tz is not None:
        df_before = df[df["date"].dt.tz_localize(None) <= selected_timestamp]
    else:
        df_before = df[df["date"] <= selected_timestamp]
        
    if len(df_before) < LOOKBACK:
        st.warning(f"⚠️ Limited data: {len(df_before)}/{LOOKBACK} samples")
        current_lookback = min(LOOKBACK, len(df_before))
        if current_lookback == 0:
            st.error("No data for selected time")
            st.stop()
        window_df = df_before.tail(current_lookback)
    else:
        window_df = df_before.tail(LOOKBACK)
        current_lookback = LOOKBACK
    
    # Ensure all columns exist
    for col in FEATURE_COLS:
        if col not in window_df.columns:
            window_df[col] = 0
    
    # Scale data
    try:
        X_scaled = scaler.transform(window_df[FEATURE_COLS])
    except:
        X_scaled = window_df[FEATURE_COLS].values
    
    # Make prediction
    X_input = X_scaled.reshape(1, current_lookback, len(FEATURE_COLS)).astype(np.float32)
    
    try:
        risk_prob = float(model.predict(X_input, verbose=0)[0][0])
    except:
        # Simulate risk based on PO4 levels
        current_po4 = window_df["T1_PO4"].iloc[-1] if len(window_df) > 0 else 0
        risk_prob = min(current_po4 / 3.0, 1.0)  # Simple linear mapping
    
    risk_category, risk_icon, risk_class = get_risk_category(risk_prob)
    current_po4 = window_df["T1_PO4"].iloc[-1] if len(window_df) > 0 else 0
    
    # Display metrics
    st.markdown('<div class="sub-header">⚠️ Current Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Probability", f"{risk_prob:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_icon} {risk_category}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current PO₄", f"{current_po4:.2f} mg/L")
        st.caption(f"Threshold: {operational_threshold} mg/L")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Samples", f"{current_lookback}")
        st.caption("Lookback window")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gauges
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_po4,
            title={"text": "PO₄ Level (mg/L)"},
            gauge={
                "axis": {"range": [0, 3]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, operational_threshold], "color": "green"},
                    {"range": [operational_threshold, regulatory_threshold], "color": "orange"},
                    {"range": [regulatory_threshold, 3], "color": "red"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Risk Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ]
            },
            number={"suffix": "%"}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if len(window_df) > 30:
            recent_change = (window_df["T1_PO4"].iloc[-1] - window_df["T1_PO4"].iloc[-30]) * 30
        else:
            recent_change = 0
        
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=recent_change,
            title={"text": "PO₄ Rate (mg/L/hr)"},
            delta={"reference": 0}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 2: PROCESS TRENDS
# ==============================
with tab2:
    st.markdown('<div class="sub-header">📈 Process Trends</div>', unsafe_allow_html=True)
    
    if len(window_df) < 2:
        st.warning("Not enough data for trends")
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=["Phosphate (T1_PO4)", "Oxygen (T1_O2)", "Influent Flow (IN_Q)"]
        )
        
        fig.add_trace(
            go.Scatter(x=window_df["date"], y=window_df["T1_PO4"], name="PO₄"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=window_df["date"], y=window_df["T1_O2"], name="O₂"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=window_df["date"], y=window_df["IN_Q"], name="Flow"),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(window_df, x="date", y=["TEMPERATURE", "IN_METAL_Q"],
                         title="Temperature & Chemical Dose")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "MAX_CF" in window_df.columns:
                fig = px.line(window_df, x="date", y=["MAX_CF"],
                             title="Control Factor")
                st.plotly_chart(fig, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("© ChemTech 2026 Project | Demonstration System")

