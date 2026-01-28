import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime, time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="WWTP Phosphorus Spike Monitor",
    layout="wide"
)

st.title("🚨 Phosphorus Spike Early Warning System")
st.caption("BiLSTM-based risk prediction | 2-minute SCADA data")

# =====================================================
# LOAD MODEL + CONFIG
# =====================================================
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.pkl")
    config = joblib.load("model_config.pkl")

    class DeepBiLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_size=96,
                num_layers=2,
                dropout=0.3,
                batch_first=True,
                bidirectional=True
            )
            self.norm = nn.LayerNorm(192)
            self.fc = nn.Sequential(
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(96, 1)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            h = self.norm(h)
            return self.sigmoid(self.fc(h))

    model = DeepBiLSTM(len(config["feature_cols"]))
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    return model, scaler, config

model, scaler, config = load_model()
FEATURE_COLS = config["feature_cols"]
LOOKBACK = config["lookback"]
THRESHOLD = config["threshold"]

# =====================================================
# LOAD DATA (LOCAL OR UPLOAD)
# =====================================================
st.sidebar.header("📂 Data Source")

DATA_PATH = "wwtp_training_subset.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    uploaded = st.sidebar.file_uploader("Upload wwtp.csv", type=["csv"])
    if uploaded is None:
        st.info("Upload wwtp.csv to continue")
        st.stop()
    df = pd.read_csv(uploaded)

# =====================================================
# DATE HANDLING (SAFE)
# =====================================================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].dt.tz is not None:
    df["date"] = df["date"].dt.tz_convert(None)

df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["po4_rate"] = df["T1_PO4"].diff().fillna(0)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =====================================================
# SIDEBAR — DATE & TIME PICKERS (STREAMLIT-SAFE)
# =====================================================
st.sidebar.header("🕒 Time Window")

min_dt = df["date"].iloc[0]
max_dt = df["date"].iloc[-1]

start_date = st.sidebar.date_input(
    "Start date",
    min_value=min_dt.date(),
    max_value=max_dt.date(),
    value=min_dt.date()
)

start_time = st.sidebar.time_input(
    "Start time",
    value=time(0, 0)
)

end_date = st.sidebar.date_input(
    "End date",
    min_value=min_dt.date(),
    max_value=max_dt.date(),
    value=max_dt.date()
)

end_time = st.sidebar.time_input(
    "End time",
    value=max_dt.time()
)

start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

if start_dt >= end_dt:
    st.error("❌ Start datetime must be before end datetime")
    st.stop()

df_win = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)

if len(df_win) <= LOOKBACK:
    st.warning(
        f"Need at least {LOOKBACK+1} rows "
        f"(≈ {(LOOKBACK*2)/60:.1f} hours of data)"
    )
    st.stop()

# =====================================================
# FAST INFERENCE (LAST WINDOW ONLY)
# =====================================================
X = df_win[FEATURE_COLS].values
X_scaled = scaler.transform(X)

seq = X_scaled[-LOOKBACK:]
seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    prob = model(seq_tensor).item()

risk_state = "HIGH" if prob > THRESHOLD else "LOW"

# =====================================================
# CURRENT STATUS
# =====================================================
st.subheader("📊 Current Risk Status")

c1, c2, c3 = st.columns(3)
c1.metric("Risk Probability", f"{prob:.3f}")
c2.metric("Threshold", f"{THRESHOLD:.2f}")
c3.metric("Risk State", "🚨 HIGH" if risk_state == "HIGH" else "✅ LOW")

# =====================================================
# PO₄ TREND (SAFE PLOT)
# =====================================================
st.subheader("📈 PO₄ Trend")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_win["T1_PO4"].values)
ax.axhline(1.5, color="red", linestyle="--")
ax.set_xlabel("Samples (2-min)")
ax.set_ylabel("PO₄ (mg/L)")
ax.grid(alpha=0.3)
st.pyplot(fig)

# =====================================================
# FOOTER
# =====================================================
st.caption(
    "⚠️ This early-warning system prioritizes recall. "
    "False alarms are expected in safety-critical WWTP operation."
)

