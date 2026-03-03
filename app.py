# =========================================================
# APL LOGISTICS – PREDICTIVE LATE DELIVERY RISK SYSTEM
# Internship: Unified Mentor Pvt. Ltd.
# Project: Predictive Supply Chain Risk Intelligence
# Author: Vidit Kapoor
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="APL Logistics | Predictive Risk Intelligence",
    layout="wide",
    page_icon="favicon.jfif"
)

# ---------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------
DATA_PATH = Path("APL_Logistics.csv.gz")
APL_LOGO_PATH = Path("APL_Logo.png")
UNIFIED_LOGO_PATH = Path("unified logo.png")
TARGET = "Late_delivery_risk"

PLOTLY_FONT = dict(family="Arial", size=12, color="#EAEAEA")
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 13
TICK_SIZE = 11

# ---------------------------------------------------------
# GLOBAL UI STYLES
# ---------------------------------------------------------
st.markdown("""
<style>

section[data-testid="stSidebar"] {
    background-color: #0E1117;
    padding: 18px 14px;
}

section[data-testid="stSidebar"] input[type="search"] {
    display: none !important;
}

.kpi-card {
    background: #161B22;
    border: 1px solid #232A33;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.kpi-title {
    color: #B0B3B8;
    font-size: 14px;
}

.kpi-value {
    color: #EAEAEA;
    font-size: 28px;
    font-weight: 700;
}

.chart-card {
    background:#161B22;
    padding:18px;
    border-radius:14px;
    border:1px solid #232A33;
    margin-bottom:30px;
}

.summary-box {
    background:#111827;
    padding:24px;
    border-radius:14px;
    border:1px solid #1F2937;
    margin-top:20px;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
def render_header():
    if not APL_LOGO_PATH.exists():
        return

    encoded = base64.b64encode(APL_LOGO_PATH.read_bytes()).decode()
    st.markdown(f"""
    <div style="background:#0E1117;padding:45px 20px 35px;text-align:center;">
        <img src="data:image/png;base64,{encoded}"
             style="width:15rem;margin-bottom:20px;">
        <h1 style="color:white;font-size:39px;margin:0;font-weight:700;">
            Predictive Late Delivery Risk Intelligence
        </h1>
        <p style="color:#B0B3B8;font-size:18px;margin-top:8px;">
            APL Logistics | Proactive Supply Chain Risk Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

render_header()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    return df.sample(min(len(df), 50000), random_state=42)

df = load_data()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔎 Filters")

ship_filter = st.sidebar.multiselect(
    "Shipping Mode",
    sorted(df["Shipping Mode"].dropna().unique())
)

market_filter = st.sidebar.multiselect(
    "Market",
    sorted(df["Market"].dropna().unique())
)

region_filter = st.sidebar.multiselect(
    "Order Region",
    sorted(df["Order Region"].dropna().unique())
)

segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    sorted(df["Customer Segment"].dropna().unique())
)

threshold = st.sidebar.slider(
    "🚨 High-Risk Probability Threshold",
    0.30, 0.90, 0.70, 0.05
)

if ship_filter:
    df = df[df["Shipping Mode"].isin(ship_filter)]
if market_filter:
    df = df[df["Market"].isin(market_filter)]
if region_filter:
    df = df[df["Order Region"].isin(region_filter)]
if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df["Shipping_Pressure_Index"] = (
    df["Order Item Quantity"] /
    df["Days for shipment (scheduled)"].replace(0, np.nan)
).fillna(0)

df["Is_Express_Shipping"] = (
    df["Shipping Mode"].astype(str)
    .str.contains("express", case=False)
    .astype(int)
)

df["Order_Complexity_Score"] = (
    df["Order Item Quantity"] *
    (1 + df["Order Item Discount Rate"])
)

region_risk = df.groupby("Order Region")[TARGET].mean()
df["Region_Delay_Risk"] = (
    df["Order Region"]
    .map(region_risk)
    .fillna(df[TARGET].mean())
)

FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Shipping_Pressure_Index",
    "Region_Delay_Risk",
    "Is_Express_Shipping",
    "Order_Complexity_Score",
    "Sales",
    "Order Item Discount Rate",
    "Order Profit Per Order",
    "Benefit per order",
    "Market",
    "Order Region",
    "Customer Segment"
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = X_test.reindex(columns=X_train_enc.columns, fill_value=0)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train_enc, y_train)
y_proba = model.predict_proba(X_test_enc)[:, 1]

# ---------------------------------------------------------
# STYLE FUNCTION (ONLY CHANGE MADE HERE)
# ---------------------------------------------------------
def style(fig, title):
    fig.update_layout(
        title=title,
        font=PLOTLY_FONT,
        title_font_size=TITLE_SIZE,
        xaxis_title_font_size=AXIS_LABEL_SIZE,
        yaxis_title_font_size=AXIS_LABEL_SIZE,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117"
    )
    fig.update_xaxes(tickfont_size=TICK_SIZE)
    fig.update_yaxes(tickfont_size=TICK_SIZE)
    return fig

# ---------------------------------------------------------
# ALL ORIGINAL CHARTS BELOW REMAIN UNCHANGED
# ---------------------------------------------------------

# Confusion Matrix
st.subheader("🧮 Model Error Analysis – Confusion Matrix")

cm = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
cm_df = pd.DataFrame(
    cm,
    index=["Actual On-Time", "Actual Late"],
    columns=["Predicted On-Time", "Predicted Late"]
)

fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig_cm, "Model Error Analysis – Confusion Matrix"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Risk Distribution
st.subheader("📈 Late Delivery Risk Distribution")

fig_hist = px.histogram(
    pd.DataFrame({"Delay Probability": y_proba}),
    x="Delay Probability",
    nbins=30
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig_hist, "Late Delivery Risk Distribution"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Region Risk
st.subheader("🌍 Average Delay Risk by Region")

fig_region = px.bar(
    df.groupby("Order Region")[TARGET].mean().reset_index(),
    x="Order Region",
    y=TARGET
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig_region, "Average Delay Risk by Region"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Shipping Mode Risk
st.subheader("🚚 Average Delay Risk by Shipping Mode")

fig_mode = px.bar(
    df.groupby("Shipping Mode")[TARGET].mean().reset_index(),
    x="Shipping Mode",
    y=TARGET
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig_mode, "Average Delay Risk by Shipping Mode"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER (UNCHANGED)
# ---------------------------------------------------------
def render_footer():
    if not UNIFIED_LOGO_PATH.exists():
        return

    encoded = base64.b64encode(UNIFIED_LOGO_PATH.read_bytes()).decode()
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:25px 40px;background:#0E1117;color:white;
                font-size:13px;font-family:'Segoe UI',sans-serif;">
        <div style="display:flex;gap:12px;align-items:center;">
            <img src="data:image/png;base64,{encoded}" style="height:50px;">
            <span>Mentored by 
            <a href="https://www.linkedin.com/in/saiprasad-kagne/"
               target="_blank" style="color:#0A66C2;">
               Sai Prasad Kagne</a></span>
        </div>
        <span>
            Created by 
            <a href="https://www.linkedin.com/in/vidit-kapoor-5062b02a6"
               target="_blank" style="color:#0A66C2;">
               Vidit Kapoor</a>
        </span>
        <span>Version 1.0 | Feb 2026</span>
    </div>
    """, unsafe_allow_html=True)

render_footer()
