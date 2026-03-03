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

DATA_PATH = Path("APL_Logistics.csv.gz")
APL_LOGO_PATH = Path("APL_Logo.png")
UNIFIED_LOGO_PATH = Path("unified logo.png")
TARGET = "Late_delivery_risk"

PRIMARY_COLOR = "#1D4ED8"
SECONDARY_COLOR = "#3B82F6"
MUTED_BLUE = "#60A5FA"
CHART_BG = "#161B22"
GRID_COLOR = "#2F3542"
TEXT_COLOR = "#E5E7EB"

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
.kpi-card {
    background: #161B22;
    border: 1px solid #232A33;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.kpi-title {
    color: #9CA3AF;
    font-size: 13px;
}
.kpi-value {
    color: #EAEAEA;
    font-size: 26px;
    font-weight: 700;
}
.chart-card {
    background:#161B22;
    padding:20px;
    border-radius:14px;
    border:1px solid #232A33;
    margin-top:20px;
    margin-bottom:30px;
}
.summary-box {
    background:#111827;
    padding:24px;
    border-radius:14px;
    border:1px solid #1F2937;
    margin-top:30px;
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
        <img src="data:image/png;base64,{encoded}" style="width:15rem;margin-bottom:20px;">
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
# SIDEBAR (UNCHANGED)
# ---------------------------------------------------------
st.sidebar.header("🔎 Filters")

ship_filter = st.sidebar.multiselect("Shipping Mode", sorted(df["Shipping Mode"].dropna().unique()))
market_filter = st.sidebar.multiselect("Market", sorted(df["Market"].dropna().unique()))
region_filter = st.sidebar.multiselect("Order Region", sorted(df["Order Region"].dropna().unique()))
segment_filter = st.sidebar.multiselect("Customer Segment", sorted(df["Customer Segment"].dropna().unique()))

threshold = st.sidebar.slider("🚨 High-Risk Probability Threshold", 0.30, 0.90, 0.70, 0.05)

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
# KPI SECTION
# ---------------------------------------------------------
high_risk_count = (y_proba >= threshold).sum()
high_risk_pct = high_risk_count / len(y_proba) * 100

st.subheader("📊 Executive Risk Overview")
c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, title, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "Orders Analysed", f"{len(df):,}")
kpi(c2, "High-Risk Orders", f"{high_risk_count}")
kpi(c3, "High-Risk %", f"{high_risk_pct:.1f}%")
kpi(c4, "ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
kpi(c5, "F1 Score", f"{f1_score(y_test, y_proba >= threshold):.3f}")

# ---------------------------------------------------------
# EXECUTIVE SUMMARY (PROPER SPACING)
# ---------------------------------------------------------
st.markdown(f"""
<div class="summary-box">
<h3 style="color:white;margin-bottom:12px;">Executive Summary</h3>
<p style="color:#D1D5DB;line-height:1.6;">
The model flags <b>{high_risk_count}</b> high-risk shipments 
({high_risk_pct:.1f}% of evaluated orders).
Operational teams can prioritize these orders to proactively reduce SLA breaches 
and prevent downstream customer dissatisfaction.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
cm_df = pd.DataFrame(cm,
                     index=["Actual On-Time", "Actual Late"],
                     columns=["Predicted On-Time", "Predicted Late"])

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale=["#1F2937", "#334155", "#475569", PRIMARY_COLOR]
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_cm, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# RISK DISTRIBUTION
# ---------------------------------------------------------
fig_hist = px.histogram(
    pd.DataFrame({"Delay Probability": y_proba}),
    x="Delay Probability",
    nbins=30,
    color_discrete_sequence=[PRIMARY_COLOR]
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# REGION & MODE CHARTS
# ---------------------------------------------------------
fig_region = px.bar(
    df.groupby("Order Region")[TARGET].mean().reset_index(),
    x="Order Region",
    y=TARGET,
    color_discrete_sequence=[SECONDARY_COLOR]
)

fig_mode = px.bar(
    df.groupby("Shipping Mode")[TARGET].mean().reset_index(),
    x="Shipping Mode",
    y=TARGET,
    color_discrete_sequence=[MUTED_BLUE]
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_region, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_mode, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# HIGH-RISK ACTION QUEUE (RESTORED)
# ---------------------------------------------------------
results = X_test.copy()
results["Delay_Probability"] = y_proba
results["Risk_Category"] = pd.cut(
    results["Delay_Probability"],
    [0, 0.4, threshold, 1],
    labels=["Low", "Medium", "High"]
)

st.subheader("🚨 High-Risk Orders – Operations Action Queue")
st.dataframe(
    results[results["Risk_Category"] == "High"]
    .sort_values("Delay_Probability", ascending=False)
    .head(50),
    use_container_width=True
)

# ---------------------------------------------------------
# EXPLAINABILITY (RESTORED)
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X_train_enc.columns,
    "Impact": np.abs(model.named_steps["lr"].coef_[0])
}).sort_values("Impact", ascending=False).head(15)

fig_importance = px.bar(
    coef_df,
    x="Impact",
    y="Feature",
    orientation="h",
    color_discrete_sequence=[PRIMARY_COLOR]
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_importance, use_container_width=True)
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
