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
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12

# ---------------------------------------------------------
# GLOBAL STYLES (Sidebar + Cards + Footer Typography)
# ---------------------------------------------------------
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0B0F14 0%, #0E141B 100%);
    border-right: 1px solid #1F2933;
    padding: 28px 18px;
}
section[data-testid="stSidebar"] input[type="search"] {
    display: none !important;
}

/* Sidebar Brand */
.sidebar-brand {
    background: linear-gradient(135deg,#0A66C2,#004182);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}
.sidebar-brand h2 {
    color: white;
    font-size: 18px;
    margin: 0;
    font-weight: 700;
}
.sidebar-brand p {
    color: #DCE6F2;
    font-size: 12px;
    margin-top: 6px;
}

/* Sidebar Sections */
.sidebar-section-title {
    font-size: 12px;
    color: #9BA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 12px;
}
.sidebar-divider {
    height: 1px;
    background: #1F2933;
    margin: 20px 0;
}
.sidebar-card {
    background: #11161D;
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #1F2933;
    margin-bottom: 20px;
}

/* KPI Cards */
.kpi-card {
    background: #11161D;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1F2933;
    text-align: center;
}
.kpi-title {
    color: #9BA3AF;
    font-size: 14px;
}
.kpi-value {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 700;
}

/* Chart Cards */
.chart-card {
    background: #11161D;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #1F2933;
    margin-bottom: 30px;
}

/* Footer */
.footer-container {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 40px;
    background:#0B1220;
    color:#9CA3AF;
    font-size:13px;
    border-top:1px solid #1F2933;
    margin-top:50px;
}

.footer-container a {
    color:#3B82F6;
    text-decoration:none;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
def render_header():
    if not APL_LOGO_PATH.exists():
        st.warning("APL logo not found")
        return

    encoded = base64.b64encode(APL_LOGO_PATH.read_bytes()).decode()

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#0A66C2,#004182);
                padding:55px 20px 45px;
                border-radius:20px;
                text-align:center;
                margin-bottom:35px;">
        <img src="data:image/png;base64,{encoded}"
             style="width:14rem;margin-bottom:20px;">
        <h1 style="color:white;font-size:42px;margin:0;font-weight:800;">
            Predictive Late Delivery Risk Intelligence
        </h1>
        <p style="color:#E3EAF2;font-size:18px;margin-top:12px;">
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
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>APL Risk Engine</h2>
    <p>Predictive Intelligence Panel</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-title">Model Controls</div>', unsafe_allow_html=True)

threshold = st.sidebar.slider(
    "🚨 High-Risk Probability Threshold",
    0.30, 0.90, 0.70, 0.05
)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("🚚 Logistics")
ship_filter = st.sidebar.multiselect("Shipping Mode", sorted(df["Shipping Mode"].dropna().unique()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("🌍 Geography")
market_filter = st.sidebar.multiselect("Market", sorted(df["Market"].dropna().unique()))
region_filter = st.sidebar.multiselect("Order Region", sorted(df["Order Region"].dropna().unique()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("👥 Customer")
segment_filter = st.sidebar.multiselect("Customer Segment", sorted(df["Customer Segment"].dropna().unique()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

if ship_filter:
    df = df[df["Shipping Mode"].isin(ship_filter)]
if market_filter:
    df = df[df["Market"].isin(market_filter)]
if region_filter:
    df = df[df["Order Region"].isin(region_filter)]
if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]

# ---------------------------------------------------------
# VALIDATION BEFORE MODEL
# ---------------------------------------------------------
if len(df) < 10:
    st.error("Not enough data after filtering. Please relax filters.")
    st.stop()

if df[TARGET].nunique() < 2:
    st.error("Filtered data contains only one class. Adjust filters.")
    st.stop()

# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------
LEAKAGE_COLS = ["Days for shipping (real)", "Delivery Status"]
HIGH_CARDINALITY = [
    "Customer City","Customer State","Order City","Order State",
    "Customer Country","Order Country",
    "Customer Id","Order Customer Id",
    "Customer Fname","Customer Lname",
    "Customer Street","Customer Zipcode","Product Name"
]

df.drop(columns=[c for c in LEAKAGE_COLS + HIGH_CARDINALITY if c in df.columns], inplace=True)

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

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
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

@st.cache_data
def encode(train, test):
    train_enc = pd.get_dummies(train, drop_first=True)
    test_enc = test.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc

X_train_enc, X_test_enc = encode(X_train, X_test)

@st.cache_resource
def train_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model

model = train_model(X_train_enc, y_train)
y_proba = model.predict_proba(X_test_enc)[:, 1]

roc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_proba >= threshold)
rec = recall_score(y_test, y_proba >= threshold)
f1 = f1_score(y_test, y_proba >= threshold)

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
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
kpi(c2, "ROC-AUC", round(roc, 3))
kpi(c3, "Precision", round(prec, 3))
kpi(c4, "Recall", round(rec, 3))
kpi(c5, "F1 Score", round(f1, 3))

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
st.subheader("🧮 Model Error Analysis – Confusion Matrix")

cm = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
cm_df = pd.DataFrame(
    cm,
    index=["Actual On-Time", "Actual Late"],
    columns=["Predicted On-Time", "Predicted Late"]
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(px.imshow(cm_df, text_auto=True), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# DISTRIBUTION
# ---------------------------------------------------------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(
    px.histogram(pd.DataFrame({"Delay Probability": y_proba}),
                 x="Delay Probability",
                 nbins=30),
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# REGION & SHIPPING
# ---------------------------------------------------------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(
    px.bar(df.groupby("Order Region")[TARGET].mean().reset_index(),
           x="Order Region", y=TARGET),
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(
    px.bar(df.groupby("Shipping Mode")[TARGET].mean().reset_index(),
           x="Shipping Mode", y=TARGET),
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# HIGH-RISK QUEUE
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
# EXPLAINABILITY
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X_train_enc.columns,
    "Impact": np.abs(model.named_steps["lr"].coef_[0])
}).sort_values("Impact", ascending=False).head(15)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(
    px.bar(coef_df, x="Impact", y="Feature", orientation="h"),
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER (Fixed Rendering + Refined Typography)
# ---------------------------------------------------------
def render_footer():
    if not UNIFIED_LOGO_PATH.exists():
        return

    encoded = base64.b64encode(UNIFIED_LOGO_PATH.read_bytes()).decode()

    footer_html = f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:16px 40px;
        background:#0B1220;
        border-top:1px solid #1F2933;
        margin-top:60px;
        font-size:13px;
        color:#9CA3AF;
    ">

        <div style="display:flex;gap:10px;align-items:center;">
            <img src="data:image/png;base64,{encoded}" style="height:34px;">
            <span>
                Mentored by 
                <a href="https://www.linkedin.com/in/saiprasad-kagne/"
                   target="_blank"
                   style="color:#3B82F6;text-decoration:none;">
                   Sai Prasad Kagne
                </a>
            </span>
        </div>

        <span>
            Created by 
            <a href="https://www.linkedin.com/in/vidit-kapoor-5062b02a6"
               target="_blank"
               style="color:#3B82F6;text-decoration:none;">
               Vidit Kapoor
            </a>
        </span>

        <span style="opacity:0.7;">
            Version 1.0 | Feb 2026
        </span>

    </div>
    """

    st.markdown(footer_html, unsafe_allow_html=True)

render_footer()
