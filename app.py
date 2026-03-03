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
# SIDEBAR STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0B0F14 0%, #0E141B 100%);
    border-right: 1px solid #1F2933;
    padding: 25px 18px;
}
section[data-testid="stSidebar"] input[type="search"] {
    display: none !important;
}
.sidebar-brand {
    background: linear-gradient(135deg,#0A66C2,#004182);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 28px;
}
.sidebar-section-title {
    font-size: 12px;
    color: #9BA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 12px;
}
.sidebar-card {
    background: #11161D;
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #1F2933;
    margin-bottom: 20px;
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
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>APL Risk Engine</h2>
    <p style="color:#DCE6F2;font-size:12px;">Predictive Intelligence Panel</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-title">Model Controls</div>', unsafe_allow_html=True)

threshold = st.sidebar.slider(
    "🚨 High-Risk Probability Threshold",
    0.30, 0.90, 0.70, 0.05
)

st.sidebar.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

ship_filter = st.sidebar.multiselect("Shipping Mode", sorted(df["Shipping Mode"].dropna().unique()))
market_filter = st.sidebar.multiselect("Market", sorted(df["Market"].dropna().unique()))
region_filter = st.sidebar.multiselect("Order Region", sorted(df["Order Region"].dropna().unique()))
segment_filter = st.sidebar.multiselect("Customer Segment", sorted(df["Customer Segment"].dropna().unique()))

if ship_filter:
    df = df[df["Shipping Mode"].isin(ship_filter)]
if market_filter:
    df = df[df["Market"].isin(market_filter)]
if region_filter:
    df = df[df["Order Region"].isin(region_filter)]
if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]

# ---------------------------------------------------------
# VALIDATION (NO STOP)
# ---------------------------------------------------------
valid_data = True

if len(df) < 10:
    st.error("Not enough data after filtering. Please relax filters.")
    valid_data = False

elif df[TARGET].nunique() < 2:
    st.error("Filtered data contains only one class. Adjust filters.")
    valid_data = False

# ---------------------------------------------------------
# MODEL + VISUALS (ONLY IF VALID)
# ---------------------------------------------------------
if valid_data:

    LEAKAGE_COLS = ["Days for shipping (real)", "Delivery Status"]
    HIGH_CARDINALITY = [
        "Customer City","Customer State","Order City","Order State",
        "Customer Country","Order Country",
        "Customer Id","Order Customer Id",
        "Customer Fname","Customer Lname",
        "Customer Street","Customer Zipcode","Product Name"
    ]

    df.drop(columns=[c for c in LEAKAGE_COLS + HIGH_CARDINALITY if c in df.columns], inplace=True)

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

    roc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_proba >= threshold)
    rec = recall_score(y_test, y_proba >= threshold)
    f1 = f1_score(y_test, y_proba >= threshold)

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
# FOOTER (ALWAYS RENDERS)
# ---------------------------------------------------------
def render_footer():
    if not UNIFIED_LOGO_PATH.exists():
        return

    encoded = base64.b64encode(UNIFIED_LOGO_PATH.read_bytes()).decode()

    st.markdown(f"""
<div style="display:flex;
            justify-content:space-between;
            align-items:center;
            padding:16px 40px;
            background:#0B1220;
            border-top:1px solid #1F2933;
            margin-top:60px;
            font-size:13px;
            color:#9CA3AF;">

    <div style="display:flex;gap:10px;align-items:center;">
        <img src="data:image/png;base64,{encoded}" style="height:34px;">
        <span>
            Mentored by 
            <a href="https://www.linkedin.com/in/saiprasad-kagne/" target="_blank"
               style="color:#3B82F6;text-decoration:none;">
               Sai Prasad Kagne
            </a>
        </span>
    </div>

    <span>
        Created by 
        <a href="https://www.linkedin.com/in/vidit-kapoor-5062b02a6" target="_blank"
           style="color:#3B82F6;text-decoration:none;">
           Vidit Kapoor
        </a>
    </span>

    <span style="opacity:0.7;">
        Version 1.0 | Feb 2026
    </span>

</div>
""", unsafe_allow_html=True)

render_footer()
