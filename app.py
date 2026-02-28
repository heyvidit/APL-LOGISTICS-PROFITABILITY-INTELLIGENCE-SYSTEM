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
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
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

PLOTLY_FONT = dict(family="Arial", size=14, color="#EAEAEA")
TITLE_SIZE = 22
AXIS_LABEL_SIZE = 16
TICK_SIZE = 14

# ---------------------------------------------------------
# SIDEBAR DESIGN (UI ONLY – NO LOGIC CHANGE)
# ---------------------------------------------------------
st.markdown("""
<style>

/* Sidebar container */
section[data-testid="stSidebar"] {
    background-color: #0E1117;
    padding: 18px 12px;
}

/* Sidebar headers */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #EAEAEA;
    font-weight: 600;
}

/* Section titles */
section[data-testid="stSidebar"] h3 {
    font-size: 15px;
    margin: 18px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #232A33;
}

/* Labels */
section[data-testid="stSidebar"] label {
    font-size: 13px;
    color: #B0B3B8;
}

/* Inputs */
section[data-testid="stSidebar"] .stMultiSelect,
section[data-testid="stSidebar"] .stSelectbox {
    background-color: #161B22;
    border-radius: 8px;
    padding: 6px;
    margin-bottom: 14px;
}

/* Slider spacing */
section[data-testid="stSidebar"] .stSlider {
    padding: 8px 0 16px;
}

/* Card wrapper */
.sidebar-card {
    background-color: #161B22;
    padding: 12px 12px 6px;
    border-radius: 10px;
    margin-bottom: 16px;
    border: 1px solid #232A33;
}

/* Scrollbar */
section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 6px;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #2A2F3A;
    border-radius: 10px;
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
st.markdown("")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    return df.sample(min(len(df), 50000), random_state=42)

df = load_data()

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

df.drop(
    columns=[c for c in LEAKAGE_COLS + HIGH_CARDINALITY if c in df.columns],
    inplace=True
)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df["Shipping_Pressure_Index"] = (
    df["Order Item Quantity"] /
    df["Days for shipment (scheduled)"].replace(0, np.nan)
).fillna(0)

df["Is_Express_Shipping"] = (
    df["Shipping Mode"]
    .astype(str)
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
# SIDEBAR FILTERS (DESIGN ENHANCED – LOGIC UNCHANGED)
# ---------------------------------------------------------
st.sidebar.header("🔎 Filters")

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("🚚 Logistics")
ship_filter = st.sidebar.multiselect(
    "Shipping Mode", sorted(df["Shipping Mode"].dropna().unique())
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("🌍 Geography")
market_filter = st.sidebar.multiselect(
    "Market", sorted(df["Market"].dropna().unique())
)
region_filter = st.sidebar.multiselect(
    "Order Region", sorted(df["Order Region"].dropna().unique())
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.subheader("👥 Customer")
segment_filter = st.sidebar.multiselect(
    "Customer Segment", sorted(df["Customer Segment"].dropna().unique())
)
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
# MODEL DATA
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

# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.25,
    random_state=42
)

@st.cache_data
def encode(train, test):
    train_enc = pd.get_dummies(train, drop_first=True)
    test_enc = test.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc

X_train_enc, X_test_enc = encode(X_train, X_test)

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
@st.cache_resource
def train_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model

model = train_model(X_train_enc, y_train)
y_proba = model.predict_proba(X_test_enc)[:, 1]

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
threshold = st.sidebar.slider(
    "🚨 High-Risk Probability Threshold",
    0.30, 0.90, 0.70, 0.05
)

roc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_proba >= threshold)
rec = recall_score(y_test, y_proba >= threshold)
f1 = f1_score(y_test, y_proba >= threshold)

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
st.subheader("📊 Executive Risk Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders Analysed", f"{len(df):,}")
c2.metric("ROC-AUC", round(roc, 3))
c3.metric("Precision", round(prec, 3))
c4.metric("Recall", round(rec, 3))
c5.metric("F1 Score", round(f1, 3))

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
st.subheader("🧮 Model Error Analysis – Confusion Matrix")

y_pred = (y_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual On-Time", "Actual Late"],
    columns=["Predicted On-Time", "Predicted Late"]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues"
)

fig_cm.update_layout(
    title="Confusion Matrix at Selected Risk Threshold",
    font=PLOTLY_FONT,
    title_font_size=TITLE_SIZE,
    xaxis_title_font_size=AXIS_LABEL_SIZE,
    yaxis_title_font_size=AXIS_LABEL_SIZE
)

st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# FOOTER (ORIGINAL – UNCHANGED)
# ---------------------------------------------------------
def render_footer():
    if not UNIFIED_LOGO_PATH.exists():
        return

    encoded = base64.b64encode(UNIFIED_LOGO_PATH.read_bytes()).decode()
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:25px 40px;background:#0E1117;color:white;
                font-size:16px;font-family:Arial;">
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
