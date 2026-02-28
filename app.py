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
# ENHANCED ENTERPRISE UI / UX CSS
# ---------------------------------------------------------
st.markdown("""
<style>

/* ===== DESIGN TOKENS ===== */
:root {
    --bg-main:#0E1117;
    --bg-card:#161B22;
    --border:#232A33;

    --text-main:#E6E6E6;
    --text-muted:#9CA3AF;

    --primary:#2A7FFF;
    --accent:#22D3EE;

    --risk-low:#2ECC71;
    --risk-med:#F1C40F;
    --risk-high:#E74C3C;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color:var(--bg-main);
    padding:18px 14px;
}
section[data-testid="stSidebar"] h3 {
    font-size:15px;
    font-weight:600;
    margin:18px 0 10px;
    padding-bottom:6px;
    border-bottom:1px solid var(--border);
    color:var(--text-main);
}
section[data-testid="stSidebar"] label {
    font-size:13px;
    color:var(--text-muted);
}
.sidebar-card {
    background:var(--bg-card);
    padding:14px;
    border-radius:12px;
    margin-bottom:16px;
    border:1px solid var(--border);
    transition:border .2s ease;
}
.sidebar-card:hover {
    border-color:var(--primary);
}

/* ===== KPI CARDS ===== */
.kpi-card {
    background:linear-gradient(180deg,#161B22,#141922);
    border:1px solid var(--border);
    border-radius:16px;
    padding:22px 18px;
    text-align:center;
    position:relative;
    transition:.15s ease;
}
.kpi-card:hover {
    transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(0,0,0,.35);
}
.kpi-card::before {
    content:"";
    position:absolute;
    top:0; left:0;
    width:100%; height:3px;
    background:linear-gradient(90deg,var(--primary),var(--accent));
    border-radius:16px 16px 0 0;
}
.kpi-title {
    color:var(--text-muted);
    font-size:13px;
}
.kpi-value {
    color:var(--text-main);
    font-size:28px;
    font-weight:700;
}

/* ===== CHART CARDS ===== */
.chart-card {
    background:linear-gradient(180deg,#161B22,#141922);
    padding:20px;
    border-radius:16px;
    border:1px solid var(--border);
    margin-bottom:30px;
}

/* ===== DATAFRAME ===== */
div[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
    border:1px solid var(--border);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-thumb {
    background:#2A2F3A;
    border-radius:8px;
}
::-webkit-scrollbar-thumb:hover {
    background:#3A4150;
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

df.drop(columns=[c for c in LEAKAGE_COLS + HIGH_CARDINALITY if c in df.columns], inplace=True)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df["Shipping_Pressure_Index"] = (
    df["Order Item Quantity"] /
    df["Days for shipment (scheduled)"].replace(0, np.nan)
).fillna(0)

df["Is_Express_Shipping"] = df["Shipping Mode"].astype(str).str.contains("express", case=False).astype(int)

df["Order_Complexity_Score"] = df["Order Item Quantity"] * (1 + df["Order Item Discount Rate"])

region_risk = df.groupby("Order Region")[TARGET].mean()
df["Region_Delay_Risk"] = df["Order Region"].map(region_risk).fillna(df[TARGET].mean())

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔎 Filters")

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

if ship_filter: df = df[df["Shipping Mode"].isin(ship_filter)]
if market_filter: df = df[df["Market"].isin(market_filter)]
if region_filter: df = df[df["Order Region"].isin(region_filter)]
if segment_filter: df = df[df["Customer Segment"].isin(segment_filter)]

# ---------------------------------------------------------
# MODEL DATA
# ---------------------------------------------------------
FEATURES = [
    "Days for shipment (scheduled)","Order Item Quantity",
    "Shipping_Pressure_Index","Region_Delay_Risk",
    "Is_Express_Shipping","Order_Complexity_Score",
    "Sales","Order Item Discount Rate",
    "Order Profit Per Order","Benefit per order",
    "Market","Order Region","Customer Segment"
]

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
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
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model

model = train_model(X_train_enc, y_train)
y_proba = model.predict_proba(X_test_enc)[:, 1]

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
threshold = st.sidebar.slider("🚨 High-Risk Probability Threshold", 0.30, 0.90, 0.70, 0.05)

roc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_proba >= threshold)
rec = recall_score(y_test, y_proba >= threshold)
f1 = f1_score(y_test, y_proba >= threshold)

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
st.subheader("📊 Executive Risk Overview")
c1,c2,c3,c4,c5 = st.columns(5)

def kpi(col, title, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1,"Orders Analysed",f"{len(df):,}")
kpi(c2,"ROC-AUC",round(roc,3))
kpi(c3,"Precision",round(prec,3))
kpi(c4,"Recall",round(rec,3))
kpi(c5,"F1 Score",round(f1,3))

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
st.subheader("🧮 Model Error Analysis – Confusion Matrix")
cm = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
cm_df = pd.DataFrame(cm, index=["Actual On-Time","Actual Late"], columns=["Predicted On-Time","Predicted Late"])
fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_cm, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# CHART STYLING HELPER
# ---------------------------------------------------------
def style(fig, title):
    fig.update_layout(
        title=title,
        font=PLOTLY_FONT,
        title_font_size=TITLE_SIZE,
        xaxis_title_font_size=AXIS_LABEL_SIZE,
        yaxis_title_font_size=AXIS_LABEL_SIZE,
        margin=dict(l=40,r=20,t=50,b=40)
    )
    fig.update_xaxes(tickfont_size=TICK_SIZE)
    fig.update_yaxes(tickfont_size=TICK_SIZE)
    return fig

# ---------------------------------------------------------
# VISUALS
# ---------------------------------------------------------
for fig, title in [
    (px.histogram(pd.DataFrame({"Delay Probability":y_proba}), x="Delay Probability", nbins=30),
     "Late Delivery Risk Distribution"),
    (px.bar(df.groupby("Order Region")[TARGET].mean().reset_index(), x="Order Region", y=TARGET),
     "Average Delay Risk by Region"),
    (px.bar(df.groupby("Shipping Mode")[TARGET].mean().reset_index(), x="Shipping Mode", y=TARGET),
     "Average Delay Risk by Shipping Mode")
]:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(style(fig, title), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# HIGH-RISK ACTION QUEUE
# ---------------------------------------------------------
results = X_test.copy()
results["Delay_Probability"] = y_proba
results["Risk_Category"] = pd.cut(results["Delay_Probability"], [0,0.4,threshold,1], labels=["Low","Medium","High"])

st.subheader("🚨 High-Risk Orders – Operations Action Queue")
st.dataframe(
    results[results["Risk_Category"]=="High"].sort_values("Delay_Probability",ascending=False).head(50),
    use_container_width=True
)

# ---------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature":X_train_enc.columns,
    "Impact":np.abs(model.named_steps["lr"].coef_[0])
}).sort_values("Impact",ascending=False).head(15)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(px.bar(coef_df, x="Impact", y="Feature", orientation="h"),
                      "Key Drivers of Late Delivery Risk"),
                use_container_width=True)
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
                font-size:16px;font-family:Arial;">
        <div style="display:flex;gap:12px;align-items:center;">
            <img src="data:image/png;base64,{encoded}" style="height:50px;">
            <span>Mentored by 
            <a href="https://www.linkedin.com/in/saiprasad-kagne/" target="_blank" style="color:#0A66C2;">
               Sai Prasad Kagne</a></span>
        </div>
        <span>
            Created by 
            <a href="https://www.linkedin.com/in/vidit-kapoor-5062b02a6" target="_blank" style="color:#0A66C2;">
               Vidit Kapoor</a>
        </span>
        <span>Version 1.0 | Feb 2026</span>
    </div>
    """, unsafe_allow_html=True)

render_footer()
