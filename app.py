# =========================================================
# APL LOGISTICS – PROFITABILITY INTELLIGENCE SYSTEM
# Internship: Unified Mentor Pvt. Ltd.
# Project: Customer & Product Profitability Analytics
# Author: Vidit Kapoor
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="APL Logistics | Profitability Intelligence",
    layout="wide",
    page_icon="favicon.jfif"
)

# ---------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------
DATA_PATH = Path("APL_Logistics.csv.gz")
APL_LOGO_PATH = Path("APL_Logo.png")
UNIFIED_LOGO_PATH = Path("unified logo.png")

PRIMARY_COLOR = "#1D4ED8"
CHART_BG = "#161B22"
GRID_COLOR = "#2F3542"
TEXT_COLOR = "#E5E7EB"

# ---------------------------------------------------------
# GLOBAL STYLES (same feel as old project)
# ---------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #0E1117;
    padding: 18px 14px;
}
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
    margin-top:30px;
    margin-bottom:40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER (same style)
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
            Profitability Intelligence Dashboard
        </h1>
        <p style="color:#B0B3B8;font-size:18px;margin-top:8px;">
            APL Logistics | Revenue, Margin & Customer Insights
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
    df = df.sample(min(len(df), 50000), random_state=42)

    # Cleaning
    df = df[df["Sales"] > 0]

    # Metrics
    df["Profit Margin"] = df["Order Profit Per Order"] / df["Sales"]
    df["Discount Impact"] = df["Order Item Discount"] / df["Sales"]

    return df

df = load_data()

# ---------------------------------------------------------
# SIDEBAR (UPDATED – PROFITABILITY FOCUSED)
# ---------------------------------------------------------
st.sidebar.header("📊 Business Filters")

# Core Business Filters
segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    sorted(df["Customer Segment"].dropna().unique())
)

category_filter = st.sidebar.multiselect(
    "Product Category",
    sorted(df["Category Name"].dropna().unique())
)

market_filter = st.sidebar.multiselect(
    "Market",
    sorted(df["Market"].dropna().unique())
)

region_filter = st.sidebar.multiselect(
    "Order Region",
    sorted(df["Order Region"].dropna().unique())
)

# Financial Controls
st.sidebar.markdown("### 💸 Pricing Controls")

discount_slider = st.sidebar.slider(
    "Max Discount Rate",
    0.0, 0.5, 0.2
)

profit_filter = st.sidebar.selectbox(
    "Profitability Filter",
    ["All", "Profitable Only", "Loss-Making Only"]
)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]

if category_filter:
    df = df[df["Category Name"].isin(category_filter)]

if market_filter:
    df = df[df["Market"].isin(market_filter)]

if region_filter:
    df = df[df["Order Region"].isin(region_filter)]

# Discount filter
df = df[df["Order Item Discount Rate"] <= discount_slider]

# Profit filter
if profit_filter == "Profitable Only":
    df = df[df["Order Profit Per Order"] > 0]
elif profit_filter == "Loss-Making Only":
    df = df[df["Order Profit Per Order"] < 0]# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
total_sales = df["Sales"].sum()
total_profit = df["Order Profit Per Order"].sum()
profit_margin = (total_profit / total_sales) * 100
avg_discount = df["Order Item Discount Rate"].mean()

st.subheader("📊 Executive Financial Overview")

c1, c2, c3, c4 = st.columns(4)

def kpi(col, title, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "Total Revenue", f"${total_sales:,.0f}")
kpi(c2, "Total Profit", f"${total_profit:,.0f}")
kpi(c3, "Profit Margin", f"{profit_margin:.2f}%")
kpi(c4, "Avg Discount", f"{avg_discount:.2f}")

# ---------------------------------------------------------
# STYLE FUNCTION
# ---------------------------------------------------------
def style(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=TEXT_COLOR)),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig

# ---------------------------------------------------------
# CATEGORY ANALYSIS
# ---------------------------------------------------------
st.subheader("📦 Revenue vs Profit by Category")

cat = df.groupby("Category Name").agg({
    "Sales": "sum",
    "Order Profit Per Order": "sum"
}).reset_index()

fig1 = px.bar(cat, x="Category Name", y="Sales", color_discrete_sequence=[PRIMARY_COLOR])
fig2 = px.bar(cat, x="Category Name", y="Order Profit Per Order", color_discrete_sequence=["#22C55E"])

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig1, "Revenue by Category"), use_container_width=True)
st.plotly_chart(style(fig2, "Profit by Category"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# CUSTOMER ANALYSIS
# ---------------------------------------------------------
st.subheader("🧍 Customer Profitability")

customer = df.groupby("Customer Id").agg({
    "Sales": "sum",
    "Order Profit Per Order": "sum"
}).reset_index()

fig3 = px.scatter(customer, x="Sales", y="Order Profit Per Order",
                  color="Order Profit Per Order", color_continuous_scale="Blues")

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig3, "Customer Value Distribution"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TOP & LOSS CUSTOMERS
# ---------------------------------------------------------
st.subheader("🏆 Top Customers")
st.dataframe(customer.sort_values("Order Profit Per Order", ascending=False).head(10))

st.subheader("⚠️ Loss-Making Customers")
st.dataframe(customer[customer["Order Profit Per Order"] < 0].head(10))

# ---------------------------------------------------------
# DISCOUNT IMPACT
# ---------------------------------------------------------
st.subheader("💸 Discount Impact")

fig4 = px.scatter(df,
                  x="Order Item Discount Rate",
                  y="Profit Margin",
                  opacity=0.5,
                  color_discrete_sequence=[PRIMARY_COLOR])

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig4, "Discount vs Profit Margin"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

high_discount = df[df["Order Item Discount Rate"] > discount_slider]
st.write(f"Orders above {discount_slider*100:.0f}% discount: {len(high_discount)}")

# ---------------------------------------------------------
# REGION ANALYSIS
# ---------------------------------------------------------
st.subheader("🌍 Profit by Region")

region_df = df.groupby("Order Region").agg({
    "Sales": "sum",
    "Order Profit Per Order": "sum"
}).reset_index()

fig5 = px.bar(region_df, x="Order Region", y="Order Profit Per Order",
              color_discrete_sequence=[PRIMARY_COLOR])

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(style(fig5, "Profit by Region"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------
st.markdown(f"""
<div class="summary-box">
<h3 style="color:white;">Executive Insights</h3>
<p style="color:#D1D5DB;">
Total revenue is <b>${total_sales:,.0f}</b> with profit of <b>${total_profit:,.0f}</b>.
Profit margin stands at <b>{profit_margin:.2f}%</b>.
Higher discounts are reducing profitability, and certain customers and regions
are contributing negatively to margins.
</p>
</div>
""", unsafe_allow_html=True)

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
