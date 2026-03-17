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

PRIMARY_COLOR = "#2A82E9"
TEXT_COLOR = "#E5E7EB"

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {background-color:#0E1117;padding:18px;}
.kpi-card {background:#161B22;border-radius:14px;padding:18px;text-align:center;}
.kpi-title {color:#9CA3AF;font-size:13px;}
.kpi-value {color:#EAEAEA;font-size:26px;font-weight:700;}
.chart-card {background:#161B22;padding:18px;border-radius:14px;margin-bottom:30px;}
.summary-box {background:#111827;padding:24px;border-radius:14px;margin-top:30px;}
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
    <div style="background:#0E1117;padding:45px;text-align:center;">
        <img src="data:image/png;base64,{encoded}" style="width:15rem;">
        <h1 style="color:white;">Profitability Intelligence Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

render_header()

# ---------------------------------------------------------
# LOAD DATA + VALIDATION
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = df.sample(min(len(df), 50000), random_state=42)

    original_rows = len(df)
    df = df[df["Sales"] > 0]
    cleaned_rows = original_rows - len(df)

    df["Profit Margin"] = df["Order Profit Per Order"] / df["Sales"]

    return df, cleaned_rows

df, cleaned_rows = load_data()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📊 Business Filters")

segment_filter = st.sidebar.multiselect("Customer Segment", sorted(df["Customer Segment"].dropna().unique()))
category_filter = st.sidebar.multiselect("Product Category", sorted(df["Category Name"].dropna().unique()))
market_filter = st.sidebar.multiselect("Market", sorted(df["Market"].dropna().unique()))
region_filter = st.sidebar.multiselect("Order Region", sorted(df["Order Region"].dropna().unique()))

st.sidebar.markdown("### 💸 Pricing Controls")
discount_slider = st.sidebar.slider("Max Discount Rate", 0.0, 0.5, 0.2)
profit_filter = st.sidebar.selectbox("Profitability Filter", ["All", "Profitable Only", "Loss-Making Only"])

if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]
if category_filter:
    df = df[df["Category Name"].isin(category_filter)]
if market_filter:
    df = df[df["Market"].isin(market_filter)]
if region_filter:
    df = df[df["Order Region"].isin(region_filter)]

df = df[df["Order Item Discount Rate"] <= discount_slider]

if profit_filter == "Profitable Only":
    df = df[df["Order Profit Per Order"] > 0]
elif profit_filter == "Loss-Making Only":
    df = df[df["Order Profit Per Order"] < 0]

# ---------------------------------------------------------
# STYLE FUNCTION (TRANSPARENT)
# ---------------------------------------------------------
def style(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=TEXT_COLOR)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR)
    )
    return fig

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Customers", "Products", "Discounts", "Regions"
])

# ---------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------
with tab1:
    total_sales = df["Sales"].sum()
    total_profit = df["Order Profit Per Order"].sum()
    profit_margin = (total_profit / total_sales) * 100
    avg_discount = df["Order Item Discount Rate"].mean()

    st.info(f"Data Validation: Removed {cleaned_rows} invalid records")

    st.subheader("📊 Executive Financial Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Revenue", f"${total_sales:,.0f}")
    c2.metric("Profit", f"${total_profit:,.0f}")
    c3.metric("Margin", f"{profit_margin:.2f}%")
    c4.metric("Avg Discount", f"{avg_discount:.2f}")

# ---------------------------------------------------------
# CUSTOMERS TAB
# ---------------------------------------------------------
with tab2:
    st.subheader("🧍 Customer Profitability")

    customer = df.groupby("Customer Id").agg({
        "Sales": "sum",
        "Order Profit Per Order": "sum"
    }).reset_index()

    total_customer_profit = customer["Order Profit Per Order"].sum()
    customer["Customer Value Index"] = customer["Order Profit Per Order"] / total_customer_profit

    fig3 = px.scatter(customer, x="Sales", y="Order Profit Per Order",
                      color_discrete_sequence=[PRIMARY_COLOR])

    st.plotly_chart(style(fig3, "Customer Value Distribution"))

    st.subheader("🏆 Top Customers")
    st.dataframe(customer.sort_values("Order Profit Per Order", ascending=False).head(10))

    st.subheader("⚠️ Loss-Making Customers")
    st.dataframe(customer[customer["Order Profit Per Order"] < 0].head(10))

    st.subheader("💎 Highest Value Customers (CVI)")
    st.dataframe(customer.sort_values("Customer Value Index", ascending=False).head(10))

st.subheader("🔥 Pareto Analysis (Top Customers)")

# Sort customers
customer = customer.sort_values("Order Profit Per Order", ascending=False)

# Cumulative %
customer["Cumulative %"] = (
    customer["Order Profit Per Order"].cumsum() /
    customer["Order Profit Per Order"].sum()
)

# Top 15 for clarity
top_n = customer.head(15)

fig_pareto = px.bar(
    top_n,
    x="Customer Id",
    y="Order Profit Per Order",
    color_discrete_sequence=[PRIMARY_COLOR]
)

# Add cumulative line
fig_pareto.add_scatter(
    x=top_n["Customer Id"],
    y=top_n["Cumulative %"],
    mode="lines+markers",
    name="Cumulative %",
    yaxis="y2"
)

# Layout fix
fig_pareto.update_layout(
    yaxis2=dict(
        overlaying="y",
        side="right",
        range=[0, 1]
    ),
    xaxis_tickangle=-45
)

# ✅ SAME indentation level
st.plotly_chart(fig_pareto, use_container_width=True)
# ---------------------------------------------------------
# PRODUCTS TAB
# ---------------------------------------------------------
with tab3:
    st.subheader("📦 Revenue vs Profit by Category")

    cat = df.groupby("Category Name").agg({
        "Sales": "sum",
        "Order Profit Per Order": "sum"
    }).reset_index()

    fig1 = px.bar(cat, x="Category Name", y="Sales", color_discrete_sequence=[PRIMARY_COLOR])
    fig2 = px.bar(cat, x="Category Name", y="Order Profit Per Order", color_discrete_sequence=[PRIMARY_COLOR])

    st.plotly_chart(style(fig1, "Revenue by Category"))
    st.plotly_chart(style(fig2, "Profit by Category"))

    st.subheader("📦 Product-Level Profitability")

    product = df.groupby("Product Name").agg({
        "Sales": "sum",
        "Order Profit Per Order": "sum"
    }).reset_index()

    product["Profit Margin"] = product["Order Profit Per Order"] / product["Sales"]

    st.dataframe(product.sort_values("Order Profit Per Order", ascending=False).head(10))
    st.dataframe(product[product["Order Profit Per Order"] < 0].head(10))

# ---------------------------------------------------------
# DISCOUNT TAB
# ---------------------------------------------------------
with tab4:
    st.subheader("💸 Discount vs Profit Margin")

    fig4 = px.scatter(df, x="Order Item Discount Rate", y="Profit Margin",
                      color_discrete_sequence=[PRIMARY_COLOR])

    st.plotly_chart(style(fig4, "Discount Impact"))

    st.subheader("📊 Threshold Analysis")

    df["Discount Bin"] = pd.cut(df["Order Item Discount Rate"], bins=5).astype(str)

    analysis = df.groupby("Discount Bin")["Profit Margin"].mean().reset_index()

    fig_discount = px.bar(analysis, x="Discount Bin", y="Profit Margin",
                          color_discrete_sequence=[PRIMARY_COLOR])

    st.plotly_chart(style(fig_discount, "Threshold"))

# ---------------------------------------------------------
# REGION TAB
# ---------------------------------------------------------
with tab5:
    st.subheader("🌍 Region Analysis")

    region = df.groupby("Order Region").agg({
        "Sales": "sum",
        "Order Profit Per Order": "sum"
    }).reset_index()

    fig = px.bar(region, x="Order Region", y=["Sales", "Order Profit Per Order"],
                 barmode="group")

    st.plotly_chart(style(fig, "Region Performance"))

    st.subheader("🌎 Market Analysis")

    market = df.groupby("Market").agg({
        "Sales": "sum",
        "Order Profit Per Order": "sum"
    }).reset_index()

    fig2 = px.bar(market, x="Market", y=["Sales", "Order Profit Per Order"],
                  barmode="group")

    st.plotly_chart(style(fig2, "Market Performance"))

# ---------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------
with tab1:

    # 🔥 TOP CUSTOMER CONTRIBUTION
    customer_summary = df.groupby("Customer Id")["Order Profit Per Order"].sum().sort_values(ascending=False)
    top_20_pct = int(len(customer_summary) * 0.2)
    top_contribution = customer_summary.head(top_20_pct).sum() / customer_summary.sum() * 100

    # 🔥 WORST CATEGORY
    cat_summary = df.groupby("Category Name")["Order Profit Per Order"].sum().reset_index()
    worst_category = cat_summary.sort_values("Order Profit Per Order").iloc[0]["Category Name"]

    # 🔥 DISCOUNT IMPACT
    discount_bins = pd.cut(df["Order Item Discount Rate"], bins=5).astype(str)
    discount_analysis = df.groupby(discount_bins)["Profit Margin"].mean().reset_index()

    negative_bins = discount_analysis[discount_analysis["Profit Margin"] < 0]

    if not negative_bins.empty:
        discount_warning = f"Profit turns negative at discount range {negative_bins.iloc[0][0]}"
    else:
        discount_warning = "No negative profit zones detected across discount ranges"

    # 🔥 FINAL INSIGHT BOX
    st.markdown(f"""
    <div class="summary-box">
    <h3 style="color:white;">Executive Insights</h3>
    <p style="color:#D1D5DB; line-height:1.7;">
    Total revenue stands at <b>${total_sales:,.0f} </b> with a net profit of 
    <b>${total_profit:,.0f} </b>, resulting in a profit margin of 
    <b>{profit_margin:.2f}%</b>.
    <br><br>
    Customer analysis reveals that the top 20% of customers contribute 
    approximately <b>{top_contribution:.1f}%</b> of total profit, indicating a strong 
    concentration of value among a small customer segment.
    <br><br>
    Product-level analysis highlights <b>{worst_category}</b> as a low-performing category, 
    contributing negatively to overall profitability.
    <br><br>
    Discount analysis shows that <b>{discount_warning}</b>, suggesting that aggressive 
    discounting strategies may be eroding margins.    
    </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
def render_footer():
    if not UNIFIED_LOGO_PATH.exists():
        return
    encoded = base64.b64encode(UNIFIED_LOGO_PATH.read_bytes()).decode()
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-top:30px;padding:25px 40px;background:#0E1117;color:white;
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
