# =========================================================
# APL LOGISTICS – PROFITABILITY INTELLIGENCE SYSTEM
# Internship: Unified Mentor Pvt. Ltd.
# Project: Customer & Product Profitability Analytics
# Author: Vidit Kapoor
# Version: 3.0 (Fixed & Enhanced)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

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
CHART_BG = "#161B22"
GRID_COLOR = "#2F3542"
TEXT_COLOR = "#E5E7EB"
LOSS_COLOR = "#EF4444"
GAIN_COLOR = "#22C55E"
WARN_COLOR = "#F59E0B"

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {background-color:#0E1117;padding:18px;}
.kpi-card {background:#161B22;border-radius:14px;padding:14px 10px;text-align:center;margin-bottom:8px;}
.kpi-title {color:#9CA3AF;font-size:11px;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.kpi-value {color:#EAEAEA;font-size:20px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.kpi-sub {color:#6B7280;font-size:11px;margin-top:4px;}
.chart-card {background:#161B22;padding:18px;border-radius:14px;margin-bottom:30px;}
.summary-box {background:#111827;padding:24px;border-radius:14px;margin-top:30px;}
.data-note {background:#1F2937;border-left:4px solid #2A82E9;padding:10px 16px;border-radius:6px;color:#9CA3AF;font-size:13px;margin-bottom:16px;}
.loss-badge {background:#1F2937;color:#EF4444;border:1px solid #EF4444;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;}
.gain-badge {background:#1F2937;color:#22C55E;border:1px solid #22C55E;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;}
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
    
    # Fix mojibake encoding
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x
        )
    df = df.sample(min(len(df), 50000), random_state=42)

    original_rows = len(df)
    df = df[df["Sales"] > 0]
    cleaned_rows = original_rows - len(df)

    # Parse order date if available (for trend charts)
    date_cols = [c for c in df.columns if "date" in c.lower() or "Date" in c]
    for col in date_cols:
        try:
            df["Order Date Parsed"] = pd.to_datetime(df[col], errors="coerce")
            if df["Order Date Parsed"].notna().sum() > 100:
                break
        except Exception:
            pass
    else:
        # No date column found — create a synthetic one for trend demo
        df["Order Date Parsed"] = pd.NaT

    return df, cleaned_rows

df, cleaned_rows = load_data()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📊 Business Filters")

segment_filter = st.sidebar.multiselect("Customer Segment", sorted(df["Customer Segment"].dropna().unique()))
category_filter = st.sidebar.multiselect("Product Category", sorted(df["Category Name"].dropna().unique()))
market_filter   = st.sidebar.multiselect("Market", sorted(df["Market"].dropna().unique()))
region_filter   = st.sidebar.multiselect("Order Region", sorted(df["Order Region"].dropna().unique()))

# FIX 4: Product-level filter added
product_filter = st.sidebar.multiselect("Product", sorted(df["Product Name"].dropna().unique()))

st.sidebar.markdown("### 💸 Pricing Controls")
discount_slider = st.sidebar.slider("Max Discount Rate", 0.0, 0.5, 0.5)
profit_filter   = st.sidebar.selectbox("Profitability Filter", ["All", "Profitable Only", "Loss-Making Only"])

if cleaned_rows > 0:
    st.sidebar.markdown(f"""
    <div class="data-note">🧹 {cleaned_rows:,} zero-sales rows removed during data cleaning.</div>
    """, unsafe_allow_html=True)

if segment_filter:
    df = df[df["Customer Segment"].isin(segment_filter)]
if category_filter:
    df = df[df["Category Name"].isin(category_filter)]
if market_filter:
    df = df[df["Market"].isin(market_filter)]
if region_filter:
    df = df[df["Order Region"].isin(region_filter)]
if product_filter:
    df = df[df["Product Name"].isin(product_filter)]

df = df[df["Order Item Discount Rate"] <= discount_slider]

if profit_filter == "Profitable Only":
    df = df[df["Order Profit Per Order"] > 0]
elif profit_filter == "Loss-Making Only":
    df = df[df["Order Profit Per Order"] < 0]

# Compute Profit Margin AFTER filtering
df = df.copy()
df["Profit Margin"] = df["Order Profit Per Order"] / df["Sales"]

# ---------------------------------------------------------
# STYLE FUNCTION
# ---------------------------------------------------------
def style(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT_COLOR)) if title else {},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(t=50 if title else 20, b=40, l=40, r=20),
    )
    return fig

# ---------------------------------------------------------
# HELPER: money formatter
# ---------------------------------------------------------
def fmt_money(v):
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def kpi(col, title, value, sub=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🧍 Customers", "📦 Products", "💸 Discounts", "🌍 Regions"
])

# ==========================================================
# TAB 1 — OVERVIEW
# ==========================================================
with tab1:
    total_sales    = df["Sales"].sum()
    total_profit   = df["Order Profit Per Order"].sum()
    profit_margin  = (total_profit / total_sales * 100) if total_sales else 0
    avg_discount   = df["Order Item Discount Rate"].mean()
    order_count    = len(df)
    avg_order_value= total_sales / order_count if order_count else 0

    # FIX 1: Discount Impact Ratio KPI
    # Defined as: average margin loss per unit of discount rate applied
    disc_corr = df[["Order Item Discount Rate", "Profit Margin"]].dropna()
    if len(disc_corr) > 10 and disc_corr["Order Item Discount Rate"].std() > 0:
        from numpy.polynomial.polynomial import polyfit
        coef = np.polyfit(disc_corr["Order Item Discount Rate"], disc_corr["Profit Margin"], 1)
        discount_impact_ratio = coef[0]  # slope: margin change per 1 unit discount rate
    else:
        discount_impact_ratio = 0.0

    st.subheader("📊 Executive Financial Overview")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    kpi(c1, "Total Revenue",         fmt_money(total_sales))
    kpi(c2, "Total Profit",          fmt_money(total_profit))
    kpi(c3, "Profit Margin %",       f"{profit_margin:.2f}%")
    kpi(c4, "Avg Discount",          f"{avg_discount:.2%}")
    kpi(c5, "Total Orders",          f"{order_count:,}")
    kpi(c6, "Avg Order Value",       fmt_money(avg_order_value))
    # FIX 1: Discount Impact Ratio as a proper KPI card
    kpi(c7, "Discount Impact Ratio", f"{discount_impact_ratio:.2f}",
        sub="Margin Δ per unit discount")

    st.markdown("<br>", unsafe_allow_html=True)

    # FIX 3: Revenue vs Profit TIME-SERIES trend chart
    has_dates = df["Order Date Parsed"].notna().sum() > 100 if "Order Date Parsed" in df.columns else False

    if has_dates:
        st.markdown("#### 📈 Revenue & Profit Trend Over Time")
        df["YearMonth"] = df["Order Date Parsed"].dt.to_period("M").astype(str)
        trend = df.groupby("YearMonth").agg(
            Revenue=("Sales", "sum"),
            Profit=("Order Profit Per Order", "sum")
        ).reset_index().sort_values("YearMonth")

        fig_trend = go.Figure()
        fig_trend.add_scatter(x=trend["YearMonth"], y=trend["Revenue"],
                              mode="lines+markers", name="Revenue",
                              line=dict(color=PRIMARY_COLOR, width=2))
        fig_trend.add_scatter(x=trend["YearMonth"], y=trend["Profit"],
                              mode="lines+markers", name="Profit",
                              line=dict(color=GAIN_COLOR, width=2))
        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor=GRID_COLOR, tickangle=-45),
            yaxis=dict(gridcolor=GRID_COLOR),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=40, b=60)
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="chart_trend")
    else:
        st.info("ℹ️ No parseable date column found in dataset — time-series trend chart will appear once an order date column is available.", icon="📅")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Revenue vs Profit by Segment")
        seg = df.groupby("Customer Segment").agg(
            Revenue=("Sales", "sum"),
            Profit=("Order Profit Per Order", "sum")
        ).reset_index()
        fig_seg = px.bar(seg, x="Customer Segment", y=["Revenue", "Profit"],
                         barmode="group",
                         color_discrete_sequence=[PRIMARY_COLOR, GAIN_COLOR])
        st.plotly_chart(style(fig_seg), use_container_width=True, key="chart_7")

    with col_right:
        st.markdown("#### Profit Margin % by Customer Segment")
        seg["Margin %"] = seg["Profit"] / seg["Revenue"] * 100
        fig_margin = px.bar(seg, x="Customer Segment", y="Margin %",
                            color_discrete_sequence=["#1E6BBF"])
        fig_margin.update_traces(marker_line_color="#2A82E9", marker_line_width=1.5)
        st.plotly_chart(style(fig_margin), use_container_width=True, key="chart_8")

    # Profit Concentration by Category (Pareto)
    st.markdown("#### Profit Concentration by Category")
    cat_conc = df.groupby("Category Name")["Order Profit Per Order"].sum().sort_values(ascending=False).reset_index()
    cat_conc["Cumulative %"] = cat_conc["Order Profit Per Order"].cumsum() / cat_conc["Order Profit Per Order"].sum() * 100
    cat_conc["Rank"] = range(1, len(cat_conc) + 1)

    fig_conc = go.Figure()
    fig_conc.add_bar(
        x=cat_conc["Rank"], y=cat_conc["Order Profit Per Order"],
        name="Profit", marker_color=PRIMARY_COLOR,
        customdata=cat_conc["Category Name"],
        hovertemplate="<b>%{customdata}</b><br>Profit: $%{y:,.0f}<extra></extra>"
    )
    fig_conc.add_scatter(
        x=cat_conc["Rank"], y=cat_conc["Cumulative %"],
        mode="lines+markers", name="Cumulative %", yaxis="y2",
        line=dict(color=GAIN_COLOR, width=2), marker=dict(size=5),
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
    )
    fig_conc.update_layout(
        yaxis=dict(title="Profit ($)", gridcolor=GRID_COLOR),
        yaxis2=dict(overlaying="y", side="right", range=[0, 110],
                    tickformat=".0f", title="Cumulative %", gridcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Category Rank (hover bar for name)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=40, b=50)
    )
    st.plotly_chart(fig_conc, use_container_width=True, key="chart_1")

    # FIX 2: Loss-making CATEGORIES explicit section
    st.markdown("#### ⚠️ Loss-Making Categories")
    cat_profit = df.groupby("Category Name").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum"),
        Orders=("Sales", "count")
    ).reset_index()
    cat_profit["Margin %"] = cat_profit["Profit"] / cat_profit["Revenue"] * 100
    loss_cats = cat_profit[cat_profit["Profit"] < 0].sort_values("Profit")

    if loss_cats.empty:
        st.success("✅ No loss-making categories detected with current filters.")
    else:
        st.warning(f"⚠️ {len(loss_cats)} category/categories are operating at a net loss.")
        # Color-coded table
        loss_display = loss_cats[["Category Name", "Revenue", "Profit", "Margin %", "Orders"]].copy()
        st.dataframe(
            loss_display.style
            .format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%"})
            .applymap(lambda v: "color:#EF4444;font-weight:600" if isinstance(v, (int, float)) and v < 0 else "",
                      subset=["Profit", "Margin %"]),
            use_container_width=True
        )
        fig_loss_cat = px.bar(
            loss_cats, x="Profit", y="Category Name", orientation="h",
            color="Profit",
            color_continuous_scale=["#EF4444", "#F59E0B", "#2A82E9"],
            title="Loss-Making Categories — Net Profit"
        )
        st.plotly_chart(style(fig_loss_cat), use_container_width=True, key="chart_loss_cat")

    # Executive Summary
    customer_summary = df.groupby("Customer Id")["Order Profit Per Order"].sum().sort_values(ascending=False)
    top_20_pct = max(int(len(customer_summary) * 0.2), 1)
    top_contribution = customer_summary.head(top_20_pct).sum() / customer_summary.sum() * 100 if customer_summary.sum() != 0 else 0

    cat_summary = df.groupby("Category Name")["Order Profit Per Order"].sum().reset_index()
    worst_category = cat_summary.sort_values("Order Profit Per Order").iloc[0]["Category Name"] if len(cat_summary) else "N/A"
    loss_cat_count = len(loss_cats)

    disc_bins = pd.cut(df["Order Item Discount Rate"], bins=5).astype(str)
    discount_analysis = df.groupby(disc_bins)["Profit Margin"].mean().reset_index()
    negative_bins = discount_analysis[discount_analysis["Profit Margin"] < 0]

    if not negative_bins.empty:
        discount_warning = f"Profit turns negative at discount range {negative_bins.iloc[0].iloc[0]}"
    else:
        discount_warning = "No negative profit zones detected across discount ranges"

    st.markdown(f"""
    <div class="summary-box">
    <h3 style="color:white;">📋 Executive Insights</h3>
    <p style="color:#D1D5DB; line-height:1.9;">
    Total revenue stands at <b>{fmt_money(total_sales)}</b> with a net profit of
    <b>{fmt_money(total_profit)}</b>, resulting in a profit margin of <b>{profit_margin:.2f}%</b>
    across <b>{order_count:,}</b> orders.
    <br><br>
    The <b>Discount Impact Ratio</b> of <b>{discount_impact_ratio:.2f}</b> indicates that for every
    1-unit increase in discount rate, profit margin shifts by {discount_impact_ratio:.2f} units —
    a key signal of discount-driven margin erosion.
    <br><br>
    Customer analysis reveals that the top 20% of customers contribute approximately
    <b>{top_contribution:.1f}%</b> of total profit, confirming a strong Pareto concentration.
    <br><br>
    Product-level analysis highlights <b>{worst_category}</b> as the lowest-performing category.
    In total, <b>{loss_cat_count} product {'category is' if loss_cat_count == 1 else 'categories are'}</b>
    operating at a net loss, requiring immediate pricing or cost intervention.
    <br><br>
    Discount analysis shows that <b>{discount_warning}</b>, suggesting aggressive
    discounting may be eroding margins significantly.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# TAB 2 — CUSTOMERS
# ==========================================================
with tab2:
    st.subheader("🧍 Customer Profitability Analysis")

    customer = df.groupby("Customer Id").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum"),
        Orders=("Sales", "count")
    ).reset_index()

    total_customer_profit = customer["Profit"].sum()
    customer["Customer Value Index"] = (
        customer["Profit"] / total_customer_profit if total_customer_profit != 0 else 0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Customer Segment Contribution")
        seg_contrib = df.groupby("Customer Segment").agg(
            Revenue=("Sales", "sum"),
            Profit=("Order Profit Per Order", "sum")
        ).reset_index()
        fig_pie = px.pie(seg_contrib, names="Customer Segment", values="Profit",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig_pie, use_container_width=True, key="chart_2")

    with col2:
        st.markdown("#### Revenue vs Profit per Segment")
        fig_seg2 = px.bar(seg_contrib, x="Customer Segment", y=["Revenue", "Profit"],
                          barmode="group",
                          color_discrete_sequence=[PRIMARY_COLOR, GAIN_COLOR])
        st.plotly_chart(style(fig_seg2), use_container_width=True, key="chart_9")

    st.markdown("#### Customer Value Distribution")
    fig3 = px.scatter(customer, x="Revenue", y="Profit",
                      size="Orders", color="Profit",
                      color_continuous_scale=["#0D3B6E", "#2A82E9", "#7EC8F8"],
                      hover_data=["Customer Id", "Orders"])
    st.plotly_chart(style(fig3), use_container_width=True, key="chart_10")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏆 Top 10 Customers by Profit")
        st.dataframe(
            customer.sort_values("Profit", ascending=False).head(10)
            [["Customer Id", "Revenue", "Profit", "Orders", "Customer Value Index"]]
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Customer Value Index": "{:.4f}"}),
            use_container_width=True
        )
    with col_b:
        st.markdown("#### ⚠️ Top 10 Loss-Making Customers")
        st.dataframe(
            customer[customer["Profit"] < 0].sort_values("Profit").head(10)
            [["Customer Id", "Revenue", "Profit", "Orders"]]
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}"}),
            use_container_width=True
        )

    # Pareto
    st.markdown("#### 🔥 Pareto Analysis — Top 40 Customers")
    customer_pareto = customer.sort_values("Profit", ascending=False).reset_index(drop=True)
    top_n = customer_pareto.head(40).copy()
    top_n["Cumulative %"] = top_n["Profit"].cumsum() / customer_pareto["Profit"].sum()

    contribution = top_n["Profit"].sum() / customer_pareto["Profit"].sum() * 100
    st.success(f"Top 40 customers contribute **{contribution:.2f}%** of total profit")

    fig_pareto = go.Figure()
    fig_pareto.add_bar(x=top_n["Customer Id"].astype(str), y=top_n["Profit"],
                       name="Profit", marker_color=PRIMARY_COLOR)
    fig_pareto.add_scatter(x=top_n["Customer Id"].astype(str), y=top_n["Cumulative %"],
                           mode="lines+markers", name="Cumulative %", yaxis="y2",
                           line=dict(color=GAIN_COLOR))
    fig_pareto.add_hline(y=0.8, line_dash="dash", line_color=LOSS_COLOR,
                          annotation_text="80% Threshold",
                          annotation_position="top right", yref="y2")
    fig_pareto.update_layout(
        yaxis=dict(title="Profit ($)", gridcolor=GRID_COLOR),
        yaxis2=dict(overlaying="y", side="right", range=[0, 1.1],
                    tickformat=".0%", title="Cumulative %", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(type="category", showticklabels=False,
                   title="Customers ranked by profit (hover bar for Customer ID)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR), legend=dict(orientation="h", y=1.08),
        margin=dict(t=40, b=60)
    )
    st.plotly_chart(fig_pareto, use_container_width=True, key="chart_3")


# ==========================================================
# TAB 3 — PRODUCTS
# ==========================================================
with tab3:
    st.subheader("📦 Product & Category Profitability")

    cat = df.groupby("Category Name").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum"),
        Orders=("Sales", "count")
    ).reset_index()
    cat["Margin %"] = cat["Profit"] / cat["Revenue"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(cat.sort_values("Revenue", ascending=True),
                      x="Revenue", y="Category Name", orientation="h",
                      color_discrete_sequence=[PRIMARY_COLOR])
        st.plotly_chart(style(fig1, "Revenue by Category"), use_container_width=True, key="chart_11")
    with col2:
        fig2 = px.bar(cat.sort_values("Profit", ascending=True),
                      x="Profit", y="Category Name", orientation="h",
                      color="Profit",
                      color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"])
        st.plotly_chart(style(fig2, "Profit by Category"), use_container_width=True, key="chart_12")

    # FIX 2: Explicit loss-making category callout in Products tab
    st.markdown("#### ⚠️ Loss-Making Categories Summary")
    loss_cats_prod = cat[cat["Profit"] < 0].sort_values("Profit")
    if loss_cats_prod.empty:
        st.success("✅ No loss-making categories with current filters.")
    else:
        cols_loss = st.columns(min(len(loss_cats_prod), 4))
        for i, (_, row) in enumerate(loss_cats_prod.iterrows()):
            with cols_loss[i % 4]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{row['Category Name']}</div>
                    <div class="kpi-value" style="color:#EF4444;">{fmt_money(row['Profit'])}</div>
                    <div class="kpi-sub">Margin: {row['Margin %']:.1f}% | {int(row['Orders'])} orders</div>
                </div>
                """, unsafe_allow_html=True)

    # Category Heatmap
    st.markdown("#### 🗺️ Category Profitability Heatmap (by Market)")
    pivot = df.groupby(["Category Name", "Market"])["Order Profit Per Order"].sum().unstack(fill_value=0)
    fig_heat = px.imshow(pivot,
                         color_continuous_scale=["#EF4444", "#161B22", "#0D3B6E", "#2A82E9", "#7EC8F8"],
                         aspect="auto", text_auto=".0f")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
                            xaxis=dict(tickangle=-30))
    st.plotly_chart(fig_heat, use_container_width=True, key="chart_4")

    # Product level
    st.markdown("#### 📋 Product-Level Profitability")
    product = df.groupby("Product Name").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum"),
        Orders=("Sales", "count")
    ).reset_index()
    product["Margin %"] = product["Profit"] / product["Revenue"] * 100

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🏆 Top 10 Products by Profit**")
        st.dataframe(
            product.sort_values("Profit", ascending=False).head(10)
            [["Product Name", "Revenue", "Profit", "Margin %"]]
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%"}),
            use_container_width=True
        )
    with col_b:
        st.markdown("**⚠️ Loss-Making Products**")
        st.dataframe(
            product[product["Profit"] < 0].sort_values("Profit").head(10)
            [["Product Name", "Revenue", "Profit", "Margin %"]]
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%"}),
            use_container_width=True
        )

    # Bubble chart
    st.markdown("#### Revenue vs Margin Bubble Chart (Top 30 Products)")
    top30 = product.nlargest(30, "Revenue")
    fig_bubble = px.scatter(top30, x="Revenue", y="Margin %", size="Orders",
                             color="Margin %",
                             color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"],
                             hover_name="Product Name",
                             hover_data={"Revenue": ":$,.0f", "Margin %": ":.1f", "Orders": True})
    fig_bubble.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#2A82E9")))
    st.plotly_chart(style(fig_bubble), use_container_width=True, key="chart_13")


# ==========================================================
# TAB 4 — DISCOUNTS
# ==========================================================
with tab4:
    st.subheader("💸 Discount Impact Analysis")

    # FIX 1: Discount Impact Ratio shown prominently at top of this tab
    disc_kpi_col1, disc_kpi_col2, disc_kpi_col3 = st.columns(3)
    kpi(disc_kpi_col1, "Discount Impact Ratio",
        f"{discount_impact_ratio:.2f}",
        sub="Margin Δ per 1-unit discount rate")
    avg_margin_no_disc = df[df["Order Item Discount Rate"] == 0]["Profit Margin"].mean() if (df["Order Item Discount Rate"] == 0).any() else profit_margin
    avg_margin_with_disc = df[df["Order Item Discount Rate"] > 0]["Profit Margin"].mean() if (df["Order Item Discount Rate"] > 0).any() else profit_margin
    kpi(disc_kpi_col2, "Avg Margin (No Discount)",   f"{avg_margin_no_disc*100:.2f}%")
    kpi(disc_kpi_col3, "Avg Margin (With Discount)",  f"{avg_margin_with_disc*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.scatter(df.sample(min(5000, len(df)), random_state=1),
                          x="Order Item Discount Rate", y="Profit Margin",
                          color="Profit Margin",
                          color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"],
                          opacity=0.5)
        st.plotly_chart(style(fig4, "Discount Rate vs Profit Margin"), use_container_width=True, key="chart_14")

    with col2:
        df["Discount Bin"] = pd.cut(df["Order Item Discount Rate"], bins=5).astype(str)
        discount_analysis = df.groupby("Discount Bin").agg(
            Avg_Margin=("Profit Margin", "mean"),
            Orders=("Sales", "count")
        ).reset_index()
        fig_disc = px.bar(discount_analysis, x="Discount Bin", y="Avg_Margin",
                          color="Avg_Margin",
                          color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"])
        st.plotly_chart(style(fig_disc, "Avg Margin by Discount Range"), use_container_width=True, key="chart_15")

    # What-If Simulator
    st.markdown("---")
    st.markdown("### 🔮 What-If Discount Scenario Simulator")
    st.markdown("Adjust the discount rate to see projected impact on revenue and profit.")

    wcol1, wcol2 = st.columns([1, 2])
    with wcol1:
        what_if_discount = st.slider("Simulated Discount Rate", 0.0, 0.5, 0.1, step=0.01)
        base_discount    = df["Order Item Discount Rate"].mean()
        base_revenue     = df["Sales"].sum()
        base_profit      = df["Order Profit Per Order"].sum()

        discount_delta       = what_if_discount - base_discount
        projected_revenue    = base_revenue * (1 + discount_delta * 0.5)
        projected_margin_chg = -discount_delta * 1.8
        projected_profit     = base_profit + base_revenue * projected_margin_chg
        projected_margin_pct = (projected_profit / projected_revenue * 100) if projected_revenue else 0

        delta_profit = projected_profit - base_profit
        delta_color  = GAIN_COLOR if delta_profit >= 0 else LOSS_COLOR

        st.markdown(f"""
        <div class="kpi-card" style="margin-top:16px;">
            <div class="kpi-title">Projected Revenue</div>
            <div class="kpi-value">{fmt_money(projected_revenue)}</div>
        </div>
        <div class="kpi-card" style="margin-top:8px;">
            <div class="kpi-title">Projected Profit</div>
            <div class="kpi-value" style="color:{delta_color};">{fmt_money(projected_profit)}</div>
        </div>
        <div class="kpi-card" style="margin-top:8px;">
            <div class="kpi-title">Projected Margin</div>
            <div class="kpi-value" style="color:{delta_color};">{projected_margin_pct:.2f}%</div>
        </div>
        <div class="kpi-card" style="margin-top:8px;">
            <div class="kpi-title">Profit Impact</div>
            <div class="kpi-value" style="color:{delta_color};">{"+" if delta_profit >= 0 else ""}{fmt_money(delta_profit)}</div>
        </div>
        """, unsafe_allow_html=True)

    with wcol2:
        rates        = np.arange(0.0, 0.51, 0.01)
        proj_profits = []
        proj_margins = []
        for r in rates:
            d    = r - base_discount
            rev  = base_revenue * (1 + d * 0.5)
            prof = base_profit + base_revenue * (-d * 1.8)
            proj_profits.append(prof)
            proj_margins.append((prof / rev * 100) if rev else 0)

        what_if_df = pd.DataFrame({
            "Discount Rate":    rates,
            "Projected Profit": proj_profits,
            "Projected Margin %": proj_margins
        })

        fig_whatif = go.Figure()
        fig_whatif.add_scatter(x=what_if_df["Discount Rate"], y=what_if_df["Projected Profit"],
                               mode="lines", name="Projected Profit",
                               line=dict(color=PRIMARY_COLOR, width=2))
        fig_whatif.add_scatter(x=what_if_df["Discount Rate"], y=what_if_df["Projected Margin %"],
                               mode="lines", name="Projected Margin %",
                               line=dict(color=GAIN_COLOR, width=2, dash="dash"),
                               yaxis="y2")
        fig_whatif.add_vline(x=what_if_discount, line_dash="dot", line_color="white",
                              annotation_text=f"Selected: {what_if_discount:.0%}",
                              annotation_position="top right")
        fig_whatif.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Margin %"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            legend=dict(orientation="h", y=1.1),
            title="Profit & Margin Sensitivity Curve"
        )
        st.plotly_chart(fig_whatif, use_container_width=True, key="chart_5")

        st.info(
            "⚠️ **Model Note**: Projections use a simplified elasticity model "
            "(revenue elasticity = 0.5, margin sensitivity = 1.8×). "
            "Replace with your regression coefficients for precise forecasting."
        )


# ==========================================================
# TAB 5 — REGIONS
# ==========================================================
with tab5:
    st.subheader("🌍 Regional & Market Profitability")

    region = df.groupby("Order Region").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum")
    ).reset_index()
    region["Margin %"] = region["Profit"] / region["Revenue"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig_r = px.bar(region.sort_values("Revenue"), x="Revenue", y="Order Region",
                       orientation="h", color_discrete_sequence=[PRIMARY_COLOR])
        st.plotly_chart(style(fig_r, "Revenue by Region"), use_container_width=True, key="chart_16")
    with col2:
        fig_r2 = px.bar(region.sort_values("Profit"), x="Profit", y="Order Region",
                        orientation="h", color="Profit",
                        color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"])
        st.plotly_chart(style(fig_r2, "Profit by Region"), use_container_width=True, key="chart_17")

    # Strong revenue but weak profit — new highlight
    st.markdown("#### 🔍 Markets with Strong Revenue but Weak Profit")
    market = df.groupby("Market").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum")
    ).reset_index()
    market["Margin %"] = market["Profit"] / market["Revenue"] * 100

    rev_median    = market["Revenue"].median()
    margin_median = market["Margin %"].median()
    concern_markets = market[(market["Revenue"] >= rev_median) & (market["Margin %"] < margin_median)]

    if not concern_markets.empty:
        st.warning(f"⚠️ {len(concern_markets)} market(s) show high revenue but below-median profit margins — potential over-discounting or cost inefficiencies.")
        for _, row in concern_markets.iterrows():
            st.markdown(f"""
            <div class="kpi-card" style="display:inline-block;width:220px;margin-right:12px;">
                <div class="kpi-title">{row['Market']}</div>
                <div class="kpi-value">{fmt_money(row['Revenue'])}</div>
                <div class="kpi-sub" style="color:#EF4444;">Margin: {row['Margin %']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    fig_m = px.bar(market, x="Market", y=["Revenue", "Profit"],
                   barmode="group", color_discrete_sequence=[PRIMARY_COLOR, GAIN_COLOR])
    st.plotly_chart(style(fig_m, "Market Revenue vs Profit"), use_container_width=True, key="chart_18")

    fig_mmargin = px.bar(market.sort_values("Margin %"), x="Market", y="Margin %",
                          color="Margin %",
                          color_continuous_scale=["#EF4444", "#161B22", "#2A82E9"],
                          text_auto=".1f")
    st.plotly_chart(style(fig_mmargin, "Profit Margin % by Market"), use_container_width=True, key="chart_19")

    # World Map
    st.markdown("#### 🗺️ Profit by Country (World Map)")
    country = df.groupby("Order Country").agg(
        Revenue=("Sales", "sum"),
        Profit=("Order Profit Per Order", "sum")
    ).reset_index()
    country["Margin %"] = country["Profit"] / country["Revenue"] * 100

    fig_map = px.choropleth(
        country, locations="Order Country", locationmode="country names",
        color="Profit", hover_name="Order Country",
        hover_data={"Revenue": ":,.0f", "Profit": ":,.0f", "Margin %": ":.1f"},
        color_continuous_scale=["#EF4444", "#161B22", "#0D3B6E", "#2A82E9", "#7EC8F8"]
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                 showcoastlines=True, coastlinecolor="#374151")
    )
    st.plotly_chart(fig_map, use_container_width=True, key="chart_6")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🏆 Top 10 Countries by Profit**")
        st.dataframe(
            country.sort_values("Profit", ascending=False).head(10)
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%"}),
            use_container_width=True
        )
    with col_b:
        st.markdown("**⚠️ Bottom 10 Countries by Profit**")
        st.dataframe(
            country.sort_values("Profit").head(10)
            .style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%"}),
            use_container_width=True
        )


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
        <span>Version 3.0 | Mar 2026</span>
    </div>
    """, unsafe_allow_html=True)

render_footer()
