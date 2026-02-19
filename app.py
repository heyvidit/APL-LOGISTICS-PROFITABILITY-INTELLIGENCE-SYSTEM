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
        page_icon="favicon.jfif"   # ✅ favicon added
    )

    # ---------------------------------------------------------
    # GLOBAL CONSTANTS
    # ---------------------------------------------------------
    DATA_PATH = Path("APL_Logistics.csv.gz")
    APL_LOGO_PATH = Path("APL_Logo.png")
    UNIFIED_LOGO_PATH = Path("unified logo.png")
    TARGET = "Late_delivery_risk"

    # Chart typography (balanced & professional)
    PLOTLY_FONT = dict(family="Arial", size=14, color="#EAEAEA")
    TITLE_SIZE = 22
    AXIS_LABEL_SIZE = 16
    TICK_SIZE = 14

    # ---------------------------------------------------------
    # GLOBAL SIDEBAR STYLING
    # ---------------------------------------------------------
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] h3 {
        font-size: 17px;
        margin-top: 20px;
    }
    section[data-testid="stSidebar"] label {
        font-size: 14px;
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
    # LOAD DATA (CACHED & FAST)
    # ---------------------------------------------------------
    @st.cache_data
    def load_data():
        df = pd.read_csv(DATA_PATH, encoding="latin1")
        return df.sample(min(len(df), 50000), random_state=42)

    df = load_data()

    # ---------------------------------------------------------
    # DATA CLEANING (NO LEAKAGE)
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
    # SIDEBAR FILTERS
    # ---------------------------------------------------------
    st.sidebar.header("🔎 Filters")

    st.sidebar.subheader("🚚 Logistics")
    ship_filter = st.sidebar.multiselect(
        "Shipping Mode", sorted(df["Shipping Mode"].dropna().unique())
    )

    st.sidebar.subheader("🌍 Geography")
    market_filter = st.sidebar.multiselect(
        "Market", sorted(df["Market"].dropna().unique())
    )
    region_filter = st.sidebar.multiselect(
        "Order Region", sorted(df["Order Region"].dropna().unique())
    )

    st.sidebar.subheader("👥 Customer")
    segment_filter = st.sidebar.multiselect(
        "Customer Segment", sorted(df["Customer Segment"].dropna().unique())
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
    fig_cm.update_xaxes(tickfont_size=TICK_SIZE)
    fig_cm.update_yaxes(tickfont_size=TICK_SIZE)

    st.plotly_chart(fig_cm, use_container_width=True)

    # ---------------------------------------------------------
    # CHART STYLING HELPER
    # ---------------------------------------------------------
    def style(fig, title):
        fig.update_layout(
            title=title,
            font=PLOTLY_FONT,
            title_font_size=TITLE_SIZE,
            xaxis_title_font_size=AXIS_LABEL_SIZE,
            yaxis_title_font_size=AXIS_LABEL_SIZE
        )
        fig.update_xaxes(tickfont_size=TICK_SIZE)
        fig.update_yaxes(tickfont_size=TICK_SIZE)
        return fig

    # ---------------------------------------------------------
    # VISUALS
    # ---------------------------------------------------------
    st.plotly_chart(
        style(
            px.histogram(
                pd.DataFrame({"Delay Probability": y_proba}),
                x="Delay Probability",
                nbins=30
            ),
            "Late Delivery Risk Distribution"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        style(
            px.bar(
                df.groupby("Order Region")[TARGET].mean().reset_index(),
                x="Order Region",
                y=TARGET
            ),
            "Average Delay Risk by Region"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        style(
            px.bar(
                df.groupby("Shipping Mode")[TARGET].mean().reset_index(),
                x="Shipping Mode",
                y=TARGET
            ),
            "Average Delay Risk by Shipping Mode"
        ),
        use_container_width=True
    )

    # ---------------------------------------------------------
    # HIGH-RISK ACTION QUEUE
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

    st.plotly_chart(
        style(
            px.bar(
                coef_df,
                x="Impact",
                y="Feature",
                orientation="h"
            ),
            "Key Drivers of Late Delivery Risk"
        ),
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
