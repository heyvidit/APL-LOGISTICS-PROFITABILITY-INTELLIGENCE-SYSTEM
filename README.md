# 📊 APL Logistics – Profitability Intelligence Dashboard

## 📌 Project Overview
This project focuses on **profitability-driven analytics** for APL Logistics.  
Instead of just analyzing sales, it uncovers **where the business is actually making or losing money**.

The dashboard provides insights into:
- Customer profitability
- Product & category performance
- Discount impact on margins
- Regional profit distribution

---

## 🎯 Problem Statement
Despite having large volumes of order and sales data, the organization lacked:

- Visibility into **customer-level profitability**
- Understanding of **discount-driven margin erosion**
- Identification of **high-value vs low-value customers**
- Insights into **loss-making products and regions**

This project addresses these gaps by shifting focus from **revenue to profit intelligence**.

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy** – Data processing
- **Plotly** – Data visualization
- **Streamlit** – Interactive dashboard

---

## 📂 Key Features

### 📊 Executive Overview
- Total Revenue
- Total Profit
- Profit Margin (%)
- Average Discount

---

### 🧍 Customer Analytics
- Customer profitability analysis
- Top & loss-making customers
- 🔥 **Customer Value Index (CVI)**
- 🔥 **Pareto Analysis (Top 20% customers contribution)**

---

### 📦 Product & Category Analysis
- Revenue vs Profit comparison
- Category-level performance
- 🔥 **Category Margin Heatmap**

---

### 💸 Discount Impact Analysis
- Discount vs Profit Margin relationship
- Identification of margin erosion zones

---

### 🌍 Regional Analysis
- Profit distribution across regions
- Identification of underperforming markets

---

### 🎛️ Interactive Filters
- Customer Segment
- Product Category
- Market & Region
- Discount Threshold
- Profitability filter

---

## 📈 Key Metrics

| KPI | Description |
|----|------------|
| Total Revenue | Overall sales generated |
| Total Profit | Net profit across orders |
| Profit Margin | Profit as % of sales |
| Customer Value Index | Profit contribution per customer |
| Category Margin | Profitability by category |
| Discount Impact | Effect of discount on margin |

---

## 🔥 Key Insights (Example)
- Top 20% customers contribute majority of total profit  
- High discounts significantly reduce profit margins  
- Certain categories generate high revenue but low profit  
- Some regions are revenue-heavy but profit-negative  

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
