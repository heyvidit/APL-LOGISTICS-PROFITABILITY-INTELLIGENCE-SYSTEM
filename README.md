# 📦 APL Logistics – Predictive Late Delivery Risk Intelligence

## Overview
This project delivers a **predictive analytics system** that identifies shipments at risk of late delivery *before dispatch*.  
It enables logistics teams to move from **reactive delay handling** to **proactive operational planning** using data-driven risk intelligence.

**Organization:** APL Logistics (KWE Group)  
**Internship:** Unified Mentor Pvt. Ltd.  
**Author:** Vidit Kapoor  
**Date:** February 2026  

---

## 🎯 Business Problem
Late deliveries in global logistics networks lead to:
- SLA breaches and financial penalties
- Customer dissatisfaction and churn
- High operational costs due to last-minute fixes

APL Logistics lacked:
- Early warning signals for shipment delays
- Quantitative risk scores for prioritization
- Explainable insights into delay drivers

This system addresses those gaps.

---

## 🧠 Solution Summary
The application:
- Predicts **Late Delivery Probability (0–1)** for each order
- Classifies orders into **Low / Medium / High Risk**
- Highlights **key drivers of delay risk**
- Provides an **operations action queue** for high-risk shipments

---

## 📊 Dashboard Features
### Executive Overview
- Total orders analyzed
- ROC-AUC, Precision, Recall, F1 Score

### Risk Analytics
- Late delivery risk distribution
- Region-wise delay risk
- Shipping mode risk comparison

### Operations Panel
- High-risk order list (priority queue)
- Risk threshold adjustment
- Filters by region, market, shipping mode, and customer segment

### Explainability
- Feature importance from Logistic Regression
- Transparent and interpretable risk drivers

---

## 🧪 Data Science Methodology
### Data Preprocessing
- Removed data leakage variables
- Dropped high-cardinality identifiers
- Encoded categorical variables (One-Hot Encoding)
- Scaled numerical features
- Handled class imbalance using class weights

### Feature Engineering
- Shipping Pressure Index
- Regional Delay Risk Score
- Express Shipping Flag
- Order Complexity Score

### Modeling
- Logistic Regression (baseline, interpretable)
- Stratified train-test split
- Evaluation using ROC-AUC, Precision, Recall, F1 Score, Confusion Matrix

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## 📁 Project Structure
```
├── app.py
├── APL_Logistics.csv
├── APL_Logo.png
├── unified logo.png
├── favicon.jfif
├── README.md
```

---

## ▶️ How to Run Locally
1. Create a virtual environment (recommended)
2. Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```
3. Run the application:
```bash
streamlit run app.py
```

---

## 🚀 Business Impact
- Enables proactive rerouting and prioritization
- Reduces SLA breaches and last-minute costs
- Improves customer communication
- Builds trust through explainable AI

---

## 📌 Disclaimer
This system provides **decision-support intelligence**.  
Final operational decisions should incorporate human expertise and contextual judgment.

---

## 👤 Credits
**Mentor:** Sai Prasad Kagne  
**Created by:** Vidit Kapoor  

---

## 📄 License
This project is developed for academic and internship evaluation purposes.
