# 📦 Demand Forecasting & Inventory Optimization

A machine learning system that predicts short-term product demand and generates inventory reorder recommendations using historical retail sales data.

---

## 🔍 Overview

This project forecasts near-term daily demand for individual Store–Product combinations and integrates the predictions with inventory planning logic to support data-driven replenishment decisions.

The system combines:

- Time-based demand forecasting

- Feature engineering (lags & rolling averages)

- Inventory-aware reorder calculations

- Interactive Streamlit deployment

---

## ⚙️ Key Features

- Short-horizon demand prediction (1–14 days ahead)

- Store & product-level forecasting

- Inventory-aware reorder recommendation

- Clean two-page Streamlit interface

---

## 🛠 Tech Stack

- Python

- Numpy

- Pandas

- Scikit-learn

- Streamlit

- Joblib

---

## How to Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
