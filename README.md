# 📦 Demand Forecasting & Inventory Optimization

A machine learning system that predicts short-term product demand and generates inventory reorder recommendations using historical retail sales data.

---

## 🔍 Overview

Retail businesses often face challenges such as stock-outs and excess inventory due to inaccurate demand estimation.  
This project builds a demand forecasting model that predicts near-term product demand and integrates the predictions with inventory planning logic to support better replenishment decisions.

The system forecasts daily demand for individual **Store–Product combinations** and converts those predictions into **inventory reorder recommendations**.

---

## ⚙️ Key Features

- Short-horizon demand prediction (1–14 days ahead)
- Store and product-level forecasting
- Inventory-aware reorder recommendations
- Interactive Streamlit web application
- End-to-end pipeline from data preprocessing to deployment

---

## 🧠 Machine Learning Approach

### Feature Engineering
To capture demand patterns and short-term trends, the following features were created:

- Calendar features (day of week, month, year)
- Lag features (7-day and 14-day past demand)
- Rolling averages (7-day and 14-day demand trends)

### Model
A **Random Forest Regressor** was used to model demand patterns because it performs well on tabular data and can capture non-linear relationships between features and demand.

### Forecasting Method
Short-term recursive forecasting is used to estimate demand for future dates within a limited prediction horizon.

---

## 📦 Inventory Recommendation Logic

Predicted demand is combined with operational constraints to generate reorder recommendations.

Lead Time Demand  
`Predicted Demand × Lead Time`

Safety Stock  
`Predicted Demand × Safety Buffer`

Reorder Quantity  
`max(0, Lead Time Demand + Safety Stock − Current Inventory)`

This connects machine learning predictions with practical inventory planning decisions.

---

## 🛠 Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Streamlit  
- Joblib  

---


