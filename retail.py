# app.py
# Run with:
# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    layout="wide"
)

# -----------------------------
# Load model & artifacts
# -----------------------------
def must_load(path):
    if not Path(path).exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

# Load model
if Path("demand_model.pkl").exists():
    model = must_load("demand_model.pkl")
elif Path("model.pkl").exists():
    model = must_load("model.pkl")
else:
    st.error("Model file not found (model.pkl or demand_model.pkl)")
    st.stop()

FEATURES = must_load("model_features.pkl")

@st.cache_data
def load_data():
    if not Path("train.csv").exists():
        st.error("train.csv not found")
        st.stop()
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df

df = load_data()

# -----------------------------
# Forecast logic (short horizon)
# -----------------------------
def forecast_to_date(df_si, target_date, features, model):
    if len(df_si) < 30:
        return None, "Not enough historical data"

    last_date = df_si["date"].iloc[-1]
    if target_date <= last_date:
        return None, "Select a future date"

    history = df_si["demand"].astype(float).tolist()
    cur_date = last_date
    pred = None

    while cur_date < target_date:
        cur_date += pd.Timedelta(days=1)

        lag_7 = history[-7] if len(history) >= 7 else np.mean(history)
        lag_14 = history[-14] if len(history) >= 14 else np.mean(history)
        roll_7 = np.mean(history[-7:])
        roll_14 = np.mean(history[-14:])

        row = {
            "store": int(df_si["store"].iloc[-1]),
            "item": int(df_si["item"].iloc[-1]),
            "dayofweek": cur_date.dayofweek,
            "month": cur_date.month,
            "year": cur_date.year,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_mean_7": roll_7,
            "rolling_mean_14": roll_14,
        }

        X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)
        history.append(pred)

    return pred, None

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Overview", "Forecast & Reorder"])

# =============================
# PAGE 1: OVERVIEW
# =============================
if page == "Overview":
    st.title("📦 Demand Forecasting and Inventory Optimization")

    st.markdown("""
This application predicts **short-term daily product demand** using historical retail data
and generates **inventory reorder recommendations**.

**Key idea**
- Forecast near-future demand (1–14 days)
- Combine predictions with inventory, lead time, and safety buffer
- Support better replenishment decisions
""")

# =============================
# PAGE 2: FORECAST & REORDER
# =============================
else:
    st.title("🔮 Forecast & Reorder")

    # ---------- Row 1: Identity ----------
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        store_id = st.selectbox("Store", sorted(df["store"].unique()))
    with r1c2:
        item_id = st.selectbox("Item (Product)", sorted(df["item"].unique()))

    df_si = df[(df["store"] == store_id) & (df["item"] == item_id)].sort_values("date")
    last_date = df_si["date"].iloc[-1]

    # ---------- Row 2: Forecast controls ----------
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        horizon_days = st.selectbox(
            "Forecast window (days ahead)",
            [1, 3, 7, 14],
            index=3
        )

    min_date = (last_date + pd.Timedelta(days=1)).date()
    max_date = (last_date + pd.Timedelta(days=horizon_days)).date()

    with r2c2:
        forecast_date = st.date_input(
            "Forecast Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

    # ---------- Row 3: Current inventory (FULL WIDTH) ----------
    current_inventory = st.slider(
        "Current Inventory (units)",
        min_value=0,
        max_value=1000,
        value=50,
        step=10
    )

    # ---------- Row 4: Lead time & Safety ----------
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        lead_time_days = st.selectbox(
            "Lead Time (days)",
            [1, 3, 5, 7, 10, 14],
            index=2
        )
    with r4c2:
        safety_days = st.selectbox(
            "Safety Buffer (days)",
            [0, 1, 2, 3, 5, 7],
            index=2
        )

    st.caption(f"Last available date: {last_date.date()}")

    # ---------- Action ----------
    st.markdown("###")
    run = st.button("Predict", use_container_width=True)

    if run:
        target_dt = pd.to_datetime(forecast_date)
        pred_demand, err = forecast_to_date(df_si, target_dt, FEATURES, model)

        if err:
            st.error(err)
            st.stop()

        lead_time_demand = pred_demand * lead_time_days
        safety_stock = pred_demand * safety_days
        reorder_qty = max(
            0.0,
            lead_time_demand + safety_stock - current_inventory
        )

        st.subheader("Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast Date", str(target_dt.date()))
        c2.metric("Predicted Demand", f"{pred_demand:.2f}")
        c3.metric("Reorder Needed", "YES" if reorder_qty > 0 else "NO")
        c4.metric("Reorder Quantity", f"{reorder_qty:.2f}")

