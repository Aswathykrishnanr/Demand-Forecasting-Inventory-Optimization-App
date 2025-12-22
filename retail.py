import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

st.title("Retail Demand Forecasting & Inventory Optimization App")

df = pd.read_csv("FMCG.csv")

df["sku"] = df["sku"].astype(str).str.strip()

#unique SKUs
sku_list = sorted(df["sku"].unique())

selected_sku = st.selectbox("Select Product (SKU)", sku_list)

st.write("You selected:", selected_sku)

row = df[df["sku"] == selected_sku].sort_values("date").iloc[-1]

st.subheader("Product Details")
st.write("Brand:", row["brand"])
st.write("Category:", row["category"])
st.write("Segment:", row["segment"])
st.write("Pack Type:", row["pack_type"])
st.write("Price:", row["price_unit"])

import datetime

date_input = st.date_input("Select Date", datetime.date.today())
stock_input = st.slider("Current Stock",min_value=0,max_value=int(df["units_sold"].quantile(0.95) * 2),value=100,step=1)

hist = df[df["sku"] == selected_sku].sort_values("date")

if len(hist) < 30:
    st.error("Not enough historical data for forecasting (need at least 30 records).")
    st.stop()

lag_1 = hist["units_sold"].iloc[-1]
lag_7 = hist["units_sold"].iloc[-7:].mean()
rolling_7 = hist["units_sold"].iloc[-7:].mean()
rolling_30 = hist["units_sold"].iloc[-30:].mean()

df_input = pd.DataFrame([{
    "sku": row["sku"],
    "brand": row["brand"],
    "segment": row["segment"],
    "category": row["category"],
    "channel": row["channel"],
    "region": row["region"],
    "pack_type": row["pack_type"],
    "price_unit": row["price_unit"],
    "promotion_flag": row["promotion_flag"],
    "delivery_days": row["delivery_days"],
    "stock_available": stock_input,
    "year": date_input.year,
    "month": date_input.month,
    "week": date_input.isocalendar()[1],
    "day": date_input.day,
    "weekday": date_input.weekday(),
    "is_weekend": 1 if date_input.weekday() >= 5 else 0,

    
    "lag_1": lag_1,
    "lag_7": lag_7,
    "rolling_7": rolling_7,
    "rolling_30": rolling_30
}])

cat_cols = ["sku", "brand", "segment", "category", "channel", "region", "pack_type"]

model, encoders = load_artifacts()

for col in cat_cols:
    df_input[col] = encoders[col].transform(df_input[col])

df_input = df_input[model.feature_names_in_]

prediction = int(round(model.predict(df_input)[0]))

st.success(f"Predicted Demand: {prediction:.2f}")

# -----------------------------
# INVENTORY CALCULATIONS
# -----------------------------

#Safety Stock
demand_std = df["units_sold"].std()
safety_stock = demand_std * 1.65  

predicted_units = prediction
lead_time = row["delivery_days"]

reorder_point = (predicted_units * lead_time) + safety_stock
recommended_inventory = predicted_units + safety_stock

st.subheader("Inventory Recommendations")

st.write(f"**Safety Stock:** {safety_stock:.2f} units")
st.write(f"**Reorder Point (ROP):** {reorder_point:.2f} units")
st.write(f"**Recommended Inventory Level:** {recommended_inventory:.2f} units")
days_of_cover = stock_input / max(prediction, 1)

st.write(f"Stock will last approximately {days_of_cover:.1f} days")


#reorder?
if stock_input < reorder_point:
    st.error("⚠️ Stock is below Reorder Point — Reorder Required!")
else:
    st.success("Stock is sufficient — No immediate reorder needed.")

hist["rolling_7"] = hist["units_sold"].rolling(7).mean()
st.subheader("📊 Recent Sales")


avg_7 = hist["units_sold"].tail(7).mean()
avg_30 = hist["units_sold"].tail(30).mean()

st.write("7-Day Avg Sales:", round(avg_7, 2))
st.write("30-Day Avg Sales:", round(avg_30, 2))



