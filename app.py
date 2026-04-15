import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.caption("AI-powered system for real-time fraud detection")

# ---------- DATA ----------
city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}

city_population = {
    "Delhi": 19000000,
    "Mumbai": 20000000,
    "Bangalore": 12000000,
    "Chennai": 11000000,
    "Kolkata": 14000000
}

category_map = {
    "Food": 0,
    "Shopping": 1,
    "Travel": 2,
    "Entertainment": 3,
    "Health": 4
}

merchant_map = {
    "Amazon": 0,
    "Walmart": 1,
    "Target": 2,
    "Starbucks": 3,
    "eBay": 4
}

# Load model
model = joblib.load("model.pkl")

# ---------- INPUT ----------
st.subheader("🔍 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amt = st.number_input("💰 Transaction Amount", min_value=0.0)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    category_name = st.selectbox("🛒 Category", list(category_map.keys()))
    merchant_name = st.selectbox("🏬 Merchant", list(merchant_map.keys()))

with col2:
    customer_city = st.selectbox("📍 Customer City", list(city_coords.keys()))
    merchant_city = st.selectbox("📍 Merchant City", list(city_coords.keys()))
    hour = st.slider("⏰ Hour", 0, 23)
    day = st.slider("📅 Day", 1, 31)
    month = st.slider("📆 Month", 1, 12)

# ---------- PROCESS ----------
category = category_map[category_name]
merchant = merchant_map[merchant_name]
gender_val = 0 if gender == "Male" else 1

lat, long = city_coords[customer_city]
merch_lat, merch_long = city_coords[merchant_city]

distance = geodesic((lat, long), (merch_lat, merch_long)).km

# ---------- BUTTON ----------
if st.button("🚀 Analyze Transaction"):

    city_pop = city_population[customer_city]
    unix_time = 1700000000

    data = pd.DataFrame([[
        merchant, category, amt, gender_val,
        lat, long, city_pop, unix_time,
        merch_lat, merch_long,
        hour, day, month, distance
    ]], columns=[
        'merchant', 'category', 'amt', 'gender',
        'lat', 'long', 'city_pop', 'unix_time',
        'merch_lat', 'merch_long',
        'hour', 'day', 'month', 'distance'
    ])

    # Prediction
    prob = model.predict_proba(data)[0][1]

    # Small adjustment (optional)
    if amt > 50000:
        prob += 0.1
    elif amt < 1000:
        prob -= 0.05

    prob = max(0, min(prob, 1))

    # ---------- RESULT ----------
    st.markdown("## 📊 Transaction Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Amount", f"₹{amt}")

    with col2:
        st.metric("⚠️ Fraud Probability", f"{prob:.2f}")

    # Risk classification
    if prob > 0.4:
        st.error("🚨 High Risk Transaction (Fraudulent)")
    elif prob > 0.25:
        st.warning("⚠️ Medium Risk Transaction (Suspicious)")
    else:
        st.success("✅ Low Risk / Legitimate Transaction")
