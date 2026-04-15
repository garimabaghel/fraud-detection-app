import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #FF4B4B;
    text-align: center;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}
.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
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

# ---------- INPUT SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
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

st.markdown('</div>', unsafe_allow_html=True)

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

    prob = model.predict_proba(data)[0][1]

    # Adjust probability
    if amt > 50000:
        prob += 0.2
    elif amt < 1000:
        prob -= 0.1

    prob = max(0, min(prob, 1))

    # ---------- RESULT UI ----------
    # ---------- RESULT UI ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Transaction Risk Analysis")

# Progress Bar
st.progress(int(prob * 100))

# Big Metric
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Amount", f"₹{amt}")

with col2:
    st.metric("📍 Distance (km)", f"{distance:.2f}")

with col3:
    st.metric("⚠️ Fraud Probability", f"{prob:.2f}")

st.markdown("---")

# Risk Level Display
if prob > 0.4:
    st.markdown(
        '<div class="result-card" style="background: linear-gradient(90deg,#ff4b4b,#ff0000);">🚨 HIGH RISK FRAUD TRANSACTION</div>',
        unsafe_allow_html=True
    )
elif prob > 0.25:
    st.markdown(
        '<div class="result-card" style="background: linear-gradient(90deg,#f39c12,#e67e22);">⚠️ MEDIUM RISK TRANSACTION</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="result-card" style="background: linear-gradient(90deg,#2ecc71,#27ae60);">✅ LOW RISK TRANSACTION</div>',
        unsafe_allow_html=True
    )

# ---------- EXPLANATION (VERY IMPORTANT FOR VIVA) ----------
st.markdown("### 🧠 Why this result?")

reasons = []

if distance > 1000:
    reasons.append("📍 Large distance between customer and merchant")

if hour < 6 or hour > 22:
    reasons.append("🌙 Transaction at unusual time")

if amt > 50000:
    reasons.append("💰 High transaction amount")

if amt < 1000:
    reasons.append("🔍 Very small test transaction (common in fraud)")

if not reasons:
    reasons.append("✔️ Transaction pattern looks normal")

for r in reasons:
    st.write("•", r)

st.markdown('</div>', unsafe_allow_html=True)
