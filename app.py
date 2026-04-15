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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Result")

    st.progress(int(prob * 100))
    st.write(f"**Fraud Probability:** {prob:.2f}")

    if prob > 0.4:
        st.markdown(
            '<div class="result-card" style="background-color:#ff4b4b;">🚨 HIGH RISK FRAUD</div>',
            unsafe_allow_html=True
        )
    elif prob > 0.25:
        st.markdown(
            '<div class="result-card" style="background-color:#f39c12;">⚠️ MEDIUM RISK</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card" style="background-color:#2ecc71;">✅ LOW RISK</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
