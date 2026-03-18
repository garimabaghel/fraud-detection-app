import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic
import time

# City mapping
city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}

# City population
city_population = {
    "Delhi": 19000000,
    "Mumbai": 20000000,
    "Bangalore": 12000000,
    "Chennai": 11000000,
    "Kolkata": 14000000
}

# Category mapping
category_map = {
    "Food": 0,
    "Shopping": 1,
    "Travel": 2,
    "Entertainment": 3,
    "Health": 4
}

# Merchant mapping
merchant_map = {
    "Amazon": 0,
    "Walmart": 1,
    "Target": 2,
    "Starbucks": 3,
    "eBay": 4
}

# Load model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details below:")

# Inputs
amt = st.number_input("Transaction Amount")

gender = st.selectbox("Gender", ["Male", "Female"])

category_name = st.selectbox("Category", list(category_map.keys()))
merchant_name = st.selectbox("Merchant", list(merchant_map.keys()))

# ✅ FIXED
category = category_map[category_name]
merchant = merchant_map[merchant_name]

customer_city = st.selectbox("Customer City", list(city_coords.keys()))
merchant_city = st.selectbox("Merchant City", list(city_coords.keys()))

lat, long = city_coords[customer_city]
merch_lat, merch_long = city_coords[merchant_city]

hour = st.slider("Hour", 0, 23)
day = st.slider("Day", 1, 31)
month = st.slider("Month", 1, 12)

# Convert gender
gender_val = 0 if gender == "Male" else 1

# Distance
distance = geodesic((lat, long), (merch_lat, merch_long)).km

# Prediction
if st.button("Check Fraud"):

    city_pop = city_population[customer_city]
    unix_time = int(time.time())

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

    st.write("Input Data:", data)

    prob = model.predict_proba(data)[0][1]
    st.write(f"Fraud Probability: {prob:.2f}")

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
