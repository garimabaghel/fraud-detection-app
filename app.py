import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic
# Category mapping (example)
category_map = {
    "Food": 0,
    "Shopping": 1,
    "Travel": 2,
    "Entertainment": 3,
    "Health": 4
}

# Merchant mapping (example)
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

category = st.number_input("Category (encoded value)")
merchant = st.number_input("Merchant (encoded value)")

lat = st.number_input("Customer Latitude")
long = st.number_input("Customer Longitude")

merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

hour = st.slider("Hour", 0, 23)
day = st.slider("Day", 1, 31)
month = st.slider("Month", 1, 12)

# Convert gender
if gender == "Male":
    gender_val = 0
else:
    gender_val = 1

# Auto calculate distance
distance = geodesic((lat, long), (merch_lat, merch_long)).km

# Prediction
if st.button("Check Fraud"):

    data = pd.DataFrame([[
        merchant, category, amt, gender_val,
        lat, long, merch_lat, merch_long,
        hour, day, month, distance
    ]], columns=[
        'merchant', 'category', 'amt', 'gender',
        'lat', 'long', 'merch_lat', 'merch_long',
        'hour', 'day', 'month', 'distance'
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
