import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

amt = st.number_input("Amount")
gender = st.number_input("Gender (0=Male, 1=Female)")
category = st.number_input("Category")
merchant = st.number_input("Merchant")

lat = st.number_input("Customer Latitude")
long = st.number_input("Customer Longitude")

merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

hour = st.number_input("Hour")
day = st.number_input("Day")
month = st.number_input("Month")

distance = st.number_input("Distance")

if st.button("Check Fraud"):

    data = pd.DataFrame([[
        merchant, category, amt, gender,
        lat, long, merch_lat, merch_long,
        hour, day, month, distance
    ]], columns=[
        'merchant', 'category', 'amt', 'gender',
        'lat', 'long', 'merch_lat', 'merch_long',
        'hour', 'day', 'month', 'distance'
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legit Transaction")
