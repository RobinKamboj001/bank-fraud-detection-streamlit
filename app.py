import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load assets
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("🏦 Bank Fraud Detection System")
st.write("Enter transaction details:")

# User Inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", 18, 100)
State = st.number_input("State Code", 0, 50)
City = st.number_input("City Code", 0, 500)
Bank_Branch = st.number_input("Branch Code", 0, 500)
Account_Type = st.selectbox("Account Type", ["Savings", "Checking", "Business"])
Transaction_Time = st.number_input("Transaction Time")
Transaction_Amount = st.number_input("Transaction Amount")
Merchant_ID = st.number_input("Merchant ID")
Transaction_Type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
day = st.number_input("Day", 1, 31)
month = st.number_input("Month", 1, 12)
weekday = st.number_input("Weekday", 0, 6)

# Encoding (must match training)
gender_map = {"Male":0, "Female":1}
account_map = {"Savings":0, "Checking":1, "Business":2}
txn_type_map = {"Online":0, "POS":1, "ATM":2}

# Create empty input with all required columns
input_data = pd.DataFrame(
    np.zeros((1, len(columns))),
    columns=columns
)

# Fill only known user inputs
input_data['Gender'] = gender_map[Gender]
input_data['Age'] = Age
input_data['State'] = State
input_data['City'] = City
input_data['Bank_Branch'] = Bank_Branch
input_data['Account_Type'] = account_map[Account_Type]
input_data['Transaction_Time'] = Transaction_Time
input_data['Transaction_Amount'] = Transaction_Amount
input_data['Merchant_ID'] = Merchant_ID
input_data['Transaction_Type'] = txn_type_map[Transaction_Type]
input_data['day'] = day
input_data['month'] = month
input_data['weekday'] = weekday


# Prediction
if st.button("Check Fraud"):
    X_scaled = scaler.transform(input_data)
    pred = model.predict(X_scaled)[0]

    if pred == -1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Transaction is Safe")
