# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 00:16:54 2025

@author: Administrator
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# =============================
# Load trained model & scaler
# =============================
model = pickle.load(open("logistic_model.pkl", "rb"))   # apna trained model pickle karo
scaler = pickle.load(open("scaler.pkl", "rb"))         # scaler bhi pickle karo

# =============================
# Streamlit UI
# =============================
st.title("📊 Customer Churn Prediction Dashboard")

st.write("Enter customer details to predict churn probability:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# =============================
# Prepare input for model
# =============================
# Convert to DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone],
    'InternetService': [internet],
    'Contract': [contract],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# One-hot encoding like training
input_data = pd.get_dummies(input_data, drop_first=True)

# Align columns with training data
# (important step: ensure same feature order)
all_cols = pickle.load(open("columns.pkl", "rb"))  # save training columns in pickle
input_data = input_data.reindex(columns=all_cols, fill_value=0)

# Scale numeric values
input_scaled = scaler.transform(input_data)

# =============================
# Predict churn
# =============================
if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]  # probability of churn
    prob_percent = round(prob*100, 2)

    if prob > 0.5:
        st.error(f"⚠️ This customer is likely to churn ({prob_percent}% probability)")
    else:
        st.success(f"✅ This customer will not churn ({prob_percent}% probability)")
