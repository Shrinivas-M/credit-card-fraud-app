# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a transaction CSV or enter details manually to predict fraud risk.")

# File upload
uploaded_file = st.file_uploader("📂 Upload CSV with transaction data", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Check required columns
    if 'Time' in input_df.columns and 'Amount' in input_df.columns:
        input_df['scaled_amount'] = scaler.transform(input_df['Amount'].values.reshape(-1, 1))
        input_df['scaled_time'] = scaler.transform(input_df['Time'].values.reshape(-1, 1))
        input_df.drop(['Time', 'Amount'], axis=1, inplace=True)

        input_df = input_df[['scaled_amount', 'scaled_time'] + [col for col in input_df.columns if col not in ['scaled_amount', 'scaled_time']]]

        # Predict
        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)[:, 1]

        input_df['Prediction'] = prediction
        input_df['Fraud Probability'] = prediction_prob

        st.success("✅ Predictions complete!")
        st.dataframe(input_df[['Prediction', 'Fraud Probability']])
    else:
        st.warning("⚠️ CSV must contain 'Time' and 'Amount' columns.")

else:
    st.info("📌 Upload a file to begin.")
