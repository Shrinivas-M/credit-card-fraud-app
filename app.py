import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# Sidebar details
with st.sidebar:
    st.header("ℹ️ App Info")
    st.markdown("""
    **Project**: Credit Card Fraud Detection  
    **Model**: Random Forest / Logistic Regression / XGBoost  
    **Author**: Shrinivas M  
    **Version**: 1.0  
    **Data**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    """)
    st.markdown("📁 Use the main panel to upload data or test manually.")

st.markdown("Upload a transaction CSV file or manually enter a single transaction below to predict fraud.")

# === CSV Upload ===
uploaded_file = st.file_uploader("📂 Upload CSV with transaction data", type=["csv"])

def preprocess_input(df):
    if 'Time' in df.columns and 'Amount' in df.columns:
        df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
    if 'Class' in df.columns:
        df.drop('Class', axis=1, inplace=True)
    return df[['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['scaled_amount', 'scaled_time']]]

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        df_processed = preprocess_input(df.copy())
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]

        df['Prediction'] = predictions
        df['Fraud Probability'] = probabilities

        st.success("✅ Predictions complete!")
        st.dataframe(df[['Prediction', 'Fraud Probability']].head(10))

        # Inside your Streamlit app after predictions or loading data
        if 'Class' in df.columns:
           class_counts = df['Class'].value_counts()

           labels = ['Not Fraud', 'Fraud']
           values = [class_counts.get(0, 0), class_counts.get(1, 0)]  # Ensure both classes exist
           # Charts
           st.subheader("📊 Fraud Distribution")
           # Get prediction counts and ensure both classes are present
           pred_counts = df['Prediction'].value_counts().to_dict()
           values = [pred_counts.get(0, 0), pred_counts.get(1, 0)]  # Always returns two values
           labels = ['Non-Fraud', 'Fraud']
           fig1, ax1 = plt.subplots()
           ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
           ax1.axis('equal')
           st.pyplot(fig1)
        else:
           st.warning("⚠️ 'Class' column not found for distribution plot.")
        st.subheader("📉 Prediction Probability Histogram")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Fraud Probability'], bins=30, kde=True, color='purple', ax=ax2)
        ax2.set_title("Fraud Probability Distribution")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")

# === Manual Single Input ===
st.markdown("---")
st.subheader("📝 Manual Input for Single Transaction")

with st.form("manual_input"):
    time_val = st.number_input("Transaction Time", min_value=0.0, step=1.0)
    amount_val = st.number_input("Transaction Amount", min_value=0.01, step=0.01)
    v_features = [st.number_input(f"V{i}", value=0.0, step=0.01) for i in range(1, 29)]
    submitted = st.form_submit_button("Predict")

    if submitted:
        scaled_amount = scaler.transform(np.array([[amount_val]]))[0][0]
        scaled_time = scaler.transform(np.array([[time_val]]))[0][0]
        input_array = np.array([[scaled_amount, scaled_time] + v_features])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        label = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
        color = "red" if prediction == 1 else "green"
        st.markdown(f"### 🎯 Prediction: **:{color}[{label}]**")
        st.markdown(f"**Probability of Fraud:** `{probability:.4f}`")