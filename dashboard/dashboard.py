import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

st.title("💳 Credit Card Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    def fetch_prediction(row):
        try:
            response = requests.post("http://localhost:8000/predict", json=row.to_dict(), timeout=10)
            response.raise_for_status()
            return response.json().get("prediction", "N/A")
        except Exception as e:
            return f"Error: {e}"

    if st.button("🔍 Predict Fraud"):
        st.info("⏳ Sending data to FastAPI server for predictions...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            predictions = list(executor.map(fetch_prediction, [row for _, row in df.iterrows()]))

        df['Prediction'] = predictions

        st.subheader("✅ Prediction Results")
        st.write(df)

        error_count = sum(1 for p in predictions if isinstance(p, str) and p.startswith("Error"))
        if error_count:
            st.warning(f"⚠️ {error_count} rows failed to get predictions.")
