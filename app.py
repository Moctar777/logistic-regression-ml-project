import streamlit as st
import joblib
import numpy as np
import os

# Load model with proper path handling
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model not found at {model_path}. Please train the model first.")
    st.stop()

st.title("Student Performance Predictor")

hours = st.slider("Study Hours", 0, 10)
sleep = st.slider("Sleep Hours", 0, 10)
attendance = st.slider("Attendance (%)", 0, 100)

if st.button("Predict"):
    features = np.array([[hours, sleep, attendance]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"Pass ✅ Probability: {prob:.2f}")
    else:
        st.error(f"Fail ❌ Probability: {prob:.2f}")
