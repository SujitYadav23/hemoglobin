import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("hemoglobin_rf_model.pkl")

st.title("Hemoglobin Level Predictor")

# User input fields
mid_s_mean = st.number_input("Mid S Mean")
mid_s_median = st.number_input("Mid S Median")
mid_a_mean = st.number_input("Mid A Mean")
mid_a_median = st.number_input("Mid A Median")
mid_g_median = st.number_input("Mid G Median")
mid_g_mean = st.number_input("Mid G Mean")
mid_b_median = st.number_input("Mid B Median")
mid_b_mean = st.number_input("Mid B Mean")

# Predict button
if st.button("Predict Hemoglobin"):
    features = np.array([[mid_s_mean, mid_s_median, mid_a_mean, mid_a_median, 
                          mid_g_median, mid_g_mean, mid_b_median, mid_b_mean]])
    prediction = model.predict(features)
    st.write(f"Predicted Hemoglobin Level: {prediction[0]:.2f}")
