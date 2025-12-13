import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Fuel Prediction App")

# Load trained model
with open("fuell.pkl", "rb") as f:
    model = pickle.load(f)

st.title("⛽ Fuel Prediction App")

x_value = st.number_input(
    "Enter input value",
    min_value=0.0,
    step=0.1
)

if st.button("Predict"):
    input_data = np.array([[x_value]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]:.2f}")




