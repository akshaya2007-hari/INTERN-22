import streamlit as st
import pickle
import numpy as np

# Load model
with open("fuel11.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Fuel Prediction App")

# ONE input only
x_value = st.number_input("Enter input value", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[x_value]])   # ONLY 1 FEATURE
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]:.2f}")
else:
    st.info("Enter a value and click Predict")
