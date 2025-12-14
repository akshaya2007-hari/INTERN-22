import streamlit as st
import pickle
import numpy as np
from sklearn.exceptions import NotFittedError

st.title("Prediction App")

with open("fuel(1).pkl", "rb") as f:
    model = pickle.load(f)

x = st.number_input("Enter value")

if st.button("Predict"):
    try:
        input_data = np.array([[x]])
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
    except NotFittedError:
        st.error("❌ Model is not trained. Please retrain the model and upload a fitted .pkl file.")







