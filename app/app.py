import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('salary_model.pkl', 'rb'))

st.title("Salary Predictor")
experience = st.number_input("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")

    