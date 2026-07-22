import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("decision_tree_model.pkl",'rb'))

st.set_page_config(
    page_title="Salary Prediction",
    
    layout="centered"
)

st.title("💼 Salary Prediction using Decision Tree")
st.write("Enter the employee details below and click **Predict Salary**.")

st.markdown("---")



experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)


# Prediction Button
if st.button("Predict Salary"):





    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

    st.balloons()

st.markdown("---")
st.caption("Built with using Streamlit and Decision Tree")

    