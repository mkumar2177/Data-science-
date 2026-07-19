import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Logistic Regression Classifier",
    page_icon="🤖",
    layout="centered"
)

# Load Model
model = pickle.load(open("model.pkl","rb"))

st.title("🤖 Logistic Regression Prediction")

st.write("Enter the values below.")

st.divider()

x1 = st.number_input("Enter X1 Value")

x2 = st.number_input("Enter X2 Value")

if st.button("Predict"):

    prediction = model.predict([[x1,x2]])

    if prediction[0]==1:
        st.success("Prediction : Class 1")
    else:
        st.error("Prediction : Class 0")