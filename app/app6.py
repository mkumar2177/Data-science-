import streamlit as st
import pickle
import numpy as np

artifacts = pickle.load(open("artifacts.pkl","rb"))

model = artifacts["model"]

encoder = artifacts["encoder"]

st.set_page_config(

page_title="Heart Disease Prediction",



)

st.title("Heart Disease Prediction")

st.write("Decision Tree Classification")

age = st.number_input("Age",20,100,40)

sex = st.selectbox("Sex",["Female","Male"])

cp = st.number_input("Chest Pain Type",1,4,2)

bp = st.number_input("Blood Pressure",80,250,120)

chol = st.number_input("Cholesterol",100,700,200)

fbs = st.selectbox("FBS over 120",["No","Yes"])

ekg = st.number_input("EKG Results",0,2,1)

hr = st.number_input("Maximum Heart Rate",60,220,150)

angina = st.selectbox("Exercise Angina",["No","Yes"])

oldpeak = st.number_input("ST Depression",0.0,10.0,1.0)

slope = st.number_input("Slope of ST",1,3,2)

vessels = st.number_input("Number of Vessels",0,4,0)

thal = st.number_input("Thallium",3,7,3)

sex = 1 if sex=="Male" else 0

fbs = 1 if fbs=="Yes" else 0

angina = 1 if angina=="Yes" else 0

if st.button("Predict"):

    data=np.array([[

        age,

        sex,

        cp,

        bp,

        chol,

        fbs,

        ekg,

        hr,

        angina,

        oldpeak,

        slope,

        vessels,

        thal

    ]])

    prediction=model.predict(data)

    result=encoder.inverse_transform(prediction)[0]

    if result=="Presence":

        st.error("High Risk of Heart Disease")

    else:

        st.success("No Heart Disease Detected")