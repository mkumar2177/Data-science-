import streamlit as st
import pickle
import numpy as np

artifacts = pickle.load(open("artifacts.pkl","rb"))

model = artifacts["model"]

encoders = artifacts["encoders"]

st.set_page_config(
    page_title="Loan Eligibility Prediction",
   
)

st.title("Loan Eligibility Prediction")
st.write("Random Forest Classification")

gender = st.selectbox(
    "Gender",
    list(encoders["Gender"].classes_)
)

married = st.selectbox(
    "Married",
    list(encoders["Married"].classes_)
)

dependents = st.selectbox(
    "Dependents",
    list(encoders["Dependents"].classes_)
)

education = st.selectbox(
    "Education",
    list(encoders["Education"].classes_)
)

self_emp = st.selectbox(
    "Self Employed",
    list(encoders["Self_Employed"].classes_)
)

income = st.number_input(
    "Applicant Income",
    0,
    100000,
    5000
)

co_income = st.number_input(
    "Coapplicant Income",
    0,
    100000,
    1500
)

loan_amount = st.number_input(
    "Loan Amount",
    1,
    1000,
    150
)

loan_term = st.number_input(
    "Loan Term",
    12,
    480,
    360
)

credit = st.selectbox(
    "Credit History",
    [0,1]
)

property_area = st.selectbox(
    "Property Area",
    list(encoders["Property_Area"].classes_)
)

gender = encoders["Gender"].transform([gender])[0]
married = encoders["Married"].transform([married])[0]
dependents = encoders["Dependents"].transform([dependents])[0]
education = encoders["Education"].transform([education])[0]
self_emp = encoders["Self_Employed"].transform([self_emp])[0]
property_area = encoders["Property_Area"].transform([property_area])[0]

if st.button("Predict Loan Status"):

    data = np.array([[

        gender,

        married,

        dependents,

        education,

        self_emp,

        income,

        co_income,

        loan_amount,

        loan_term,

        credit,

        property_area

    ]])

    prediction = model.predict(data)

    result = encoders["Loan_Status"].inverse_transform(prediction)[0]

    if result == "Y":

        st.success(" Loan Approved")

    else:

        st.error(" Loan Rejected")