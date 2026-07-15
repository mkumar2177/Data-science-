import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("house_price_model.pkl", "rb"))

st.set_page_config(page_title="House Price Prediction")

st.title("House Price Prediction System")

st.write("Enter the details below to predict the house price.")

area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

floors = st.number_input("Floors", min_value=1, max_value=5, value=2)

yearbuilt = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015)

location = st.selectbox("Location", ["Downtown", "Suburban", "Urban", "Rural"])

condition = st.selectbox("Condition", ["Fair", "Good", "Excellent"])

garage = st.selectbox("Garage", ["No", "Yes"])

# Encoding (same as training)
location_map = {
    "Downtown":0,
    "Rural":1,
    "Suburban":2,
    "Urban":3
}

condition_map = {
    "Excellent":0,
    "Fair":1,
    "Good":2
}

garage_map = {
    "No":0,
    "Yes":1
}

if st.button("Predict Price"):

    data = np.array([[

        area,

        bedrooms,

        bathrooms,

        floors,

        yearbuilt,

        location_map[location],

        condition_map[condition],

        garage_map[garage]

    ]])

    prediction = model.predict(data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.0f}")