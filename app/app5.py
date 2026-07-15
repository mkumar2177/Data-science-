import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("car_price_model.pkl","rb"))

st.set_page_config(
    page_title="Car Price Prediction",
    
    layout="centered"
)

st.title("Car Price Prediction")
st.write("Predict the selling price of a used car using Random Forest Regression.")

year = st.number_input("Manufacturing Year",1990,2025,2018)

km = st.number_input("Kilometers Driven",0,500000,50000)

fuel = st.selectbox(
    "Fuel Type",
    ["Diesel","Petrol","CNG","LPG","Electric"]
)

seller = st.selectbox(
    "Seller Type",
    ["Dealer","Individual","Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

mileage = st.number_input("Mileage (km/l)",5.0,40.0,20.0)

engine = st.number_input("Engine (CC)",700.0,5000.0,1200.0)

power = st.number_input("Max Power (BHP)",20.0,400.0,80.0)

seats = st.number_input("Seats",2,10,5)

# Replace these mappings with the values printed from your LabelEncoders
fuel_map = {
    "CNG":0,
    "Diesel":1,
    "Electric":2,
    "LPG":3,
    "Petrol":4
}

seller_map = {
    "Dealer":0,
    "Individual":1,
    "Trustmark Dealer":2
}

transmission_map = {
    "Automatic":0,
    "Manual":1
}

owner_map = {
    "First Owner":0,
    "Fourth & Above Owner":1,
    "Second Owner":2,
    "Test Drive Car":3,
    "Third Owner":4
}

if st.button("Predict Car Price"):

    features = np.array([[
        year,
        km,
        fuel_map[fuel],
        seller_map[seller],
        transmission_map[transmission],
        owner_map[owner],
        mileage,
        engine,
        power,
        seats
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted Selling Price: ₹ {prediction[0]:,.0f}")