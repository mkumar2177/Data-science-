import streamlit as st
import pickle
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

model = load_model("fraud_model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

le_type = encoders["TransactionType"]
le_location = encoders["Location"]

st.title("Credit Card Fraud Detection")
st.caption("Analyze transactions using an Artificial Neural Network")

left, right = st.columns(2)

with left:
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    merchant = st.number_input("Merchant ID", min_value=1, step=1)
    transaction = st.selectbox("Transaction Type", le_type.classes_)
    location = st.selectbox("Location", le_location.classes_)

with right:
    current_year = datetime.now().year

    year = st.selectbox("Year",
                        [current_year-2, current_year-1,
                         current_year, current_year+1])

    month = st.selectbox("Month", range(1,13))
    day = st.selectbox("Day", range(1,32))
    hour = st.selectbox("Hour", range(24))

predict = st.button("Analyze Transaction", use_container_width=True)

if predict:

    with st.spinner("Analyzing transaction..."):
        time.sleep(1)

        transaction = le_type.transform([transaction])[0]
        location = le_location.transform([location])[0]

        data = np.array([[amount,
                          merchant,
                          transaction,
                          location,
                          year,
                          month,
                          day,
                          hour]])

        data = scaler.transform(data)

        probability = model.predict(data, verbose=0)[0][0]
        prediction = probability >= 0.5
        confidence = probability if prediction else 1 - probability

    if prediction:
        st.error("Fraud Detected")
    else:
        st.success("Legitimate Transaction")

    st.metric("Confidence", f"{confidence*100:.2f}%")