import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Heart Disease Prediction",
   
    layout="wide"
)

# Load Model
model = pickle.load(open("artifacts.pkl","rb"))

# ---------------- CSS ----------------

st.markdown("""
<style>

.main{
background:#0E1117;
}

h1{
text-align:center;
color:#FF4B4B;
}

.stButton>button{
width:100%;
height:60px;
font-size:22px;
border-radius:15px;
background:#FF4B4B;
color:white;
font-weight:bold;
}

.stButton>button:hover{
background:#ff0000;
}

.result{
padding:20px;
border-radius:15px;
font-size:22px;
font-weight:bold;
text-align:center;
}

</style>
""",unsafe_allow_html=True)

st.title("Heart Disease Prediction System")

st.write("Predict whether a patient has heart disease using Machine Learning.")

left,right=st.columns(2)

with left:

    age=st.number_input("Age",1,100,45)

    sex=st.selectbox("Gender",
                     ["Female","Male"])

    cp=st.selectbox("Chest Pain Type",
                    [0,1,2,3])

    trestbps=st.number_input("Resting Blood Pressure",80,250,120)

    chol=st.number_input("Cholesterol",100,600,240)

    fbs=st.selectbox("Fasting Blood Sugar",
                     [0,1])

    restecg=st.selectbox("Rest ECG",
                         [0,1,2])

with right:

    thalach=st.number_input("Maximum Heart Rate",60,220,150)

    exang=st.selectbox("Exercise Induced Angina",
                       [0,1])

    oldpeak=st.number_input("Old Peak",0.0,10.0,1.5)

    slope=st.selectbox("Slope",[0,1,2])

    ca=st.selectbox("Major Vessels",[0,1,2,3,4])

    thal=st.selectbox("Thal",[0,1,2,3])

if sex=="Male":
    sex=1
else:
    sex=0

data=np.array([[age,sex,cp,trestbps,chol,fbs,
restecg,thalach,exang,oldpeak,slope,ca,thal]])

if st.button("❤️ Predict Heart Disease"):

    prediction=model.predict(data)

    if hasattr(model,"predict_proba"):
        prob=model.predict_proba(data)[0][1]
    else:
        prob=0.80 if prediction[0]==1 else 0.20

    st.divider()

    st.subheader("Prediction Probability")

    st.progress(int(prob*100))

    st.metric("Risk Probability",
              f"{prob*100:.2f}%")

    if prediction[0]==1:

        st.error("🔴 High Risk of Heart Disease")

        st.info("""
### Recommendation

✔ Consult a Cardiologist

✔ Maintain Healthy Diet

✔ Regular Exercise

✔ Avoid Smoking

✔ Regular Health Checkups
""")

    else:

        st.success("🟢 Low Risk of Heart Disease")

        st.balloons()

        st.info("""
### Recommendation

✔ Continue Healthy Lifestyle

✔ Exercise Daily

✔ Balanced Diet

✔ Routine Checkup
""")