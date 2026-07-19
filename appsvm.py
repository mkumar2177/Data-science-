import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model
# ----------------------------

with open("artifacts.pkl","rb") as file:
    artifacts = pickle.load(file)

model = artifacts["model"]
scaler = artifacts["scaler"]

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#1d2b64,#f8cdda);
}

h1,h2,h3{
color:white;
text-align:center;
}

div[data-testid="stSidebar"]{
background:#141E30;
}

div[data-testid="stSidebar"] *{
color:white;
}

.stButton>button{

background:#ff1744;

color:white;

font-size:22px;

border-radius:15px;

height:60px;

width:100%;

font-weight:bold;

}

.stButton>button:hover{

background:#00c853;

color:white;

}

.block{

background:white;

padding:20px;

border-radius:20px;

box-shadow:0px 0px 20px gray;

}

.result{

padding:20px;

border-radius:15px;

font-size:30px;

font-weight:bold;

text-align:center;

}

</style>
""",unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png",width=120)

st.sidebar.title("❤️ Heart Disease Prediction")

st.sidebar.write("---")

st.sidebar.success("Support Vector Machine")

st.sidebar.info("Machine Learning Project")

st.sidebar.write("---")

st.sidebar.write("Developer")

st.sidebar.write("Mohd Khalid Umar")

st.sidebar.write("Data Science Project")

# ----------------------------
# Title
# ----------------------------

st.markdown("<h1>❤️ Heart Disease Prediction System ❤️</h1>",unsafe_allow_html=True)

st.write("")

# ----------------------------
# Input
# ----------------------------

col1,col2,col3=st.columns(3)

with col1:

    age=st.number_input("Age",20,100)

    sex=st.selectbox("Gender",[0,1],format_func=lambda x:"Female" if x==0 else "Male")

    chest=st.selectbox("Chest Pain Type",[1,2,3,4])

    bp=st.number_input("Blood Pressure",80,250)

    chol=st.number_input("Cholesterol",100,600)

with col2:

    fbs=st.selectbox("FBS over 120",[0,1])

    ekg=st.selectbox("EKG Results",[0,1,2])

    hr=st.number_input("Maximum Heart Rate",60,220)

    angina=st.selectbox("Exercise Angina",[0,1])

    depression=st.number_input("ST Depression",0.0,10.0)

with col3:

    slope=st.selectbox("Slope of ST",[1,2,3])

    vessels=st.selectbox("Number of Vessels",[0,1,2,3])

    thallium=st.selectbox("Thallium",[3,6,7])

st.write("")

# ----------------------------
# Prediction
# ----------------------------

if st.button("❤️ Predict Heart Disease"):

    sample=np.array([[

        age,

        sex,

        chest,

        bp,

        chol,

        fbs,

        ekg,

        hr,

        angina,

        depression,

        slope,

        vessels,

        thallium

    ]])

    sample=scaler.transform(sample)

    prediction=model.predict(sample)[0]

    probability=model.predict_proba(sample)

    confidence=np.max(probability)*100

    st.write("")

    if prediction==1:

        st.markdown(f"""
        <div class="result" style="background:#ff5252;color:white;">
        ❤️ Heart Disease Detected
        </div>
        """,unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result" style="background:#00c853;color:white;">
        💚 No Heart Disease
        </div>
        """,unsafe_allow_html=True)

    st.write("")

    st.subheader("Prediction Confidence")

    st.progress(int(confidence))

    st.success(f"Confidence : {confidence:.2f}%")

    st.write("")

    st.subheader("Prediction Probability")

    st.write(probability)

    st.balloons()

# ----------------------------
# Footer
# ----------------------------

st.write("---")

st.markdown(
"""
<center>

<h3 style="color:white;">
Support Vector Machine (SVM)

Heart Disease Prediction

Developed using Streamlit ❤️
</h3>

</center>
""",
unsafe_allow_html=True
)