# ==========================================
# Breast Cancer Prediction using KNN
# Professional Medical Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# Load Model
# ==========================================

with open("artifacts.pkl","rb") as file:
    artifacts = pickle.load(file)

model = artifacts["model"]
scaler = artifacts["scaler"]
best_k = artifacts["best_k"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("brca.csv")

# Drop ID Column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Encode Target
le = LabelEncoder()
df["y"] = le.fit_transform(df["y"])

# Features & Target
X = df.drop("y", axis=1)
y = df["y"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale Test Data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""

<style>

.stApp{

background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);

}

/* Main Title */

.main-title{

font-size:42px;

font-weight:bold;

text-align:center;

color:white;

padding:15px;

}

/* Cards */

.card{

background:white;

padding:18px;

border-radius:15px;

box-shadow:0px 8px 20px rgba(0,0,0,0.3);

text-align:center;

transition:0.3s;

}

.card:hover{

transform:scale(1.04);

}

.card-title{

font-size:18px;

color:gray;

}

.card-value{

font-size:30px;

font-weight:bold;

color:#1565C0;

}

/* Prediction Box */

.result{

padding:20px;

border-radius:15px;

font-size:28px;

font-weight:bold;

text-align:center;

color:white;

}

/* Button */

.stButton>button{

width:100%;

height:60px;

font-size:22px;

border-radius:12px;

background:#1565C0;

color:white;

font-weight:bold;

border:none;

}

.stButton>button:hover{

background:#00C853;

color:white;

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#102027;

}

section[data-testid="stSidebar"] *{

color:white;

}

/* Inputs */

input{

border-radius:8px !important;

}

</style>

""",unsafe_allow_html=True)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🩺 Medical Dashboard")

st.sidebar.markdown("---")

st.sidebar.success("Model : K-Nearest Neighbors")

st.sidebar.info(f"Best K Value : {best_k}")

st.sidebar.write("")

st.sidebar.subheader("📚 Project Workflow")

st.sidebar.write("✔ Data Cleaning")

st.sidebar.write("✔ Label Encoding")

st.sidebar.write("✔ Feature Scaling")

st.sidebar.write("✔ KNN Classification")

st.sidebar.write("✔ Prediction")

st.sidebar.write("✔ Streamlit Deployment")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Developer")

st.sidebar.write("Mohd Khalid Umar")

st.sidebar.write("B.Tech IT")

st.sidebar.write("ADGIPS | GGSIPU")

st.sidebar.markdown("---")

st.sidebar.info("Machine Learning Project")



st.markdown(
"""
<div class="main-title">

🩺 Breast Cancer Prediction System

</div>
""",
unsafe_allow_html=True
)

st.markdown(
"<center><h4 style='color:white;'>K-Nearest Neighbors (KNN) Classification</h4></center>",
unsafe_allow_html=True
)

st.write("")



col1,col2,col3,col4=st.columns(4)

with col1:

    st.markdown(f"""

    <div class="card">

    <div class="card-title">Algorithm</div>

    <div class="card-value">KNN</div>

    </div>

    """,unsafe_allow_html=True)

with col2:

    st.markdown(f"""

    <div class="card">

    <div class="card-title">Best K</div>

    <div class="card-value">{best_k}</div>

    </div>

    """,unsafe_allow_html=True)

with col3:

    st.markdown("""

    <div class="card">

    <div class="card-title">Target</div>

    <div class="card-value">Cancer</div>

    </div>

    """,unsafe_allow_html=True)

with col4:

    st.markdown("""

    <div class="card">

    <div class="card-title">Prediction</div>

    <div class="card-value">AI</div>

    </div>

    """,unsafe_allow_html=True)

st.write("")
st.write("")
# ==========================================
# Patient Information
# ==========================================

st.markdown(
"""
<h2 style='color:white;text-align:center;'>
📝 Patient Medical Information
</h2>
""",
unsafe_allow_html=True
)

st.write("")

col1, col2, col3 = st.columns(3)

# ==========================================
# Column 1
# ==========================================

with col1:

    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        value=14.12,
        format="%.3f",
        help="Mean radius of tumor"
    )

    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        value=19.28,
        format="%.3f"
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        value=91.96,
        format="%.3f"
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        value=654.90,
        format="%.3f"
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        min_value=0.0,
        value=0.096,
        format="%.5f"
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        min_value=0.0,
        value=0.104,
        format="%.5f"
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        value=0.089,
        format="%.5f"
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        value=0.048,
        format="%.5f"
    )

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        min_value=0.0,
        value=0.181,
        format="%.5f"
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        min_value=0.0,
        value=0.062,
        format="%.5f"
    )

# ==========================================
# Column 2
# ==========================================

with col2:

    radius_se = st.number_input(
        "Radius SE",
        min_value=0.0,
        value=0.405,
        format="%.3f"
    )

    texture_se = st.number_input(
        "Texture SE",
        min_value=0.0,
        value=1.216,
        format="%.3f"
    )

    perimeter_se = st.number_input(
        "Perimeter SE",
        min_value=0.0,
        value=2.866,
        format="%.3f"
    )

    area_se = st.number_input(
        "Area SE",
        min_value=0.0,
        value=40.34,
        format="%.3f"
    )

    smoothness_se = st.number_input(
        "Smoothness SE",
        min_value=0.0,
        value=0.007,
        format="%.5f"
    )

    compactness_se = st.number_input(
        "Compactness SE",
        min_value=0.0,
        value=0.025,
        format="%.5f"
    )

    concavity_se = st.number_input(
        "Concavity SE",
        min_value=0.0,
        value=0.031,
        format="%.5f"
    )

    concave_points_se = st.number_input(
        "Concave Points SE",
        min_value=0.0,
        value=0.012,
        format="%.5f"
    )

    symmetry_se = st.number_input(
        "Symmetry SE",
        min_value=0.0,
        value=0.020,
        format="%.5f"
    )

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE",
        min_value=0.0,
        value=0.003,
        format="%.5f"
    )

# ==========================================
# Column 3
# ==========================================

with col3:

    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        value=16.26,
        format="%.3f"
    )

    texture_worst = st.number_input(
        "Texture Worst",
        min_value=0.0,
        value=25.68,
        format="%.3f"
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        value=107.26,
        format="%.3f"
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        value=880.58,
        format="%.3f"
    )

    smoothness_worst = st.number_input(
        "Smoothness Worst",
        min_value=0.0,
        value=0.132,
        format="%.5f"
    )

    compactness_worst = st.number_input(
        "Compactness Worst",
        min_value=0.0,
        value=0.254,
        format="%.5f"
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        value=0.272,
        format="%.5f"
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        value=0.115,
        format="%.5f"
    )

    symmetry_worst = st.number_input(
        "Symmetry Worst",
        min_value=0.0,
        value=0.290,
        format="%.5f"
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst",
        min_value=0.0,
        value=0.084,
        format="%.5f"
    )

st.write("")

predict_button = st.button("🔍 Predict Breast Cancer")
# ==========================================
# Prediction Logic
# ==========================================

if predict_button:

    # Create Input Array
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,
        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Prediction Probability
    probability = model.predict_proba(input_scaled)

    confidence = np.max(probability) * 100

    benign_probability = probability[0][0] * 100
    malignant_probability = probability[0][1] * 100

    st.write("")
    st.write("---")

    st.subheader("🩺 Prediction Result")

    # ==============================
    # Result Card
    # ==============================

    if prediction[0] == 0:

        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#00C853,#43A047);
        padding:25px;
        border-radius:18px;
        color:white;
        text-align:center;
        box-shadow:0px 8px 25px rgba(0,0,0,0.35);
        ">
        <h1>🟢 Benign Tumor</h1>
        <h3>No Breast Cancer Detected</h3>
        <h2>Confidence : {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#D50000,#FF1744);
        padding:25px;
        border-radius:18px;
        color:white;
        text-align:center;
        box-shadow:0px 8px 25px rgba(0,0,0,0.35);
        ">
        <h1>🔴 Malignant Tumor</h1>
        <h3>Breast Cancer Detected</h3>
        <h2>Confidence : {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==================================
    # Confidence Section
    # ==================================

    st.subheader("📊 Prediction Confidence")

    st.progress(int(confidence))

    st.success(f"Overall Confidence : {confidence:.2f}%")

    st.write("")

    # ==================================
    # Probability Cards
    # ==================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div style="
        background:#E8F5E9;
        padding:20px;
        border-radius:15px;
        text-align:center;
        ">
        <h2 style="color:#2E7D32;">🟢 Benign</h2>
        <h1 style="color:#1B5E20;">
        {benign_probability:.2f}%
        </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div style="
        background:#FFEBEE;
        padding:20px;
        border-radius:15px;
        text-align:center;
        ">
        <h2 style="color:#C62828;">🔴 Malignant</h2>
        <h1 style="color:#B71C1C;">
        {malignant_probability:.2f}%
        </h1>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==================================
    # Quick Summary
    # ==================================

    st.subheader("📋 Prediction Summary")

    summary = {
        "Algorithm": "K-Nearest Neighbors (KNN)",
        "Best K Value": best_k,
        "Prediction":
            "Benign" if prediction[0] == 0 else "Malignant",
        "Confidence":
            f"{confidence:.2f}%"
    }

    st.table(pd.DataFrame(summary.items(),
                          columns=["Parameter", "Value"]))

    # ==========================================
    # Probability Visualization
    # ==========================================

    st.write("")
    st.subheader("📊 Prediction Probability")

    probability_df = pd.DataFrame({

        "Class":["Benign","Malignant"],

        "Probability":[
            benign_probability,
            malignant_probability
        ]

    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

    st.write("")

    # ==========================================
    # Patient Input Summary
    # ==========================================

    st.subheader("📝 Patient Input Summary")

    patient_data = pd.DataFrame({

        "Feature":[

            "Radius Mean",
            "Texture Mean",
            "Perimeter Mean",
            "Area Mean",
            "Smoothness Mean",
            "Compactness Mean",
            "Concavity Mean",
            "Concave Points Mean",
            "Symmetry Mean",
            "Fractal Dimension Mean",

            "Radius SE",
            "Texture SE",
            "Perimeter SE",
            "Area SE",
            "Smoothness SE",
            "Compactness SE",
            "Concavity SE",
            "Concave Points SE",
            "Symmetry SE",
            "Fractal Dimension SE",

            "Radius Worst",
            "Texture Worst",
            "Perimeter Worst",
            "Area Worst",
            "Smoothness Worst",
            "Compactness Worst",
            "Concavity Worst",
            "Concave Points Worst",
            "Symmetry Worst",
            "Fractal Dimension Worst"

        ],

        "Value":[

            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,

            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,

            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst

        ]

    })

    st.dataframe(patient_data,
                 use_container_width=True)

    st.write("")

    # ==========================================
    # Download Report
    # ==========================================

    report = pd.DataFrame({

        "Prediction":[
            "Benign"
            if prediction[0]==0
            else
            "Malignant"
        ],

        "Confidence (%)":[
            round(confidence,2)
        ],

        "Benign Probability (%)":[
            round(benign_probability,2)
        ],

        "Malignant Probability (%)":[
            round(malignant_probability,2)
        ],

        "Best K":[best_k]

    })

    csv = report.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv,

        file_name="Breast_Cancer_Report.csv",

        mime="text/csv"

    )

    st.write("")

    # ==========================================
    # Recommendation Box
    # ==========================================

    st.subheader("💡 AI Recommendation")

    if prediction[0]==0:

        st.success("""

✅ Prediction indicates a **Benign Tumor**.

• Continue regular medical check-ups.

• Maintain a healthy lifestyle.

• Follow your doctor's advice.

• This prediction is generated by a machine learning model and should not replace professional medical diagnosis.

""")

        st.balloons()

    else:

        st.error("""

⚠️ Prediction indicates a **Malignant Tumor**.

• Please consult an oncologist or qualified healthcare professional promptly.

• Do not rely solely on this prediction.

• Further diagnostic tests (such as imaging or biopsy) may be necessary.

• This model is intended only as an educational machine learning project.

""")

    st.write("")

    # ==========================================
    # Model Information
    # ==========================================

    st.subheader("🤖 Model Information")

    info1,info2,info3=st.columns(3)

    with info1:
        st.metric(
            "Algorithm",
            "KNN"
        )

    with info2:
        st.metric(
            "Best K",
            best_k
        )

    with info3:
        st.metric(
            "Classes",
            "2"
        )

    st.write("")

    # ==========================================
    # Footer
    # ==========================================

    st.markdown("---")

    st.markdown("""

<div style="text-align:center;color:white;">

<h3>
🩺 Breast Cancer Prediction System
</h3>

<p>
Developed using
<b>Python • Scikit-Learn • KNN • Streamlit</b>
</p>

<p>
Created by <b>Mohd Khalid Umar</b>
</p>

</div>

""",
unsafe_allow_html=True)

# ==========================================
# Dataset Information Section
# ==========================================

st.write("")
st.markdown("---")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📊 Dataset Information
</h2>
""",
unsafe_allow_html=True
)

# Load Dataset
dataset = pd.read_csv("brca.csv")

# Remove ID Column
if "Unnamed: 0" in dataset.columns:
    dataset.drop("Unnamed: 0", axis=1, inplace=True)

rows = dataset.shape[0]
columns = dataset.shape[1]
features = columns - 1

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    Total Records
    </div>

    <div class="card-value">
    {rows}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    Features
    </div>

    <div class="card-value">
    {features}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">

    <div class="card-title">
    Classes
    </div>

    <div class="card-value">
    2
    </div>

    </div>
    """, unsafe_allow_html=True)

st.write("")
# ==========================================
# Model Performance
# ==========================================

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📈 Model Performance
</h2>
""",
unsafe_allow_html=True
)

accuracy_percent = round(accuracy * 100, 2)
precision_percent = round(precision * 100, 2)
recall_percent = round(recall * 100, 2)
f1_percent = round(f1 * 100, 2)

a1, a2, a3, a4 = st.columns(4)

with a1:
    st.metric(
        "Accuracy",
        f"{accuracy_percent}%"
    )

with a2:
    st.metric(
        "Precision",
        f"{precision_percent}%"
    )

with a3:
    st.metric(
        "Recall",
        f"{recall_percent}%"
    )

with a4:
    st.metric(
        "F1 Score",
        f"{f1_percent}%"
    )

st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📉 Confusion Matrix
</h2>
""",
unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📄 Classification Report
</h2>
""",
unsafe_allow_html=True
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)
st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📋 Dataset Preview
</h2>
""",
unsafe_allow_html=True
)

st.dataframe(
    dataset.head(10),
    use_container_width=True
)
csv_dataset = dataset.to_csv(index=False)

st.download_button(

    label="📥 Download Dataset",

    data=csv_dataset,

    file_name="Breast_Cancer_Dataset.csv",

    mime="text/csv"

)
st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
📊 Target Distribution
</h2>
""",
unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(5,5))

dataset["y"].value_counts().plot(

    kind="pie",

    autopct="%1.1f%%",

    colors=["lightgreen","red"],

    ax=ax

)

ax.set_ylabel("")

st.pyplot(fig)


st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
🔥 Correlation Heatmap
</h2>
""",
unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(

    dataset.corr(numeric_only=True),

    cmap="coolwarm",

    ax=ax

)

st.pyplot(fig)

st.write("")

st.markdown(
"""
<h2 style='text-align:center;color:white;'>
🔥 Correlation Heatmap
</h2>
""",
unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(

    dataset.corr(numeric_only=True),

    cmap="coolwarm",

    ax=ax

)

st.pyplot(fig)
st.write("")

st.success("✅ Machine Learning Model Loaded Successfully")

st.info("🩺 This project is built for educational purposes only.")

st.caption(
"Predictions should always be verified by qualified healthcare professionals."
)
st.sidebar.markdown("---")

st.sidebar.success("✅ Project Completed")

st.sidebar.write("")

st.sidebar.caption("Version : 2.0")

st.sidebar.caption("Developer")

st.sidebar.caption("Mohd Khalid Umar")
with st.expander("💡 Tips for Better Prediction"):

    st.write("""

✔ Enter realistic medical values.

✔ Check all values before prediction.

✔ This application is intended for educational use.

✔ Do not use this prediction as a medical diagnosis.

""")
st.markdown("""

<style>

div[data-testid="metric-container"]{

background:white;

padding:15px;

border-radius:15px;

box-shadow:0px 5px 15px rgba(0,0,0,0.25);

}

div[data-testid="metric-container"]:hover{

transform:scale(1.03);

transition:0.3s;

}

</style>

""",unsafe_allow_html=True)
    


