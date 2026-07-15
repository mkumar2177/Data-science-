import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Logistic Regression Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #131722 100%); }

    .big-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #06b6d4;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pred-box-1 {
        background: linear-gradient(135deg, #059669, #10b981);
        border-radius: 16px; padding: 28px; text-align: center;
        color: white; font-size: 1.6rem; font-weight: 700;
    }
    .pred-box-0 {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        border-radius: 16px; padding: 28px; text-align: center;
        color: white; font-size: 1.6rem; font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background: #131722;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stMetricValue"] { color: #06b6d4; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# DATA / MODEL LOADING
# ---------------------------------------------------------------
DATA_PATH = "data.csv"
MODEL_PATH = "model.pkl"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def train_model(df, test_size, random_state):
    X = df[["X_1", "X_2"]]
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test

if "df" not in st.session_state:
    if os.path.exists(DATA_PATH):
        st.session_state.df = load_data(DATA_PATH)
    else:
        st.session_state.df = None

if "model" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        st.session_state.model = pickle.load(open(MODEL_PATH, "rb"))
        st.session_state.model_source = "Loaded from model.pkl"
    else:
        st.session_state.model = None
        st.session_state.model_source = None

# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Explore Data", "⚙️ Train Model", "🔮 Predict", "📈 Model Performance", "📂 Batch Predict"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data Source")
uploaded_csv = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])
if uploaded_csv is not None:
    st.session_state.df = pd.read_csv(uploaded_csv)
    st.session_state.df.columns = [c.strip() for c in st.session_state.df.columns]
    st.sidebar.success(f"Loaded {uploaded_csv.name}")

st.sidebar.markdown("---")
st.sidebar.info("Made with Streamlit · Logistic Regression")

# ---------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<p class="big-title">Logistic Regression Studio</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Train, explore, evaluate and predict — all in one place.</p>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    df = st.session_state.df
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df) if df is not None else 0}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        status = "✅ Ready" if st.session_state.model is not None else "❌ Not trained"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem">{status}</div><div class="metric-label">Model Status</div></div>', unsafe_allow_html=True)
    with col4:
        classes = df["y"].nunique() if df is not None else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{classes}</div><div class="metric-label">Classes</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### 👋 Kaise use kare")
    st.markdown("""
    1. **📊 Explore Data** — dataset dekho, distributions aur scatter plot
    2. **⚙️ Train Model** — apne settings ke saath model train karo (ya pehle se trained model.pkl use ho jayega)
    3. **🔮 Predict** — sliders se X_1, X_2 daal ke live prediction lo
    4. **📈 Model Performance** — accuracy, confusion matrix, ROC curve dekho
    5. **📂 Batch Predict** — pura CSV upload karke ek saath predictions le lo
    """)

    if df is not None:
        st.markdown("### 🔍 Quick Preview")
        st.dataframe(df.head(), use_container_width=True)

# ---------------------------------------------------------------
# EXPLORE DATA PAGE
# ---------------------------------------------------------------
elif page == "📊 Explore Data":
    st.markdown('<p class="big-title">Explore Data</p>', unsafe_allow_html=True)
    df = st.session_state.df

    if df is None:
        st.warning("Pehle data.csv upload karo sidebar se.")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Scatter Plot"])

        with tab1:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(df, use_container_width=True, height=400)
            with c2:
                st.markdown("#### Summary")
                st.write(df.describe())
                st.markdown("#### Missing values")
                st.write(df.isnull().sum())

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.histplot(df["X_1"], kde=True, ax=ax, color="#06b6d4")
                ax.set_title("Distribution of X_1")
                st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.histplot(df["X_2"], kde=True, ax=ax, color="#7c3aed")
                ax.set_title("Distribution of X_2")
                st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="y", data=df, ax=ax, palette=["#ef4444", "#10b981"])
            ax.set_title("Class Balance (y)")
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(7, 6))
            scatter = ax.scatter(df["X_1"], df["X_2"], c=df["y"], cmap="coolwarm", edgecolor="k", alpha=0.8)
            ax.set_xlabel("X_1")
            ax.set_ylabel("X_2")
            ax.set_title("X_1 vs X_2 colored by class y")
            legend1 = ax.legend(*scatter.legend_elements(), title="Class")
            ax.add_artist(legend1)
            st.pyplot(fig)

# ---------------------------------------------------------------
# TRAIN MODEL PAGE
# ---------------------------------------------------------------
elif page == "⚙️ Train Model":
    st.markdown('<p class="big-title">Train Model</p>', unsafe_allow_html=True)
    df = st.session_state.df

    if df is None:
        st.warning("Pehle data.csv upload karo sidebar se.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05)
        with c2:
            random_state = st.number_input("Random state", value=42, step=1)
        with c3:
            st.write("")
            st.write("")
            train_btn = st.button("🚀 Train Model", use_container_width=True)

        if train_btn:
            with st.spinner("Training..."):
                clf, X_train, X_test, y_train, y_test = train_model(df, test_size, int(random_state))
                st.session_state.model = clf
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model_source = "Freshly trained this session"
                pickle.dump(clf, open(MODEL_PATH, "wb"))

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained! Test accuracy: **{acc:.2%}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            m3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            m4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")

            st.info("Model `model.pkl` me save ho gaya — Predict aur Model Performance pages me use hoga.")
        elif st.session_state.model is not None:
            st.info(f"Model status: {st.session_state.model_source}. Naye settings ke saath retrain karne ke liye button dabao.")

# ---------------------------------------------------------------
# PREDICT PAGE
# ---------------------------------------------------------------
elif page == "🔮 Predict":
    st.markdown('<p class="big-title">Live Prediction</p>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Pehle model train karo ⚙️ Train Model page se, ya model.pkl uplaod/present hona chahiye.")
    else:
        clf = st.session_state.model
        df = st.session_state.df

        x1_min, x1_max = (-1.0, 1.0) if df is None else (float(df["X_1"].min()), float(df["X_1"].max()))
        x2_min, x2_max = (-1.0, 1.0) if df is None else (float(df["X_2"].min()), float(df["X_2"].max()))

        col_input, col_result = st.columns([1, 1])

        with col_input:
            st.markdown("#### Input Features")
            x1 = st.slider("X_1", x1_min, x1_max, 0.0, 0.01)
            x2 = st.slider("X_2", x2_min, x2_max, 0.0, 0.01)
            st.caption("Ya direct value type kar sakte ho neeche")
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.number_input("X_1 value", value=float(x1), format="%.6f")
            with c2:
                x2 = st.number_input("X_2 value", value=float(x2), format="%.6f")

            predict_btn = st.button("🔮 Predict", use_container_width=True)

        with col_result:
            st.markdown("#### Result")
            if predict_btn or True:
                input_df = pd.DataFrame([[x1, x2]], columns=["X_1", "X_2"])
                pred = clf.predict(input_df)[0]
                proba = clf.predict_proba(input_df)[0]

                if pred == 1:
                    st.markdown(f'<div class="pred-box-1">Class 1 ✅<br><span style="font-size:1rem">Confidence: {proba[1]:.1%}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="pred-box-0">Class 0 ❌<br><span style="font-size:1rem">Confidence: {proba[0]:.1%}</span></div>', unsafe_allow_html=True)

                st.write("")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(["Class 0", "Class 1"], proba, color=["#ef4444", "#10b981"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Class Probabilities")
                st.pyplot(fig)

        # Decision boundary visualization
        if df is not None:
            st.markdown("#### Decision Boundary")
            xx, yy = np.meshgrid(
                np.linspace(x1_min - 0.2, x1_max + 0.2, 200),
                np.linspace(x2_min - 0.2, x2_max + 0.2, 200)
            )
            grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["X_1", "X_2"])
            Z = clf.predict(grid).reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
            ax.scatter(df["X_1"], df["X_2"], c=df["y"], cmap="coolwarm", edgecolor="k", s=30)
            ax.scatter([x1], [x2], c="yellow", marker="*", s=400, edgecolor="black", label="Your input")
            ax.set_xlabel("X_1")
            ax.set_ylabel("X_2")
            ax.legend()
            st.pyplot(fig)

# ---------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------------------
elif page == "📈 Model Performance":
    st.markdown('<p class="big-title">Model Performance</p>', unsafe_allow_html=True)

    if st.session_state.model is None or st.session_state.df is None:
        st.warning("Pehle model train karo aur data load karo.")
    else:
        clf = st.session_state.model
        df = st.session_state.df

        if "X_test" not in st.session_state:
            X = df[["X_1", "X_2"]]
            y = df["y"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        m4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")

        st.write("")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
            st.pyplot(fig)

        with c2:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="#06b6d4", label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.markdown("#### Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": ["X_1", "X_2"],
            "Coefficient": clf.coef_[0]
        })
        st.dataframe(coef_df, use_container_width=True)
        st.caption(f"Intercept: {clf.intercept_[0]:.4f}")

# ---------------------------------------------------------------
# BATCH PREDICT PAGE
# ---------------------------------------------------------------
elif page == "📂 Batch Predict":
    st.markdown('<p class="big-title">Batch Prediction</p>', unsafe_allow_html=True)
    st.write("Ek CSV upload karo jisme **X_1** aur **X_2** columns ho, predictions milengi neeche.")

    if st.session_state.model is None:
        st.warning("Pehle model train karo.")
    else:
        batch_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="batch")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            batch_df.columns = [c.strip() for c in batch_df.columns]

            if not {"X_1", "X_2"}.issubset(batch_df.columns):
                st.error("CSV me X_1 aur X_2 columns hone chahiye.")
            else:
                clf = st.session_state.model
                preds = clf.predict(batch_df[["X_1", "X_2"]])
                probas = clf.predict_proba(batch_df[["X_1", "X_2"]])

                result_df = batch_df.copy()
                result_df["Predicted_y"] = preds
                result_df["Confidence"] = probas.max(axis=1)

                st.success(f"✅ {len(result_df)} rows predict ho gaye!")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
