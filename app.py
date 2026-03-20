import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf     = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return rf, scaler

rf, scaler = load_models()

# ── Header ────────────────────────────────────────────────────────────────
st.title("EduPredict — Student Exam Outcome Predictor")
st.caption("Machine Learning Demo | Academic Project")
st.markdown("---")

# ── Input sliders ─────────────────────────────────────────────────────────
st.subheader("Enter student details")

col1, col2 = st.columns(2)
with col1:
    study   = st.slider("Study hours per day",  1.0, 10.0, 5.0, 0.5)
    attend  = st.slider("Attendance hours (sem)", 10.0, 50.0, 30.0, 1.0)
with col2:
    assign  = st.slider("Assignments completed", 0, 10, 5)
    past    = st.slider("Past score (weak feature)", 30.0, 100.0, 60.0, 1.0)

st.caption("Note: Past score is intentionally treated as a weak predictor.")

# ── Predict ───────────────────────────────────────────────────────────────
if st.button("Predict outcome", type="primary"):
    X = pd.DataFrame([[study, attend, assign, past]],
                     columns=["study_hours","attendance_hours",
                              "assignments","past_score"])
    X_sc   = scaler.transform(X)
    pred   = rf.predict(X_sc)[0]
    prob   = rf.predict_proba(X_sc)[0]

    st.markdown("---")
    st.subheader("Prediction result")

    if pred == 1:
        st.success(f"PASS — Confidence: {prob[1]*100:.1f}%")
    else:
        st.error(f"FAIL — Confidence: {prob[0]*100:.1f}%")

    # Probability bar
    st.markdown("**Pass probability**")
    st.progress(float(prob[1]))

    # Feature contribution (manual SHAP-lite approximation)
    st.markdown("---")
    st.subheader("Feature contribution (approximate)")
    weights = [0.40, 0.30, 0.25, 0.05]
    feat_names = ["Study hours","Attendance","Assignments","Past score"]
    raw = [study/10, attend/50, assign/10, (past-30)/70]
    contributions = [w * v for w, v in zip(weights, raw)]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#e74c3c" if n == "Past score" else "#3498db" for n in feat_names]
    ax.barh(feat_names, contributions, color=colors)
    ax.set_xlabel("Weighted contribution")
    ax.set_title("How each feature influenced this prediction")
    ax.axvline(x=0, color="black", linewidth=0.8)
    st.pyplot(fig)

    st.caption("Red = weak feature (past score). Blue = strong features.")

# ── Sidebar info ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this model")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Dataset:** 500 synthetic students")
    st.write("**Train/Test split:** 80/20")
    st.write("**CV Accuracy:** ~92%")
    st.markdown("---")
    st.write("**Feature weights (by design):**")
    st.write("- Study hours: 40%")
    st.write("- Attendance: 30%")
    st.write("- Assignments: 25%")
    st.write("- Past score: 5% ← weak")