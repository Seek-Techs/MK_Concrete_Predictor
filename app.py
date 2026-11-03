from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "operational_model.joblib"
META_PATH = ROOT / "artifacts" / "operational_model_metadata.json"

st.set_page_config(page_title="MK Concrete Strength Predictor", page_icon="🧱", layout="centered")
st.title("Metakaolin-Blended Concrete")
st.subheader("Mean Compressive Strength Predictor")
st.write("Estimate mean cube compressive strength from metakaolin (MK) replacement and curing age.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Required model artifacts are missing. Please check the repository's artifacts directory.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Model information")
    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Operational fit:** {metadata['fit_data']}")
    st.write(f"**Training observations:** {metadata['n_fit_observations']}")
    st.write(f"**Input domain:** MK {metadata['mk_min_pct']}–{metadata['mk_max_pct']}%; age {metadata['age_min_days']}–{metadata['age_max_days']} days")
    st.caption("The deployed model is refitted on all 28 experimental group means. The reserved holdout metrics apply to the separate frozen 22-observation research model, not this refitted artifact.")

st.info("Use only within the observed input ranges. Estimates are model outputs, not guaranteed batch strengths or structural acceptance results.")
with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        mk = st.number_input("MK replacement (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
    with c2:
        age = st.selectbox("Curing age (days)", options=[7, 28, 56, 90], index=1)
    submitted = st.form_submit_button("Estimate mean strength", type="primary")

if submitted:
    X = pd.DataFrame([{"MK_replacement_pct": float(mk), "curing_age_days": int(age)}])
    prediction = float(model.predict(X)[0])
    st.metric("Predicted mean compressive strength", f"{prediction:.2f} MPa")
    st.caption(f"Inputs: {mk:g}% MK replacement, {age} days curing.")
    st.caption("Rounded for display. This is a point estimate; no calibrated prediction interval is provided.")

with st.expander("Model assumptions and limitations"):
    st.markdown("""
- The target is the **group mean cube compressive strength** for a tested MK–age condition.
- Predictors are MK replacement percentage and curing age.
- Model: degree-2 polynomial features followed by standardisation and ordinary least squares.
- This operational artifact is refitted on the 28 group means to support use within the study's tested domain.
- The study has only 28 experimental-condition means. The six-condition holdout is small and comes from the same experimental programme.
- The reported holdout results (MAE 1.7153 MPa, RMSE 1.9585 MPa, R² 0.9368) belong to the research model trained on 22 points; they are not a validation score for this full-data operational model.
- Results do not establish performance for other materials, batches, laboratories, mix designs, temperatures, ages, or MK percentages outside the observed range.
- A predicted group mean is not an individual-cube prediction, design strength, characteristic strength, or acceptance criterion.
- No prediction interval is supplied. Do not use this tool alone for engineering design, quality acceptance, or safety-critical decisions.
""")
st.caption("Research demonstration | The deployed operational model is pre-trained. Retraining requires authorised access to the research dataset.")
