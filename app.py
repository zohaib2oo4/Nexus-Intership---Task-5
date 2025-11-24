import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# PAGE CONFIG (must be FIRST Streamlit command)
# -------------------------
st.set_page_config(
    page_title="SuperMarket Prediction App",
    layout="wide",
)

# -------------------------
# LOAD MODEL
# -------------------------
MODEL_PATH = "models/pipeline_model_20251124_143200.joblib"
THRESHOLD = 253.848

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------
# UI HEADER
# -------------------------
st.title("🛒 SuperMarket ML Prediction Web App")
st.write("Upload a CSV file to generate predictions using your trained machine learning pipeline.")

# -------------------------
# FILE UPLOADER
# -------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load new CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Run prediction
    try:
        preds = model.predict(df)
    except Exception as e:
        st.error(f"⚠ Error during prediction: {e}")
        st.stop()

    # Add prediction columns
    df["PredictedValue"] = preds
    df["Class"] = (df["PredictedValue"] >= THRESHOLD).astype(int)

    st.subheader("📊 Prediction Results")
    st.dataframe(df)

    # Summary
    st.subheader("📈 Summary")
    st.write(f"**Threshold used:** {THRESHOLD}")
    st.write(df["Class"].value_counts())

    # Download button
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Predictions CSV",
        data=csv_download,
        file_name="predictions_output.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a CSV file to begin.")
