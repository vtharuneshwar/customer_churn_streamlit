import streamlit as st
import pandas as pd
from PIL import Image

st.title("📊 Model Performance")

# Load metrics
metrics = pd.read_csv("model_metrics.csv")

st.subheader("Evaluation Metrics")
st.table(metrics)

st.subheader("Confusion Matrix")
cm_img = Image.open("confusion_matrix.png")
st.image(cm_img, caption="Confusion Matrix - Random Forest", use_container_width=True)

st.subheader("ROC Curve")
roc_img = Image.open("roc_curve.png")
st.image(roc_img, caption="ROC Curve", use_container_width=True)
