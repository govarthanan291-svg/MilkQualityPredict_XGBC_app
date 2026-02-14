import streamlit as st
import numpy as np
import joblib

# MUST be first Streamlit command
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)

# Load model bundle
@st.cache_resource
def load_bundle():
    return joblib.load("milk_quality_xgb.pkl")

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]

# UI
st.markdown("<h1 style='text-align:center;color:#00FFFF;'>🥛 Milk Quality Prediction</h1>", unsafe_allow_html=True)

ph = st.number_input("pH Level (0-14)", 0.0, 14.0, 6.8)
temp = st.number_input("Temperature (0-100)", 0.0, 100.0, 25.0)

taste = st.selectbox("Taste", ["Bad", "Good"])
odor = st.selectbox("Odor", ["Bad", "Good"])
fat = st.selectbox("Fat Content", ["Not Optimal", "Optimal"])
turbidity = st.selectbox("Turbidity", ["High", "Low"])

colour = st.number_input("Colour Value (0-255)", 0, 255, 150)

# Encoding
taste = 1 if taste == "Good" else 0
odor = 1 if odor == "Good" else 0
fat = 1 if fat == "Optimal" else 0
turbidity = 0 if turbidity == "Low" else 1

if st.button("🔍 Predict Milk Quality"):
    input_data = np.array([[ph, temp, taste, odor, fat, turbidity, colour]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]

    if pred == 0:
        st.error("🥛 Milk Quality: LOW")
    elif pred == 1:
        st.warning("🥛 Milk Quality: MEDIUM")
    else:
        st.success("🥛 Milk Quality: HIGH")
