import streamlit as st
import numpy as np
import joblib

# -------------------------------------------------
# Page Config (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("milk_quality_xgb.pkl")
    return model

model = load_model()

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #E3F2FD;
}
.main {
    background-color: #E3F2FD;
}
.title {
    text-align:center;
    font-size:38px;
    font-weight:bold;
    color:#0D47A1;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#1565C0;
}
.box {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 0px 10px gray;
}
.stButton>button {
    background-color:#1976D2;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}
.stButton>button:hover {
    background-color:#0D47A1;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<div class='title'>🥛 Milk Quality Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Using XGBoost Machine Learning Model</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Input Fields
# -------------------------------------------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

ph = st.number_input("pH Level (0 - 14)", min_value=0.0, max_value=14.0, value=6.5)
temperature = st.number_input("Temperature (0 - 100)", min_value=0.0, max_value=100.0, value=25.0)
taste = st.selectbox("Taste", ["Good", "Bad"])
odor = st.selectbox("Odor", ["Good", "Bad"])
fat = st.selectbox("Fat Content", ["Optimal", "Not Optimal"])
turbidity = st.selectbox("Turbidity", ["High", "Low"])
colour = st.number_input("Colour Value (0 - 255)", min_value=0, max_value=255, value=150)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Convert Inputs
# -------------------------------------------------
taste = 1 if taste == "Good" else 0
odor = 1 if odor == "Good" else 0
fat = 1 if fat == "Optimal" else 0
turbidity = 1 if turbidity == "High" else 0

input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Milk Quality"):

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.error("🥛 Milk Quality: LOW")
    elif prediction == 1:
        st.warning("🥛 Milk Quality: MEDIUM")
    else:
        st.success("🥛 Milk Quality: HIGH")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("Developed by Govarthanan | XGBoost Milk Quality Project")
