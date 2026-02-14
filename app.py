import streamlit as st
import numpy as np
import joblib

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)

# --------------------------------
# Load Model + Scaler
# --------------------------------
@st.cache_resource
def load_objects():
    data = joblib.load("milk_quality_xgb.pkl")
    return data["model"], data["scaler"]

model, scaler = load_objects()

# --------------------------------
# UI Style
# --------------------------------
st.markdown("""
<style>
body {background-color:#E3F2FD;}
.title{text-align:center;font-size:36px;font-weight:bold;color:#0D47A1;}
.box{background:white;padding:25px;border-radius:15px;}
.stButton>button{background:#1976D2;color:white;width:100%;height:50px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🥛 Milk Quality Prediction</div>", unsafe_allow_html=True)

# --------------------------------
# Inputs
# --------------------------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

ph = st.number_input("pH (0-14)",0.0,14.0,6.5)
temperature = st.number_input("Temperature",0.0,100.0,25.0)
taste = st.selectbox("Taste",["Good","Bad"])
odor = st.selectbox("Odor",["Good","Bad"])
fat = st.selectbox("Fat",["Optimal","Not Optimal"])
turbidity = st.selectbox("Turbidity",["High","Low"])
colour = st.number_input("Colour",0,255,150)

st.markdown("</div>", unsafe_allow_html=True)

# Encode
taste = 1 if taste=="Good" else 0
odor = 1 if odor=="Good" else 0
fat = 1 if fat=="Optimal" else 0
turbidity = 1 if turbidity=="High" else 0

input_data = np.array([[ph,temperature,taste,odor,fat,turbidity,colour]])

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict Quality"):
    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]

    if pred==0:
        st.error("🥛 Milk Quality: LOW")
    elif pred==1:
        st.warning("🥛 Milk Quality: MEDIUM")
    else:
        st.success("🥛 Milk Quality: HIGH")
