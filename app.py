import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("milk_quality_xgb.pkl")
    return bundle["model"], bundle["scaler"]

model, scaler = load_model()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
}
.main {
    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0b2545;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#1f4e79;
}
.card {
    background:white;
    padding:30px;
    border-radius:18px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.2);
}
.stButton>button {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color:white;
    height:55px;
    width:100%;
    font-size:18px;
    border-radius:12px;
    border:none;
}
.stButton>button:hover {
    background: linear-gradient(to right, #24c6dc, #514a9d);
}
.footer {
    text-align:center;
    color:#0b2545;
}
.range {
    color:#555;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<div class='title'>🥛 Milk Quality Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>XGBoost Machine Learning Model</div>", unsafe_allow_html=True)

# -------------------------------
# Input Card
# -------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("**pH Level**  <span class='range'>(Min: 0  |  Max: 14)</span>", unsafe_allow_html=True)
    ph = st.slider("", 0.0, 14.0, 6.5)

    st.markdown("**Temperature (°C)**  <span class='range'>(Min: 0  |  Max: 100)</span>", unsafe_allow_html=True)
    temperature = st.slider("", 0.0, 100.0, 25.0)

    taste = st.selectbox("Taste", [0,1], format_func=lambda x: "Good" if x==1 else "Bad")
    odor = st.selectbox("Odor", [0,1], format_func=lambda x: "Good" if x==1 else "Bad")
    fat = st.selectbox("Fat", [0,1], format_func=lambda x: "Optimal" if x==1 else "Not Optimal")
    turbidity = st.selectbox("Turbidity", [0,1], format_func=lambda x: "High" if x==1 else "Low")

    st.markdown("**Colour Value**  <span class='range'>(Min: 0  |  Max: 255)</span>", unsafe_allow_html=True)
    colour = st.slider("", 0, 255, 200)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Milk Quality"):
    input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 2:
        st.success("🥇 HIGH QUALITY MILK")
    elif prediction == 1:
        st.warning("🥈 MEDIUM QUALITY MILK")
    else:
        st.error("🥉 LOW QUALITY MILK")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<div class='footer'>Developed by Govarthanan | XGBoost Project</div>", unsafe_allow_html=True)
