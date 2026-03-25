import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from utils import get_risk_level, dynamic_guidance

st.set_page_config(
    page_title="Heart Risk System",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #000000;
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #00b4d8;
    font-weight: bold;
}

/* Labels */
label {
    color: white !important;
    font-weight: 500;
}

/* Input boxes (white) */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
}

/* Input text */
input, select {
    color: black !important;
}

/* Button */
.stButton > button {
    background-color: #00b4d8;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    border: none;
}

/* Button hover */
.stButton > button:hover {
    background-color: #0096c7;
}

/* General text */
p, span {
    color: white;
}

/* Footer */
.footer-text {
    color: grey;
    font-size: 13px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)



#Load trained model 
import os 
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path= os.path.join(BASE_DIR, "models", "heart_model.pkl")
model= joblib.load(model_path)

# Risk level function
def get_risk_level(prob, low, high):
    if prob < low:
        return "Low Risk"
    elif prob < high:
        return "Moderate Risk"
    else:
        return "High Risk"

def explain_risk(data):
    reasons = []

    if data["Cholesterol"].values[0] > 240:
        reasons.append("Elevated serum cholesterol levels (hyperlipidemia)")

    if data["BP"].values[0] > 140:
        reasons.append("Elevated blood pressure(hypertension)")

    if data["Max HR"].values[0] < 120:
        reasons.append("Reduced maximum heart rate during stress testing (possible impaired cardiac response))")

    if data["ST depression"].values[0] > 2:
        reasons.append("Significant ST depression during exercise (possible ischemia)")
    
    if data["Exercise angina"].values[0] == 1:
        reasons.append("Presence of exercise-induced angina, indicating cardiac stress intolerance")

    if data["Number of vessels fluro"].values[0] >= 2:
        reasons.append("Evidence of multiple major vessels with reduced blood flow (possible coronary artery blockage)")

    return reasons

def give_advice(level):

    if level == "High Risk":
        return[
            "Immediate clinical evaluation is strongly recommended",
            "Refer patient to a cardiologist for further assessment",
            "Consider ECG, echocardiography, and stress testing",
            "Initiate lifestyle modifications (diet, exercise, smoking cessation)",
            "Monitor blood pressure and lipid profile closely"
        ]
    elif level == "Moderate Risk":
        return[
            "Patient requires medical follow-up and risk factor management",
            "Encourage regular cardiovascular check-ups",
            "Improve diet and physical activity",
            "Monitor symptoms such as chest pain or fatigue"
        ]
    else:
        return[
            "Low cardiovascular risk detected",
            "Maintain a healthy lifestyle",
            "Regular medical check-ups recommended"
        ]



# App Title
st.title("Heart Disease Clinical Risk System")
st.markdown("AI- assisted cardiovascular screening tool")
st.info("⚠️ This tool is for screening purposes only. Not a medical diagnosis.")
st.divider()

# INPUTS
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])

with col2:
    st.subheader("Clinical Measurements")
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ])
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("FBS > 120", ["No", "Yes"])
    ekg = st.selectbox("EKG Results", [
        "Normal",
        "ST-T Abnormality",
        "Left Ventricular Hypertrophy"
    ])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])
    st_dep = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    vessels = st.number_input("Number of vessels fluro", 0, 4, 0)
    thal = st.selectbox("Thallium", [
        "Unknown",
        "Normal",
        "Fixed Defect", 
        "Reversible Defect"
    ])

#Making it user friendly

sex = 0 if sex == "Female" else 1

fbs= 0 if fbs== "No" else 1 

cp_map= {
    "Typical Angina":0,
    "Atypical Angina":1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp=cp_map[cp]

ekg_map= {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
ekg= ekg_map[ekg]

exang= 0 if exang == "No" else 1

thal_map = {
    "Unknown": 0,
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}
thal= thal_map[thal]

st.divider()
predict_btn= st.button("Run Risk Assessment")
st.divider()

# PREDICTION
if predict_btn:

    # Create dataframe 
    data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Chest pain type": cp,
        "BP": bp,
        "Cholesterol": chol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exang,
        "ST depression": st_dep,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thal
    }])

    # Predict probability
    prob = model.predict_proba(data)[0][1]

    # Risk thresholds
    low = 0.33
    high = 0.66

    level = get_risk_level(prob, low, high)

    # OUTPUT
    if predict_btn:
        st.subheader("Risk Assessment Result")

        st.metric("Risk Probability", f"{prob*100:.1f}%")

        #Progressbar 
        st.progress(int(prob * 100))

    if level == "High Risk":
        st.error(f"🚨 High Risk- Immediate medical attention recommended")

    elif level == "Moderate Risk":
        st.warning(f"⚠️ Moderate Risk- Lifestyle changes advised")

    else:
        st.success(f"✅ Low Risk")

    #Function why a patient is high risk
    st.divider()
    st.subheader("Clinical Interpretation")
    reasons = explain_risk(data)
    if reasons:
       for r in reasons:
         st.write(f"- {r}")
    else:
        st.write("No significant clinical risk factors identified")
        
    
    #Health advice 
    st.subheader("Clinical Recommendations")
    advice = give_advice(level)
    for a in advice:
        st.markdown(f"- {a}")

st.divider()
st.caption("Developed for educational and research purposes")


st.info("Note: ECG input is simplified and does not represent full clinical ECG diagnosis.")
