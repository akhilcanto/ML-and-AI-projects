import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Prediction")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
    

model = load_model()

st.title("Diabetes Prediction")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0.0, 300.0, 110.0)
    bp = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
    skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
with col2:
    insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    x = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    pred = int(model.predict(x)[0])

    if pred == 1:
        st.error("Higher risk of diabetes")
    else:
        st.success("Lower risk of diabetes")

    if hasattr(model, "predict_proba"):
        st.write(f"Probability: {model.predict_proba(x)[0][1]:.1%}")