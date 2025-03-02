import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("student_depression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("student_depression_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Student Depression Prediction")

# Collect user inputs one by one
age = st.number_input("Enter your Age:", min_value=10, max_value=100, step=1)
if st.button("Next", key="age_next"):
    st.session_state.age = age
    st.session_state.step = 1

if "step" in st.session_state and st.session_state.step >= 1:
    cgpa = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.1)
    if st.button("Next", key="cgpa_next"):
        st.session_state.cgpa = cgpa
        st.session_state.step = 2

if "step" in st.session_state and st.session_state.step >= 2:
    work_study_hours = st.number_input("Enter Work/Study Hours per Day:", min_value=0, max_value=24, step=1)
    if st.button("Next", key="work_hours_next"):
        st.session_state.work_study_hours = work_study_hours
        st.session_state.step = 3

if "step" in st.session_state and st.session_state.step >= 3:
    academic_pressure = st.slider("Rate your Academic Pressure (1-10):", min_value=1, max_value=10)
    if st.button("Next", key="academic_pressure_next"):
        st.session_state.academic_pressure = academic_pressure
        st.session_state.step = 4

if "step" in st.session_state and st.session_state.step >= 4:
    financial_stress = st.slider("Rate your Financial Stress (1-10):", min_value=1, max_value=10)
    if st.button("Predict", key="predict"):
        st.session_state.financial_stress = financial_stress
        st.session_state.step = 5

# Make prediction
if "step" in st.session_state and st.session_state.step == 5:
    user_input = np.array([
        [st.session_state.age, st.session_state.cgpa, st.session_state.work_study_hours,
         st.session_state.academic_pressure, st.session_state.financial_stress]
    ])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    result = "Likely Depressed" if prediction == 1 else "Not Depressed"
    st.success(f"Prediction: {result}")
