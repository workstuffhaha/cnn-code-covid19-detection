import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("Models/diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Models/diabetes_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Diabetes Prediction")
st.write("Answer the following questions to assess diabetes risk.")

# Define questions
questions = [
    {"text": "Enter Patient ID:", "type": "number", "key": "id", "min": 1, "max": 9999, "step": 1},
    {"text": "Enter Number of Pregnancies:", "type": "number", "key": "pregnancies", "min": 0, "max": 20, "step": 1},
    {"text": "Enter Glucose Level:", "type": "number", "key": "glucose", "min": 0, "max": 200, "step": 1},
    {"text": "Enter Blood Pressure:", "type": "number", "key": "blood_pressure", "min": 0, "max": 150, "step": 1},
    {"text": "Enter Skin Thickness:", "type": "number", "key": "skin_thickness", "min": 0, "max": 150, "step": 1},
    {"text": "Enter Insulin Level:", "type": "number", "key": "insulin", "min": 0, "max": 1000, "step": 1},
    {"text": "Enter BMI:", "type": "number", "key": "bmi", "min": 0.0, "max": 100.0, "step": 0.1},
    {"text": "Enter Diabetes Pedigree Function:", "type": "number", "key": "dpf", "min": 0.0, "max": 3.0, "step": 0.01},
    {"text": "Enter Age:", "type": "number", "key": "age", "min": 0, "max": 100, "step": 1}
]

# Session state for question tracking
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# Display questions one at a time
q = questions[st.session_state.current_question]
response = st.number_input(q["text"], min_value=q["min"], max_value=q["max"], step=q["step"], key=q["key"])

if st.button("Next"):
    st.session_state.responses[q["key"]] = response
    if st.session_state.current_question < len(questions) - 1:
        st.session_state.current_question += 1
        st.rerun()

# Prediction step
if st.session_state.current_question == len(questions) - 1 and st.button("Predict"):
    # Extract user responses
    user_input = [
        st.session_state.responses.get("pregnancies", 0),
        st.session_state.responses.get("glucose", 0),
        st.session_state.responses.get("blood_pressure", 0),
        st.session_state.responses.get("skin_thickness", 0),
        st.session_state.responses.get("insulin", 0),
        st.session_state.responses.get("bmi", 0.0),
        st.session_state.responses.get("dpf", 0.0),
        st.session_state.responses.get("age", 0)
    ]

    # Transform input and predict
    transformed_input = scaler.transform([user_input])
    prediction = model.predict(transformed_input)

    result = "Yes, you might have diabetes." if prediction[0] == 1 else "No, you do not have diabetes."
    st.success(result)

# Restart Button
if st.button("Restart Test"):
    st.session_state.current_question = 0
    st.session_state.responses = {}
    st.rerun()
