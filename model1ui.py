import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model and scaler
model_path = "Models/student_depression_model.pkl"
scaler_path = "Models/student_depression_scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
else:
    st.error("Model or scaler file not found. Please train the model first.")
    st.stop()

# Title
st.title("Student Depression Prediction")
st.write("Answer the following questions to assess the risk of depression.")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.user_inputs = []

def reset_chat():
    st.session_state.step = 0
    st.session_state.user_inputs = []

# Questions for input
questions = [
    ("Enter Age:", "number", 10, 100),
    ("Enter CGPA:", "number", 0.0, 4.0),
    ("Enter Financial Stress Level (1-10):", "number", 1, 10),
    ("Enter Work/Study Hours per Day:", "number", 0, 24),
    ("Enter Academic Pressure Level (1-10):", "number", 1, 10)
]

# Column names for DataFrame
columns = ["Age", "CGPA", "Financial Stress", "Work/Study Hours per Day", "Academic Pressure"]

if st.session_state.step < len(questions):
    question, q_type, *options = questions[st.session_state.step]
    st.write(question)
    
    if q_type == "number":
        response = st.number_input("", min_value=options[0], max_value=options[1], key=st.session_state.step)
    
    if st.button("Next"):
        st.session_state.user_inputs.append(response)
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == len(questions):
    st.write("### All inputs received! Click 'Predict' to see the result.")
    if st.button("Predict"):
        user_data_df = pd.DataFrame([st.session_state.user_inputs], columns=columns)
        processed_data = scaler.transform(user_data_df)
        prediction = model.predict(processed_data)
        result = "High Risk of Depression" if prediction[0] == 1 else "Low Risk of Depression"
        st.success(result)
    
    if st.button("Restart Test"):
        reset_chat()
        st.rerun()
