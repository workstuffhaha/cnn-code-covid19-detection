import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("Models/student_depression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Models/student_depression_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Student Depression Prediction")
st.write("Provide the following details to assess the risk of depression.")

# Sleep Duration Mapping
sleep_mapping = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours (Ideal)": 3,
    "More than 8 hours": 4
}

# User Inputs
academic_pressure = st.slider("Academic Pressure (0-5):", 0, 5, 2)
sleep_duration = st.selectbox("Select Sleep Duration:", list(sleep_mapping.keys()))
financial_stress = st.slider("Financial Stress (1-5):", 1, 5, 3)

# Convert Sleep Duration
sleep_value = sleep_mapping[sleep_duration]

# Prediction Button
if st.button("Predict"):
    user_input = np.array([academic_pressure, sleep_value, financial_stress]).reshape(1, -1)
    transformed_input = scaler.transform(user_input)
    prediction = model.predict(transformed_input)
    result = "Yes, you might have depression." if prediction[0] == 1 else "No, you do not have depression."
    st.success(result)
