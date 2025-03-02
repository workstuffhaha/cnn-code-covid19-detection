import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("Models/student_depression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("Models/student_depression_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define input sequence
questions = [
    ("Age", "Enter your age:"),
    ("CGPA", "Enter your CGPA:"),
    ("Financial Stress", "Rate your financial stress (1-10):"),
    ("Work/Study Hours per Day", "How many hours do you work/study per day?"),
    ("Academic Pressure", "Rate your academic pressure (1-10):")
]

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = []

# Display questions one by one
if st.session_state.step < len(questions):
    feature, question = questions[st.session_state.step]
    user_input = st.number_input(question, min_value=0.0, step=0.1, format="%.1f")
    
    if st.button("Next"):
        st.session_state.inputs.append(user_input)
        st.session_state.step += 1
        st.experimental_rerun()
else:
    # Prepare input for prediction
    X_new = np.array(st.session_state.inputs).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)[0]
    
    # Display result
    st.subheader("Prediction Result")
    st.write("Depression Risk:" , "Yes" if prediction == 1 else "No")
    
    # Reset button
    if st.button("Restart"):
        st.session_state.step = 0
        st.session_state.inputs = []
        st.experimental_rerun()
