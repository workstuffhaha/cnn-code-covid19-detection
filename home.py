import streamlit as st

# Set page config
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

# Title
st.title("Chatbot Prediction System")
st.write("Choose a prediction model:")

# Buttons for navigation
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Heart Disease Prediction ❤️"):
        st.switch_page("pages/model1UI.py")

with col2:
    if st.button("Student Depression Prediction 🎓"):
        st.switch_page("pages/model2UI.py")

with col3:
    if st.button("Diabetes Prediction 💬"):
        st.switch_page("pages/model3UI.py")

with col4:
    if st.button("COVID-19 Detection 🦠"):
        st.switch_page("pages/model4UI.py")
