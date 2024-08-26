import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('diabetes_rf_model.pkl')

# Load the scaler used for feature scaling
scaler = joblib.load('scaler.pkl')

# Define the app layout
st.title('Diabetes Prediction App')

st.write('Enter the following information to predict diabetes:')

# Create input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=300, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

# Create a DataFrame for the user input
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
})

# Apply feature scaling
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.write('The model predicts that the patient has diabetes.')
    else:
        st.write('The model predicts that the patient does not have diabetes.')
