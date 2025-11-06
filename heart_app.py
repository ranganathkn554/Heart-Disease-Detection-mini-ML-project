import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter your health details to predict heart disease risk.")

# Input fields
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
chol = st.number_input("Serum Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1=Yes, 0=No)", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0=Normal,1=Fixed,2=Reversible)", [0, 1, 2])

# Prepare input
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal],
    'cp': [cp]
})

# One-hot encode cp
user_input = pd.get_dummies(user_input, columns=['cp'], drop_first=True)

# Match training columns
user_input = user_input.reindex(columns=feature_columns, fill_value=0)

# Scale
user_scaled = scaler.transform(user_input)

# Predict
if st.button("🔍 Predict Heart Disease"):
    try:
        prob = model.predict_proba(user_scaled)[0][1]
        prediction = model.predict(user_scaled)[0]

        st.write(f"### 🧾 Prediction Probability: {prob*100:.2f}%")
        if prob > 0.7:
            st.error("🚨 High chance of heart disease!")
        elif prob > 0.4:
            st.warning("⚠️ Moderate risk of heart disease.")
        else:
            st.success("💚 Low chance of heart disease.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

