import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
heart_dataset = pd.read_csv("high_accuracy_heart_disease.csv")

# Splitting features and target
X = heart_dataset.iloc[:, :-1]
y = heart_dataset['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("💓 Heart Disease Prediction App")

st.sidebar.header("Enter Patient Details")

# User input fields
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise (0-2)", [0, 1, 2])
ca = st.sidebar.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: This person is likely to have heart disease.")
    else:
        st.success("✅ Low Risk: This person is unlikely to have heart disease.")

# Show model accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train)) * 100
test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

st.sidebar.subheader("📊 Model Accuracy")
st.sidebar.write(f"🔹 Training Accuracy: {train_accuracy:.2f}%")
st.sidebar.write(f"🔹 Testing Accuracy: {test_accuracy:.2f}%")
