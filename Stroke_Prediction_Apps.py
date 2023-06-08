import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from streamlit_lottie import st_lottie

cleaned_data = pd.read_csv("Stroke_cleaned_dataset.csv")

X = cleaned_data.drop("stroke", axis=1)
y = cleaned_data["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC()
model.fit(X_train, y_train)

gender_mapping = {"Female": 0, "Male": 1}
hypertension_mapping = {"No": 0, "Yes": 1}
heart_disease_mapping = {"No": 0, "Yes": 1}
ever_married_mapping = {"No": 0, "Yes": 1}
work_type_mapping = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
Residence_type_mapping = {"Rural": 0, "Urban": 1}
smoking_status_mapping = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

def preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                     avg_glucose_level, bmi, smoking_status):
    gender = gender_mapping[gender]
    hypertension = hypertension_mapping[hypertension]
    heart_disease = heart_disease_mapping[heart_disease]
    ever_married = ever_married_mapping[ever_married]
    work_type = work_type_mapping[work_type]
    Residence_type = Residence_type_mapping[Residence_type]
    smoking_status = smoking_status_mapping[smoking_status]

    return np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                     avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

def main():
    # Load and display the Lottie animation
    lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_yfk4tei3.json")
    
    st.title("Stroke Risk Prediction")

    # User input
    gender = st.selectbox("Gender", ("Female", "Male"))
    age = st.slider("Age", 0, 100, 50)
    hypertension = st.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.selectbox("Heart Disease", ("No", "Yes"))
    ever_married = st.selectbox("Ever Married", ("No", "Yes"))
    work_type = st.selectbox("Work Type", ("Govt_job", "Never_worked", "Private", "Self-employed", "children"))
    Residence_type = st.selectbox("Residence Type", ("Rural", "Urban"))
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    smoking_status = st.selectbox("Smoking Status", ("Unknown", "formerly smoked", "never smoked", "smokes"))

    if st.button("Predict"):
        # Preprocess input
        input_features = preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type,
                                          Residence_type, avg_glucose_level, bmi, smoking_status)

        # Make prediction
        prediction = model.predict(input_features)

        # Display prediction
        if prediction == 0:
            st.write("Low stroke risk")
        else:
            st.write("High stroke risk")

if __name__ == "__main__":
    main()
