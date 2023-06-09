import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image

# cleaned_data = pd.read_csv("Stroke_cleaned_dataset.csv")
csv_url = "https://raw.githubusercontent.com/Peilin-CodeVersion/Stroke_Prediction/main/Stroke_cleaned_dataset.csv"  
cleaned_data = pd.read_csv(csv_url)

X = cleaned_data.drop("stroke", axis=1)
y = cleaned_data["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC()
model.fit(X_train, y_train)

# Create the Streamlit app
def main():
    # Set the app title and header image
    header_image_url = "https://raw.githubusercontent.com/Peilin-CodeVersion/Stroke_Prediction/main/Stroke_Risk_Predictor.jpg" 

    # Display the header image
    st.image(header_image_url, use_column_width=True)

    # Define the user inputs
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

    # Create a button to start the prediction
    if st.button("Predict Stroke Risk"):
        # Save the user inputs into a dataset
        user_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [Residence_type],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "smoking_status": [smoking_status]
        })

        # Encode the user dataset
        user_data["gender"] = user_data["gender"].map({"Female": 0, "Male": 1}).astype(int)
        user_data["hypertension"] = user_data["hypertension"].map({"No": 0, "Yes": 1}).astype(int)
        user_data["heart_disease"] = user_data["heart_disease"].map({"No": 0, "Yes": 1}).astype(int)
        user_data["ever_married"] = user_data["ever_married"].map({"No": 0, "Yes": 1}).astype(int)
        user_data["work_type"] = user_data["work_type"].map({"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}).astype(int)
        user_data["Residence_type"] = user_data["Residence_type"].map({"Rural": 0, "Urban": 1}).astype(int)
        user_data["smoking_status"] = user_data["smoking_status"].map({"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}).astype(int)
       

        # Use the dataset to predict the stroke using the trained SVM model
        stroke_prediction = model.predict(user_data)
        
        # Apply additional conditions to change prediction to high stroke risk
        if bmi > 26 and ever_married == "Yes" and hypertension == "Yes" and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1
        elif age >= 50 and ever_married == "Yes" and hypertension == "Yes" and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1
        elif ever_married == "Yes" and hypertension == "Yes" and smoking_status != "smokes" and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1
        elif ever_married == "Yes" and hypertension == "Yes" and smoking_status != "formerly smoked" and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1
        elif hypertension == "Yes" and heart_disease == "Yes" and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1
        elif hypertension == "Yes" and ever_married == "Yes" and age >= 50 and age > 10 and avg_glucose_level >50 and BMI >5 :
            stroke_prediction[0] = 1

        # Show the risk of getting a stroke
        st.subheader("Stroke Risk Prediction Result:")
        if stroke_prediction[0] == 0:
            st.write("Congrats, You have a lower risk of experiencing a stroke ")
        else:
            st.write("Hhere is a higher risk of experiencing a stroke. We recommend regular body check-ups")

if __name__ == "__main__":
    main()

