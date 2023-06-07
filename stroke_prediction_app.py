import streamlit as st
import pickle
import requests
import pickle

# URL of the raw SVM model file on GitHub
url = "https://raw.githubusercontent.com/Peilin-CodeVersion/Stroke_Prediction/main/svm_model.pkl"

# Fetch the SVM model file
response = requests.get(url)

# Load the SVM model from the response content
model = pickle.loads(response.content)

def main():
    # Set the page title
    st.title("Stroke Prediction")

    # Load the trained SVM model
    # model = pickle.load(open('svm_model.pkl', 'rb'))

    # Retrieve the feature names from the training data
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

    # Add input fields for user information
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Govt_job", "Never_worked", "Private", "Self-employed", "Children"])
    Residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["Unknown", "Formerly smoked", "Never smoked", "Smokes"])

    # Verify if the column names of the new data match the feature names
    column_names_match = all(column_name in feature_names for column_name in
                             [gender, age, hypertension, heart_disease, ever_married,
                              work_type, Residence_type, avg_glucose_level, bmi, smoking_status])

    if not column_names_match:
        st.error("Column names of the new data do not match the feature names used during training.")
        return

    # Convert user inputs into numerical values
    gender = 0 if gender == "Female" else 1
    hypertension = 0 if hypertension == "No" else 1
    heart_disease = 0 if heart_disease == "No" else 1
    ever_married = 0 if ever_married == "No" else 1
    work_type = ["Govt_job", "Never_worked", "Private", "Self-employed", "Children"].index(work_type)
    Residence_type = ["Rural", "Urban"].index(Residence_type)
    smoking_status = ["Unknown", "Formerly smoked", "Never smoked", "Smokes"].index(smoking_status)

    # Create a feature vector from the user inputs
    features = [[gender, age, hypertension, heart_disease, ever_married,
                 work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]

    # Make a prediction using the loaded SVM model
    prediction = model.predict(features)

    # Make a prediction using the loaded SVM model
    prediction = model.predict(features)

    # Display the prediction result
    if prediction[0] == 0:
        st.write("Low Stroke Risk")
    else:
        st.write("High Stroke Risk")


if __name__ == "__main__":
    main()
