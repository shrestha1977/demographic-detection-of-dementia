import streamlit as st
import pandas as pd
import pickle

# Load saved model, scaler, and encoders
with open('rfc.pkl', 'rb') as f:
    rfc = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title("Dementia Probability Prediction")


# Define input widgets for each feature (replace names with your actual feature names)
# For categorical features, use the original classes from label encoders
def get_categorical_options(col):
    le = label_encoders[col]
    return le.classes_


# Inputs (add/change according to your features)
diabetic = st.selectbox('Diabetic Status', [0, 1])
alcohol_level = st.number_input("Alcohol Level", value=0.0849, format="%.6f")
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=120, value=70)
blood_oxygen = st.number_input("Blood Oxygen Level (%)", value=96.23, format="%.2f")
body_temp = st.number_input("Body Temperature (°C)", value=36.22, format="%.2f")
weight = st.number_input("Weight (kg)", value=57.56, format="%.2f")
mri_delay = st.number_input("MRI Delay", value=36.42, format="%.2f")
prescription = st.selectbox("Prescription", get_categorical_options("Prescription"))
dosage = st.number_input("Dosage in mg", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=65)
education_level = st.selectbox("Education Level", get_categorical_options("Education_Level"))
dominant_hand = st.selectbox("Dominant Hand", get_categorical_options("Dominant_Hand"))
gender = st.selectbox("Gender", get_categorical_options("Gender"))
family_history = st.selectbox("Family History", get_categorical_options("Family_History"))
smoking_status = st.selectbox("Smoking Status", get_categorical_options("Smoking_Status"))
apoe_e4 = st.selectbox("APOE ε4 Status", get_categorical_options("APOE_ε4"))
physical_activity = st.selectbox("Physical Activity", get_categorical_options("Physical_Activity"))
depression_status = st.selectbox("Depression Status", get_categorical_options("Depression_Status"))
cognitive_scores = st.number_input("Cognitive Test Scores", min_value=0, max_value=10, value=5)
medication_history = st.selectbox("Medication History", get_categorical_options("Medication_History"))
nutrition_diet = st.selectbox("Nutrition/Diet Quality", get_categorical_options("Nutrition_Diet"))
sleep_quality = st.selectbox("Sleep Quality", get_categorical_options("Sleep_Quality"))
chronic_conditions = st.selectbox("Chronic Health Conditions", get_categorical_options("Chronic_Health_Conditions"))

# Prepare input DataFrame
input_dict = {
    'Diabetic': diabetic,
    'AlcoholLevel': alcohol_level,
    'HeartRate': heart_rate,
    'BloodOxygenLevel': blood_oxygen,
    'BodyTemperature': body_temp,
    'Weight': weight,
    'MRI_Delay': mri_delay,
    'Prescription': prescription,
    'Dosage in mg': dosage,
    'Age': age,
    'Education_Level': education_level,
    'Dominant_Hand': dominant_hand,
    'Gender': gender,
    'Family_History': family_history,
    'Smoking_Status': smoking_status,
    'APOE_ε4': apoe_e4,
    'Physical_Activity': physical_activity,
    'Depression_Status': depression_status,
    'Cognitive_Test_Scores': cognitive_scores,
    'Medication_History': medication_history,
    'Nutrition_Diet': nutrition_diet,
    'Sleep_Quality': sleep_quality,
    'Chronic_Health_Conditions': chronic_conditions
}

df_input = pd.DataFrame([input_dict])

# Encode categorical columns using saved encoders
for col in label_encoders.keys():
    le = label_encoders[col]
    # Transform using the fitted encoder; handle unseen categories gracefully
    try:
        df_input[col] = le.transform(df_input[col])
    except ValueError:
        st.error(f"Value '{df_input[col][0]}' not recognized in {col}. Please select a valid option.")
        st.stop()

# Scale the input
X_scaled = scaler.transform(df_input)

if st.button("Predict Dementia Probability"):
    proba = rfc.predict_proba(X_scaled)[0][1]
    st.write(f"Probability of having dementia: **{proba:.4f}**")
