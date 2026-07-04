import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Files
# -----------------------------
model = joblib.load("models/credit_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳")

st.title("💳 Credit Risk Prediction")
st.write("Enter the applicant details below.")

# -----------------------------
# User Inputs
# -----------------------------
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    encoders["person_home_ownership"].classes_
)

person_emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    value=2.0
)

loan_intent = st.selectbox(
    "Loan Purpose",
    encoders["loan_intent"].classes_
)

loan_grade = st.selectbox(
    "Loan Grade",
    encoders["loan_grade"].classes_
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

loan_int_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default",
    encoders["cb_person_default_on_file"].classes_
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0,
    value=5
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [
            encoders["person_home_ownership"].transform(
                [person_home_ownership]
            )[0]
        ],
        "person_emp_length": [person_emp_length],
        "loan_intent": [
            encoders["loan_intent"].transform(
                [loan_intent]
            )[0]
        ],
        "loan_grade": [
            encoders["loan_grade"].transform(
                [loan_grade]
            )[0]
        ],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [
            encoders["cb_person_default_on_file"].transform(
                [cb_person_default_on_file]
            )[0]
        ],
        "cb_person_cred_hist_length": [
            cb_person_cred_hist_length
        ]
    })

    input_data = input_data[feature_names]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 0:
        st.success("✅ Good Credit (Low Risk)")
    else:
        st.error("❌ Bad Credit (High Risk)")

    st.write(f"### Confidence: {max(probability)*100:.2f}%")

    st.progress(float(max(probability)))