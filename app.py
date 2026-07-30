import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("models/credit_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")
features = joblib.load("models/feature_names.pkl")

st.title("💳 Credit Risk Prediction App")

st.write("Enter customer details:")

# =========================
# INPUTS
# =========================
person_age = st.number_input("Age", 18, 100, 30)
person_income = st.number_input("Income", 0, 1000000, 50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length", 0, 50, 2)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
loan_amnt = st.number_input("Loan Amount", 0, 500000, 10000)
loan_int_rate = st.number_input("Interest Rate", 0.0, 50.0, 10.0)
loan_percent_income = st.number_input("Loan % Income", 0.0, 1.0, 0.2)
cb_default = st.selectbox("Default History", ["N", "Y"])
cb_hist = st.number_input("Credit History Length", 0, 50, 5)

# =========================
# ENCODING FUNCTION
# =========================
def encode(col, val):
    return encoders[col].transform([val])[0]

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    input_data = pd.DataFrame([[
        person_age,
        person_income,
        encode("person_home_ownership", person_home_ownership),
        person_emp_length,
        encode("loan_intent", loan_intent),
        encode("loan_grade", loan_grade),
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        encode("cb_person_default_on_file", cb_default),
        cb_hist
    ]], columns=features)

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 0:
        st.success("✅ Low Risk (Good Credit)")
    else:
        st.error("❌ High Risk (Bad Credit)")

    st.write("Probability of Default:", round(prob, 2))