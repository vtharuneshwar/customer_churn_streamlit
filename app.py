import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Telecom Customer Churn Prediction")

st.write("Enter customer details")

tenure = st.number_input("Tenure Months", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict"):
    input_dict = {col: 0 for col in model_columns}

    input_dict["Tenure Months"] = tenure
    input_dict["Monthly Charges"] = monthly
    input_dict["Total Charges"] = total
    input_dict["Contract"] = contract
    input_dict["Internet Service"] = internet

    input_df = pd.DataFrame([input_dict])

    input_processed = preprocessor.transform(input_df)
    pred = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    if pred == 1:
        st.error(f"Customer will Churn (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will Stay (Probability: {1-prob:.2f})")
