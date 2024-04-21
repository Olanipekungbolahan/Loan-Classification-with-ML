import numpy as np
import pickle
import pandas as pd
import streamlit as st 

scaler = pickle.load(open('scaler.pkl','rb')) # Load your scaler
rf_model = pickle.load(open('rf_model.pkl','rb'))  # Load your trained model

def predict_Loan_Default(credit_policy, purpose, interest_rate, installment, log_annual_inc, debt_to_income_ratio, fico_credit_score, days_with_credit_on_the_line, revolving_balance, revolving_utilisation, six_mths_inquiry, two_years_delinquency, public_record):
    
    # Perform One Hot Encoding for categorical feature 'purpose'
    input_df_encoded = pd.get_dummies(input_df, columns=['purpose'])
    
    # Normalize the data using the loaded scaler
    input_normalized = scaler.transform(input_df_encoded)
    prediction = rf_model.predict([[credit_policy, purpose, interest_rate, installment, log_annual_inc, debt_to_income_ratio, fico_credit_score, days_with_credit_on_the_line, revolving_balance, revolving_utilisation, six_mths_inquiry, two_years_delinquency, public_record]])
    return prediction

def main():
    st.title("Loan Default Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Loan Default Predictor </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    credit_policy = st.text_input("Credit Policy", "Type Here")
    purpose = st.text_input("Purpose", "Type Here")
    interest_rate = st.text_input("Interest Rate", "Type Here")
    installment = st.text_input("Installment", "Type Here")
    log_annual_inc = st.text_input("Log Annual Income", "Type Here")
    debt_to_income_ratio = st.text_input("Debt to Income Ratio", "Type Here")
    fico_credit_score = st.text_input("FICO Credit Score", "Type Here")
    days_with_credit_on_the_line = st.text_input("Days with Credit on the Line", "Type Here")
    revolving_balance = st.text_input("Revolving Balance", "Type Here")
    revolving_utilisation = st.text_input("Revolving Utilisation", "Type Here")
    six_mths_inquiry = st.text_input("6 Months Inquiry", "Type Here")
    two_years_delinquency = st.text_input("2 Years Delinquency", "Type Here")
    public_record = st.text_input("Public Record", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_Loan_Default(credit_policy, purpose, interest_rate, installment, log_annual_inc, debt_to_income_ratio, fico_credit_score, days_with_credit_on_the_line, revolving_balance, revolving_utilisation, six_mths_inquiry, two_years_delinquency, public_record)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
