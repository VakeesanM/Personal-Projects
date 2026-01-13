import streamlit as st
import pandas as pd
import joblib


model = joblib.load("fruad_decection_model_pipleine.pkl")
st.title("Fruad Dectectoin App")
st.markdown("please enter the transaction info:")
st.divider()

tranfer_type = st.selectbox("Transaction Type", ["PAYMENT", 'CASH_IN', "CASH_OUT", "TRANSFER", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value = 0.0)
oldbalanceOrg = st.number_input("Old Balance For Sender", min_value=0.0, value = 0.0)
newbalanceOrig= st.number_input("New Balance for Sender", min_value=0.0, value = 0.0)
oldbalanceDest = st.number_input("Old Balance For Receviver",min_value=0.0, value = 0.0)
newbalanceDest = st.number_input("New Balance For Receviver", min_value=0.0, value = 0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": tranfer_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest' : oldbalanceDest,
        'newbalanceDest' : newbalanceDest
    }])

    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("This is most likely fraud")
    else:
        st.success("This is most likely not fraud")
