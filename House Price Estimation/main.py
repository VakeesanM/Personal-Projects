import joblib
import numpy as np
import streamlit as st


model = joblib.load('regression_model.pkl')

st.title("Fruad Dectectoin App")
st.markdown("Please enter the House info:")
st.divider()

long = st.number_input("longitude:")
lat = st.number_input("Latitude:")
house_age = st.number_input("House Age:")
room = st.number_input("Number of Rooms:")
bedroom = st.number_input("Number of Bedrooms:")
pop = st.number_input("Population:")
household = st.number_input("Number of Houses in Block:")
income = st.number_input("Average Income of Block:")
ocean_proximity = st.selectbox("Transaction Type", ["<1H OCEAN", 'INLAND', "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Predict"):
    ocean=0
    inland=0
    island=0
    near_bay = 0
    nocean = 0

    if ocean_proximity == "<1H OCEAN":
        nocean= 1
    elif ocean_proximity == "INLAND":
        inland = 1
    elif ocean_proximity == "NEAR OCEAN":
        near_bay = 1
    elif ocean_proximity == "NEAR BAY":
        near_bay = 1
    elif ocean_proximity == "ISLAND":
        island = 1

    Data = [long,lat,house_age,room,bedroom,pop,household,income,ocean,inland,island,near_bay,nocean]
    Data.append(Data[3]/Data[2])
    Data.append(Data[2]/Data[5])
    Data = np.array(Data).reshape(1,-1)
    Estimate = model.predict(Data).astype(float)
    st.success(f"Estimated Cost: ${Estimate}")

