import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("🍷 Wine Quality Predictor")

st.write("Enter wine chemical properties:")

fixed_acidity = st.number_input("fixed acidity")
volatile_acidity = st.number_input("volatile acidity")
citric_acid = st.number_input("citric acid")
residual_sugar = st.number_input("residual sugar")
chlorides = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free sulfur dioxide")
total_sulfur_dioxide = st.number_input("total sulfur dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates = st.number_input("sulphates")
alcohol = st.number_input("alcohol")

if st.button("Predict Quality"):

    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH,
        sulphates, alcohol
    ]], columns=[
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH",
        "sulphates", "alcohol"
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"🍷 Predicted Wine Quality: {prediction}")