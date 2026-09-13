import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page title and configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write("Predict the estimated selling price of a car using Machine Learning (Random Forest).")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

df = load_data()

# Model Training Setup
@st.cache_resource
def train_model(data):
    X = data.drop(columns=["name", "selling_price"])
    y = data["selling_price"]

    categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
    numerical_cols = ["year", "km_driven"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X, y)
    return model

model = train_model(df)

# User Inputs UI
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2015, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])

with col2:
    seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission Type", options=["Manual", "Automatic"])
    owner = st.selectbox(
        "Owner History",
        options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
    )

# Prediction Button
if st.button("Predict Price 💰", type="primary"):
    input_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
    })

    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Selling Price: ₹{prediction:,.2f}")
