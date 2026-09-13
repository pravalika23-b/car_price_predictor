import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page Setup
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        background-color: #E3F2FD;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #2196F3;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")
    # Extract brand name for quick filtering
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0])
    return df

df = load_data()

# Model Training Setup
@st.cache_resource
def train_model(data):
    X = data.drop(columns=["selling_price", "brand"])
    y = data["selling_price"]

    categorical_cols = ["name", "fuel", "seller_type", "transmission", "owner"]
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

# Header UI
st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Estimate market resale value using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar Info
st.sidebar.header("📌 About Model")
st.sidebar.info(
    """
    **Algorithm:** Random Forest Regressor  
    **Dataset Size:** 4,300+ Records  
    **Features:** Model, Year, KMs Driven, Fuel, Transmission, Owner & Seller Type.
    """
)

# Main Form Layout
st.subheader("📋 Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    # Select Brand first, then filter Car Models
    brands = sorted(df["brand"].unique().tolist())
    selected_brand = st.selectbox("Select Car Brand", options=brands, index=brands.index("Maruti") if "Maruti" in brands else 0)

    # Filter models based on selected brand
    filtered_models = sorted(df[df["brand"] == selected_brand]["name"].unique().tolist())
    car_name = st.selectbox("Select Car Model", options=filtered_models)

    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2016, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)

with col2:
    fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox(
        "Owner History",
        options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
    )

st.markdown("---")

# Predict Button & Result Card
if st.button("🔍 Estimate Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "name": [car_name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
    })

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="color: #0D47A1; margin-bottom: 5px;">Estimated Selling Price</h3>
            <h1 style="color: #1565C0; margin-top: 0px;">₹ {prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
