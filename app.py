import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Page Setup
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom Styling (Simple Clean Theme)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-card {
        background-color: #E3F2FD;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #2196F3;
        margin-top: 20px;
    }
    .dep-card {
        background-color: #2196F3;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #E53935;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")
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
st.markdown('<p class="sub-header">Estimate resale value and depreciation using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar Info
st.sidebar.header("📌 About Model")
st.sidebar.info(
    """
    **Algorithm:** Random Forest Regressor  
    **Dataset Size:** 4,300+ Records  
    **Features:** Model, Year, Original Price, KMs Driven, Fuel, Transmission, Owner & Seller Type.
    """
)

# Input Form Layout
st.subheader("📋 Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    brands = sorted(df["brand"].unique().tolist())
    selected_brand = st.selectbox("Select Car Brand", options=brands, index=brands.index("Maruti") if "Maruti" in brands else 0)

    filtered_models = sorted(df[df["brand"] == selected_brand]["name"].unique().tolist())
    car_name = st.selectbox("Select Car Model", options=filtered_models)

    original_price = st.number_input(
        "Original Purchase Price (in ₹)",
        min_value=50000,
        max_value=10000000,
        value=600000,
        step=25000,
        help="Amount the car was originally bought for."
    )

    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2016, step=1)

with col2:
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
    fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox(
        "Owner History",
        options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
    )

st.markdown("---")

# Predict Button & Results
if st.button("🔍 Estimate Price & Depreciation", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "name": [car_name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
    })

    # Prediction
    predicted_price = model.predict(input_data)[0]

    # Depreciation Calculations
    depreciation_amt = original_price - predicted_price
    depreciation_pct = (depreciation_amt / original_price) * 100 if original_price > 0 else 0
    current_year = datetime.datetime.now().year
    car_age = max(1, current_year - year)

    # Main Price Display Card
    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="color: #0D47A1; margin-bottom: 5px;">Estimated Selling Price</h3>
            <h1 style="color: #1565C0; margin-top: 0px;">₹ {predicted_price:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Depreciation Breakdown Below
    if depreciation_amt > 0:
        st.markdown(
            f"""
            <div class="dep-card">
                <h4 style="color: #D32F2F; margin-top: 0; margin-bottom: 10px;">📉 Depreciation Breakdown</h4>
                <p style="margin: 4px 0;"><strong>Original Price:</strong> ₹ {original_price:,.2f}</p>
                <p style="margin: 4px 0;"><strong>Value Retained:</strong> {100 - depreciation_pct:.1f}%</p>
                <p style="margin: 4px 0;"><strong>Total Value Lost:</strong> ₹ {depreciation_amt:,.2f} ({depreciation_pct:.1f}%)</p>
                <p style="margin: 4px 0;"><strong>Average Annual Loss:</strong> ₹ {depreciation_amt / car_age:,.2f} / year</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="dep-card" style="border-left-color: #388E3C;">
                <h4 style="color: #388E3C; margin-top: 0; margin-bottom: 10px;">📈 Value Appreciation</h4>
                <p style="margin: 0;">This vehicle model has retained or gained value relative to your entered purchase price.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
