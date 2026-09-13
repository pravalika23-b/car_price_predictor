import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Page Configuration
st.set_page_config(
    page_title="Car Price & Depreciation Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Impressive UI
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-box h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .header-box p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    
    /* Result Cards */
    .price-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
        margin-top: 15px;
    }
    .price-card h3 {
        color: #f0fdf4;
        margin-bottom: 5px;
        font-size: 1.2rem;
    }
    .price-card h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    
    .dep-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #e63946;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    
    /* Card Container */
    .form-container {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
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

# Top Header Banner
st.markdown("""
    <div class="header-box">
        <h1>🚗 Smart Used Car Price & Value Predictor</h1>
        <p>AI-Powered Valuation & Depreciation Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.title("📊 Project Stats")
    st.markdown("---")
    st.metric(label="Total Dataset Cars", value=f"{len(df):,}")
    st.metric(label="Unique Brands", value=f"{df['brand'].nunique()}")
    st.metric(label="ML Model", value="Random Forest")
    st.markdown("---")
    st.info("💡 **Tip:** Entering the original purchase price provides a detailed depreciation breakdown!")

# Input Form Section
st.markdown('<div class="form-container">', unsafe_allow_html=True)
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
        help="Enter how much the car was originally bought for."
    )

    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2017, step=1)

with col2:
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
    fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox(
        "Owner History",
        options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
    )

st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
if st.button("🚀 Calculate Estimated Market Value", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "name": [car_name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
    })

    # Model Prediction
    predicted_price = model.predict(input_data)[0]

    # Calculate Depreciation Metrics
    depreciation_amt = original_price - predicted_price
    depreciation_pct = (depreciation_amt / original_price) * 100 if original_price > 0 else 0
    current_year = datetime.datetime.now().year
    car_age = max(1, current_year - year)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.markdown(
            f"""
            <div class="price-card">
                <h3>Estimated Resale Value</h3>
                <h1>₹ {predicted_price:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        if depreciation_amt > 0:
            st.markdown(
                f"""
                <div class="dep-card">
                    <h4 style="color:#e63946; margin-top:0;">📉 Value Depreciation Summary</h4>
                    <p style="margin:5px 0;"><strong>Original Price:</strong> ₹ {original_price:,.2f}</p>
                    <p style="margin:5px 0;"><strong>Value Retained:</strong> {100 - depreciation_pct:.1f}%</p>
                    <p style="margin:5px 0;"><strong>Total Depreciation:</strong> ₹ {depreciation_amt:,.2f} ({depreciation_pct:.1f}%)</p>
                    <p style="margin:5px 0;"><strong>Est. Annual Loss:</strong> ₹ {depreciation_amt / car_age:,.2f}/yr</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="dep-card" style="border-left-color: #2a9d8f;">
                    <h4 style="color:#2a9d8f; margin-top:0;">📈 Value Appreciation Summary</h4>
                    <p>This vehicle model has retained or gained market value relative to your input original price!</p>
                </div>
                """,
                unsafe_allow_html=True
            )
