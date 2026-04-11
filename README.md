**Car Price Predictor**
Project Description: Car Price Predictor with Machine Learning models.
Interactive web app to estimate used car resale prices using ML-inspired logic.Data Science Essentials project.

Features:
- Predict prices based on 8 car features
- Animated results with depreciation %
- Model comparison (Random Forest best @ 0.96 R²)

**Example:** Maruti Swift 2018 → ₹9.85L → **₹7.25L predicted**

Tech & Logic:

Frontend: Pure HTML/CSS/JS
Prediction: present_price × factors × (1 - penalties)
Factors: Fuel/Trans | Penalties: Age/Kms/Owners
Backend companion: car_price_prediction.py

ML Highlights:

**Feature Importance:**
Present Price 87% | Year 62% | Kms 48% | Fuel 22%
**Pipeline:** Data → Preprocess → Train → Evaluate (Random Forest)
**Metrics:** R² 0.96 | MAE 0.82

Files:

simple_car.html     ← Main app
car_price_prediction.py ← Python ML
car_data.csv        ← Dataset
README.md           ← This file

Built for Data Science learning.

