from google.colab import files

uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get the filename automatically
filename = list(uploaded.keys())[0]

# Read into a DataFrame
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Step 4: Verify it loaded correctly
print("Shape:", df.shape)
print(df.head())



print("\n── Column Info ──")
print(df.info())
 
print("\n── Basic Statistics ──")
print(df.describe())
 
print("\n── Missing Values ──")
print(df.isnull().sum())
 
print("\n── Column Names ──")
print(df.columns.tolist())


# ============================================================
# PHASE 3 ── Exploratory Data Analysis (EDA)
# ============================================================

# ── 1. Distribution of Selling Price ──
plt.figure(figsize=(8, 4))
sns.histplot(df['selling_price'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Car selling Price')
plt.xlabel('selling Price (in lakhs)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot_1_price_distribution.png')
plt.show()

# ── 2. Selling Price by Fuel Type ──
plt.figure(figsize=(7, 4))
sns.boxplot(x='fuel', y='selling_price', data=df, palette='Set2')
plt.title('Selling Price by Fuel Type')
plt.tight_layout()
plt.savefig('plot_2_price_by_fuel.png')
plt.show()

# ── 3. Year vs Selling Price ──
plt.figure(figsize=(8, 4))
sns.scatterplot(x='year', y='selling_price', data=df,
                hue='fuel', alpha=0.7)
plt.title('Year vs Selling Price')
plt.tight_layout()
plt.savefig('plot_3_year_vs_price.png')
plt.show()

# ── 4. Driven (KMs) vs Selling Price ──
plt.figure(figsize=(8, 4))
sns.scatterplot(x='km_driven', y='selling_price', data=df,
                alpha=0.5, color='coral')
plt.title('Kms Driven vs Selling Price')
plt.tight_layout()
plt.savefig('plot_4_kms_vs_price.png')
plt.show()

# ── 5. Correlation Heatmap (numeric columns only) ──
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_5_heatmap.png')
plt.show()

# ============================================================
# PHASE 3 ── Exploratory Data Analysis (EDA)
# ============================================================

# ── 1. Distribution of Selling Price ──
plt.figure(figsize=(8, 4))
sns.histplot(df['selling_price'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Car selling Price')
plt.xlabel('selling Price (in lakhs)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot_1_price_distribution.png')
plt.show()

# ── 2. Selling Price by Fuel Type ──
plt.figure(figsize=(7, 4))
sns.boxplot(x='fuel', y='selling_price', data=df, palette='Set2')
plt.title('Selling Price by Fuel Type')
plt.tight_layout()
plt.savefig('plot_2_price_by_fuel.png')
plt.show()

# ── 3. Year vs Selling Price ──
plt.figure(figsize=(8, 4))
sns.scatterplot(x='year', y='selling_price', data=df,
                hue='fuel', alpha=0.7)
plt.title('Year vs Selling Price')
plt.tight_layout()
plt.savefig('plot_3_year_vs_price.png')
plt.show()

# ── 4. Driven (KMs) vs Selling Price ──
plt.figure(figsize=(8, 4))
sns.scatterplot(x='km_driven', y='selling_price', data=df,
                alpha=0.5, color='coral')
plt.title('Kms Driven vs Selling Price')
plt.tight_layout()
plt.savefig('plot_4_kms_vs_price.png')
plt.show()

# ── 5. Correlation Heatmap (numeric columns only) ──
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_5_heatmap.png')
plt.show() 


# ============================================================
# PHASE 4 ── Data Preprocessing
# ============================================================

# ── Drop unnecessary columns (if any) ──
# Uncomment below if your dataset has a 'Car_Name' column
# df.drop(columns=['Car_Name'], inplace=True)

# ── Handle missing values ──
df.dropna(inplace=True)   # Drop rows with any missing value
# OR fill with median: df.fillna(df.median(numeric_only=True), inplace=True)

# ── Encode categorical columns ──
le = LabelEncoder()

for col in ['fuel', 'seller_type', 'transmission']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

print("\nAfter encoding:")
print(df.head())

# ── Define features (X) and target (y) ──
# Adjust column names to match your dataset
X = df.drop(columns=['selling_price', 'name', 'owner'])
y = df['selling_price']

# ── Feature Scaling ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train-Test Split (80% train, 20% test) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# ============================================================
# PHASE 5 ── Model Building
# ============================================================

# ── Model 1: Linear Regression ──
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ── Model 2: Decision Tree ──
dt = DecisionTreeRegressor(random_state=42, max_depth=6)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# ── Model 3: Random Forest ──
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nAll 3 models trained successfully!")


# ============================================================
# PHASE 6 ── Model Evaluation
# ============================================================

def evaluate_model(name, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n── {name} ──")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []
results.append(evaluate_model("Linear Regression",  y_test, y_pred_lr))
results.append(evaluate_model("Decision Tree",       y_test, y_pred_dt))
results.append(evaluate_model("Random Forest",       y_test, y_pred_rf))

# ── Results Table ──
results_df = pd.DataFrame(results)
print("\n── Model Comparison Table ──")
print(results_df.to_string(index=False))

# ── Bar Chart: R² Score Comparison ──
plt.figure(figsize=(7, 4))
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plot_6_r2_comparison.png')
plt.show()

# ── Actual vs Predicted (Random Forest) ──
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.savefig('plot_7_actual_vs_predicted.png')
plt.show()

# ── Feature Importance (Random Forest) ──
feature_names = X.columns # Correctly get feature names from the X DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df,
            palette='magma')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('plot_8_feature_importance.png')
plt.show()

print("\n── Feature Importance ──")
print(importance_df.to_string(index=False))


# ============================================================
# PHASE 7 ── Predict a New Car Price
# ============================================================

# Example: predict price for a car with custom input
# Make sure features match your training columns exactly
sample_input = pd.DataFrame([{    # Ensure column names match X exactly
    'year':         2017,
    'km_driven':   35000,
    'fuel':    1,     # 1 = Petrol (after encoding)
    'seller_type':  0,     # 0 = Dealer
    'transmission': 1      # 1 = Manual
}])

sample_scaled = scaler.transform(sample_input)
predicted_price = rf.predict(sample_scaled)
print(f"\nPredicted Selling Price: ₹ {predicted_price[0]:.2f}")


# ============================================================
# PHASE 8 ── Conclusion
# ============================================================
 
print("""
╔══════════════════════════════════════════════╗
║           PROJECT SUMMARY                   ║
╠══════════════════════════════════════════════╣
║ Dataset   : Car Price Prediction (Kaggle)   ║
║ Target    : Selling_Price                   ║
║ Best Model: Random Forest Regressor         ║
║ Key Features: Year, Present_Price,          ║
║               Kms_Driven, Fuel_Type         ║
╚══════════════════════════════════════════════╝
""")