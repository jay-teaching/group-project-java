"""Script to retrain the model with underscored feature names."""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
DATA_PATH = Path("input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_SAVE_PATH = Path("models/telco_logistic_regression.joblib")
SELECTED_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TechSupport_yes",
    "Contract_one_year",
    "Contract_two_year",
    "TotalCharges",
    "Partner_yes",
    "StreamingTV_yes",
    "StreamingTV_no_internet_service"
]

print("Loading data...")
telco_df = pd.read_csv(DATA_PATH)

print("Preprocessing data...")
cleaned = telco_df.copy()
if "customerID" in cleaned.columns:
    cleaned = cleaned.drop(columns=["customerID"])

cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
cleaned = cleaned.dropna()

for column in cleaned.select_dtypes(include="object"):
    cleaned[column] = cleaned[column].str.lower().str.strip()

X = pd.get_dummies(cleaned.drop(columns=["Churn"]), drop_first=True, dtype=int)

# Rename columns to use underscores instead of spaces
X.columns = X.columns.str.replace(' ', '_')

print(f"Available features: {X.columns.tolist()}")
print(f"Selected features: {SELECTED_FEATURES}")

# Select features
X = X[SELECTED_FEATURES]
y = cleaned["Churn"].map({"yes": 1, "no": 0}).to_numpy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, stratify=y, random_state=42
)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model
print(f"Saving model to {MODEL_SAVE_PATH}...")
joblib.dump({"model": model, "scaler": scaler}, MODEL_SAVE_PATH)

print("✓ Model retrained and saved successfully!")
print(f"✓ Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"✓ Test accuracy: {model.score(X_test, y_test):.4f}")
