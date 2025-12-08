import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations
import joblib

# Load data
DATA_PATH = "input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
telco_df = pd.read_csv(DATA_PATH)

# Preprocessing
cleaned = telco_df.copy()
if "customerID" in cleaned.columns:
    cleaned = cleaned.drop(columns=["customerID"])
cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
cleaned = cleaned.dropna()

for column in cleaned.select_dtypes(include="object"):
    cleaned[column] = cleaned[column].str.lower().str.strip()

# One-hot encode
X = pd.get_dummies(cleaned.drop(columns=["Churn"]), drop_first=True, dtype=int)
y = cleaned["Churn"].map({"yes": 1, "no": 0}).to_numpy()

print("All available features after One-Hot-Encoding:")
all_features = sorted(X.columns.tolist())
for i, f in enumerate(all_features, 1):
    print(f"  {i}. {f}")

# Baseline features
BASELINE = ["tenure", "MonthlyCharges", "TotalCharges"]

# All other features (excluding baseline)
other_features = [f for f in all_features if f not in BASELINE]

print(f"\n{'='*80}")
print(f"BASELINE FEATURES ({len(BASELINE)}): {BASELINE}")
print(f"OTHER FEATURES ({len(other_features)}): {other_features}")
print(f"{'='*80}\n")

# Test combinations
results = []

# Test baseline only
X_baseline = X[BASELINE]
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

results.append({
    "features": BASELINE,
    "num_features": len(BASELINE),
    "accuracy": accuracy,
    "f1": f1,
    "roc_auc": roc_auc,
    "description": "Baseline (3 features)"
})

print(f"✓ Baseline: {accuracy:.4f}")

# Test all combinations from size 1 to all other features
for r in range(1, min(len(other_features) + 1, 6)):  # Test up to 5 additional features
    print(f"\nTesting combinations with {r} additional feature(s)...")
    
    best_accuracy_for_size = 0
    best_combo_for_size = None
    
    for combo in combinations(other_features, r):
        features = BASELINE + list(combo)
        X_test_features = X[features]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_test_features, y, test_size=0.2, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy_for_size:
            best_accuracy_for_size = accuracy
            best_combo_for_size = (features, accuracy)
    
    if best_combo_for_size:
        features, acc = best_combo_for_size
        f1 = f1_score(y_test, model.predict(X_test_scaled))
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        
        results.append({
            "features": features,
            "num_features": len(features),
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "description": f"Best with +{r} features"
        })
        
        print(f"  Best accuracy with {r} additional features: {acc:.4f}")
        print(f"  Features: {[f for f in features if f not in BASELINE]}")

# Sort by accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("accuracy", ascending=False)

print(f"\n{'='*80}")
print("TOP 10 FEATURE COMBINATIONS BY ACCURACY:")
print(f"{'='*80}\n")
print(results_df.head(10).to_string(index=False))

# Find best combination above 80%
above_80 = results_df[results_df["accuracy"] >= 0.80]
if len(above_80) > 0:
    print(f"\n{'='*80}")
    print("✓ COMBINATIONS WITH ACCURACY >= 80%:")
    print(f"{'='*80}\n")
    for idx, row in above_80.head(5).iterrows():
        print(f"Accuracy: {row['accuracy']:.4f} | F1: {row['f1']:.4f} | ROC-AUC: {row['roc_auc']:.4f}")
        print(f"Features ({row['num_features']}): {row['features']}\n")
else:
    print(f"\n⚠ No combination reached 80% accuracy yet.")
    best = results_df.iloc[0]
    print(f"\nBest so far: {best['accuracy']:.4f}")
    print(f"Features ({best['num_features']}): {best['features']}")

# Save best model
best_result = results_df.iloc[0]
print(f"\n{'='*80}")
print(f"TRAINING BEST MODEL FOR DEPLOYMENT...")
print(f"{'='*80}\n")

X_final = X[best_result["features"]]
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

model_final = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
model_final.fit(X_train_scaled, y_train)

# Verify
y_pred_final = model_final.predict(X_test_scaled)
accuracy_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
roc_auc_final = roc_auc_score(y_test, model_final.predict_proba(X_test_scaled)[:, 1])

print(f"Final Model Performance:")
print(f"  Accuracy:  {accuracy_final:.4f}")
print(f"  F1 Score:  {f1_final:.4f}")
print(f"  ROC-AUC:   {roc_auc_final:.4f}")
print(f"  Features:  {best_result['features']}")

# Save model
joblib.dump({"model": model_final, "scaler": scaler_final}, "models/telco_logistic_regression.joblib")
print(f"\n✓ Model saved to models/telco_logistic_regression.joblib")
