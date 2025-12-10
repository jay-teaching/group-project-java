import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from datetime import datetime

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================
print("Loading and preprocessing data...")
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

all_features = sorted(X.columns.tolist())
print(f"Total features available: {len(all_features)}")
print(f"Dataset shape: {X.shape}")
print()

# ============================================================================
# STEP 1: TRAIN RANDOM FOREST AND GET FEATURE IMPORTANCES
# ============================================================================
print("=" * 80)
print("STEP 1: RANDOM FOREST - FEATURE IMPORTANCE RANKING")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importance_df = pd.DataFrame(
    {"feature": all_features, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\nRandom Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20).to_string(index=False))
print()

# ============================================================================
# STEP 2: TEST MODELS WITH TOP K FEATURES
# ============================================================================
print("=" * 80)
print("STEP 2: CROSS-VALIDATION WITH TOP K FEATURES")
print("=" * 80)

k_values = [5, 10, 15, 20]
cv_results = []

for k in k_values:
    print(f"\n--- Testing with Top {k} Features ---")
    top_k_features = feature_importance_df.head(k)["feature"].tolist()

    print(f"Features: {top_k_features}")

    X_k = X[top_k_features]

    # Standardize
    scaler_k = StandardScaler()
    X_k_scaled = scaler_k.fit_transform(X_k)

    # Cross-validation with Logistic Regression
    lr_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

    # Multiple metrics
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

    cv_scores = cross_validate(lr_model, X_k_scaled, y, cv=5, scoring=scoring)

    mean_accuracy = cv_scores["test_accuracy"].mean()
    mean_f1 = cv_scores["test_f1"].mean()
    mean_roc_auc = cv_scores["test_roc_auc"].mean()

    std_accuracy = cv_scores["test_accuracy"].std()
    std_f1 = cv_scores["test_f1"].std()
    std_roc_auc = cv_scores["test_roc_auc"].std()

    print("Cross-Validation Results (5-fold):")
    print(f"  Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  F1 Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  ROC-AUC:   {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")

    cv_results.append(
        {
            "k": k,
            "features": top_k_features,
            "num_features": k,
            "accuracy_mean": mean_accuracy,
            "accuracy_std": std_accuracy,
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "roc_auc_mean": mean_roc_auc,
            "roc_auc_std": std_roc_auc,
        }
    )

# ============================================================================
# STEP 3: TRAIN FINAL MODEL WITH BEST K
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: SELECT BEST K AND TRAIN FINAL MODEL")
print("=" * 80)

# Find best k by accuracy
best_cv = max(cv_results, key=lambda x: x["accuracy_mean"])
best_k = best_cv["k"]
best_features = best_cv["features"]

print(f"\nBest K: {best_k}")
print(f"Features: {best_features}")
print(f"Cross-validation Accuracy: {best_cv['accuracy_mean']:.4f}")

# Train final model
X_final = X[best_features]
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)

final_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
final_model.fit(X_train_final_scaled, y_train_final)

y_pred_final = final_model.predict(X_test_final_scaled)
y_proba_final = final_model.predict_proba(X_test_final_scaled)[:, 1]

final_accuracy = accuracy_score(y_test_final, y_pred_final)
final_f1 = f1_score(y_test_final, y_pred_final)
final_roc_auc = roc_auc_score(y_test_final, y_proba_final)
final_cm = confusion_matrix(y_test_final, y_pred_final)
final_report = classification_report(y_test_final, y_pred_final)

print("\nFinal Model Performance (Test Set):")
print(f"  Accuracy:  {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"  ROC-AUC:   {final_roc_auc:.4f}")

# ============================================================================
# STEP 4: OPTIONAL - RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RECURSIVE FEATURE ELIMINATION (RFE) REFINEMENT")
print("=" * 80)

print(f"\nPerforming RFE with top {best_k} features...")
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(
    X[best_features], y, test_size=0.2, stratify=y, random_state=42
)

lr_base = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
rfe = RFE(lr_base, n_features_to_select=best_k, step=1)
X_train_rfe_selected = rfe.fit_transform(X_train_rfe, y_train_rfe)
X_test_rfe_selected = rfe.transform(X_test_rfe)

rfe_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
rfe_model.fit(X_train_rfe_selected, y_train_rfe)
y_pred_rfe = rfe_model.predict(X_test_rfe_selected)

rfe_accuracy = accuracy_score(y_test_rfe, y_pred_rfe)
rfe_f1 = f1_score(y_test_rfe, y_pred_rfe)
rfe_roc_auc = roc_auc_score(
    y_test_rfe, rfe_model.predict_proba(X_test_rfe_selected)[:, 1]
)

rfe_selected_features = [
    feat for feat, selected in zip(best_features, rfe.support_) if selected
]

print(f"RFE Selected Features ({len(rfe_selected_features)}): {rfe_selected_features}")
print("RFE Model Performance:")
print(f"  Accuracy:  {rfe_accuracy:.4f} ({rfe_accuracy * 100:.2f}%)")
print(f"  F1 Score:  {rfe_f1:.4f}")
print(f"  ROC-AUC:   {rfe_roc_auc:.4f}")

# ============================================================================
# GENERATE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("TELCO CHURN PREDICTION - FEATURE IMPORTANCE SELECTION REPORT")
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("METHODOLOGY:")
report_lines.append("1. Train Random Forest to get feature importances")
report_lines.append("2. Test Logistic Regression with Top 5, 10, 15, 20 features")
report_lines.append("3. Use 5-fold cross-validation for reliability")
report_lines.append("4. Select best K by accuracy")
report_lines.append("5. Train final model on best features")
report_lines.append("6. Optional: Apply RFE for feature refinement")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("RANDOM FOREST - TOP 20 FEATURE IMPORTANCES")
report_lines.append("=" * 80)
report_lines.append("")

for count, (idx, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
    report_lines.append(
        f"{count:2d}. {row['feature']:40s} -> {row['importance']:.4f}"
    )
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("CROSS-VALIDATION RESULTS - TOP K FEATURES")
report_lines.append("=" * 80)
report_lines.append("")

for result in cv_results:
    report_lines.append(f"K = {result['k']} Features:")
    report_lines.append(
        f"  Accuracy:  {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}"
    )
    report_lines.append(
        f"  F1 Score:  {result['f1_mean']:.4f} ± {result['f1_std']:.4f}"
    )
    report_lines.append(
        f"  ROC-AUC:   {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f}"
    )
    report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("SELECTED BEST MODEL")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(f"Best K Selected: {best_k}")
report_lines.append(f"Number of Features: {len(best_features)}")
report_lines.append("")

report_lines.append("Features:")
for feat in best_features:
    report_lines.append(f"  - {feat}")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("FINAL MODEL PERFORMANCE (TEST SET)")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(f"Accuracy:  {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
report_lines.append(f"F1 Score:  {final_f1:.4f}")
report_lines.append(f"ROC-AUC:   {final_roc_auc:.4f}")
report_lines.append("")

report_lines.append("Confusion Matrix:")
report_lines.append(str(final_cm))
report_lines.append("")

report_lines.append("Classification Report:")
report_lines.append(final_report)
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("RFE REFINEMENT RESULTS")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(f"RFE Selected Features ({len(rfe_selected_features)}):")
for feat in rfe_selected_features:
    report_lines.append(f"  - {feat}")
report_lines.append("")

report_lines.append("RFE Model Performance:")
report_lines.append(f"  Accuracy:  {rfe_accuracy:.4f} ({rfe_accuracy * 100:.2f}%)")
report_lines.append(f"  F1 Score:  {rfe_f1:.4f}")
report_lines.append(f"  ROC-AUC:   {rfe_roc_auc:.4f}")
report_lines.append("")

report_lines.append("=" * 80)

# Save report
report_text = "\n".join(report_lines)
with open("feature_importance_output.txt", "w") as f:
    f.write(report_text)

print("\n✓ Report saved to feature_importance_output.txt")
print(f"\n{report_text}")
