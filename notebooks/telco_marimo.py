import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import joblib
    import marimo as mo
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Telco churn – baseline logistic regression

    Edit the constants below, run the notebook top-to-bottom, and inspect the metrics.
    """)
    return


@app.cell
def _():
    DATA_PATH = "input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    MAX_ITER = 1000
    MODEL_SAVE_PATH = "models/telco_logistic_regression.joblib"
    SAVE_MODEL = True

    # Features configuration
    BASELINE_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
    EXTENDED_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService_fiber optic', 'MultipleLines_no phone service', 'MultipleLines_yes', 'PaymentMethod_electronic check', 'TechSupport_yes', "Contract_one_year",
        "Contract_two_year",
        "StreamingTV_yes",
        "StreamingTV_no_internet_service"]
        

    SOLVER = "lbfgs"
    TEST_SIZE = 0.2
    C_VALUE = 1.0
    return (
        BASELINE_FEATURES,
        C_VALUE,
        DATA_PATH,
        EXTENDED_FEATURES,
        MAX_ITER,
        MODEL_SAVE_PATH,
        SAVE_MODEL,
        SOLVER,
        TEST_SIZE,
    )


@app.cell
def _(DATA_PATH):
    telco_df = pd.read_csv(DATA_PATH)
    telco_df.head()
    return (telco_df,)


@app.function
def preprocess_telco(df: pd.DataFrame, selected_features: list):
    cleaned = df.copy()
    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"], errors="coerce"
    )
    cleaned = cleaned.dropna()

    for column in cleaned.select_dtypes(include="object"):
        cleaned[column] = cleaned[column].str.lower().str.strip()

    # One-hot encode categorical features
    X = pd.get_dummies(cleaned.drop(columns=["Churn"]), drop_first=True, dtype=int)

    print(f"Available features after encoding: {X.columns.tolist()}")
    print(f"Selected features for modeling: {selected_features}")

    # Choose only the selected features that exist
    available_features = [f for f in selected_features if f in X.columns]
    missing_features = [f for f in selected_features if f not in X.columns]

    if missing_features:
        print(f"Warning: Missing features {missing_features}")

    X = X[available_features]
    y = cleaned["Churn"].map({"yes": 1, "no": 0}).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return cleaned, X_scaled, y, scaler, X.columns.tolist()


@app.cell
def _(BASELINE_FEATURES, EXTENDED_FEATURES, telco_df):
    # Preprocess data for both feature sets
    cleaned_df, X_baseline, y, scaler_baseline, baseline_feature_names = preprocess_telco(telco_df, BASELINE_FEATURES)
    cleaned_df, X_extended, y, scaler_extended, extended_feature_names = preprocess_telco(telco_df, EXTENDED_FEATURES)

    print(f"\nBaseline features ({len(baseline_feature_names)}): {baseline_feature_names}")
    print(f"Extended features ({len(extended_feature_names)}): {extended_feature_names}")
    return X_baseline, X_extended, scaler_extended, y


@app.cell
def _(C_VALUE, MAX_ITER, SOLVER, TEST_SIZE, X_baseline, X_extended, y):
    # Train baseline model (3 features)
    X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
        X_baseline,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )

    model_baseline = LogisticRegression(
        solver=SOLVER, C=C_VALUE, max_iter=MAX_ITER, random_state=42
    )
    model_baseline.fit(X_train_baseline, y_train)

    y_pred_baseline = model_baseline.predict(X_test_baseline)
    y_proba_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]

    metrics_baseline = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "f1": f1_score(y_test, y_pred_baseline),
        "roc_auc": roc_auc_score(y_test, y_proba_baseline),
        "confusion": confusion_matrix(y_test, y_pred_baseline),
    }

    # Train extended model (9 features)
    X_train_extended, X_test_extended, _, _ = train_test_split(
        X_extended,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )

    model_extended = LogisticRegression(
        solver=SOLVER, C=C_VALUE, max_iter=MAX_ITER, random_state=42
    )
    model_extended.fit(X_train_extended, y_train)

    y_pred_extended = model_extended.predict(X_test_extended)
    y_proba_extended = model_extended.predict_proba(X_test_extended)[:, 1]

    metrics_extended = {
        "accuracy": accuracy_score(y_test, y_pred_extended),
        "f1": f1_score(y_test, y_pred_extended),
        "roc_auc": roc_auc_score(y_test, y_proba_extended),
        "confusion": confusion_matrix(y_test, y_pred_extended),
    }
    return metrics_baseline, metrics_extended, model_extended


@app.cell(hide_code=True)
def _(metrics_baseline, metrics_extended):
    # Create comparison table
    comparison_data = {
        "Metric": ["Accuracy", "F1 Score", "ROC AUC"],
        "Baseline (3 Features)": [
            f"{metrics_baseline['accuracy']:.4f}",
            f"{metrics_baseline['f1']:.4f}",
            f"{metrics_baseline['roc_auc']:.4f}",
        ],
        "Extended (9 Features)": [
            f"{metrics_extended['accuracy']:.4f}",
            f"{metrics_extended['f1']:.4f}",
            f"{metrics_extended['roc_auc']:.4f}",
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)

    mo.md("""
    ## Model Performance Comparison
    """)
    return (comparison_df,)


@app.cell
def _(comparison_df):
    comparison_df
    return


@app.cell
def _(MODEL_SAVE_PATH, SAVE_MODEL, model_extended, scaler_extended):
    if SAVE_MODEL:
        # Save the extended model (with 9 features) for production
        joblib.dump({"model": model_extended, "scaler": scaler_extended}, MODEL_SAVE_PATH)
        print(f"Extended model saved to {MODEL_SAVE_PATH}")
    return


if __name__ == "__main__":
    app.run()
