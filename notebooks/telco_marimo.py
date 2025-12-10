import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    import joblib
    import marimo as mo
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
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
    SELECTED_FEATURES = [
        "tenure",
        "MonthlyCharges",
        "TechSupport_yes",
        "Contract_one year",
        "Contract_two year",
        "TotalCharges",
        "Partner_yes",
        "StreamingTV_yes",
        "StreamingTV_no internet service",
    ]
    SOLVER = "lbfgs"
    TEST_SIZE = 0.2
    C_VALUE = 1.0
    return (
        C_VALUE,
        DATA_PATH,
        MAX_ITER,
        MODEL_SAVE_PATH,
        SAVE_MODEL,
        SELECTED_FEATURES,
        SOLVER,
        TEST_SIZE,
    )


@app.cell
def _(DATA_PATH):
    telco_df = pd.read_csv(DATA_PATH)
    telco_df.head()
    return (telco_df,)


@app.cell
def _(SELECTED_FEATURES):
    def preprocess_telco(df: pd.DataFrame):
        cleaned = df.copy()
        if "customerID" in cleaned.columns:
            cleaned = cleaned.drop(columns=["customerID"])
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )
        cleaned = cleaned.dropna()

        for column in cleaned.select_dtypes(include="object"):
            cleaned[column] = cleaned[column].str.lower().str.strip()

        X = pd.get_dummies(cleaned.drop(columns=["Churn"]), drop_first=True, dtype=int)

        print("Available features after encoding:", X.columns.tolist())
        print("Selected features for modeling:", SELECTED_FEATURES)

        # Choose features
        X = X[SELECTED_FEATURES]
        y = cleaned["Churn"].map({"yes": 1, "no": 0}).to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return cleaned, X_scaled, y, scaler, X.columns.tolist()

    return (preprocess_telco,)


@app.cell
def _(preprocess_telco, telco_df):
    cleaned_df, X_scaled, y, scaler, feature_names = preprocess_telco(telco_df)
    print("Chosen features:", feature_names)
    cleaned_df.head()
    return X_scaled, scaler, y


@app.cell
def _(C_VALUE, MAX_ITER, SOLVER, TEST_SIZE, X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )

    model = LogisticRegression(
        solver=SOLVER, C=C_VALUE, max_iter=MAX_ITER, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    return metrics, model


@app.cell
def _(metrics):
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    roc = metrics["roc_auc"]

    mo.vstack(
        [
            mo.md("## Performance Summary"),
            mo.md(
                f"""- Accuracy: {acc:.3f}
    - F1 Score: {f1:.3f}
    - ROC AUC: {roc:.3f}
    """
            ),
        ]
    )
    return


@app.cell
def _(metrics):
    cm = metrics["confusion"]

    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(["Actual: No churn", "Actual: Churn"]),
        columns=pd.Index(["Pred: No churn", "Pred: Churn"]),
    )

    mo.vstack(
        [
            mo.md("## Confusion Matrix"),
            cm_df,
        ]
    )
    return


@app.cell
def _(metrics):
    print(metrics["report"])
    return


@app.cell
def _(MODEL_SAVE_PATH, SAVE_MODEL, model, scaler):
    if SAVE_MODEL:
        joblib.dump({"model": model, "scaler": scaler}, MODEL_SAVE_PATH)
    return


if __name__ == "__main__":
    app.run()
