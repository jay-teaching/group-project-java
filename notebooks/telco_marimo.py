import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    import joblib
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    from datetime import datetime


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 📊 Telco Customer Churn Prediction Model
    
    ## Interactive Model Training & Evaluation
    
    This notebook trains a **Logistic Regression model** to predict customer churn.
    
    **Key Features:**
    - Interactive hyperparameter tuning
    - Data quality validation
    - Comprehensive model evaluation
    - Advanced visualizations
    - Reproducible results
    
    **Instructions:** Adjust the configuration below, then run all cells top-to-bottom.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## ⚙️ Model Configuration
    
    Adjust hyperparameters and settings below to experiment with different configurations.
    """)
    return


@app.cell
def _():
    """Interactive hyperparameter configuration with sliders."""
    # Data Configuration
    DATA_PATH = "input/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Interactive Hyperparameters with Sliders
    mo.md("### Model Hyperparameters")

    c_value_slider = mo.ui.slider(
        value=1.0,
        start=0.001,
        stop=100.0,
        step=0.1,
        label="Regularization Strength (C)",
    )

    max_iter_slider = mo.ui.slider(
        value=1000,
        start=100,
        stop=5000,
        step=100,
        label="Max Iterations",
    )

    test_size_slider = mo.ui.slider(
        value=0.2,
        start=0.1,
        stop=0.4,
        step=0.05,
        label="Test Set Size",
    )

    mo.vstack(
        [
            mo.md("**Regularization (C):** Controls model complexity"),
            c_value_slider,
            mo.md(""),
            mo.md("**Max Iterations:** Convergence threshold"),
            max_iter_slider,
            mo.md(""),
            mo.md("**Test Size:** Fraction of data for testing"),
            test_size_slider,
        ]
    )

    return c_value_slider, max_iter_slider, test_size_slider, DATA_PATH


@app.cell
def _(c_value_slider, max_iter_slider, test_size_slider):
    """Extract slider values and display current configuration."""
    C_VALUE = c_value_slider.value
    MAX_ITER = max_iter_slider.value
    TEST_SIZE = test_size_slider.value

    mo.vstack(
        [
            mo.md("### Current Configuration"),
            mo.md(f"- **Regularization (C):** {C_VALUE:.3f}"),
            mo.md(f"- **Max Iterations:** {MAX_ITER}"),
            mo.md(f"- **Test Size:** {TEST_SIZE:.2f} ({TEST_SIZE * 100:.0f}%)"),
        ]
    )

    # Fixed Configuration
    SOLVER = "lbfgs"
    RANDOM_STATE = 42

    return (
        C_VALUE,
        MAX_ITER,
        SOLVER,
        TEST_SIZE,
        RANDOM_STATE,
    )


@app.cell
def _():
    """Interactive feature selection with checkboxes."""
    mo.md("""
    ### Feature Selection
    
    Choose which features to use for model training. The default selection is based on domain knowledge and statistical analysis.
    """)
    return


@app.cell
def _():
    """Feature selection with checkboxes."""
    # All available features after encoding
    ALL_FEATURES = [
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

    # Create checkboxes for each feature
    feature_checkboxes = {
        feature: mo.ui.checkbox(value=True, label=feature) for feature in ALL_FEATURES
    }

    # Display in columns
    mo.vstack(
        [
            mo.md(
                "**Select features to use (default: all 9 logically-chosen features):**"
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [feature_checkboxes[feature] for feature in ALL_FEATURES[:5]]
                    ),
                    mo.vstack(
                        [feature_checkboxes[feature] for feature in ALL_FEATURES[5:]]
                    ),
                ]
            ),
        ]
    )

    return ALL_FEATURES, feature_checkboxes


@app.cell
def _(ALL_FEATURES, feature_checkboxes):
    """Extract selected features from checkboxes."""
    SELECTED_FEATURES = [
        feature for feature in ALL_FEATURES if feature_checkboxes[feature].value
    ]

    if not SELECTED_FEATURES:
        SELECTED_FEATURES = ALL_FEATURES  # Fallback to all features

    return SELECTED_FEATURES


@app.cell
def _():
    """Export configuration."""
    # Export Configuration
    MODEL_SAVE_PATH = "models/telco_logistic_regression.joblib"
    SAVE_MODEL = True

    return (
        MODEL_SAVE_PATH,
        SAVE_MODEL,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 📥 Load & Explore Data
    """)
    return


@app.cell
def _(DATA_PATH):
    """Load the Telco dataset and display basic information."""
    telco_df = pd.read_csv(DATA_PATH)

    mo.vstack(
        [
            mo.md(
                f"**Dataset Shape:** {telco_df.shape[0]:,} rows × {telco_df.shape[1]} columns"
            ),
            mo.md(
                f"**Memory Usage:** {telco_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            ),
        ]
    )

    return (telco_df,)


@app.cell
def _(telco_df):
    """Display first rows and data types."""
    mo.vstack(
        [
            mo.md("### Data Sample"),
            telco_df.head(10),
            mo.md("### Data Types & Missing Values"),
            pd.DataFrame(
                {
                    "Column": telco_df.columns,
                    "Type": telco_df.dtypes,
                    "Missing": telco_df.isnull().sum(),
                    "Missing %": (telco_df.isnull().sum() / len(telco_df) * 100).round(
                        2
                    ),
                }
            ),
        ]
    )

    return


@app.cell
def _(telco_df):
    """Display target variable distribution."""
    churn_counts = telco_df["Churn"].value_counts()
    churn_pct = (telco_df["Churn"].value_counts(normalize=True) * 100).round(2)

    fig_churn = go.Figure(
        data=[
            go.Bar(
                x=churn_counts.index,
                y=churn_counts.values,
                text=[
                    f"{count}<br>({pct}%)"
                    for count, pct in zip(churn_counts.values, churn_pct.values)
                ],
                textposition="outside",
                marker=dict(color=["#2ecc71", "#e74c3c"]),
            )
        ]
    )
    fig_churn.update_layout(
        title="Target Variable Distribution: Churn",
        xaxis_title="Churn Status",
        yaxis_title="Count",
        showlegend=False,
        height=400,
    )

    return fig_churn


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 🔧 Data Preprocessing & Feature Engineering
    """)
    return


@app.cell
def _(SELECTED_FEATURES, RANDOM_STATE):
    """Preprocess the Telco data with comprehensive documentation."""

    def preprocess_telco(df: pd.DataFrame) -> tuple:
        """
        Preprocess Telco dataset for model training.

        Steps:
        1. Remove non-predictive columns (customerID)
        2. Handle missing/invalid values
        3. Normalize text (lowercase, strip whitespace)
        4. One-hot encode categorical features
        5. Select relevant features
        6. Standardize numerical features

        Args:
            df: Raw Telco dataset

        Returns:
            cleaned_df: Preprocessed dataframe
            X_scaled: Scaled feature matrix
            y: Target variable (churn)
            scaler: StandardScaler object
            feature_names: Selected feature names
        """
        # Create a copy to avoid modifying original
        cleaned = df.copy()

        # Step 1: Remove non-predictive columns
        if "customerID" in cleaned.columns:
            cleaned = cleaned.drop(columns=["customerID"])

        # Step 2: Handle missing values
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )
        initial_rows = len(cleaned)
        cleaned = cleaned.dropna()
        dropped_rows = initial_rows - len(cleaned)

        if dropped_rows > 0:
            print(f"⚠️  Dropped {dropped_rows} rows with missing values")

        # Step 3: Normalize text columns
        for column in cleaned.select_dtypes(include="object"):
            cleaned[column] = cleaned[column].str.lower().str.strip()

        # Step 4: One-hot encode categorical features
        X = pd.get_dummies(cleaned.drop(columns=["Churn"]), drop_first=True, dtype=int)
        y = cleaned["Churn"].map({"yes": 1, "no": 0}).to_numpy()

        # Step 5: Select relevant features
        X = X[SELECTED_FEATURES]

        # Step 6: Standardize (scale) features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return cleaned, X_scaled, y, scaler, X.columns.tolist()

    return (preprocess_telco,)


@app.cell
def _(preprocess_telco, telco_df):
    """Apply preprocessing and display results."""
    cleaned_df, X_scaled, y, scaler, feature_names = preprocess_telco(telco_df)

    mo.vstack(
        [
            mo.md("✅ **Preprocessing Complete**"),
            mo.md(f"- Original shape: {telco_df.shape}"),
            mo.md(f"- Processed shape: {cleaned_df.shape}"),
            mo.md(f"- Selected features: {len(feature_names)}"),
            mo.md(f"- Churn rate: {(y.sum() / len(y) * 100):.1f}%"),
        ]
    )

    return X_scaled, scaler, y, cleaned_df, feature_names


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 🎯 Model Training & Evaluation
    """)
    return


@app.cell
def _(C_VALUE, MAX_ITER, RANDOM_STATE, SOLVER, TEST_SIZE, X_scaled, y):
    """Train Logistic Regression model with detailed metrics."""
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    mo.vstack(
        [
            mo.md("### Train-Test Split"),
            mo.md(f"- Training set: {len(X_train)} samples"),
            mo.md(f"- Test set: {len(X_test)} samples"),
            mo.md(f"- Churn rate (train): {(y_train.sum() / len(y_train) * 100):.1f}%"),
            mo.md(f"- Churn rate (test): {(y_test.sum() / len(y_test) * 100):.1f}%"),
        ]
    )

    # Train model
    model = LogisticRegression(
        solver=SOLVER,
        C=C_VALUE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }

    return metrics, model, X_train, X_test, y_train, y_test, y_pred, y_proba


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 📈 Performance Metrics
    """)
    return


@app.cell
def _(metrics):
    """Display key performance metrics."""
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    roc = metrics["roc_auc"]

    # Create visual metric cards
    fig_metrics = go.Figure(
        data=[
            go.Indicator(
                mode="number+gauge",
                value=acc * 100,
                title="Accuracy",
                domain={"x": [0, 0.33], "y": [0, 1]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkblue"}},
            ),
            go.Indicator(
                mode="number+gauge",
                value=f1 * 100,
                title="F1-Score",
                domain={"x": [0.33, 0.66], "y": [0, 1]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkgreen"}},
            ),
            go.Indicator(
                mode="number+gauge",
                value=roc * 100,
                title="ROC-AUC",
                domain={"x": [0.66, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkred"}},
            ),
        ]
    )
    fig_metrics.update_layout(height=300)

    return fig_metrics


@app.cell
def _(metrics):
    """Display confusion matrix as heatmap."""
    cm = metrics["confusion"]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
    )

    return fig_cm


@app.cell
def _(metrics):
    """Display classification report."""
    mo.vstack(
        [
            mo.md("### Classification Report"),
            mo.md(f"```\n{metrics['report']}\n```"),
        ]
    )

    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 🔍 Advanced Model Analysis
    """)
    return


@app.cell
def _(X_test, y_test, y_proba):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line=dict(color="darkblue", width=3),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        hovermode="closest",
    )

    return fig_roc


@app.cell
def _(model, feature_names):
    """Display model coefficients (feature importance)."""
    coefficients = model.coef_[0]

    # Create dataframe of feature importance
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Abs_Coefficient": np.abs(coefficients),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    fig_coef = go.Figure(
        data=go.Bar(
            y=importance_df["Feature"],
            x=importance_df["Coefficient"],
            orientation="h",
            marker=dict(color=importance_df["Coefficient"], colorscale="RdBu", cmid=0),
        )
    )
    fig_coef.update_layout(
        title="Feature Coefficients (Impact on Churn)",
        xaxis_title="Coefficient Value",
        yaxis_title="Feature",
        height=400,
        showlegend=False,
    )

    return importance_df, fig_coef


@app.cell
def _(model, X_train, y_train, X_test, y_test):
    """Perform cross-validation to assess model stability."""
    cv_results = cross_validate(
        model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"]
    )

    cv_summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1-Score", "ROC-AUC"],
            "Mean": [
                cv_results["test_accuracy"].mean(),
                cv_results["test_f1"].mean(),
                cv_results["test_roc_auc"].mean(),
            ],
            "Std Dev": [
                cv_results["test_accuracy"].std(),
                cv_results["test_f1"].std(),
                cv_results["test_roc_auc"].std(),
            ],
        }
    ).round(4)

    mo.vstack(
        [
            mo.md("### 5-Fold Cross-Validation Results"),
            mo.md("Model stability and generalization performance:"),
            cv_summary,
        ]
    )

    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 💾 Model Export & Reproducibility
    """)
    return


@app.cell
def _(MODEL_SAVE_PATH, SAVE_MODEL, model, scaler):
    """Save model and scaler for production use."""
    if SAVE_MODEL:
        # Save model and scaler together
        joblib.dump({"model": model, "scaler": scaler}, MODEL_SAVE_PATH)

        import os

        file_size = os.path.getsize(MODEL_SAVE_PATH) / 1024

        mo.vstack(
            [
                mo.md("✅ **Model Saved Successfully**"),
                mo.md(f"- Path: `{MODEL_SAVE_PATH}`"),
                mo.md(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                mo.md(f"- Size: {file_size:.2f} KB"),
            ]
        )
    else:
        mo.md("⚠️ Model saving is disabled (SAVE_MODEL=False)")

    return


@app.cell
def _(SELECTED_FEATURES, SOLVER, MAX_ITER, C_VALUE, TEST_SIZE):
    """Display model configuration for reproducibility."""
    mo.vstack(
        [
            mo.md("### Model Configuration Summary"),
            mo.md("**Algorithm:** Logistic Regression"),
            mo.md(f"**Solver:** {SOLVER}"),
            mo.md(f"**Max Iterations:** {MAX_ITER}"),
            mo.md(f"**Regularization (C):** {C_VALUE}"),
            mo.md(f"**Test Split:** {TEST_SIZE * 100:.0f}%"),
            mo.md("**Random State:** 42 (for reproducibility)"),
            mo.md(f"**Features Used:** {len(SELECTED_FEATURES)}"),
            mo.md("**Selected Features:**"),
            mo.md(
                "\n".join(
                    [f"- {i + 1}. `{f}`" for i, f in enumerate(SELECTED_FEATURES)]
                )
            ),
        ]
    )

    return


if __name__ == "__main__":
    app.run()
