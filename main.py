from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Telco Churn Prediction API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
MODEL_PATH = Path("models/telco_logistic_regression.joblib")
model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    try:
        data = joblib.load(MODEL_PATH)
        model = data["model"]
        scaler = data["scaler"]
    Contract_one_year: int  # 0 or 1
    Contract_two_year: int  # 0 or 1
    TotalCharges: float
    Partner_yes: int  # 0 or 1
    StreamingTV_yes: int  # 0 or 1
    StreamingTV_no_internet_service: int  # 0 or 1

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: str
    confidence: float

@app.get("/")
def read_root():
    return {
        "message": "Telco Churn Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(input_data: PredictionInput) -> PredictionOutput:
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare features in correct order
    features = np.array([
        input_data.tenure,
        input_data.MonthlyCharges,
        input_data.TechSupport_yes,
        input_data.Contract_one_year,
        input_data.Contract_two_year,
        input_data.TotalCharges,
        input_data.Partner_yes,
        input_data.StreamingTV_yes,
        input_data.StreamingTV_no_internet_service
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    churn_proba = model.predict_proba(features_scaled)[0, 1]
    churn_pred = model.predict(features_scaled)[0]
    
    return PredictionOutput(
        churn_probability=float(churn_proba),
        churn_prediction="Yes" if churn_pred == 1 else "No",
        confidence=float(max(churn_proba, 1 - churn_proba))
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)