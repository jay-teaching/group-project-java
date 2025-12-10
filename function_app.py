import azure.functions as func
import logging
import json
from prediction import make_prediction

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="predict")
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # Get parameters from query string
    tenure = req.params.get("tenure")
    monthly_charges = req.params.get("MonthlyCharges")
    tech_support = req.params.get("TechSupport_yes")
    contract_one_year = req.params.get("Contract_one_year")
    contract_two_year = req.params.get("Contract_two_year")
    total_charges = req.params.get("TotalCharges")
    partner = req.params.get("Partner_yes")
    streaming_tv = req.params.get("StreamingTV_yes")
    streaming_no_internet = req.params.get("StreamingTV_no_internet_service")

    # Validate required parameters
    if not all(
        [
            tenure,
            monthly_charges,
            tech_support,
            contract_one_year,
            contract_two_year,
            total_charges,
            partner,
            streaming_tv,
            streaming_no_internet,
        ]
    ):
        return func.HttpResponse(
            json.dumps({
                "error": "Please provide all required parameters: tenure, MonthlyCharges, TechSupport_yes, "
                "Contract_one_year, Contract_two_year, TotalCharges, Partner_yes, "
                "StreamingTV_yes, StreamingTV_no_internet_service"
            }),
            status_code=400,
            mimetype="application/json"
        )

    try:
        # Convert to appropriate types and make prediction
        churn_prob = make_prediction(
            tenure=float(tenure),
            MonthlyCharges=float(monthly_charges),
            TechSupport_yes=int(tech_support),
            Contract_one_year=int(contract_one_year),
            Contract_two_year=int(contract_two_year),
            TotalCharges=float(total_charges),
            Partner_yes=int(partner),
            StreamingTV_yes=int(streaming_tv),
            StreamingTV_no_internet_service=int(streaming_no_internet),
        )

        # Determine churn prediction (Yes/No) based on threshold
        churn_prediction = "Yes" if churn_prob > 0.5 else "No"
        
        # Calculate confidence (distance from 0.5 threshold)
        confidence = abs(churn_prob - 0.5) * 2  # Scale to 0-1 range

        # Return JSON response matching Streamlit app expectations
        response_data = {
            "churn_probability": churn_prob,
            "churn_prediction": churn_prediction,
            "confidence": confidence
        }

        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Error making prediction: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

