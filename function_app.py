import azure.functions as func
import logging
from prediction import make_prediction

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="predict")
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get parameters from query string
    tenure = req.params.get('tenure')
    monthly_charges = req.params.get('MonthlyCharges')
    tech_support = req.params.get('TechSupport_yes')
    contract_one_year = req.params.get('Contract_one_year')
    contract_two_year = req.params.get('Contract_two_year')
    total_charges = req.params.get('TotalCharges')
    partner = req.params.get('Partner_yes')
    streaming_tv = req.params.get('StreamingTV_yes')
    streaming_no_internet = req.params.get('StreamingTV_no_internet_service')

    # Validate required parameters
    if not all([tenure, monthly_charges, tech_support, contract_one_year, 
                contract_two_year, total_charges, partner, streaming_tv, 
                streaming_no_internet]):
        return func.HttpResponse(
            "Please provide all required parameters: tenure, MonthlyCharges, TechSupport_yes, "
            "Contract_one_year, Contract_two_year, TotalCharges, Partner_yes, "
            "StreamingTV_yes, StreamingTV_no_internet_service",
            status_code=400
        )

    try:
        # Convert to appropriate types and make prediction
        prediction = make_prediction(
            tenure=float(tenure),
            MonthlyCharges=float(monthly_charges),
            TechSupport_yes=int(tech_support),
            Contract_one_year=int(contract_one_year),
            Contract_two_year=int(contract_two_year),
            TotalCharges=float(total_charges),
            Partner_yes=int(partner),
            StreamingTV_yes=int(streaming_tv),
            StreamingTV_no_internet_service=int(streaming_no_internet)
        )
        
        return func.HttpResponse(
            f"Churn probability: {prediction:.4f}",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return func.HttpResponse(
            f"Error making prediction: {str(e)}",
            status_code=500
        )
