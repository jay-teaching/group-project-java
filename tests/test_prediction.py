import prediction


def test_make_prediction_simple():
    """A simple test to check if make_prediction returns a float.

    This test must be modified as per the actual model used.

    """
    # Use dictionary unpacking to pass features with spaces in their names
    features = {
        "tenure": 2,
        "MonthlyCharges": 12.3,
        "TechSupport_yes": 0,
        "Contract_one year": 1,
        "Contract_two year": 0,
        "TotalCharges": 24.6,
        "Partner_yes": 1,
        "StreamingTV_yes": 0,
        "StreamingTV_no internet service": 0
    }
    result = prediction.make_prediction(**features)
    assert isinstance(result, float)
