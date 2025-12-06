import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Title and description
st.title("📊 Telco Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Information")
    
    # Numeric inputs
    with st.form("prediction_form"):
        st.markdown("### Basic Information")
        
        col_a, col_b = st.columns(2)
        with col_a:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=24,
                help="Number of months the customer has stayed with the company"
            )
            
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=65.0,
                step=0.01,
                help="The amount charged to the customer monthly"
            )
        
        with col_b:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=1500.0,
                step=0.01,
                help="Total amount charged to the customer"
            )
        
        st.markdown("### Services & Features")
        
        col_c, col_d = st.columns(2)
        with col_c:
            tech_support = st.checkbox("Has Tech Support", value=False)
            partner = st.checkbox("Has Partner", value=True)
            streaming_tv = st.checkbox("Streaming TV Service", value=False)
        
        with col_d:
            no_internet = st.checkbox("No Internet Service", value=False)
        
        st.markdown("### Contract Type")
        contract_type = st.radio(
            "Select contract type",
            ["Month-to-month", "One year", "Two year"],
            horizontal=True
        )
        
        # Submit button
        submit_button = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    
    if submit_button:
        # Prepare the payload
        contract_one_year = 1 if contract_type == "One year" else 0
        contract_two_year = 1 if contract_type == "Two year" else 0
        
        payload = {
            "tenure": float(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TechSupport_yes": int(tech_support),
            "Contract_one_year": contract_one_year,
            "Contract_two_year": contract_two_year,
            "TotalCharges": float(total_charges),
            "Partner_yes": int(partner),
            "StreamingTV_yes": int(streaming_tv),
            "StreamingTV_no_internet_service": int(no_internet)
        }
        
        try:
            # Make API request
            with st.spinner("Making prediction..."):
                response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display prediction
                churn_prob = result["churn_probability"]
                churn_pred = result["churn_prediction"]
                confidence = result["confidence"]
                
                # Color-coded result
                if churn_pred == "Yes":
                    st.error("⚠️ **High Churn Risk**")
                    st.markdown("This customer is likely to churn. Consider retention strategies.")
                else:
                    st.success("✅ **Low Churn Risk**")
                    st.markdown("This customer is likely to stay. Focus on maintaining satisfaction.")
                
                # Metrics
                st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
                st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                
                # Progress bar
                st.progress(churn_prob)
                
            else:
                st.error(f"API Error: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API")
            st.info("Make sure the FastAPI backend is running at http://localhost:8000")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application predicts customer churn for a telecommunications company using a 
    logistic regression model trained on the IBM Telco Customer Churn dataset.
    
    ### Model Features
    - Tenure
    - Monthly Charges
    - Total Charges
    - Tech Support
    - Contract Type
    - Partner Status
    - Streaming TV
    - Internet Service
    
    ### API Status
    """)
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.markdown("**Model Accuracy:** ~77.5%")
