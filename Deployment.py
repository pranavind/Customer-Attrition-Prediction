import streamlit as st
from joblib import load
import numpy as np
import warnings

# Load the trained Logistic Regression model
model = load(r"C:\\Users\\HP\\OneDrive\Desktop\\ml project\\logistic_regression_model.joblib")

# Create a Streamlit app
st.title("Customer Churn Prediction App")

# Input fields for feature values on the main screen
st.header("Enter Customer Information")
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=20000, value=50)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)
payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
online_security = st.selectbox("Online Security", ('No', 'Yes', 'No internet service'))
tech_support = st.selectbox("Tech Support", ('No', 'Yes', 'No internet service'))

# Map input values to numeric (if label encoding is required)
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3,
}
internet_service = label_mapping.get(internet_service, 0)  # Use 0 as default value
contract = label_mapping.get(contract, 0)  # Use 0 as default value
payment_method = label_mapping.get(payment_method, 0)  # Use 0 as default value
online_security = label_mapping.get(online_security, 0)  # Use 0 as default value
tech_support = label_mapping.get(tech_support, 0)  # Use 0 as default value

# Make a prediction using the model
try:
  # Suppress the warning
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    prediction = model.predict(np.array([[tenure, internet_service, contract, monthly_charges, total_charges, payment_method, online_security, tech_support]]))
  
  # Display the prediction result on the main screen
  st.header("Prediction Result")
  if prediction[0] == 0:
      st.success("This customer is likely to stay.")
  else:
      st.error("This customer is likely to churn.")
except Exception as e:  # Add specific exception handling if needed
  st.error(f"An error occurred: {e}")

# Add any additional Streamlit components or UI elements as needed.



