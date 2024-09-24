Customer Churn Analysis
This repository contains a comprehensive Python workflow for analyzing customer churn data, including data loading, preprocessing, exploration, and modeling. The project aims to identify key factors contributing to customer churn, allowing businesses to improve retention strategies.

Project Overview
Data Loading and Preprocessing:

Loads customer data from a CSV file.
Cleans the dataset by removing duplicates and handling missing values.
Inspects data types, unique values, and creates a new feature for customer tenure.
Churn Rate Calculation:

Computes the churn rate to determine the percentage of customers who have left.
Visualizes categorical variables related to churn using count plots.
Feature Engineering:

Encodes categorical features and categorizes customer tenure into defined intervals.
Visualization:

Utilizes Seaborn and Matplotlib for data visualization, including average charges for churned vs. non-churned customers.
Modeling:

Evaluates several machine learning models:
Logistic Regression (79% accuracy)
Random Forest (80% accuracy)
Support Vector Machine (78% accuracy)
Decision Tree (81% accuracy)
Utilizes cross-validation to assess model performance.
Model Saving:

Saves the trained models using joblib for future predictions.
Features
User input interface for predictions.
Enhanced error handling and user feedback.
Displays input values for verification before prediction.
Loading spinner during model processing.
Documentation of model input parameters.
Reset button for clearing inputs.
