import streamlit as st
import data_prep as dp
import eda as ed
import feature_eng as fe
import modeling as md
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Customer Churn Analysis")

# Load the dataset from a file uploader or a fixed path
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess the data
    df = dp.load_data(uploaded_file)
    
    # Ensure DataFrame is not empty and has consistent data types
    if df.empty:
        st.warning("The DataFrame is empty. Please upload a valid dataset.")
    else:
        # Handle mixed data types
        df = df.infer_objects()  # Convert object columns to appropriate types
        df = dp.handle_missing_values(df)
        df = dp.encode_categorical(df)
        df = fe.create_features(df)

        # Display sample data
        st.write("Sample Data:")
        st.write(df.head())
        
        # Calculate churn rate and visualize churn distribution
        churn_rate = ed.calculate_churn_rate(df)
        st.subheader("Churn Rate")
        st.write(f"Churn Rate: {churn_rate:.2f}%")
        
        churn_dist_plot = ed.visualize_churn_distribution(df)
        st.pyplot(churn_dist_plot)

        # Feature engineering and data scaling
        features, target = fe.select_features(df)

        # Standardize features to avoid convergence issues
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Split data and train models
        X_train, X_test, y_train, y_test = md.split_data(features, target)
        log_reg = LogisticRegression(max_iter=500)  # Increase iterations
        rf = RandomForestClassifier()  # Random Forest doesn't usually have convergence issues
        
        # Train models
        log_reg.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # Evaluate models
        log_reg_metrics = md.evaluate_model(log_reg, X_test, y_test)
        rf_metrics = md.evaluate_model(rf, X_test, y_test)

        st.subheader("Model Performance")
        st.write("Logistic Regression Metrics:")
        st.write(log_reg_metrics)
        
        st.write("Random Forest Metrics:")
        st.write(rf_metrics)
