"""
ML Assignment 2 - Streamlit Web Application
Interactive app to demonstrate classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍷 Wine Quality Classification - ML Models Comparison</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Model Comparison", "🔮 Predict on New Data", "📈 Dataset Info"]
)

# Load models and metrics
@st.cache_data
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    scaler = None
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {str(e)}")
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[model_name] = joblib.load(filepath)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
    
    return models, scaler

@st.cache_data
def load_metrics():
    """Load saved metrics"""
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        try:
            import json
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")
            return None
    return None

# Load data
try:
    models, scaler = load_models()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models/metrics: {str(e)}")
    models, scaler = {}, None
    metrics = None

# Page 1: Model Comparison
if page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    
    # Check if models/metrics exist
    if not metrics and not models:
        st.error("""
        ⚠️ **Models and metrics not found!**
        
        Please ensure you have:
        1. Trained the models by running: `python train_models.py`
        2. The `model/` directory contains:
           - `metrics.json`
           - `*.pkl` files for all 6 models
           - `scaler.pkl`
        
        If you're running this on Streamlit Cloud, make sure the model files are committed to GitHub.
        """)
    elif metrics:
        # Display metrics table
        st.subheader("Evaluation Metrics Table")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'ML Model Name': model_name,
                'Accuracy': f"{model_metrics['accuracy']:.4f}",
                'AUC': f"{model_metrics['auc']:.4f}",
                'Precision': f"{model_metrics['precision']:.4f}",
                'Recall': f"{model_metrics['recall']:.4f}",
                'F1': f"{model_metrics['f1']:.4f}",
                'MCC': f"{model_metrics['mcc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Model selection for detailed view
        st.subheader("Select Model for Detailed Analysis")
        selected_model = st.selectbox(
            "Choose a model",
            list(metrics.keys())
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Metrics")
            selected_metrics = metrics[selected_model]
            st.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
            st.metric("AUC Score", f"{selected_metrics['auc']:.4f}")
            st.metric("Precision", f"{selected_metrics['precision']:.4f}")
            st.metric("Recall", f"{selected_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{selected_metrics['f1']:.4f}")
            st.metric("MCC Score", f"{selected_metrics['mcc']:.4f}")
        
        with col2:
            st.markdown("### 📈 Confusion Matrix")
            cm = np.array(selected_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        report = selected_metrics['classification_report']
        if isinstance(report, dict):
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("⚠️ No metrics found. Please train the models first using train_models.py")

# Page 2: Predict on New Data
elif page == "🔮 Predict on New Data":
    st.header("Upload Test Data and Make Predictions")
    
    st.info("💡 Upload a CSV file with test data. The file should have the same features as the training data (excluding the target column).")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload test data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(test_data)} rows")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(test_data.head(), use_container_width=True)
            
            # Model selection
            if models:
                st.subheader("Select Model for Prediction")
                selected_model = st.selectbox(
                    "Choose a model",
                    list(models.keys())
                )
                
                if st.button("🔮 Make Predictions"):
                    try:
                        model = models[selected_model]
                        
                        # Check if model needs scaled data
                        needs_scaling = selected_model in ['Logistic Regression', 'KNN', 'Naive Bayes']
                        
                        if needs_scaling and scaler:
                            X_test = scaler.transform(test_data)
                        else:
                            X_test = test_data.values
                        
                        # Make predictions
                        predictions = model.predict(X_test)
                        prediction_proba = model.predict_proba(X_test)
                        
                        # Display results
                        st.subheader("Predictions")
                        results_df = pd.DataFrame({
                            'Prediction': predictions,
                            'Confidence': np.max(prediction_proba, axis=1)
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
            else:
                st.warning("⚠️ No models found. Please train the models first.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Show example format
        st.subheader("Expected Data Format")
        st.info("""
        The CSV file should contain the following columns (for Wine Quality dataset):
        - fixed acidity
        - volatile acidity
        - citric acid
        - residual sugar
        - chlorides
        - free sulfur dioxide
        - total sulfur dioxide
        - density
        - pH
        - sulphates
        - alcohol
        """)

# Page 3: Dataset Info
elif page == "📈 Dataset Info":
    st.header("Dataset Information")
    
    st.markdown("""
    ### Wine Quality Dataset
    
    **Source:** UCI Machine Learning Repository
    
    **Description:**
    This dataset contains chemical properties of red wine samples and their quality ratings.
    The quality is rated on a scale from 3 to 8.
    
    **Features (11 input features):**
    1. Fixed Acidity
    2. Volatile Acidity
    3. Citric Acid
    4. Residual Sugar
    5. Chlorides
    6. Free Sulfur Dioxide
    7. Total Sulfur Dioxide
    8. Density
    9. pH
    10. Sulphates
    11. Alcohol
    
    **Target:**
    - Quality (3-8 scale)
    
    **Dataset Statistics:**
    - Total Instances: 1,599
    - Features: 11
    - Classes: 6 (quality levels 3-8)
    
    **Models Implemented:**
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor (KNN)
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    
    **Evaluation Metrics:**
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient (MCC)
    """)
    
    # Try to load and display dataset statistics
    try:
        if os.path.exists('model/test_data.csv'):
            test_data = pd.read_csv('model/test_data.csv')
            st.subheader("Feature Statistics")
            st.dataframe(test_data.describe(), use_container_width=True)
    except:
        pass

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>ML Assignment 2 - Classification Models Comparison</div>",
    unsafe_allow_html=True
)

