# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models to predict wine quality based on chemical properties. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and deployment through an interactive Streamlit web application.

The assignment requires:
- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Wine Quality (Red Wine) from UCI Machine Learning Repository

**Source:** [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Description:**
The dataset contains chemical properties of red wine samples and their quality ratings. The quality is rated on a scale from 3 to 8, making this a multi-class classification problem.

**Features (11 input features):**
1. **Fixed Acidity** - Most acids involved with wine or fixed or nonvolatile
2. **Volatile Acidity** - The amount of acetic acid in wine
3. **Citric Acid** - Found in small quantities, adds freshness and flavor
4. **Residual Sugar** - The amount of sugar remaining after fermentation stops
5. **Chlorides** - The amount of salt in the wine
6. **Free Sulfur Dioxide** - The free form of SO2 exists in equilibrium between molecular SO2 and bisulfite ion
7. **Total Sulfur Dioxide** - Amount of free and bound forms of S02
8. **Density** - The density of water is close to that of water depending on the percent alcohol and sugar content
9. **pH** - Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic)
10. **Sulphates** - A wine additive which can contribute to sulfur dioxide gas (S02) levels
11. **Alcohol** - The percent alcohol content of the wine

**Target Variable:**
- **Quality** - Score between 3 and 8 (6 classes)

**Dataset Statistics:**
- **Total Instances:** 1,599
- **Features:** 11
- **Classes:** 6 (quality levels 3-8)
- **Missing Values:** None

## Models Used

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5906 | 0.7555 | 0.5695 | 0.5906 | 0.5673 | 0.3250 |
| Decision Tree | 0.6062 | 0.6974 | 0.6097 | 0.6062 | 0.6066 | 0.3944 |
| KNN | 0.6094 | 0.7476 | 0.5841 | 0.6094 | 0.5959 | 0.3733 |
| Naive Bayes | 0.5625 | 0.7377 | 0.5745 | 0.5625 | 0.5681 | 0.3299 |
| Random Forest | 0.6750 | 0.8375 | 0.6504 | 0.6750 | 0.6599 | 0.4764 |
| XGBoost | 0.6531 | 0.8153 | 0.6480 | 0.6531 | 0.6434 | 0.4453 |

### Observations about Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression achieved 59.06% accuracy with 75.55% AUC. While it provides a linear decision boundary and is fast to train, its performance is limited by the non-linear relationships in the wine quality data. The model struggles with the multi-class classification task, particularly for minority classes (quality 3, 4, 8). |
| Decision Tree | Decision Tree achieved 60.62% accuracy with 69.74% AUC. The model can capture non-linear relationships but shows signs of overfitting. It performs better than Logistic Regression but still struggles with class imbalance, especially for rare quality levels. |
| kNN | K-Nearest Neighbors achieved 60.94% accuracy with 74.76% AUC, performing slightly better than Decision Tree. The model benefits from feature scaling and captures local patterns effectively. However, it's computationally expensive and sensitive to the choice of k parameter. |
| Naive Bayes | Naive Bayes achieved the lowest accuracy (56.25%) but a reasonable AUC (73.77%). The model's assumption of feature independence doesn't hold well for wine quality data, where chemical properties are correlated. Despite this, it provides a fast baseline model. |
| Random Forest (Ensemble) | Random Forest achieved the best performance (67.50% accuracy, 83.75% AUC). By combining multiple decision trees, it effectively reduces overfitting and handles non-linear relationships. The ensemble approach provides robust predictions and better generalization compared to individual models. |
| XGBoost (Ensemble) | XGBoost achieved the second-best performance (65.31% accuracy, 81.53% AUC). It effectively captures complex patterns and feature interactions. While slightly lower than Random Forest in this case, it shows strong performance with good generalization. The model benefits from gradient boosting's ability to correct errors iteratively. |

## Project Structure

```
ml-classification-comparison/
├── streamlit_app.py          # Main Streamlit application
├── train_models.py           # Script to train all models
├── download_dataset.py       # Script to download dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── model/                    # Directory for saved models
│   ├── *.pkl                # Trained model files
│   ├── scaler.pkl           # Feature scaler
│   ├── metrics.json         # Evaluation metrics
│   ├── test_data.csv        # Test dataset
│   └── test_labels.csv      # Test labels
└── .github/
    └── workflows/
        └── deploy.yml       # GitHub Actions workflow
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-classification-comparison
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   python download_dataset.py
   ```
   Alternatively, download manually from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) and save as `winequality-red.csv` in the project root.

4. **Train the models**
   ```bash
   python train_models.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train all 6 models
   - Calculate evaluation metrics
   - Save models and metrics to the `model/` directory

5. **Run the Streamlit app locally**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Streamlit Application Features

1. **Model Comparison Page**
   - View comparison table of all models
   - Select a model to see detailed metrics
   - View confusion matrix and classification report

2. **Predict on New Data Page**
   - Upload CSV file with test data
   - Select a model for prediction
   - View predictions and download results

3. **Dataset Info Page**
   - Learn about the dataset
   - View feature statistics

## Deployment

### Streamlit Community Cloud

**Quick Deployment Steps:**

1. **Push your code to GitHub** (already done ✅)
   ```bash
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App**
   - Click "New App" button
   - Select repository: `shivam2003-dev/ml-classification-comparison`
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Deployment takes 2-5 minutes
   - Monitor the deployment logs
   - Your app will be live at: `https://ml-classification-comparison.streamlit.app`

**📖 Detailed Guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for complete step-by-step instructions and troubleshooting.

**✅ Deployment Checklist:** See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) to ensure everything is ready.

### Automatic Updates

- Streamlit Cloud automatically redeploys when you push to the `main` branch
- Just push your changes: `git push origin main`
- Wait 2-5 minutes for automatic redeployment

## Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **AUC Score**: Area Under the ROC Curve (one-vs-rest for multi-class)
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **MCC Score**: Matthews Correlation Coefficient, a balanced measure for multi-class classification

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

## Author

[Your Name]

## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.

## Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- BITS Pilani for the assignment framework

