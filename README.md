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
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| kNN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest (Ensemble) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost (Ensemble) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

*Note: The metrics above are placeholders. After running `train_models.py`, these values will be updated with actual results.*

### Observations about Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression provides a linear decision boundary and works well when the relationship between features and target is approximately linear. It's fast to train and interpretable. Performance may be limited for complex non-linear relationships. |
| Decision Tree | Decision Trees are easy to interpret and can capture non-linear relationships. However, they are prone to overfitting, especially with deep trees. They may not generalize well to unseen data. |
| kNN | K-Nearest Neighbors is a simple, instance-based learning algorithm. It can capture local patterns but is sensitive to the choice of k and feature scaling. It can be computationally expensive for large datasets. |
| Naive Bayes | Naive Bayes assumes feature independence, which may not hold in practice. It's fast and works well with small datasets. Performance depends on how well the independence assumption is satisfied. |
| Random Forest (Ensemble) | Random Forest combines multiple decision trees to reduce overfitting and improve generalization. It typically performs better than a single decision tree and is robust to outliers. It can handle non-linear relationships effectively. |
| XGBoost (Ensemble) | XGBoost is a gradient boosting algorithm that often achieves state-of-the-art performance. It's highly effective at capturing complex patterns and interactions between features. It may require more tuning but generally provides excellent results. |

*Note: Detailed observations will be added after model training and evaluation.*

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

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository
6. Choose branch (usually `main`)
7. Select `streamlit_app.py` as the main file
8. Click "Deploy"

The app will be live in a few minutes!

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

