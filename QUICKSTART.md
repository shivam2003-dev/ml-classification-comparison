# Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset & Train Models
```bash
python download_dataset.py
python train_models.py
```

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

Or use the automated setup:
```bash
chmod +x setup.sh
./setup.sh
```

## 📋 Manual Steps

### 1. Download Dataset
The Wine Quality dataset will be automatically downloaded. If it fails, download manually:
- URL: https://archive.ics.uci.edu/ml/datasets/wine+quality
- Save as `winequality-red.csv` in the project root

### 2. Train Models
```bash
python train_models.py
```
This creates:
- `model/*.pkl` - Trained models
- `model/scaler.pkl` - Feature scaler
- `model/metrics.json` - Evaluation metrics
- `model/test_data.csv` - Test dataset

### 3. Update README (Optional)
```bash
python update_readme_metrics.py
```
Updates README.md with actual metrics from training.

### 4. Run App Locally
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 in your browser.

## 🌐 Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New App"
5. Select repository
6. Choose `app.py` or `streamlit_app.py` as main file
7. Deploy!

## 📁 Project Structure

```
ml-assignment2/
├── app.py                 # Streamlit app (for Cloud)
├── streamlit_app.py       # Main Streamlit app
├── train_models.py        # Train all 6 models
├── download_dataset.py    # Download dataset
├── update_readme_metrics.py  # Update README
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
├── setup.sh              # Automated setup
└── model/                # Saved models (created after training)
    ├── *.pkl
    ├── scaler.pkl
    ├── metrics.json
    └── *.csv
```

## ✅ Verification Checklist

- [ ] All dependencies installed
- [ ] Dataset downloaded (`winequality-red.csv` exists)
- [ ] Models trained (`model/` directory has .pkl files)
- [ ] Metrics calculated (`model/metrics.json` exists)
- [ ] Streamlit app runs locally
- [ ] All 6 models appear in app
- [ ] Can upload test data and make predictions

## 🐛 Troubleshooting

### Issue: Dataset not found
**Solution:** Run `python download_dataset.py` or download manually

### Issue: Models not loading in Streamlit
**Solution:** Make sure you've run `python train_models.py` first

### Issue: Import errors
**Solution:** Install all dependencies: `pip install -r requirements.txt`

### Issue: Streamlit Cloud deployment fails
**Solution:** 
- Check `requirements.txt` has all dependencies
- Ensure `app.py` or `streamlit_app.py` exists
- Verify model files are in `model/` directory (or handle missing models gracefully)

## 📊 Expected Results

After training, you should see:
- 6 trained models (.pkl files)
- Metrics for each model (Accuracy, AUC, Precision, Recall, F1, MCC)
- Test dataset saved for predictions

## 🎯 Next Steps

1. Train models: `python train_models.py`
2. Review metrics in `model/metrics.json`
3. Update README: `python update_readme_metrics.py`
4. Test app locally: `streamlit run streamlit_app.py`
5. Push to GitHub
6. Deploy on Streamlit Cloud

