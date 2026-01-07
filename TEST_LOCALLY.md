# 🧪 Test App Locally

## Quick Test

1. **Verify all components:**
   ```bash
   python test_app.py
   ```
   Should show: ✅ ALL TESTS PASSED!

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Or:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in browser:**
   - The app will automatically open at: http://localhost:8501
   - Or manually go to: http://localhost:8501

## What to Check

### ✅ Model Comparison Page
- [ ] All 6 models appear in the comparison table
- [ ] Metrics are displayed (Accuracy, AUC, Precision, Recall, F1, MCC)
- [ ] Can select different models from dropdown
- [ ] Confusion matrix displays for selected model
- [ ] Classification report shows

### ✅ Predict on New Data Page
- [ ] File uploader appears
- [ ] Can select a model from dropdown
- [ ] Can upload a CSV file
- [ ] Predictions are generated
- [ ] Can download predictions

### ✅ Dataset Info Page
- [ ] Dataset information displays
- [ ] Feature statistics show (if test data exists)

## Troubleshooting

### If models don't load:
1. Check models exist: `ls -la model/`
2. Run test: `python test_app.py`
3. Re-train if needed: `python train_models.py`

### If Streamlit doesn't start:
1. Install dependencies: `pip install -r requirements.txt`
2. Check Python version: `python --version` (should be 3.8+)

### If you see errors:
- Check the error message in the terminal
- Verify all files are in place
- Make sure you're in the project directory

## Expected Output

When you run `streamlit run app.py`, you should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Then the app should open automatically in your browser!

