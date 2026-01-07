# 📊 Test Datasets for Upload Testing

This directory contains test CSV files that you can use to test the file upload and prediction functionality in the Streamlit app.

## Available Test Files

### 1. `test_data_sample.csv` (Small - 10 samples)
- **Purpose:** Quick testing with a small dataset
- **Size:** 10 wine samples
- **Use Case:** Fast testing, quick predictions
- **Best for:** Initial testing, verifying the upload feature works

### 2. `test_data_large.csv` (Medium - 30 samples)
- **Purpose:** More comprehensive testing
- **Size:** 30 wine samples
- **Use Case:** Testing with multiple predictions
- **Best for:** Verifying batch predictions work correctly

## How to Use

### In Streamlit App:

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to "🔮 Predict on New Data" page**

3. **Click "Choose a CSV file" button**

4. **Select one of the test files:**
   - `test_data_sample.csv` (for quick test)
   - `test_data_large.csv` (for more samples)

5. **Select a model** from the dropdown

6. **Click "🔮 Make Predictions"**

7. **View results:**
   - Predictions table will show quality predictions (3-8)
   - Confidence scores for each prediction
   - Download predictions as CSV

## File Format

All test files follow the same format as the training data:

```csv
fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4
...
```

**Important:** 
- ✅ Include header row (column names)
- ✅ All 11 features must be present
- ✅ No target column (quality) - that's what we're predicting!
- ✅ Values should be numeric

## Expected Predictions

The models will predict wine quality on a scale of **3-8**:
- **3-4:** Low quality
- **5-6:** Medium quality (most common)
- **7-8:** High quality

## Testing Checklist

- [ ] Upload `test_data_sample.csv` successfully
- [ ] Data preview shows correctly
- [ ] Can select different models
- [ ] Predictions are generated
- [ ] Predictions are in range 3-8
- [ ] Confidence scores are shown
- [ ] Can download predictions as CSV
- [ ] Try with `test_data_large.csv` (30 samples)

## Troubleshooting

### Error: "File format incorrect"
- Check that the CSV has all 11 columns
- Verify column names match exactly (case-sensitive)
- Ensure no extra columns

### Error: "Could not make predictions"
- Make sure a model is selected
- Check that models are loaded (should see 6 models available)
- Verify the data has numeric values only

### Predictions seem wrong
- This is normal - these are test samples
- Different models may give different predictions
- Check confidence scores - higher = more certain

## Creating Your Own Test Data

You can create your own test CSV file with this format:

```csv
fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
7.5,0.75,0.10,2.0,0.080,12.0,35.0,0.9980,3.50,0.60,9.5
```

**Value Ranges (typical):**
- fixed acidity: 4.0 - 16.0
- volatile acidity: 0.1 - 1.6
- citric acid: 0.0 - 1.0
- residual sugar: 0.5 - 15.0
- chlorides: 0.01 - 0.6
- free sulfur dioxide: 1 - 72
- total sulfur dioxide: 6 - 289
- density: 0.99 - 1.00
- pH: 2.7 - 4.0
- sulphates: 0.3 - 2.0
- alcohol: 8.0 - 15.0

## Notes

- These test files are based on real wine quality data samples
- Predictions may vary between models
- Use these for testing the app functionality, not for actual wine quality assessment
- The actual quality values for these samples are not included (that's what we're predicting!)

