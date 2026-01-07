"""
Quick test script to verify the app can load all models and metrics
"""

import os
import json
import joblib

def test_app_components():
    """Test if all app components can be loaded"""
    print("=" * 60)
    print("Testing ML Classification App Components")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Test 1: Check model directory exists
    print("\n1. Checking model directory...")
    if os.path.exists('model'):
        print("   ✓ model/ directory exists")
    else:
        errors.append("model/ directory not found")
        print("   ✗ model/ directory NOT found")
        return errors, warnings
    
    # Test 2: Check metrics.json
    print("\n2. Checking metrics.json...")
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"   ✓ metrics.json loaded successfully")
            print(f"   ✓ Found {len(metrics)} models in metrics")
            for model_name in metrics.keys():
                print(f"      - {model_name}")
        except Exception as e:
            errors.append(f"Could not load metrics.json: {str(e)}")
            print(f"   ✗ Error loading metrics.json: {str(e)}")
    else:
        errors.append("metrics.json not found")
        print("   ✗ metrics.json NOT found")
    
    # Test 3: Check scaler
    print("\n3. Checking scaler...")
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"   ✓ scaler.pkl loaded successfully")
            print(f"   ✓ Scaler type: {type(scaler).__name__}")
        except Exception as e:
            warnings.append(f"Could not load scaler: {str(e)}")
            print(f"   ⚠ Warning loading scaler: {str(e)}")
    else:
        warnings.append("scaler.pkl not found (may not be needed for all models)")
        print("   ⚠ scaler.pkl NOT found")
    
    # Test 4: Check all model files
    print("\n4. Checking model files...")
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    loaded_models = 0
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   ✓ {model_name}: {size_kb:.1f} KB - {type(model).__name__}")
                loaded_models += 1
            except Exception as e:
                errors.append(f"Could not load {model_name}: {str(e)}")
                print(f"   ✗ {model_name}: Error - {str(e)}")
        else:
            errors.append(f"{model_name} file not found: {filepath}")
            print(f"   ✗ {model_name}: File NOT found")
    
    # Test 5: Check label encoder (if exists)
    print("\n5. Checking label encoder...")
    label_encoder_path = 'model/label_encoder.pkl'
    if os.path.exists(label_encoder_path):
        try:
            le = joblib.load(label_encoder_path)
            print(f"   ✓ label_encoder.pkl loaded successfully")
        except Exception as e:
            warnings.append(f"Could not load label encoder: {str(e)}")
            print(f"   ⚠ Warning loading label encoder: {str(e)}")
    else:
        print("   ℹ label_encoder.pkl not found (may not be needed)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Models loaded: {loaded_models}/6")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"   - {error}")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    if not errors and loaded_models == 6:
        print("\n✅ ALL TESTS PASSED! App should work correctly.")
        return True
    elif not errors:
        print("\n⚠ Some models missing, but app should still work.")
        return True
    else:
        print("\n❌ ERRORS DETECTED. Please fix before deploying.")
        return False

if __name__ == "__main__":
    success = test_app_components()
    exit(0 if success else 1)

