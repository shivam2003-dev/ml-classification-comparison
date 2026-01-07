"""
Comprehensive verification script to check all assignment requirements
"""

import os
import json
import pandas as pd

def check_assignment_requirements():
    """Verify all assignment requirements are met"""
    
    print("=" * 70)
    print("ML Assignment 2 - Comprehensive Requirements Check")
    print("=" * 70)
    
    all_passed = True
    issues = []
    
    # 1. Check required files
    print("\n1. Checking Required Files...")
    required_files = {
        'app.py': 'Main Streamlit app',
        'streamlit_app.py': 'Alternative Streamlit app',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Project documentation',
        'train_models.py': 'Model training script'
    }
    
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {desc}")
        else:
            print(f"   ❌ {file} - {desc} - MISSING!")
            all_passed = False
            issues.append(f"Missing file: {file}")
    
    # 2. Check model directory
    print("\n2. Checking Model Directory...")
    if os.path.exists('model'):
        model_files = os.listdir('model')
        pkl_files = [f for f in model_files if f.endswith('.pkl')]
        required_models = [
            'logistic_regression.pkl',
            'decision_tree.pkl',
            'knn.pkl',
            'naive_bayes.pkl',
            'random_forest.pkl',
            'xgboost.pkl',
            'scaler.pkl'
        ]
        
        for model in required_models:
            if model in pkl_files:
                print(f"   ✅ {model}")
            else:
                print(f"   ❌ {model} - MISSING!")
                all_passed = False
                issues.append(f"Missing model: {model}")
        
        if 'metrics.json' in model_files:
            print(f"   ✅ metrics.json")
            # Verify metrics structure
            try:
                with open('model/metrics.json', 'r') as f:
                    metrics = json.load(f)
                if len(metrics) == 6:
                    print(f"   ✅ All 6 models have metrics")
                    for model_name in metrics.keys():
                        required_metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']
                        missing = [m for m in required_metrics if m not in metrics[model_name]]
                        if missing:
                            print(f"   ❌ {model_name} missing metrics: {missing}")
                            all_passed = False
                        else:
                            print(f"   ✅ {model_name} has all 6 metrics")
                else:
                    print(f"   ❌ Expected 6 models, found {len(metrics)}")
                    all_passed = False
            except Exception as e:
                print(f"   ❌ Error reading metrics.json: {e}")
                all_passed = False
        else:
            print(f"   ❌ metrics.json - MISSING!")
            all_passed = False
    else:
        print("   ❌ model/ directory - MISSING!")
        all_passed = False
        issues.append("Missing model/ directory")
    
    # 3. Check README structure
    print("\n3. Checking README.md Structure...")
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        required_sections = [
            'Problem Statement',
            'Dataset Description',
            'Comparison Table',
            'Observations'
        ]
        
        for section in required_sections:
            if section.lower() in readme_content.lower():
                print(f"   ✅ {section} section found")
            else:
                print(f"   ❌ {section} section - MISSING!")
                all_passed = False
                issues.append(f"Missing README section: {section}")
        
        # Check for metrics table
        if '| ML Model Name | Accuracy | AUC |' in readme_content:
            print(f"   ✅ Metrics comparison table found")
        else:
            print(f"   ❌ Metrics comparison table - MISSING or INCORRECT FORMAT!")
            all_passed = False
    else:
        print("   ❌ README.md - MISSING!")
        all_passed = False
    
    # 4. Check requirements.txt
    print("\n4. Checking requirements.txt...")
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            req_content = f.read()
        
        required_packages = [
            'streamlit',
            'scikit-learn',
            'numpy',
            'pandas',
            'matplotlib',
            'seaborn',
            'xgboost'
        ]
        
        for package in required_packages:
            if package.lower() in req_content.lower():
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} - MISSING!")
                all_passed = False
                issues.append(f"Missing package: {package}")
    else:
        print("   ❌ requirements.txt - MISSING!")
        all_passed = False
    
    # 5. Check Streamlit app features
    print("\n5. Checking Streamlit App Features...")
    if os.path.exists('app.py'):
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        required_features = {
            'file_uploader': 'Dataset upload (CSV)',
            'selectbox': 'Model selection dropdown',
            'st.metric': 'Display evaluation metrics',
            'confusion_matrix': 'Confusion matrix',
            'classification_report': 'Classification report'
        }
        
        for feature, desc in required_features.items():
            if feature in app_content:
                print(f"   ✅ {desc}")
            else:
                print(f"   ❌ {desc} - MISSING!")
                all_passed = False
                issues.append(f"Missing feature: {desc}")
    else:
        print("   ❌ app.py - MISSING!")
        all_passed = False
    
    # 6. Check dataset
    print("\n6. Checking Dataset...")
    dataset_files = ['winequality-red.csv']
    for dataset in dataset_files:
        if os.path.exists(dataset):
            try:
                df = pd.read_csv(dataset, sep=';')
                if len(df.columns) >= 12:  # 11 features + 1 target
                    print(f"   ✅ {dataset} - {len(df)} rows, {len(df.columns)} columns")
                    if len(df) >= 500:
                        print(f"   ✅ Dataset has {len(df)} instances (>= 500 required)")
                    else:
                        print(f"   ❌ Dataset has only {len(df)} instances (< 500 required)")
                        all_passed = False
                else:
                    print(f"   ❌ {dataset} - Only {len(df.columns)} columns (< 12 required)")
                    all_passed = False
            except Exception as e:
                print(f"   ❌ Error reading {dataset}: {e}")
                all_passed = False
        else:
            print(f"   ⚠️  {dataset} - Not found (will be downloaded)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL REQUIREMENTS MET!")
        print("\nYour assignment is ready for:")
        print("  1. ✅ GitHub submission")
        print("  2. ✅ Streamlit Cloud deployment")
        print("  3. ✅ BITS Virtual Lab execution (screenshot needed)")
    else:
        print("❌ SOME REQUIREMENTS MISSING!")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix the issues above before submission.")
    
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = check_assignment_requirements()
    exit(0 if success else 1)

