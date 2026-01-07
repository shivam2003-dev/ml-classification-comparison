"""
Script to update README.md with actual metrics after model training
"""

import json
import re

def update_readme_metrics():
    """Update the metrics table in README.md with actual values"""
    
    # Load metrics
    try:
        with open('model/metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("❌ metrics.json not found. Please run train_models.py first.")
        return
    
    # Read README
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Create metrics table rows
    model_order = [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    table_rows = []
    for model_name in model_order:
        if model_name in metrics:
            m = metrics[model_name]
            row = f"| {model_name} | {m['accuracy']:.4f} | {m['auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['mcc']:.4f} |"
            table_rows.append(row)
    
    # Replace the metrics table
    pattern = r'\| Logistic Regression \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \|.*?\| XGBoost \(Ensemble\) \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \| 0\.0000 \|'
    
    new_table = '\n'.join(table_rows)
    
    readme_content = re.sub(pattern, new_table, readme_content, flags=re.DOTALL)
    
    # Write updated README
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md updated with actual metrics!")

if __name__ == "__main__":
    update_readme_metrics()

