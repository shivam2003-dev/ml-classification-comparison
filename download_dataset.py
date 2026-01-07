"""
Script to download the Wine Quality dataset
"""

import pandas as pd
import urllib.request
import os

def download_wine_quality_dataset():
    """Download Wine Quality dataset from UCI repository"""
    
    # URL for Wine Quality Red dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    print("Downloading Wine Quality dataset...")
    try:
        # Download the dataset
        urllib.request.urlretrieve(url, "winequality-red.csv")
        print("✅ Dataset downloaded successfully!")
        
        # Load and display info (Wine Quality uses semicolon separator)
        df = pd.read_csv("winequality-red.csv", sep=';')
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataset info:")
        print(df.info())
        print(f"\nTarget distribution (Quality):")
        print(df['quality'].value_counts().sort_index())
        
        return df
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("\nAlternative: Please download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/wine+quality")
        return None

if __name__ == "__main__":
    download_wine_quality_dataset()

