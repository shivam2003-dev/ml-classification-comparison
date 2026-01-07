#!/bin/bash

# Setup script for ML Assignment 2

echo "🚀 Setting up ML Assignment 2 project..."

# Create model directory
mkdir -p model

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download dataset
echo "📥 Downloading dataset..."
python download_dataset.py

# Train models
echo "🤖 Training models..."
python train_models.py

# Update README with metrics
echo "📝 Updating README with metrics..."
python update_readme_metrics.py

echo "✅ Setup complete!"
echo "Run 'streamlit run streamlit_app.py' to start the app"

