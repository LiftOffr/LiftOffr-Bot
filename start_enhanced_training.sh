#!/bin/bash
# Script to start enhanced ML model training process

echo "=================================================================="
echo "Starting Enhanced ML Model Training for Kraken Trading Bot"
echo "=================================================================="

# Check if historical data exists
if [ ! -d "historical_data" ] || [ -z "$(ls -A historical_data)" ]; then
    echo "Historical data not found. Fetching data first..."
    ./fetch_historical_data.sh
    echo "Historical data fetching complete."
fi

# Create necessary directories
mkdir -p models
mkdir -p models/tcn models/cnn models/lstm models/gru models/bilstm models/attention models/transformer

# Install required packages (if not already installed)
echo "Checking for required packages..."
pip install tensorflow scikit-learn pandas numpy matplotlib

# Install TCN package
pip install keras-tcn

echo "=================================================================="
echo "Starting ML model training..."
echo "=================================================================="

# Run training script
python ml_data_integrator.py

echo "=================================================================="
echo "ML model training complete!"
echo "=================================================================="
echo "Trained ML models available in the 'models' directory."
echo "Model comparison results are in 'models/model_comparison.json'"