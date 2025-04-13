#!/bin/bash
# Script to fetch historical data for ML model training

echo "==================================================================="
echo "Starting Historical Data Fetching Process for ML Model Enhancement"
echo "==================================================================="

# Create required directories
mkdir -p historical_data
mkdir -p historical_data/ml_datasets
mkdir -p models
mkdir -p models/tcn models/cnn models/lstm

# Run the historical data fetcher
echo "Fetching historical data from Kraken API..."
python historical_data_fetcher.py

echo "==================================================================="
echo "Historical data fetching complete!"
echo "==================================================================="
echo "Run ml_data_integrator.py to prepare and train ML models"