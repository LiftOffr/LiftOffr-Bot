#!/bin/bash
# Train the advanced ensemble models for the Kraken Trading Bot

# Set environment
export PYTHONPATH=.:$PYTHONPATH

# Create necessary directories
mkdir -p models
mkdir -p models/tcn models/cnn models/lstm models/gru models/bilstm models/attention models/transformer models/hybrid models/ensemble

# Check if historical data exists
if [ ! -d "historical_data" ] || [ ! -f "historical_data/SOLUSD_1h.csv" ]; then
    echo "Historical data not found. Fetching historical data first..."
    bash fetch_historical_data.sh
    python fix_historical_data_timestamps.py
fi

# Run training with default parameters
echo "Starting ensemble model training..."
echo "This may take a while (1-3 hours) depending on your system..."

# Train models one by one to avoid memory issues
for model in tcn cnn lstm gru bilstm attention transformer hybrid; do
    echo "Training $model model..."
    python train_ensemble_models.py --models $model --epochs 30 --batch_size 64
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error training $model model. Continuing with next model..."
    else
        echo "$model model training completed."
    fi
    
    # Sleep briefly to allow system resources to stabilize
    sleep 5
done

echo "All models training completed!"
echo "You can now start trading with ML-enhanced strategies using:"
echo "python start_ml_enhanced_trading.py --sandbox"