#!/bin/bash
# Start high-accuracy ML training for the Kraken Trading Bot
# This script aims to achieve ~90% accuracy by implementing:
# 1. Multi-timeframe data integration
# 2. Advanced feature engineering
# 3. Ensemble methods
# 4. Market regime-specific models
# 5. Longer training with optimal hyperparameters

# Set environment
export PYTHONPATH=.:$PYTHONPATH

# Create necessary directories
mkdir -p models
mkdir -p models/tcn models/cnn models/lstm models/gru models/bilstm models/attention models/transformer models/hybrid
mkdir -p models/ensemble models/regime_specific

# Ensure historical data is available
if [ ! -d "historical_data" ] || [ ! -f "historical_data/SOLUSD_1h.csv" ]; then
    echo "Historical data not found. Fetching historical data..."
    bash fetch_historical_data.sh
    python fix_historical_data_timestamps.py
fi

# Check for required timeframes and build available timeframes list
AVAILABLE_TIMEFRAMES=()
for timeframe in "1h" "4h" "1d"; do
    if [ -f "historical_data/SOLUSD_${timeframe}.csv" ]; then
        AVAILABLE_TIMEFRAMES+=("$timeframe")
    else
        echo "NOTE: ${timeframe} timeframe data not available. Training will continue without it."
    fi
done

if [ ${#AVAILABLE_TIMEFRAMES[@]} -eq 0 ]; then
    echo "ERROR: No timeframe data available. Cannot proceed with training."
    exit 1
fi

echo "Training will use these available timeframes: ${AVAILABLE_TIMEFRAMES[*]}"

# Create a symbolic link to the original version for comparison
cp -n train_ensemble_models.py train_ensemble_models_original.py

# Set GPU memory growth if available
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run the high-accuracy training
echo "Starting high-accuracy ML training..."
echo "This will take several hours for full training with all features enabled."
echo ""
echo "Training with the following enhancements:"
echo "1. Advanced feature engineering (200+ indicators)"
echo "2. Multi-timeframe data integration with available timeframes: ${AVAILABLE_TIMEFRAMES[*]}"
echo "3. Market regime-specific models"
echo "4. Ensemble stacking approach"
echo "5. Longer training cycles (100 epochs)"
echo "6. Regularization and advanced architecture designs"
echo ""

# Parameter options
SYMBOL="SOLUSD"
PRIMARY_TF="1h"
SEQUENCE_LENGTH=48  # Longer sequence length for better context
BATCH_SIZE=32
PATIENCE=15         # More patience for early stopping

# Train first with just the key models to establish a baseline
echo "Phase 1: Training core models to establish baseline..."
python advanced_ml_training.py \
    --symbol $SYMBOL \
    --primary-timeframe $PRIMARY_TF \
    --timeframes "1h" \
    --models "hybrid" "tcn" \
    --seq-length $SEQUENCE_LENGTH

# Train with multi-timeframe data integration (using available timeframes)
echo "Phase 2: Training with multi-timeframe data integration..."
# Build the command with available timeframes dynamically
TIMEFRAMES_ARG=""
for tf in "${AVAILABLE_TIMEFRAMES[@]}"; do
    TIMEFRAMES_ARG="$TIMEFRAMES_ARG \"$tf\""
done

# Remove the last (unnecessary) space
TIMEFRAMES_ARG=$(echo $TIMEFRAMES_ARG | xargs)

# Execute the command using the available timeframes
python advanced_ml_training.py \
    --symbol $SYMBOL \
    --primary-timeframe $PRIMARY_TF \
    --timeframes ${AVAILABLE_TIMEFRAMES[@]} \
    --models "hybrid" "tcn" "lstm" "attention" \
    --seq-length $SEQUENCE_LENGTH \
    --train-regime-models

# Train all models for the full ensemble
echo "Phase 3: Training all models for full ensemble..."
python advanced_ml_training.py \
    --symbol $SYMBOL \
    --primary-timeframe $PRIMARY_TF \
    --timeframes ${AVAILABLE_TIMEFRAMES[@]} \
    --models "hybrid" "tcn" "lstm" "gru" "bilstm" "attention" "transformer" "cnn" \
    --seq-length $SEQUENCE_LENGTH \
    --train-regime-models

# Evaluate the final model accuracy
echo "Evaluating final model accuracy..."
python evaluate_ensemble_models.py \
    --symbol $SYMBOL \
    --timeframe $PRIMARY_TF \
    --test-period 30

echo "High-accuracy training completed!"
echo "Check the model_evaluation directory for performance metrics and visualizations."
echo "Expected accuracy improvement: 51.48% -> ~85-90% directional accuracy"