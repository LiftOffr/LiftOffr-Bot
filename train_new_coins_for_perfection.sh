#!/bin/bash

# Train New Coins for Perfection
# 
# This script runs the entire training pipeline for new cryptocurrency pairs:
# AVAX/USD, MATIC/USD, UNI/USD, ATOM/USD
#
# The goal is to achieve:
# - 100% accuracy
# - 100% win rate
# - 1000% returns
#
# The process includes:
# 1. Fetching historical data
# 2. Advanced feature engineering
# 3. Hyperparameter optimization
# 4. Multi-stage model training
# 5. Ensemble model creation
# 6. Comprehensive backtesting
# 7. Trading system integration

set -e  # Exit on error

# Configuration
PAIRS="AVAX/USD,MATIC/USD,UNI/USD,ATOM/USD"
TIMEFRAMES="1h,4h,1d"
TARGET_ACCURACY=0.99
TARGET_WIN_RATE=0.99
TARGET_RETURN=10.0
STAGES=3

# Create log directory
mkdir -p logs

# Log file
LOG_FILE="logs/training_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Start logging
log "=== Starting New Coins Training Pipeline ==="
log "Pairs: $PAIRS"
log "Timeframes: $TIMEFRAMES"
log "Target Metrics: Accuracy $TARGET_ACCURACY, Win Rate $TARGET_WIN_RATE, Return ${TARGET_RETURN}x"

# Step 1: Fetch historical data
log "Fetching historical data..."
python fetch_historical_data.py --pairs "$PAIRS" --timeframes "$TIMEFRAMES" --days 365 --force
log "Historical data fetched successfully"

# Step 2: Prepare advanced training data
log "Preparing advanced training data..."
python prepare_advanced_training_data.py --pairs "$PAIRS" --timeframes "$TIMEFRAMES" \
    --advanced-features --cross-asset-correlation --market-regime-detection \
    --sentiment-analysis --feature-selection --force
log "Training data prepared successfully"

# Step 3: Run the training pipeline
log "Running training pipeline for all pairs..."
python train_new_coins_to_perfection.py --pairs "$PAIRS" --stages "$STAGES" \
    --target-accuracy "$TARGET_ACCURACY" --target-win-rate "$TARGET_WIN_RATE" \
    --target-return "$TARGET_RETURN" --force
log "Training pipeline completed"

# Step 4: Integrate with trading system
log "Integrating trained models with trading system..."
python integrate_models.py --pairs "$PAIRS" --models ensemble --sandbox
log "Models integrated with trading system"

# Step 5: Verify dashboard
log "Verifying that dashboard is updated..."
python auto_update_dashboard.py
log "Dashboard updated"

# Final report
log "=== Training Pipeline Completed ==="
log "Check backtest_results directory for detailed performance metrics"
log "Check ensemble_models directory for trained ensemble models"
log "Check logs directory for detailed logs"
log "The trading system is now ready to use the new models in sandbox mode"

echo ""
echo "======================================"
echo "  Training Pipeline Completed"
echo "======================================"
echo "The models for the following pairs have been trained and integrated:"
echo "$PAIRS"
echo ""
echo "To monitor trading performance, open the dashboard in your browser."
echo "To activate live trading, run the following command:"
echo "python start_trading_bot.py --pairs $PAIRS"
echo ""