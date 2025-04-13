#!/bin/bash
# Training and backtesting script for all strategies as a full unit
# This script runs comprehensive backtesting on all trading strategies
# to identify optimal parameters and evaluate overall performance

# Set environment
export PYTHONPATH=.:$PYTHONPATH

# Create necessary directories
mkdir -p backtest_results
mkdir -p models

# Ensure historical data is available
if [ ! -d "historical_data" ] || [ ! -f "historical_data/SOLUSD_1h.csv" ]; then
    echo "Historical data not found. Fetching historical data..."
    bash fetch_historical_data.sh
    python fix_historical_data_timestamps.py
fi

# Print available timeframes
echo "Available timeframes:"
ls -la historical_data/

# First, run optimization on individual strategies
echo "Phase 1: Optimizing individual strategies through backtesting..."

# ARIMA Strategy optimization
echo "Optimizing ARIMA strategy parameters..."
python strategy_backtesting.py --strategy arima --symbol SOLUSD --timeframe 1h --optimize

# Integrated Strategy optimization
echo "Optimizing Integrated strategy parameters..."
python strategy_backtesting.py --strategy integrated --symbol SOLUSD --timeframe 1h --optimize

# Next, run ML training with the high-accuracy approach
echo "Phase 2: Training ML models with high-accuracy approach..."
bash start_high_accuracy_training.sh

# Finally, run ensemble backtesting to evaluate the full system
echo "Phase 3: Backtesting the full ensemble trading system..."
python strategy_backtesting.py --strategy ensemble --symbol SOLUSD --timeframe 1h

# Summarize results
echo "All training and backtesting complete."
echo "Results are available in the backtest_results directory."
echo "ML models are available in the models directory."
echo "Check the generated charts for visual representation of performance."