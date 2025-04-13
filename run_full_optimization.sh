#!/bin/bash
# Run full optimization and backtesting process for the trading bot

# Set up log file
LOGFILE="full_optimization_$(date +%Y%m%d_%H%M%S).log"
RESULT_DIR="optimization_results"

# Create required directories
mkdir -p $RESULT_DIR
mkdir -p historical_data
mkdir -p analysis_results
mkdir -p backtest_results
mkdir -p logs

# Log start time
echo "Starting full optimization process at $(date)" | tee -a $LOGFILE

# Step 1: Fetch historical data for multiple cryptocurrencies
echo "Step 1: Fetching historical data..." | tee -a $LOGFILE
python3 enhanced_historical_data_fetcher.py 2>&1 | tee -a $LOGFILE

# Check if data fetching was successful
if [ $? -ne 0 ]; then
    echo "Error fetching historical data. Exiting." | tee -a $LOGFILE
    exit 1
fi

# Step 2: Analyze correlations between assets
echo "Step 2: Analyzing correlations between assets..." | tee -a $LOGFILE
python3 multi_asset_correlation_analyzer.py 2>&1 | tee -a $LOGFILE

# Step 3: Run backtesting optimization for ARIMA strategy
echo "Step 3: Optimizing ARIMA strategy..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair SOL/USD --timeframe 1h --strategy arima --optimize --plot --output $RESULT_DIR/SOLUSD_1h 2>&1 | tee -a $LOGFILE

# Step 4: Run backtesting optimization for Integrated strategy
echo "Step 4: Optimizing Integrated strategy..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair SOL/USD --timeframe 1h --strategy integrated --optimize --plot --output $RESULT_DIR/SOLUSD_1h 2>&1 | tee -a $LOGFILE

# Step 5: Try additional cryptocurrencies
echo "Step 5: Optimizing strategies for BTC/USD..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair BTC/USD --timeframe 1h --strategy arima --optimize --plot --output $RESULT_DIR/BTCUSD_1h 2>&1 | tee -a $LOGFILE

echo "Step 6: Optimizing strategies for ETH/USD..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair ETH/USD --timeframe 1h --strategy arima --optimize --plot --output $RESULT_DIR/ETHUSD_1h 2>&1 | tee -a $LOGFILE

# Step 7: Run ML training (if available)
echo "Step 7: Running ML model training..." | tee -a $LOGFILE
if [ -f "advanced_ml_training.py" ]; then
    python3 advanced_ml_training.py --symbol SOLUSD --epochs 50 2>&1 | tee -a $LOGFILE
else
    echo "ML training script not found, skipping this step" | tee -a $LOGFILE
fi

# Step 8: Run optimized multi-strategy backtest
echo "Step 8: Running optimized multi-strategy backtest..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair SOL/USD --timeframe 1h --multi-strategy --plot --output $RESULT_DIR/SOLUSD_1h_multi 2>&1 | tee -a $LOGFILE

# Step 9: Run the enhanced training process (which orchestrates everything)
echo "Step 9: Running the complete enhanced training process..." | tee -a $LOGFILE
python3 run_enhanced_training.py --pairs SOL/USD BTC/USD ETH/USD --timeframes 1h 4h --skip-data 2>&1 | tee -a $LOGFILE

# Log completion
echo "Full optimization process completed at $(date)" | tee -a $LOGFILE
echo "Optimization results saved to $RESULT_DIR"
echo "Log file saved to $LOGFILE"