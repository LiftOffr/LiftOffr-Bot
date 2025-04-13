#!/bin/bash
# Run comprehensive testing and optimization process for the trading bot

# Set up log file
LOGFILE="full_optimization_$(date +%Y%m%d_%H%M%S).log"
RESULT_DIR="optimization_results"

# Create required directories
mkdir -p $RESULT_DIR
mkdir -p historical_data
mkdir -p analysis_results
mkdir -p backtest_results
mkdir -p logs
mkdir -p models

# Log start time
echo "==================================================================" | tee -a $LOGFILE
echo "COMPREHENSIVE TESTING AND OPTIMIZATION PROCESS" | tee -a $LOGFILE
echo "Starting at $(date)" | tee -a $LOGFILE
echo "==================================================================" | tee -a $LOGFILE

# Step 1: Fetch historical data for multiple cryptocurrencies (including low timeframes)
echo "Step 1: Fetching historical data including low timeframes..." | tee -a $LOGFILE
python3 enhanced_historical_data_fetcher.py 2>&1 | tee -a $LOGFILE

# Check if data fetching was successful
if [ $? -ne 0 ]; then
    echo "Error fetching historical data. Exiting." | tee -a $LOGFILE
    exit 1
fi

# Step 2: Analyze correlations between assets
echo "Step 2: Analyzing correlations between assets..." | tee -a $LOGFILE
python3 multi_asset_correlation_analyzer.py 2>&1 | tee -a $LOGFILE

# Step 3: Test low timeframe data utilization
echo "Step 3: Testing low timeframe data effectiveness..." | tee -a $LOGFILE
for pair in "SOL/USD" "BTC/USD" "ETH/USD" "ADA/USD" "DOT/USD" "LINK/USD" "MATICUSD"; do
    echo "Testing low timeframe data for $pair..." | tee -a $LOGFILE
    python3 comprehensive_backtest.py --pair $pair --timeframe 5m --strategy arima --optimize --plot --output $RESULT_DIR/$(echo $pair | tr '/' '')_5m 2>&1 | tee -a $LOGFILE
done

# Step 4: Run backtesting optimization for ARIMA strategy
echo "Step 4: Optimizing ARIMA strategy on multiple timeframes..." | tee -a $LOGFILE
for timeframe in "15m" "1h" "4h"; do
    echo "Optimizing ARIMA strategy on $timeframe timeframe..." | tee -a $LOGFILE
    python3 comprehensive_backtest.py --pair SOL/USD --timeframe $timeframe --strategy arima --optimize --plot --output $RESULT_DIR/SOLUSD_${timeframe}_arima 2>&1 | tee -a $LOGFILE
done

# Step 5: Run backtesting optimization for Integrated strategy
echo "Step 5: Optimizing Integrated strategy on multiple timeframes..." | tee -a $LOGFILE
for timeframe in "15m" "1h" "4h"; do
    echo "Optimizing Integrated strategy on $timeframe timeframe..." | tee -a $LOGFILE
    python3 comprehensive_backtest.py --pair SOL/USD --timeframe $timeframe --strategy integrated --optimize --plot --output $RESULT_DIR/SOLUSD_${timeframe}_integrated 2>&1 | tee -a $LOGFILE
done

# Step 6: Optimize for additional cryptocurrencies
echo "Step 6: Optimizing strategies for multiple cryptocurrencies..." | tee -a $LOGFILE
for pair in "BTC/USD" "ETH/USD" "ADA/USD" "DOT/USD" "LINK/USD" "MATICUSD"; do
    echo "Optimizing strategies for $pair..." | tee -a $LOGFILE
    python3 comprehensive_backtest.py --pair $pair --timeframe 1h --strategy arima --optimize --plot --output $RESULT_DIR/$(echo $pair | tr '/' '')_1h_arima 2>&1 | tee -a $LOGFILE
    python3 comprehensive_backtest.py --pair $pair --timeframe 1h --strategy integrated --optimize --plot --output $RESULT_DIR/$(echo $pair | tr '/' '')_1h_integrated 2>&1 | tee -a $LOGFILE
done

# Step 7: Run advanced ML model training 
echo "Step 7: Training ML models for all assets..." | tee -a $LOGFILE
for pair in "SOL/USD" "BTC/USD" "ETH/USD"; do
    pair_code=$(echo $pair | tr '/' '')
    if [ -f "advanced_ml_training.py" ]; then
        echo "Training advanced ML models for $pair..." | tee -a $LOGFILE
        python3 advanced_ml_training.py --symbol $pair_code --epochs 100 --use-low-timeframe --save-models 2>&1 | tee -a $LOGFILE
    else
        echo "ML training script not found, skipping this step" | tee -a $LOGFILE
    fi
done

# Step 8: Testing ML models with auto-pruning of unprofitable components
echo "Step 8: Testing and optimizing ML models with auto-pruning..." | tee -a $LOGFILE
for pair in "SOL/USD" "BTC/USD" "ETH/USD"; do
    pair_code=$(echo $pair | tr '/' '')
    echo "Testing ML models for $pair with auto-pruning..." | tee -a $LOGFILE
    python3 evaluate_ensemble_models.py --symbol $pair_code --auto-prune --save-best 2>&1 | tee -a $LOGFILE
done

# Step 9: Run optimized multi-strategy backtest with all assets
echo "Step 9: Running multi-strategy backtest for all assets..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --multi-asset --multi-strategy --plot --output $RESULT_DIR/multi_asset_multi_strategy 2>&1 | tee -a $LOGFILE

# Step 10: Test signal strength mechanism
echo "Step 10: Testing signal strength optimization..." | tee -a $LOGFILE
python3 signal_strength_optimizer.py --pairs SOL/USD BTC/USD ETH/USD --timeframes 1h 4h --plot --output $RESULT_DIR/signal_strength_test 2>&1 | tee -a $LOGFILE

# Step 11: Run the full ML-enhanced trading strategy test
echo "Step 11: Testing ML-enhanced strategy with optimized parameters..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --pair SOL/USD --timeframe 1h --strategy ml_enhanced --use-best-params --plot --output $RESULT_DIR/SOLUSD_ml_enhanced 2>&1 | tee -a $LOGFILE

# Step 12: Run the comprehensive integrated strategy test that combines everything
echo "Step 12: Testing comprehensive integrated strategy..." | tee -a $LOGFILE
python3 comprehensive_backtest.py --full-system --use-ml --use-low-timeframe --multi-asset --plot --output $RESULT_DIR/full_system_test 2>&1 | tee -a $LOGFILE

# Step 13: Run the automated optimization process for signal strength tuning
echo "Step 13: Running signal strength auto-tuning..." | tee -a $LOGFILE
python3 run_strategy_optimization.py --auto-tune-signals --pairs SOL/USD BTC/USD ETH/USD --timeframes 1h 4h 2>&1 | tee -a $LOGFILE

# Final cleanup and reporting
echo "==================================================================" | tee -a $LOGFILE
echo "OPTIMIZATION SUMMARY" | tee -a $LOGFILE
echo "==================================================================" | tee -a $LOGFILE

# Generate summary of test results
echo "Generating performance summary..." | tee -a $LOGFILE
python3 performance_metrics.py --summarize-all --output $RESULT_DIR/performance_summary.json 2>&1 | tee -a $LOGFILE

# Log completion
echo "==================================================================" | tee -a $LOGFILE
echo "Full optimization process completed at $(date)" | tee -a $LOGFILE
echo "Optimization results saved to $RESULT_DIR" | tee -a $LOGFILE
echo "Log file saved to $LOGFILE" | tee -a $LOGFILE
echo "==================================================================" | tee -a $LOGFILE

# Display best performers
echo "Top performing strategies:" | tee -a $LOGFILE
if [ -f "$RESULT_DIR/performance_summary.json" ]; then
    # Parse and display top strategies from summary file
    grep "best_strategy" $RESULT_DIR/performance_summary.json | tee -a $LOGFILE
else
    echo "Summary file not found" | tee -a $LOGFILE
fi