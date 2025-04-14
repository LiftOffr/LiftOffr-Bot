#!/bin/bash
# Run Full Optimization and Trading Script
# This script automates the entire process of:
# 1. Optimizing ML models for all assets
# 2. Running hyper-optimized multi-asset trading

# Set up logging
LOG_FILE="full_optimization_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo "STARTING FULL OPTIMIZATION AND TRADING PROCESS"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================================="

# Function to handle errors
handle_error() {
    echo "ERROR: $1"
    echo "Process aborted at $(date)"
    exit 1
}

# Step 1: Check environment and dependencies
echo "Checking environment..."
if [ ! -f "hyper_optimized_ml_training.py" ]; then
    handle_error "Required file hyper_optimized_ml_training.py not found!"
fi

if [ ! -f "run_hyper_optimized_trading.py" ]; then
    handle_error "Required file run_hyper_optimized_trading.py not found!"
fi

if [ ! -f "optimize_ml_for_all_assets.py" ]; then
    handle_error "Required file optimize_ml_for_all_assets.py not found!"
fi

# Check for API keys
if [ -z "$KRAKEN_API_KEY" ] || [ -z "$KRAKEN_API_SECRET" ]; then
    echo "WARNING: Kraken API credentials not found in environment."
    echo "Trading will run in sandbox mode."
    SANDBOX_MODE="--sandbox"
else
    echo "Kraken API credentials found."
    SANDBOX_MODE=""
fi

# Step 2: Parse command line arguments
RISK_LEVEL="moderate"
FETCH_DATA="--fetch-data"
FORCE_RETRAIN=""
ASSETS="SOL/USD ETH/USD BTC/USD"
INITIAL_CAPITAL=20000
LIVE_MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --risk-level)
            RISK_LEVEL="$2"
            shift 2
            ;;
        --no-fetch-data)
            FETCH_DATA=""
            shift
            ;;
        --retrain)
            FORCE_RETRAIN="--retrain"
            shift
            ;;
        --assets)
            ASSETS="$2"
            shift 2
            ;;
        --capital)
            INITIAL_CAPITAL="$2"
            shift 2
            ;;
        --live)
            LIVE_MODE="--live"
            SANDBOX_MODE=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "Risk Level: $RISK_LEVEL"
echo "Assets: $ASSETS"
echo "Initial Capital: $INITIAL_CAPITAL"
echo "Force Retrain: ${FORCE_RETRAIN:-No}"
echo "Fetch Data: ${FETCH_DATA:-No}"
echo "Live Mode: ${LIVE_MODE:-No}"
echo "Sandbox Mode: ${SANDBOX_MODE:-No}"

# Step 3: Run ML optimization
echo ""
echo "=========================================================="
echo "STARTING ML OPTIMIZATION"
echo "Started at: $(date)"
echo "=========================================================="

echo "Running ML optimization for all assets..."
python optimize_ml_for_all_assets.py --assets $ASSETS --risk-level $RISK_LEVEL $FORCE_RETRAIN $FETCH_DATA

if [ $? -ne 0 ]; then
    handle_error "ML optimization failed!"
fi

echo "ML optimization completed successfully at $(date)"

# Step 4: Run hyper-optimized trading
echo ""
echo "=========================================================="
echo "STARTING HYPER-OPTIMIZED TRADING"
echo "Started at: $(date)"
echo "=========================================================="

echo "Running hyper-optimized trading..."
python run_hyper_optimized_trading.py --assets $ASSETS --capital $INITIAL_CAPITAL $LIVE_MODE $SANDBOX_MODE

if [ $? -ne 0 ]; then
    handle_error "Hyper-optimized trading failed!"
fi

echo "Trading process completed successfully at $(date)"

# Final summary
echo ""
echo "=========================================================="
echo "PROCESS COMPLETED SUCCESSFULLY"
echo "Started at: $(cat "$LOG_FILE" | grep "STARTING FULL OPTIMIZATION" -A 2 | grep "Started at" | cut -d':' -f2-)"
echo "Finished at: $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================================="

exit 0