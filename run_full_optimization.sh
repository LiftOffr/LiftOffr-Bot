#!/bin/bash

# Full Optimization Pipeline for Hyper-Optimized Trading Bot
# This script runs the complete optimization pipeline for maximum profit:
# 1. Fetches the latest historical data
# 2. Trains and optimizes ML models for all assets
# 3. Generates optimized position sizing configurations
# 4. Runs comprehensive backtests to validate performance
# 5. (Optionally) Starts live trading with optimized parameters

# Default settings
RISK_LEVEL="balanced"  # balanced, aggressive, or ultra_aggressive
DATA_DAYS=90           # Days of historical data to use
CAPITAL=20000          # Initial capital for backtesting/trading
RETRAIN=false          # Whether to retrain ML models
BACKTEST=true          # Whether to run backtests
OPTIMIZE=true          # Whether to optimize parameters
LIVE=false             # Whether to start live trading
ASSETS="SOL/USD ETH/USD BTC/USD"  # Assets to trade

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --risk-level)
      RISK_LEVEL="$2"
      shift 2
      ;;
    --data-days)
      DATA_DAYS="$2"
      shift 2
      ;;
    --capital)
      CAPITAL="$2"
      shift 2
      ;;
    --retrain)
      RETRAIN=true
      shift
      ;;
    --no-backtest)
      BACKTEST=false
      shift
      ;;
    --no-optimize)
      OPTIMIZE=false
      shift
      ;;
    --live)
      LIVE=true
      shift
      ;;
    --assets)
      ASSETS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set risk level-specific configuration
echo "Setting position sizing for risk level: $RISK_LEVEL"
case $RISK_LEVEL in
  aggressive)
    cp position_sizing_config_ultra_aggressive.json position_sizing_config.json
    echo "Using aggressive position sizing configuration"
    ;;
  ultra_aggressive)
    python dynamic_position_sizing_ml.py  # Generate ultra-aggressive config
    echo "Using ultra-aggressive position sizing configuration"
    ;;
  *)
    # Use default balanced configuration
    echo "Using balanced position sizing configuration"
    ;;
esac

# Validate Kraken API keys for live trading
if $LIVE; then
  if [ -z "$KRAKEN_API_KEY" ] || [ -z "$KRAKEN_API_SECRET" ]; then
    echo "Error: Kraken API keys are required for live trading. Please set KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables."
    exit 1
  fi
  echo "Valid Kraken API keys detected for live trading"
fi

# Step 1: Fetch historical data
echo "==============================================="
echo "Step 1: Fetching historical data for $DATA_DAYS days"
echo "==============================================="
python enhanced_historical_data_fetcher.py --days $DATA_DAYS --assets $ASSETS

# Step 2: Optimize ML models (if enabled)
if $RETRAIN; then
  echo "==============================================="
  echo "Step 2: Training and optimizing ML models"
  echo "==============================================="
  python optimize_ml_for_all_assets.py --assets $ASSETS

  # Step 3: Optimize dynamic position sizing
  echo "==============================================="
  echo "Step 3: Optimizing position sizing parameters"
  echo "==============================================="
  python dynamic_position_sizing_ml.py
fi

# Step 4: Run comprehensive backtests (if enabled)
if $BACKTEST; then
  echo "==============================================="
  echo "Step 4: Running comprehensive backtests"
  echo "==============================================="
  
  if $OPTIMIZE; then
    # Run with parameter optimization
    python run_enhanced_backtesting.py --optimize --trials 10 --capital $CAPITAL --days $DATA_DAYS --save-parameters
  else
    # Run without parameter optimization
    python run_enhanced_backtesting.py --capital $CAPITAL --days $DATA_DAYS
  fi
fi

# Step 5: Start live trading (if enabled)
if $LIVE; then
  echo "==============================================="
  echo "Step 5: Starting live trading"
  echo "==============================================="
  
  # Start trading with optimized parameters
  if [ "$RISK_LEVEL" == "ultra_aggressive" ]; then
    echo "Starting live trading with ULTRA AGGRESSIVE risk profile"
    python main.py --multi-strategy "integrated,ml,arima,adaptive" --capital $CAPITAL --live --aggressive
  elif [ "$RISK_LEVEL" == "aggressive" ]; then
    echo "Starting live trading with AGGRESSIVE risk profile"
    python main.py --multi-strategy "integrated,ml,arima,adaptive" --capital $CAPITAL --live
  else
    echo "Starting live trading with BALANCED risk profile"
    python main.py --multi-strategy "arima,adaptive" --capital $CAPITAL --live
  fi
else
  echo "Live trading not enabled. Run with --live flag to start trading."
fi

echo "==============================================="
echo "Full optimization pipeline complete"
echo "==============================================="