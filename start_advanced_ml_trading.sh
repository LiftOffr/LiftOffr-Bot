#!/bin/bash
# Start Advanced ML Trading System
# This script automates the entire process of launching the advanced ML trading system

# Display banner
echo "================================================================"
echo "                 ADVANCED ML TRADING SYSTEM                     "
echo "================================================================"
echo "Starting advanced ML trading system for cryptocurrency pairs..."
echo "Target: 90% prediction accuracy and 1000%+ returns"
echo "----------------------------------------------------------------"

# Create necessary directories
mkdir -p logs
mkdir -p logs/trading
mkdir -p logs/training
mkdir -p models
mkdir -p training_data
mkdir -p backtest_results
mkdir -p sentiment_data

# 1. Ensure ML dependencies are installed
echo "[1/5] Checking ML dependencies..."
python ensure_ml_dependencies.py
if [ $? -ne 0 ]; then
  echo "Failed to ensure ML dependencies. Exiting."
  exit 1
fi
echo "ML dependencies verified."

# 2. Generate ML configurations
echo "[2/5] Generating ML configurations..."
python create_ml_config.py
if [ $? -ne 0 ]; then
  echo "Failed to create ML configurations. Exiting."
  exit 1
fi
echo "ML configurations created successfully."

# 3. Check for existing trained models
echo "[3/5] Checking for existing ML models..."
MODELS_FOUND=0
PAIRS=("SOL/USD" "ETH/USD" "BTC/USD" "DOT/USD" "LINK/USD")
for pair in "${PAIRS[@]}"; do
  pair_dir=$(echo $pair | tr '/' '_')
  model_dir="models/$pair_dir"
  
  if [ -d "$model_dir" ] && [ "$(ls -A $model_dir 2>/dev/null)" ]; then
    echo "Models found for $pair"
    MODELS_FOUND=$((MODELS_FOUND+1))
  else
    echo "No models found for $pair (will be trained)"
  fi
done

# 4. Train ML models if needed
if [ $MODELS_FOUND -lt ${#PAIRS[@]} ]; then
  echo "[4/5] Training ML models for all trading pairs..."
  python train_ml_models_all_assets.py --force-train
  if [ $? -ne 0 ]; then
    echo "Warning: ML model training encountered issues. Continuing with existing models."
  fi
else
  echo "[4/5] All models already trained. Skipping training step."
fi

# 5. Start the ML sandbox trading system
echo "[5/5] Starting ML sandbox trading system..."
RISK_FACTOR=1.5  # Slightly aggressive
MAX_LEVERAGE=20  # Default max leverage
INITIAL_CAPITAL=20000  # Default initial capital

# Start with command line options
echo "Starting ML sandbox trading with:"
echo "- Trading pairs: ${PAIRS[@]}"
echo "- Initial capital: $INITIAL_CAPITAL"
echo "- Risk factor: $RISK_FACTOR"
echo "- Max leverage: $MAX_LEVERAGE"
echo "- Features: Cross-asset correlation, sentiment analysis, continuous training"

python start_ml_sandbox_trading.py \
  --trading-pairs "${PAIRS[@]}" \
  --initial-capital $INITIAL_CAPITAL \
  --risk-factor $RISK_FACTOR \
  --max-leverage $MAX_LEVERAGE \
  --use-correlation \
  --use-sentiment \
  --continuous-training \
  --model-update-interval 4

# Script execution complete
echo "================================================================"
echo "Advanced ML Trading System launched successfully!"
echo "Trading is now running in sandbox mode with ML enhancements."
echo "To view logs, check the logs directory."
echo "To stop the system, press Ctrl+C."
echo "================================================================"