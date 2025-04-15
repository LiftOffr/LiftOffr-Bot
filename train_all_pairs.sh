#!/bin/bash
# Train models for all trading pairs one by one

echo "======================================================================"
echo "Training models for all trading pairs"
echo "======================================================================"

# Define pairs
PAIRS=(
    "SOL/USD"
    "BTC/USD"
    "ETH/USD"
    "ADA/USD"
    "DOT/USD"
    "LINK/USD"
    "AVAX/USD"
    "MATIC/USD"
    "UNI/USD"
    "ATOM/USD"
)

# Train each pair with fewer epochs to speed up training
for pair in "${PAIRS[@]}"; do
    echo "--------------------------------------"
    echo "Training model for $pair"
    echo "--------------------------------------"
    python train_and_activate_model.py --pair "$pair" --epochs 5 --batch-size 32
    echo "Completed training for $pair"
    echo ""
done

echo "======================================================================"
echo "All models trained. Restarting trading bot..."
echo "======================================================================"

# Restart the trading bot
echo "Models trained for all pairs. Restarting the trading bot..."
touch .trading_bot_restart_trigger

echo "Done!"