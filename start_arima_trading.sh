#!/bin/bash

# Script to start the Kraken trading bot with the ARIMA strategy

echo "Starting Kraken trading bot with ARIMA strategy..."

# Check command line arguments
SANDBOX_MODE="--sandbox"
if [ "$1" == "--live" ]; then
    SANDBOX_MODE="--live"
    echo "WARNING: Running in LIVE trading mode! Real trades will be executed."
    echo "Press Ctrl+C within 5 seconds to cancel..."
    sleep 5
fi

# Run the trading bot with ARIMA strategy
python main.py --strategy arima --pair SOLUSD $SANDBOX_MODE

echo "Trading bot stopped."