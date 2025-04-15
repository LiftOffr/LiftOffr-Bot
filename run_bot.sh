#!/bin/bash

# Direct shell script to run the trading bot without any Flask dependencies

echo "===================================================="
echo " DIRECT TRADING BOT LAUNCHER"
echo "===================================================="
echo

# Set environment variables to prevent Flask
export TRADING_BOT_PROCESS=1
export FLASK_APP=none
export NO_FLASK=1
export PYTHONUNBUFFERED=1

# Create data directory
mkdir -p data

# Run the direct runner
exec python -B direct_runner.py