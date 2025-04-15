#!/bin/bash

# This script runs the isolated trading bot directly with Flask set to a different port

# Set environment variables
export TRADING_BOT_PROCESS=1
export DISABLE_FLASK=1
export NO_FLASK=1
export BYPASS_MAIN=1
export FLASK_APP=isolated_bot.py
export FLASK_RUN_PORT=5001

# Print header
echo "============================================================"
echo " RUNNING TRADING BOT ON PORT 5001"
echo "============================================================"

# Create necessary directories
mkdir -p data logs

# Run the isolated bot
exec python3 -B isolated_bot.py