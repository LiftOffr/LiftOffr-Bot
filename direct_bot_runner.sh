#!/bin/bash

# This script runs the trading bot directly, completely bypassing Flask

# Set environment variables to prevent Flask loading
export TRADING_BOT_PROCESS=1
export DISABLE_FLASK=1
export NO_FLASK=1
export BYPASS_MAIN=1

# Print header
echo "============================================================"
echo " DIRECTLY RUNNING ISOLATED TRADING BOT"
echo "============================================================"

# Execute the isolated trading bot script directly
exec python3 -B isolated_trading_bot.py