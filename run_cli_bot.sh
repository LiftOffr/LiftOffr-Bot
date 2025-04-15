#!/bin/bash
# Simple shell script to run the standalone CLI bot
# This bypasses any potential Flask conflicts by using a direct Python call

echo "================================================"
echo " KRAKEN TRADING BOT LAUNCHER (SHELL SCRIPT)"
echo "================================================"
echo
echo "Launching standalone CLI bot..."
echo "This script bypasses any Flask conflicts"
echo

# Set environment variable to prevent Flask from starting
export TRADING_BOT_PROCESS=true

# Run the standalone CLI bot with Python
# The -B flag prevents writing .pyc files
python -B standalone_cli_bot.py

echo
echo "Bot exited."