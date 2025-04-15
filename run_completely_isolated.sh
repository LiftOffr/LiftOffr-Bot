#!/bin/bash

# This script runs the completely isolated trading bot without any Flask dependencies

# Set environment variables to prevent Flask loading
export TRADING_BOT_PROCESS=1
export DISABLE_FLASK=1
export NO_FLASK=1
export FLASK_RUN_PORT=5001
export PYTHONPATH=$(pwd)

# Print header
echo "============================================================"
echo " RUNNING COMPLETELY ISOLATED TRADING BOT"
echo "============================================================"

# Make sure logs directory exists
mkdir -p logs data

# Execute the isolated trading bot script directly with -B to disable bytecode
exec python3 -B completely_isolated_bot.py