#!/bin/bash
# Run Realtime Sandbox Trading Bot
# This script runs the realtime trading bot using bash to prevent 
# conflicts with Flask/port issues

echo "================================================"
echo " REALTIME SANDBOX TRADING BOT LAUNCHER"
echo "================================================"
echo
echo "This bot uses real-time market data in sandbox mode"
echo
echo "Starting bot..."
echo

# Set environment variables to prevent Flask from starting
export TRADING_BOT_PROCESS=1
export FLASK_RUN_PORT=5001

# Start the bot in Python directly to avoid any potential Flask issues
python -B realtime_sandbox_bot.py

echo
echo "Bot exited."
echo "================================================"