#!/bin/bash
# Run Isolated Trading Bot
# This script runs the isolated trading bot without any Flask components

echo "================================================"
echo " ISOLATED TRADING BOT LAUNCHER"
echo "================================================"
echo
echo "This bot uses real-time market data in sandbox mode"
echo "with NO Flask dependencies!"
echo
echo "Starting bot..."
echo

# Make sure no Flask processes are running
pkill -f "flask run" > /dev/null 2>&1 || true
pkill -f "gunicorn" > /dev/null 2>&1 || true
sleep 1

# Set environment variables to prevent Flask from starting
export TRADING_BOT_PROCESS=1 
export FLASK_RUN_PORT=8080
export FLASK_APP=none
export NO_FLASK=1
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p data

# Start the isolated bot
exec python -B isolated_bot.py