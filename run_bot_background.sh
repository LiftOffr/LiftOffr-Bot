#!/bin/bash

# This script runs the trading bot in the background without Flask
# preventing port conflicts with the dashboard

# Set environment variables to prevent Flask loading
export TRADING_BOT_PROCESS=1
export DISABLE_FLASK=1
export NO_FLASK=1
export FLASK_RUN_PORT=5001
export PYTHONPATH=$(pwd)

# Create necessary directories
mkdir -p logs data

echo "======================================================================"
echo " STARTING TRADING BOT IN BACKGROUND (FLASK ISOLATED)"
echo " Log file: logs/trading_bot.log"
echo "======================================================================"

# Run the bot in the background using nohup
nohup python3 -B no_flask_bot.py > logs/bot_stdout.log 2>&1 &
BOT_PID=$!

echo "Trading bot started with PID: $BOT_PID"
echo "To check logs, run: tail -f logs/trading_bot.log"
echo "To stop the bot, run: kill $BOT_PID"

# Write PID to a file for later reference
echo $BOT_PID > .bot_pid.txt