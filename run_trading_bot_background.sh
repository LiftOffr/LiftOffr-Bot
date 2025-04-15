#!/bin/bash

# Kill any existing trading bot processes
pkill -f "python.*run_bot.py" || true

# Ensure log directory exists
mkdir -p logs

# Run the trading bot in the background with better log handling
echo "Starting trading bot in background..."
nohup python3 -B run_bot.py > logs/trading_bot.log 2>&1 &

# Save the PID
echo $! > logs/trading_bot.pid
echo "Trading bot started with PID $(cat logs/trading_bot.pid)"
echo "View logs with: tail -f logs/trading_bot.log"