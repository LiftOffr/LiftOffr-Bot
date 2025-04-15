#!/bin/bash

# Kill any existing trading bot processes
pkill -f "python.*run_bot.py" || true

# Run the trading bot in the background
echo "Starting trading bot in background..."
nohup python3 -B run_bot.py > trading_bot.log 2>&1 &

# Save the PID
echo $! > trading_bot.pid
echo "Trading bot started with PID $(cat trading_bot.pid)"
echo "View logs with: tail -f trading_bot.log"