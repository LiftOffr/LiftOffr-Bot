#!/bin/bash

# Run the isolated trading bot in the background
echo "Starting isolated trading bot in background..."
nohup python -B isolated_trading_bot.py > trading_bot.log 2>&1 &

# Save the process ID for later reference
echo $! > trading_bot.pid
echo "Trading bot started with PID $(cat trading_bot.pid)"
echo "View logs with: tail -f trading_bot.log"