#!/bin/bash
# Script to start the integrated trading bot in sandbox mode with detailed logging

echo "Starting integrated strategy trading bot with detailed logging..."

# Kill any previous instances
kill -9 $(cat integrated_strategy.pid 2>/dev/null) >/dev/null 2>&1 || true

# Start the bot with special logging prefixes for integrated strategy
PYTHONUNBUFFERED=1 python main.py --strategy integrated --sandbox 2>&1 | tee integrated_strategy_log.txt | grep -E "【INTEGRATED|【ANALYSIS|【INDICATORS|【VOLATILITY|【BANDS|【SIGNAL|【ACTION" &

# Save the PID of the background process
echo $! > integrated_strategy.pid
echo "Bot started! View the logs in real-time with: tail -f integrated_strategy_log.txt"
echo "Or analyze logs with: python analyze_integrated_logs.py"
echo "Process ID: $(cat integrated_strategy.pid)"