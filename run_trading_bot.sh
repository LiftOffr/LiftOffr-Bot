#!/bin/bash

# This script runs the simple trading bot directly 
# without Flask, preventing port conflicts with the dashboard

echo "Starting trading bot in isolated mode..."
python simple_trading_bot.py