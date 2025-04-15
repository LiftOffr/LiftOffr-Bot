#!/bin/bash

# This script runs the trading bot directly 
# without Flask, preventing port conflicts with the dashboard

echo "Starting trading bot in isolated mode..."
python no_flask_bot.py