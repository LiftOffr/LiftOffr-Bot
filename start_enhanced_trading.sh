#!/bin/bash
# Enhanced Trading System Startup Script
# This script starts all components of the enhanced trading system

# Create necessary directories
mkdir -p logs
mkdir -p market_context
mkdir -p models
mkdir -p plots
mkdir -p static
mkdir -p templates

# Set up log rotation
echo "Setting up log rotation..."
python log_manager.py --setup-auto

# Start dashboard in background
echo "Starting dashboard on port 8050..."
nohup python dashboard.py > logs/dashboard.log 2>&1 &
echo $! > dashboard.pid
echo "Dashboard started with PID $(cat dashboard.pid)"

# Check if trading bot is already running
if [ -f "integrated_strategy.pid" ]; then
    echo "Trading bot already running with PID $(cat integrated_strategy.pid)"
else
    # Start integrated trading strategy with sandbox mode
    echo "Starting integrated trading strategy in sandbox mode..."
    nohup python main.py --strategy integrated --sandbox > logs/integrated_strategy.log 2>&1 &
    echo $! > integrated_strategy.pid
    echo "Integrated strategy started with PID $(cat integrated_strategy.pid)"
fi

# Display info
echo ""
echo "=================================================================================="
echo "Enhanced Trading System Started"
echo "=================================================================================="
echo ""
echo "Dashboard: http://localhost:8050"
echo "Log Directory: ./logs"
echo ""
echo "To view trading logs in real-time:"
echo "  tail -f logs/integrated_strategy.log"
echo ""
echo "To stop the system:"
echo "  ./stop_enhanced_trading.sh"
echo ""
echo "=================================================================================="