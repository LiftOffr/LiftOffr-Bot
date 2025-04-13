#!/bin/bash
# Enhanced Trading System Shutdown Script
# This script stops all components of the enhanced trading system

# Stop dashboard
if [ -f "dashboard.pid" ]; then
    echo "Stopping dashboard with PID $(cat dashboard.pid)..."
    kill -15 $(cat dashboard.pid) 2>/dev/null || echo "Dashboard process not found"
    rm dashboard.pid
else
    echo "Dashboard PID file not found"
fi

# Stop integrated trading strategy
if [ -f "integrated_strategy.pid" ]; then
    echo "Stopping integrated strategy with PID $(cat integrated_strategy.pid)..."
    kill -15 $(cat integrated_strategy.pid) 2>/dev/null || echo "Integrated strategy process not found"
    rm integrated_strategy.pid
else
    echo "Integrated strategy PID file not found"
fi

# Additional cleanup
echo "Cleaning up temporary files..."

# Display info
echo ""
echo "=================================================================================="
echo "Enhanced Trading System Stopped"
echo "=================================================================================="