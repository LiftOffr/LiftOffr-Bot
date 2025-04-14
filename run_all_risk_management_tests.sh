#!/bin/bash

# Run all risk management tests and optimization
# This script orchestrates the complete testing and optimization process

echo "===== RISK MANAGEMENT SYSTEM TESTING AND OPTIMIZATION ====="
echo "Starting comprehensive testing and optimization..."

# Create necessary directories
mkdir -p test_data
mkdir -p test_results
mkdir -p backtest_results
mkdir -p optimization_results
mkdir -p logs

# Step 1: Run risk management tests
echo ""
echo "===== STEP 1: Testing Risk Management System ====="
python test_risk_management.py

# Step 2: Run risk-aware optimization for all pairs
echo ""
echo "===== STEP 2: Running Risk-Aware Optimization ====="
python run_risk_aware_optimization.py

# Step 3: Apply optimized settings to the trading system
echo ""
echo "===== STEP 3: Applying Optimized Settings ====="
python optimize_with_risk_management.py

# Step 4: Run enhanced backtest with new settings
echo ""
echo "===== STEP 4: Running Enhanced Backtest ====="
python risk_enhanced_backtest.py --plot

echo ""
echo "===== ALL PROCESSES COMPLETED ====="
echo "Risk management system has been tested, optimized, and applied to the trading system."
echo "Check the results in the following directories:"
echo "  - test_results/: Risk management test results"
echo "  - optimization_results/: Optimization results"
echo "  - backtest_results/: Enhanced backtest results"
echo ""
echo "The trading bot is now using the optimized risk management system."