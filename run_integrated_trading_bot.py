#!/usr/bin/env python3
"""
Run script for integrated trading bot

This script sets a special flag to prevent Flask from starting,
then imports and runs the integrated trading bot.
"""

# Set a flag to prevent Flask from starting if it's imported
import sys
sys._called_from_test = True

# Import and run the trading bot
if __name__ == "__main__":
    # Use command line args directly
    import integrated_trading_bot
    integrated_trading_bot.main()