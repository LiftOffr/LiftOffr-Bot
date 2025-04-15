#!/usr/bin/env python3
"""
Standalone runner for the trading bot

This script imports and runs the trading logic directly from simple_trading_bot.py
without any Flask dependencies.
"""

# Set flag to prevent Flask from starting when simple_trading_bot imports main
import sys
sys._called_from_test = True

# Explicitly avoid importing main.py
if __name__ == "__main__":
    import simple_trading_bot
    simple_trading_bot.main()