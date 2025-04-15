#!/usr/bin/env python3
"""
Standalone bot runner that avoids Flask imports completely
"""
import os
import sys

# Set environment variable to indicate this is a trading bot process
os.environ["TRADING_BOT_PROCESS"] = "1"

# Run the isolated trading bot directly
print("Starting isolated trading bot...")
import isolated_trading_bot
isolated_trading_bot.main()