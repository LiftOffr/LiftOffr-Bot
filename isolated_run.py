#!/usr/bin/env python3
"""
Completely isolated runner script that bypasses any Flask imports
"""
import os
import sys
import subprocess

# Prevent Flask from loading
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["DISABLE_FLASK"] = "1"
os.environ["NO_FLASK"] = "1"

print("Starting isolated trading bot...")
print("=" * 60)

# Run the isolated trading bot as a subprocess
# This ensures complete isolation from any Flask imports
try:
    subprocess.run(
        ["python3", "-B", "isolated_trading_bot.py"], 
        check=True
    )
except KeyboardInterrupt:
    print("\nTrading bot stopped by user")
except subprocess.CalledProcessError as e:
    print(f"Trading bot exited with error code {e.returncode}")
except Exception as e:
    print(f"Error running trading bot: {e}")