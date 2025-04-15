#!/usr/bin/env python3
"""
Run Standalone Trading Bot

This script runs the standalone trading bot in a way that
doesn't trigger Flask to start, avoiding port conflicts
with the dashboard.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    # Set environment variable to prevent Flask from starting
    os.environ["TRADING_BOT_PROCESS"] = "1"
    
    print("\n" + "=" * 60)
    print(" ISOLATED TRADING BOT LAUNCHER")
    print("=" * 60 + "\n")
    
    print("Launching trading bot in isolated mode...")
    print("This will prevent port conflicts with the dashboard\n")
    
    # Launch the standalone trading bot
    try:
        # Use subprocess to completely isolate the process
        process = subprocess.Popen(
            [sys.executable, "-B", "standalone_trading_bot.py"],
            env=dict(os.environ, TRADING_BOT_PROCESS="1")
        )
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nLauncher interrupted, stopping bot...")
        if 'process' in locals():
            process.terminate()
        
    except Exception as e:
        print(f"\nError launching trading bot: {e}")
    
    print("\n" + "=" * 60)
    print(" LAUNCHER SHUTDOWN COMPLETE")
    print("=" * 60)