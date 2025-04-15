#!/usr/bin/env python3
"""
Simple launcher for the trading bot

This script simply launches the standalone trading bot directly using subprocess
without importing any modules, to avoid Flask conflicts.
"""
import os
import sys
import subprocess
import time

def main():
    """Main function to run the standalone bot"""
    print("=" * 60)
    print("TRADING BOT LAUNCHER")
    print("=" * 60)
    print("\nStarting isolated trading bot...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create command with the current Python executable
    cmd = [sys.executable, "standalone_trading_bot.py", "--sandbox"]
    
    # Run the bot
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    process = None
    try:
        # Run the bot in a subprocess with output displayed in current terminal
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Stopping trading bot...")
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Bot didn't terminate cleanly, forcing kill...")
                process.kill()
        print("Trading bot stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())