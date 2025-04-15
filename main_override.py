#!/usr/bin/env python3
"""
Main Override

This file is used to override the normal main.py import system.
It executes the direct bot runner to bypass Flask completely.
"""
import os
import sys
import time
import signal
import subprocess

print("\n" + "=" * 60)
print(" TRADING BOT LAUNCHER - SKIP FLASK")
print("=" * 60 + "\n")

# Set environment variables to prevent Flask
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["FLASK_APP"] = "none"
os.environ["NO_FLASK"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Make sure data directory exists
os.makedirs("data", exist_ok=True)

try:
    print("Executing trading bot directly...")
    
    # Exit any existing processes
    try:
        print("Checking for existing Flask processes...")
        subprocess.run(["pkill", "-f", "flask"], check=False)
        print("Checking for existing bots...")
        subprocess.run(["pkill", "-f", "direct_runner.py"], check=False)
        time.sleep(1)
    except Exception as e:
        print(f"Error cleaning up processes: {e}")
    
    print("\nStarting direct trader...\n")
    
    # Execute direct runner as a subprocess
    process = subprocess.Popen(
        [sys.executable, "-B", "direct_runner.py"],
        env=os.environ
    )
    
    def signal_handler(sig, frame):
        print("\nInterrupt received, shutting down...")
        if process.poll() is None:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for process to finish
    process.wait()
    
except Exception as e:
    print(f"Error running bot: {e}")
finally:
    print("\n" + "=" * 60)
    print(" TRADING BOT COMPLETE")
    print("=" * 60)