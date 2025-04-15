#!/usr/bin/env python3
"""
Isolated Runner Script

This is a completely isolated script that runs our trading bot
without importing any Flask-related modules at all.
"""
import os
import sys
import subprocess
import signal

# Set environment variables to prevent Flask
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["FLASK_RUN_PORT"] = "8080"
os.environ["FLASK_APP"] = "none"
os.environ["NO_FLASK"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Print some info
print("\n" + "=" * 60)
print(" ISOLATED TRADING BOT LAUNCHER - PURE PYTHON")
print("=" * 60)
print("\nThis launcher avoids importing Flask completely\n")
print("Starting bot process...\n")

# Create data directory
os.makedirs("data", exist_ok=True)

# Define a signal handler
def signal_handler(sig, frame):
    print("\nInterrupt received, shutting down...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    # Run the isolated bot as a separate process
    process = subprocess.Popen(
        [sys.executable, "-B", "isolated_bot.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ
    )
    
    # Wait for the process to complete
    process.wait()
    
    print("\nBot process exited with code:", process.returncode)
    
except Exception as e:
    print(f"Error running isolated bot: {e}")
finally:
    print("\n" + "=" * 60)
    print(" ISOLATED TRADING BOT SHUTDOWN COMPLETE")
    print("=" * 60)

# Exit with the same code as the subprocess
return_code = 0
if 'process' in locals() and process is not None and process.returncode is not None:
    return_code = process.returncode
sys.exit(return_code)