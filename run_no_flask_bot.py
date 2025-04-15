#!/usr/bin/env python
"""
Run No-Flask Bot

This is a special runner that completely bypasses the Flask/main.py import system.
It executes the trading bot in a subprocess to guarantee no Flask imports.
"""
import os
import sys
import time
import signal
import subprocess

def main():
    """Run the trading bot in an isolated subprocess"""
    
    # Create data directory if needed
    os.makedirs("data", exist_ok=True)
    
    # Set up environment to prevent Flask
    env = os.environ.copy()
    env["TRADING_BOT_PROCESS"] = "1"
    env["NO_FLASK"] = "1"
    env["FLASK_APP"] = "none"
    env["FLASK_RUN_PORT"] = "8080"
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    
    # Print banner
    print("\n" + "=" * 60)
    print(" NO-FLASK TRADING BOT")
    print("=" * 60 + "\n")
    print("Running trading bot in a completely isolated process...")
    
    # Create the isolated process
    cmd = [sys.executable, "-B", "isolated_bot.py"]
    
    try:
        # Run the process and forward all output
        bot_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
        
        # Wait for the process
        print(f"\nBot process started with PID {bot_process.pid}\n")
        
        # Handle keyboard interrupt
        def signal_handler(sig, frame):
            if bot_process.poll() is None:
                print("\nStopping bot process...")
                bot_process.terminate()
                time.sleep(2)
                if bot_process.poll() is None:
                    print("Force killing bot process...")
                    bot_process.kill()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process to finish
        bot_process.wait()
        
        # Print exit status
        if bot_process.returncode == 0:
            print("\nBot process completed successfully.")
        else:
            print(f"\nBot process exited with code {bot_process.returncode}.")
        
    except Exception as e:
        print(f"Error running bot process: {e}")
    finally:
        print("\n" + "=" * 60)
        print(" BOT PROCESS COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()