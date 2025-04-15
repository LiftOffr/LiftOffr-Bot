#!/usr/bin/env python3

"""
Start Sandbox Trader

This script starts the sandbox trader with the optimized settings
from our enhanced training process. It runs the trader in the background
so it continues to operate and optimize models during live trading.

Usage:
    python start_sandbox_trader.py
"""

import os
import sys
import time
import subprocess
import logging
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_command(command, description=None):
    """Run a command and log the output"""
    if description:
        logger.info(description)
        
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

def main():
    """Main function"""
    logger.info("Starting sandbox trader with optimized settings")
    
    # Start sandbox trader
    sandbox_process = run_command(
        ["python", "run_sandbox_trader.py", "--reset-portfolio", "--continuous-training"],
        "Starting sandbox trader"
    )
    
    if not sandbox_process:
        logger.error("Failed to start sandbox trader")
        return 1
    
    # Store PID for later reference
    with open("bot_pid.txt", "w") as f:
        f.write(str(sandbox_process.pid))
        
    logger.info(f"Sandbox trader started with PID {sandbox_process.pid}")
    logger.info("Use 'kill $(cat bot_pid.txt)' to stop the trader")
    
    # Wait briefly to make sure process starts
    time.sleep(2)
    
    # Check if process is still running
    if sandbox_process.poll() is not None:
        logger.error(f"Sandbox trader exited with code {sandbox_process.returncode}")
        logger.error(f"stdout: {sandbox_process.stdout.read()}")
        logger.error(f"stderr: {sandbox_process.stderr.read()}")
        return 1
    
    logger.info("Sandbox trader is running in the background")
    logger.info("You can check the portfolio status using: python check_metrics.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())