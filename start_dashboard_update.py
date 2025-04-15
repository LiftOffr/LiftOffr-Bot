#!/usr/bin/env python3

"""
Start Dashboard Auto Update

This script starts the auto update process for the dashboard,
which will update the portfolio data at regular intervals.

Usage:
    python start_dashboard_update.py
"""

import os
import sys
import time
import logging
import subprocess
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    try:
        # Start the auto update process
        logger.info("Starting dashboard auto update process...")
        process = subprocess.Popen(
            [sys.executable, "auto_update_dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        logger.info(f"Started auto update process with PID {process.pid}")
        
        # Monitor the process
        while True:
            # Check if process is still running
            if process.poll() is not None:
                logger.warning(f"Auto update process exited with code {process.returncode}")
                
                # Restart the process
                logger.info("Restarting auto update process...")
                process = subprocess.Popen(
                    [sys.executable, "auto_update_dashboard.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                logger.info(f"Restarted auto update process with PID {process.pid}")
            
            # Read output from the process
            output = process.stdout.readline()
            if output:
                print(output.strip())
                
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping auto update process...")
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
        logger.info("Auto update process stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())