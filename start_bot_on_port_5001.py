#!/usr/bin/env python3
"""
Start Trading Bot on Port 5001

This script starts the trading bot on port 5001 to avoid conflicts with the dashboard
"""

import os
import sys
import logging
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    # Run the trading bot on a different port to avoid conflicts
    cmd = ["python", "run_enhanced_simulation.py", "--sandbox", "--flash-crash", "--latency", "--interval", "1"]
    
    logger.info(f"Starting trading bot: {' '.join(cmd)}")
    
    # Execute the enhanced simulation directly
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted. Shutting down...")
        if process:
            process.terminate()
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())