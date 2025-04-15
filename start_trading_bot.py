#!/usr/bin/env python3
"""
Start Trading Bot

This script starts the enhanced trading bot with proper risk management.
It ensures the bot runs separately from the web dashboard.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the trading bot"""
    logger.info("Starting enhanced trading bot...")
    
    # Make sure required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Command to run with sandbox mode
    cmd = [
        "python", 
        "run_enhanced_trading_bot.py", 
        "--sandbox",
        "--verbose"
    ]
    
    try:
        # Run the trading bot
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Trading bot exited with code {return_code}")
            return 1
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping trading bot...")
        process.terminate()
        process.wait(timeout=5)
        return 0
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())