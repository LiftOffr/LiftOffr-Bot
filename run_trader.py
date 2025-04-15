#!/usr/bin/env python3
"""
Run Enhanced Trading Bot

This script runs the enhanced trading bot with fee simulation, 
liquidation risk, and dynamic parameters.
"""

import os
import time
import logging
import datetime
import threading
import subprocess
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_trading_bot():
    """Run the enhanced trading bot"""
    logger.info("Starting enhanced trading bot...")
    
    while True:
        try:
            # Run the main trading logic
            logger.info("Running trading cycle...")
            subprocess.run(["python", "run_enhanced_trading_bot.py", "--sandbox", "--verbose", "--interval", "1"], 
                           capture_output=True, 
                           text=True, 
                           check=False, 
                           timeout=60)  # 1 minute timeout
            
            # Check status of trades
            subprocess.run(["python", "check_trader_status.py"], check=False)
            
            # Wait a bit before next cycle
            logger.info("Sleeping for 5 minutes before next cycle...")
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            time.sleep(60)  # Wait a bit before retrying

if __name__ == "__main__":
    run_trading_bot()