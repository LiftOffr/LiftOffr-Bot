#!/usr/bin/env python3
"""
Run Trading Bot in Debug Mode

This script runs the enhanced trading bot with detailed logging for debugging purposes.
It captures all exceptions and logs them to help identify issues.
"""
import os
import sys
import time
import traceback
import logging
from datetime import datetime

# Configure logging to a file and stdout with timestamp
log_file = f"trading_bot_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Record start time
start_time = datetime.now()
logger.info(f"Starting trading bot debug at {start_time}")

try:
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Import necessary modules
    logger.info("Importing run_enhanced_trading_bot...")
    from run_enhanced_trading_bot import run_trading_bot
    logger.info("Successfully imported run_enhanced_trading_bot")
    
    # Create simulated args
    logger.info("Setting up arguments...")
    class Args:
        pairs = "BTC/USD,ETH/USD,SOL/USD,ADA/USD,DOT/USD,LINK/USD,AVAX/USD,MATIC/USD,UNI/USD,ATOM/USD"
        strategies = "Adaptive,ARIMA"
        sandbox = True
    
    # Run the trading bot with a safety wrapper
    logger.info(f"Starting trading for pairs: {Args.pairs}")
    logger.info("Running in sandbox mode")
    
    # Run the trading bot
    run_trading_bot(Args())
    
    # If we get here, the bot completed normally
    logger.info("Trading bot completed normally")
    
except Exception as e:
    # Catch and log any exceptions
    logger.error(f"Error running trading bot: {str(e)}")
    logger.error(traceback.format_exc())
    
finally:
    # Log end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Trading bot debug ended at {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    
    print("="*80)
    print(f"Trading bot debug completed. See log file for details: {log_file}")
    print("="*80)