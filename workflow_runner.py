#!/usr/bin/env python3
"""
Workflow runner script - Runs the simple trading bot
"""
import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Starting workflow runner...")
logger.info("Environment: " + str(os.environ))
logger.info("Args: " + str(sys.argv))

# Set environment variables to prevent Flask
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["DISABLE_FLASK"] = "1"
os.environ["NO_FLASK"] = "1"

# Run the simple trading bot script directly
try:
    logger.info("Running simple_trading_bot.py...")
    result = subprocess.run(
        [sys.executable, "simple_trading_bot.py"],
        check=True
    )
    logger.info(f"Bot exited with code {result.returncode}")
except KeyboardInterrupt:
    logger.info("Bot stopped by user")
except Exception as e:
    logger.error(f"Error running bot: {e}")
    
logger.info("Workflow runner exiting")