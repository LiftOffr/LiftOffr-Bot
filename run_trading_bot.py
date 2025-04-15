#!/usr/bin/env python3
"""
Trading Bot Runner with Port Management

This script provides an isolated way to run the trading bot
on a different port than the dashboard.
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

# Set environment variable to indicate trading bot process
os.environ["TRADING_BOT_PROCESS"] = "1"

def run_bot():
    """Run the trading bot in a separate process"""
    logger.info("Starting trading bot on port 8000")
    
    try:
        # Run the isolated bot directly
        cmd = ["python", "isolated_trading_bot.py", "--port", "8000"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run with subprocess and capture output
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        logger.info(f"Bot process completed with output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running trading bot: {e}")
        logger.error(f"Output from process:\n{e.output}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running trading bot: {e}")
        return False

def check_if_file_exists(filename):
    """Check if a file exists or try to create it"""
    if os.path.exists(filename):
        logger.info(f"Using existing file: {filename}")
        return True
    
    # Try to create a basic version of the file
    if filename == "isolated_trading_bot.py":
        logger.info("Creating basic isolated trading bot file")
        with open(filename, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Isolated Trading Bot

This bot runs independently of the Flask dashboard.
\"\"\"
import os
import sys
import time
import argparse
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Isolated Trading Bot')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logger.info(f"Starting isolated trading bot on port {args.port}")
    logger.info(f"Sandbox mode: {args.sandbox}")
    
    # Run the bot
    try:
        # Import the actual trading functions
        from trade_entry_manager import TradeEntryManager
        
        # Create trade manager
        trade_manager = TradeEntryManager()
        
        # Get available capital
        available = trade_manager.get_available_capital()
        logger.info(f"Available capital: ${available:.2f}")
        
        # Main trading loop
        while True:
            logger.info("Trading bot is running...")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in trading bot: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
        return True
    
    logger.error(f"File not found and could not be created: {filename}")
    return False

if __name__ == "__main__":
    logger.info("Trading Bot Runner: Starting")
    
    # Check for required files
    if check_if_file_exists("isolated_trading_bot.py"):
        # Run the bot
        run_bot()
    else:
        logger.error("Required files are missing, cannot run trading bot")
        sys.exit(1)