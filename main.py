from app import app  # noqa: F401
import sys
import os
import threading

# This file is required to run the Flask app on Replit
# The import above makes the Flask app instance available to Replit's server

def run_trading_bot():
    """Run the enhanced trading bot in the background"""
    import time
    import logging
    from run_enhanced_trading_bot import run_trading_bot as enhanced_run_trading_bot
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    class Args:
        pairs = "BTC/USD,ETH/USD,SOL/USD,ADA/USD,DOT/USD,LINK/USD,AVAX/USD,MATIC/USD,UNI/USD,ATOM/USD"
        sandbox = True
    
    # Run trading bot
    enhanced_run_trading_bot(Args())

# Check if this script is being run via the trading_bot workflow
if "python main.py --sandbox" in " ".join(sys.argv):
    # Start trading bot in a background thread
    trading_thread = threading.Thread(target=run_trading_bot)
    trading_thread.daemon = True
    trading_thread.start()
    print("Trading bot started in background thread")