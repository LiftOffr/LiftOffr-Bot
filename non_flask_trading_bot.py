#!/usr/bin/env python3
"""
Non-Flask Trading Bot

This script runs the integrated trading bot without importing Flask dependencies.
"""
import sys
import os
import time
import logging
import argparse
import integrated_trading_bot

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Mark a special flag to prevent Flask from importing
os.environ['NO_FLASK'] = 'true'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading bot")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                                 "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"],
                        help="Trading pairs to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Start the trading bot
        bot = integrated_trading_bot.IntegratedTradingBot(
            trading_pairs=args.pairs,
            sandbox=args.sandbox
        )
        
        # Start the bot
        bot.start()
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTrading bot stopped by user")
            bot.cleanup()
    
    except Exception as e:
        logger.exception(f"Error running trading bot: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())