#!/usr/bin/env python3
"""
Start Trading Bot

This script starts the enhanced trading bot with all 10 cryptocurrency pairs in sandbox mode.
"""
import sys
import logging
from run_enhanced_trading_bot import run_trading_bot

def main():
    """Main function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create arguments class
    class Args:
        pairs = "BTC/USD,ETH/USD,SOL/USD,ADA/USD,DOT/USD,LINK/USD,AVAX/USD,MATIC/USD,UNI/USD,ATOM/USD"
        sandbox = True
    
    # Run trading bot
    run_trading_bot(Args())

if __name__ == "__main__":
    main()