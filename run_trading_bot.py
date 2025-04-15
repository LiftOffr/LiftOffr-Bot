#!/usr/bin/env python3
"""
Run Trading Bot

This script runs the Kraken trading bot in sandbox mode without starting the web interface.
It's used to run the trading bot when the web interface is already running in another process.
"""

import os
import sys
import time
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Kraken trading bot')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    parser.add_argument('--pairs', type=str, default='SOL/USD,BTC/USD,ETH/USD,ADA/USD,DOT/USD,LINK/USD',
                        help='Comma-separated list of trading pairs')
    return parser.parse_args()

def run_trading_bot(pairs, sandbox=True):
    """Run the trading bot"""
    # Log the start
    logger.info(f"Starting trading bot with pairs: {pairs}")
    
    # Convert pairs list to comma-separated string if needed
    if isinstance(pairs, list):
        pairs_str = ','.join(pairs)
    else:
        pairs_str = pairs
    
    # Run the trading command
    cmd = [sys.executable, "kraken_trading_bot.py"]
    if sandbox:
        cmd.append("--sandbox")
    cmd.extend(["--pairs", pairs_str])
    
    # Start the process
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Log output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
                
        return_code = process.poll()
        if return_code != 0:
            logger.error(f"Trading bot exited with code {return_code}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get trading pairs from command line or environment
    pairs = args.pairs.split(',')
    
    # Run continuously
    while True:
        try:
            # Run the trading bot
            success = run_trading_bot(pairs, args.sandbox)
            
            if not success:
                # If failed, wait before retrying
                logger.warning("Trading bot run failed, will retry in 60 seconds")
                time.sleep(60)
            else:
                # If successful but exited, wait a bit before starting again
                logger.info("Trading bot completed, restarting in 5 seconds")
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == '__main__':
    main()