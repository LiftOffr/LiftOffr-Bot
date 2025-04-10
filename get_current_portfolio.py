#!/usr/bin/env python3

# A simple script to get portfolio value from memory by inspecting running bot

import os
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create a cache file for the bot to write its portfolio value to
    cache_file = "portfolio_value.txt"
    
    # Write a request file that the bot can detect
    with open("portfolio_request.txt", "w") as f:
        f.write(str(time.time()))
    
    logger.info("Request for portfolio value sent to bot...")
    logger.info("Waiting for response (up to 10 seconds)...")
    
    # Wait for up to 10 seconds for the bot to respond
    for _ in range(10):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = f.read().strip()
                    if data:
                        portfolio_data = json.loads(data)
                        logger.info(f"Current portfolio value: ${portfolio_data['value']:.2f}")
                        logger.info(f"Profit: ${portfolio_data['profit']:.2f} ({portfolio_data['profit_percent']:.2f}%)")
                        logger.info(f"Position: {portfolio_data.get('position', 'None')}")
                        logger.info(f"Entry price: ${portfolio_data.get('entry_price', 0):.2f}")
                        logger.info(f"Current price: ${portfolio_data.get('current_price', 0):.2f}")
                        logger.info(f"Trade count: {portfolio_data.get('trade_count', 0)}")
                        break
            except Exception as e:
                logger.error(f"Error reading portfolio data: {e}")
        
        # Sleep for 1 second before checking again
        time.sleep(1)
    else:
        logger.info("No response received from bot. Default portfolio value is $20,000.00")
    
    # Clean up the request file
    if os.path.exists("portfolio_request.txt"):
        os.remove("portfolio_request.txt")
    
    # Clean up the response file
    if os.path.exists(cache_file):
        os.remove(cache_file)

if __name__ == "__main__":
    main()