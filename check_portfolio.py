#!/usr/bin/env python3

# A script to check the current portfolio value from the kraken API and bot configuration

import os
import logging
import argparse
import time
import json
from kraken_api import KrakenAPI
from config import INITIAL_CAPITAL, MARGIN_PERCENT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_portfolio_status():
    """
    Attempts to find portfolio status from running bot or configuration
    
    Returns:
        dict: Dictionary containing portfolio information
    """
    # First, check if we can communicate with the running bot
    # Create a request file that will trigger the bot to write its status
    request_file = "portfolio_request.txt"
    response_file = "portfolio_value.txt"
    
    portfolio_data = {
        'value': INITIAL_CAPITAL,  # Default value
        'profit': 0.0,
        'profit_percent': 0.0,
        'trade_count': 0,
        'position': None,
        'entry_price': 0.0,
        'current_price': 0.0
    }
    
    # Check if the portfolio log file exists first (more recent data)
    try:
        if os.path.exists("portfolio_log.txt"):
            with open("portfolio_log.txt", "r") as f:
                line = f.readlines()[-1].strip()  # Get the last line
                if line:
                    # Parse line in format: timestamp,value,profit,profit_percent,trade_count,position,entry_price,current_price
                    parts = line.split(",")
                    if len(parts) >= 8:
                        portfolio_data.update({
                            'value': float(parts[1]),
                            'profit': float(parts[2]),
                            'profit_percent': float(parts[3]),
                            'trade_count': int(parts[4]),
                            'position': parts[5] if parts[5] != "None" else None,
                            'entry_price': float(parts[6]),
                            'current_price': float(parts[7])
                        })
                        logger.info(f"Retrieved portfolio data from log file")
                        return portfolio_data
    except Exception as e:
        logger.warning(f"Error reading portfolio log: {e}")
    
    # Request current status from bot
    with open(request_file, "w") as f:
        f.write(str(time.time()))
    
    logger.info("Request for portfolio value sent to bot...")
    
    # Wait for up to 5 seconds for the bot to respond
    for _ in range(5):
        if os.path.exists(response_file):
            try:
                with open(response_file, "r") as f:
                    data = f.read().strip()
                    if data:
                        # Try to parse as JSON
                        try:
                            portfolio_data = json.loads(data)
                            logger.info("Portfolio data successfully retrieved from bot")
                            
                            # Also write to log file for future reference
                            try:
                                log_entry = f"{time.time()},{portfolio_data['value']},{portfolio_data['profit']},{portfolio_data['profit_percent']},{portfolio_data['trade_count']},{portfolio_data.get('position', 'None')},{portfolio_data.get('entry_price', 0)},{portfolio_data.get('current_price', 0)}"
                                with open("portfolio_log.txt", "a") as log_file:
                                    log_file.write(log_entry + "\n")
                            except Exception as e:
                                logger.warning(f"Error writing to portfolio log: {e}")
                                
                            # Clean up files
                            if os.path.exists(request_file):
                                os.remove(request_file)
                            if os.path.exists(response_file):
                                os.remove(response_file)
                                
                            return portfolio_data
                        except json.JSONDecodeError:
                            # Try to parse as a plain number
                            try:
                                portfolio_data['value'] = float(data)
                                logger.info("Portfolio value successfully retrieved from bot")
                                return portfolio_data
                            except ValueError:
                                logger.warning(f"Could not parse portfolio data: {data}")
            except Exception as e:
                logger.error(f"Error reading portfolio data: {e}")
        
        # Sleep for 1 second before checking again
        time.sleep(1)
    
    # Clean up request file if it still exists
    if os.path.exists(request_file):
        os.remove(request_file)
    
    # No response from bot - use default values
    logger.info(f"No response received from bot. Using default portfolio value: ${INITIAL_CAPITAL:.2f}")
    
    return portfolio_data

def main():
    parser = argparse.ArgumentParser(description="Check Kraken portfolio value")
    parser.add_argument("--sandbox", action="store_true", help="Run in sandbox mode")
    parser.add_argument("--live", action="store_true", help="Connect to live Kraken API")
    parser.add_argument("--json", action="store_true", help="Output data in JSON format")
    args = parser.parse_args()
    
    # Get the portfolio status
    portfolio_data = check_portfolio_status()
    
    # Output data
    if args.json:
        print(json.dumps(portfolio_data))
    else:
        # Pretty print the portfolio information
        logger.info(f"Current portfolio value: ${portfolio_data['value']:.2f}")
        logger.info(f"Profit: ${portfolio_data['profit']:.2f} ({portfolio_data['profit_percent']:.2f}%)")
        logger.info(f"Trading portion (margin): ${portfolio_data['value'] * MARGIN_PERCENT:.2f}")
        logger.info(f"Position: {portfolio_data.get('position', 'None')}")
        logger.info(f"Entry price: ${portfolio_data.get('entry_price', 0):.2f}")
        logger.info(f"Current price: ${portfolio_data.get('current_price', 0):.2f}")
        logger.info(f"Trade count: {portfolio_data.get('trade_count', 0)}")
        
    # If live mode is requested, also try to query the Kraken API
    if args.live:
        # Set up API credentials
        api_key = os.environ.get("KRAKEN_API_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET")
        
        if api_key and api_secret:
            # In live mode, query Kraken API for account balance
            logger.info("Using live Kraken API to check portfolio value...")
            api = KrakenAPI(api_key, api_secret)
            
            try:
                # Get account balance
                balance = api.get_account_balance()
                logger.info(f"Account Balance: {balance}")
                
                # Get USD balance
                if 'ZUSD' in balance:
                    usd_balance = float(balance['ZUSD'])
                    logger.info(f"USD Balance: ${usd_balance:.2f}")
                else:
                    logger.info("No USD balance found")
                
                # Get open positions
                positions = api.get_open_positions()
                if positions:
                    logger.info(f"Open Positions: {positions}")
                    position_value = sum([float(p.get('value', 0)) for p in positions])
                    logger.info(f"Position Value: ${position_value:.2f}")
                else:
                    logger.info("No open positions")
            
            except Exception as e:
                logger.error(f"Error querying Kraken API: {e}")
        else:
            logger.error("Live mode requires KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables")
    
if __name__ == "__main__":
    main()