#!/usr/bin/env python3
"""
Update Portfolio Prices

This script updates the prices of all positions in the sandbox portfolio
with current market data from Kraken API.
"""

import json
import os
import sys
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
SANDBOX_PORTFOLIO_FILE = "config/sandbox_portfolio.json"


def load_portfolio():
    """Load the sandbox portfolio"""
    try:
        if os.path.exists(SANDBOX_PORTFOLIO_FILE):
            with open(SANDBOX_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Portfolio file not found: {SANDBOX_PORTFOLIO_FILE}")
            return None
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return None


def save_portfolio(portfolio):
    """Save the sandbox portfolio"""
    try:
        with open(SANDBOX_PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")
        return False


def get_current_prices(pairs):
    """Get current prices from Kraken API"""
    if not pairs:
        return {}

    # Map for handling different asset naming conventions
    asset_mapping = {
        "ETH/USD": "XETHZUSD",
        "BTC/USD": "XXBTZUSD",
        "SOL/USD": "SOLUSD",
        "ADA/USD": "ADAUSD",
        "DOT/USD": "DOTUSD",
        "LINK/USD": "LINKUSD",
        "AVAX/USD": "AVAXUSD",
        "MATIC/USD": "MATICUSD",
        "UNI/USD": "UNIUSD"
    }

    # Get all prices in one request
    try:
        response = requests.get("https://api.kraken.com/0/public/Ticker")
        if response.status_code != 200:
            logger.error(f"Error fetching prices: HTTP {response.status_code}")
            return {}

        data = response.json()
        if "error" in data and data["error"]:
            logger.error(f"Kraken API error: {data['error']}")
            return {}

        # Extract prices from response
        prices = {}
        for pair in pairs:
            kraken_pair = asset_mapping.get(pair, pair.replace("/", ""))
            
            # Map for Kraken's quirky response format
            for key in data.get("result", {}):
                # Try different variations of the pair name
                if key.upper() == kraken_pair.upper() or f"X{kraken_pair}".upper() == key.upper():
                    # Use the last trade price (c[0])
                    prices[pair] = float(data["result"][key]["c"][0])
                    logger.info(f"Found price for {pair}: {prices[pair]}")
                    break
            
            if pair not in prices:
                logger.warning(f"No price data found for {pair} (tried {kraken_pair})")

        return prices
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        return {}


def update_portfolio_prices():
    """Update portfolio prices with current market data"""
    # Load portfolio
    portfolio = load_portfolio()
    if not portfolio:
        logger.error("Failed to load portfolio")
        return False

    # Get pairs with open positions
    pairs = [position["pair"] for position in portfolio.get("positions", {}).values()]
    if not pairs:
        logger.info("No open positions to update")
        return True

    # Get current prices
    prices = get_current_prices(pairs)
    if not prices:
        logger.error("Failed to fetch current prices")
        return False

    # Update positions with current prices
    updated_count = 0
    total_pnl = 0.0
    
    for position_id, position in portfolio.get("positions", {}).items():
        pair = position["pair"]
        if pair in prices:
            old_price = position["current_price"]
            new_price = prices[pair]
            
            # Update position
            position["current_price"] = new_price
            
            # Calculate unrealized PnL
            if position["side"] == "long":
                price_change = new_price - position["entry_price"]
                pnl_multiplier = 1
            else:  # short
                price_change = position["entry_price"] - new_price
                pnl_multiplier = -1
                
            units = position["units"]
            leverage = position["leverage"]
            
            # Calculate unrealized PnL
            unrealized_pnl = units * price_change * pnl_multiplier * leverage
            unrealized_pnl_percent = (price_change / position["entry_price"]) * 100 * pnl_multiplier * leverage
            
            position["unrealized_pnl"] = unrealized_pnl
            position["unrealized_pnl_percent"] = unrealized_pnl_percent
            
            total_pnl += unrealized_pnl
            updated_count += 1
            
            logger.info(f"Updated {pair}: {old_price:.2f} -> {new_price:.2f} (PnL: ${unrealized_pnl:.2f}, {unrealized_pnl_percent:.2f}%)")

    # Update portfolio equity
    if "current_capital" in portfolio:
        portfolio["equity"] = portfolio["current_capital"] + total_pnl

    # Update timestamp
    portfolio["last_updated"] = datetime.now().isoformat()

    # Save updated portfolio
    if save_portfolio(portfolio):
        logger.info(f"Updated {updated_count} positions, total unrealized PnL: ${total_pnl:.2f}")
        return True
    else:
        logger.error("Failed to save updated portfolio")
        return False


def main():
    """Main function"""
    logger.info("Updating portfolio prices...")
    success = update_portfolio_prices()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())