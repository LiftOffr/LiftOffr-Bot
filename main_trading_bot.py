#!/usr/bin/env python3
"""
Main entry point for the Trading Bot in isolated mode.
"""
import os
import sys
import json
import time
import logging
import requests
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def get_kraken_price(pair):
    """Get current price from Kraken API"""
    try:
        # Replace '/' with '' in pair name for Kraken API format
        kraken_pair = pair.replace('/', '')
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and data['result']:
                pair_data = next(iter(data['result'].values()))
                if 'c' in pair_data and pair_data['c']:
                    return float(pair_data['c'][0])  # 'c' is the last trade closed price
        
        logger.warning(f"Could not get price for {pair} from Kraken API")
        return None
    except Exception as e:
        logger.error(f"Error getting price for {pair} from Kraken API: {e}")
        return None

def update_positions_with_current_prices():
    """Update positions with current prices from Kraken API"""
    positions = load_file(POSITIONS_FILE, [])
    if not positions:
        logger.info("No positions to update")
        return

    # Make a copy to avoid modifying the original
    updated_positions = []
    
    for position in positions:
        # Create a copy of the position
        updated_position = position.copy()
        
        pair = position.get('pair')
        if not pair:
            updated_positions.append(updated_position)
            continue
            
        # Get current price from Kraken API
        current_price = get_kraken_price(pair)
        
        if current_price:
            logger.info(f"Updated {pair} price: {current_price}")
            # Update current price
            updated_position['current_price'] = current_price
            
            # Calculate unrealized PnL
            entry_price = position.get('entry_price', 0)
            position_size = position.get('position_size', 0)
            direction = position.get('direction', '').lower()
            
            if entry_price and position_size:
                if direction == 'long':
                    pnl_amount = (current_price - entry_price) * position_size
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                else:  # short
                    pnl_amount = (entry_price - current_price) * position_size
                    pnl_pct = ((entry_price / current_price) - 1) * 100
                
                updated_position['unrealized_pnl_amount'] = pnl_amount
                updated_position['unrealized_pnl_pct'] = pnl_pct
        else:
            logger.warning(f"Could not update price for {pair}")
        
        updated_positions.append(updated_position)
    
    # Save updated positions
    save_file(POSITIONS_FILE, updated_positions)
    logger.info(f"Updated {len(updated_positions)} positions with current prices")
    
    # Update portfolio with new total PnL
    update_portfolio_metrics(updated_positions)

def update_portfolio_metrics(positions):
    """Update portfolio metrics based on positions"""
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    
    # Calculate total unrealized PnL
    total_pnl = 0
    for position in positions:
        if "unrealized_pnl_amount" in position:
            total_pnl += position["unrealized_pnl_amount"]
    
    # Update portfolio metrics
    portfolio["unrealized_pnl_usd"] = total_pnl
    portfolio["unrealized_pnl_pct"] = (total_pnl / portfolio.get("balance", 20000.0)) * 100 if total_pnl != 0 else 0.0
    portfolio["equity"] = portfolio.get("balance", 20000.0) + total_pnl
    
    # Save updated portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    logger.info(f"Updated portfolio metrics. Total PnL: ${total_pnl:.2f}")
    
    # Update portfolio history
    update_portfolio_history(portfolio)

def update_portfolio_history(portfolio):
    """Update portfolio history with current portfolio value"""
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Add new data point
    now = datetime.now().isoformat()
    if isinstance(history, list):
        history.append({
            "timestamp": now,
            "portfolio_value": portfolio.get("equity", 20000.0)
        })
    elif isinstance(history, dict) and "timestamps" in history and "values" in history:
        history["timestamps"].append(now)
        history["values"].append(portfolio.get("equity", 20000.0))
    else:
        # Create new history if invalid format
        history = [
            {
                "timestamp": now,
                "portfolio_value": portfolio.get("equity", 20000.0)
            }
        ]
    
    # Save updated history
    save_file(PORTFOLIO_HISTORY_FILE, history)
    logger.info("Updated portfolio history")

def run_trading_bot():
    """Main trading bot loop"""
    logger.info("Starting trading bot in isolated mode...")
    
    try:
        while True:
            logger.info("Updating positions with current prices...")
            update_positions_with_current_prices()
            
            # Sleep for a random interval between 30-60 seconds to simulate real trading activity
            sleep_time = random.randint(30, 60)
            logger.info(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in trading bot: {e}")

if __name__ == "__main__":
    # Make sure we're running as a separate process
    os.environ["TRADING_BOT_PROCESS"] = "1"
    run_trading_bot()