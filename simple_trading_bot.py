#!/usr/bin/env python3
"""
Simple Trading Bot for Kraken

This script runs a simple trading bot that:
1. Fetches real-time prices from Kraken API
2. Updates positions with current prices
3. Calculates accurate PnL
4. Updates portfolio metrics 

It avoids using Flask completely to prevent port conflicts.
"""
import os
import sys
import json
import time
import random
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default if default is not None else {}

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
                # Get the first result key (should be the pair data)
                pair_data = next(iter(data['result'].values()))
                if 'c' in pair_data and pair_data['c']:
                    # 'c' contains the last trade closed info, first element is price
                    current_price = float(pair_data['c'][0])
                    logger.info(f"Got price for {pair}: ${current_price}")
                    return current_price
        
        logger.warning(f"Could not get price for {pair} from Kraken API")
        return None
    except Exception as e:
        logger.error(f"Error getting price for {pair} from Kraken API: {e}")
        return None

def update_positions():
    """Update positions with current prices from Kraken API"""
    positions = load_file(POSITIONS_FILE, [])
    if not positions:
        logger.info("No positions to update")
        return positions
    
    # Update each position with current prices and PnL
    updated_positions = []
    
    for position in positions:
        # Create a copy to avoid modifying the original
        updated_position = position.copy()
        
        pair = position.get('pair')
        if not pair:
            updated_positions.append(updated_position)
            continue
        
        # Get current price from Kraken API
        current_price = get_kraken_price(pair)
        if not current_price:
            # If we couldn't get a price, keep the existing one
            updated_positions.append(updated_position)
            continue
        
        # Calculate unrealized PnL
        entry_price = position.get('entry_price', 0)
        position_size = position.get('position_size', 0)
        direction = position.get('direction', '').lower()
        leverage = position.get('leverage', 1.0)
        
        # Ensure position_size is positive
        if position_size < 0:
            position_size = abs(position_size)
            updated_position['position_size'] = position_size
        
        # Update current price
        updated_position['current_price'] = current_price
        
        # Calculate PnL
        if entry_price and position_size and entry_price > 0:
            # For long positions:
            # - Price goes up: profit
            # - Price goes down: loss
            if direction == 'long':
                price_change_pct = ((current_price / entry_price) - 1) * 100
                pnl_pct = price_change_pct * leverage
                pnl_amount = (position_size * price_change_pct / 100) * leverage
            # For short positions:
            # - Price goes up: loss
            # - Price goes down: profit
            else:  # short
                price_change_pct = ((entry_price / current_price) - 1) * 100
                pnl_pct = price_change_pct * leverage
                pnl_amount = (position_size * price_change_pct / 100) * leverage
            
            updated_position['unrealized_pnl_amount'] = round(pnl_amount, 2)
            updated_position['unrealized_pnl_pct'] = round(pnl_pct, 2)
            updated_position['current_value'] = round(position_size + pnl_amount, 2)
            
            # Update trade record with the same PnL information
            trade_id = position.get('open_trade_id')
            if trade_id:
                update_trade_with_pnl(trade_id, current_price, pnl_amount, pnl_pct)
            
            logger.info(f"Updated {pair} position: Price=${current_price}, PnL=${pnl_amount:.2f} ({pnl_pct:.2f}%)")
        
        updated_positions.append(updated_position)
    
    # Save updated positions
    save_file(POSITIONS_FILE, updated_positions)
    logger.info(f"Updated {len(updated_positions)} positions with current prices")
    
    return updated_positions

def update_trade_with_pnl(trade_id, current_price, pnl_amount, pnl_pct):
    """Update the corresponding trade record with PnL information"""
    trades_file = f"{DATA_DIR}/sandbox_trades.json"
    trades = load_file(trades_file, [])
    
    for trade in trades:
        if trade.get('id') == trade_id and trade.get('status') == 'open':
            # Update PnL
            trade['pnl'] = round(pnl_amount, 2)
            trade['pnl_pct'] = round(pnl_pct, 2)
            trade['current_price'] = current_price
            
    # Save updated trades
    save_file(trades_file, trades)

def update_portfolio(positions):
    """Update portfolio with current position values"""
    portfolio = load_file(PORTFOLIO_FILE, {"initial_capital": 20000.0, "balance": 20000.0, "equity": 20000.0})
    
    # Calculate total unrealized PnL
    total_pnl = 0
    for position in positions:
        if "unrealized_pnl_amount" in position:
            total_pnl += position.get("unrealized_pnl_amount", 0)
    
    # Update portfolio metrics
    initial_balance = portfolio.get("initial_capital", 20000.0)
    current_balance = portfolio.get("balance", initial_balance)
    
    # Fix negative balance values if present
    if current_balance < 0:
        current_balance = initial_balance
        portfolio["balance"] = current_balance
    
    # Set unrealized PnL values
    unrealized_pnl = total_pnl  # This is the total unrealized PnL in USD
    unrealized_pnl_pct = round((unrealized_pnl / current_balance) * 100, 2) if current_balance > 0 else 0
    
    # Update the portfolio fields
    portfolio["unrealized_pnl"] = unrealized_pnl
    portfolio["unrealized_pnl_usd"] = round(unrealized_pnl, 2)
    portfolio["unrealized_pnl_pct"] = unrealized_pnl_pct
    
    # Calculate total value and equity (balance + unrealized PnL)
    total_value = current_balance + unrealized_pnl
    portfolio["equity"] = round(total_value, 2)
    portfolio["total_value"] = round(total_value, 2)
    
    # Calculate total PnL (from initial capital)
    portfolio["total_pnl"] = round(total_value - initial_balance, 2)
    portfolio["total_pnl_pct"] = round(((total_value / initial_balance) - 1) * 100, 2)
    
    # Position metrics
    portfolio["open_positions_count"] = len(positions)
    
    # Calculate total margin used and available capital
    total_margin_used = sum(p.get("position_size", 0) for p in positions)
    available_capital = current_balance - total_margin_used
    margin_used_pct = (total_margin_used / current_balance) * 100 if current_balance > 0 else 0
    
    portfolio["available_capital"] = round(available_capital, 2)
    portfolio["margin_used_pct"] = round(margin_used_pct, 2)
    portfolio["available_margin"] = round(available_capital, 2)
    
    # Update timestamp
    portfolio["updated_at"] = datetime.now().isoformat()
    
    # Save updated portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    logger.info(f"Updated portfolio: Balance=${current_balance}, Equity=${portfolio['equity']}, Unrealized PnL=${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)")
    
    # Update portfolio history
    update_portfolio_history(portfolio)
    
    return portfolio

def update_portfolio_history(portfolio):
    """Add a new data point to portfolio history"""
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Add current equity value as a new data point
    now = datetime.now().isoformat()
    
    if len(history) > 0:
        # Only add a new point if it's been at least 5 minutes since the last update
        last_timestamp = history[-1].get("timestamp", "")
        if last_timestamp:
            last_time = datetime.fromisoformat(last_timestamp)
            time_diff = (datetime.now() - last_time).total_seconds() / 60  # in minutes
            
            if time_diff < 5:
                return  # Skip if less than 5 minutes
    
    new_point = {
        "timestamp": now,
        "portfolio_value": portfolio.get("equity", 20000.0)
    }
    
    history.append(new_point)
    
    # Keep only the last 1000 data points to avoid file growing too large
    if len(history) > 1000:
        history = history[-1000:]
    
    save_file(PORTFOLIO_HISTORY_FILE, history)
    logger.info(f"Added portfolio history point: {new_point['portfolio_value']:.2f} at {now}")

def main():
    """Main function to run the trading bot"""
    logger.info("=" * 50)
    logger.info("Starting simple trading bot with real-time Kraken prices")
    logger.info("=" * 50)
    
    try:
        # Main loop
        while True:
            # Update positions with current prices
            positions = update_positions()
            
            # Update portfolio with position values
            update_portfolio(positions)
            
            # Wait before next update (30-60 seconds)
            sleep_time = random.randint(30, 60)
            logger.info(f"Waiting {sleep_time} seconds until next update...")
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in trading bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()