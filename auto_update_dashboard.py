#!/usr/bin/env python3

"""
Auto Update Dashboard

This script automatically updates the dashboard data at regular intervals
to ensure the information stays current without requiring manual refreshes.

Usage:
    python auto_update_dashboard.py
"""

import os
import sys
import time
import json
import logging
import random
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
UPDATE_INTERVAL = 120  # Update every 2 minutes

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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def update_portfolio_history():
    """Update portfolio history with current data"""
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Create new history point
    new_point = {
        "timestamp": datetime.now().isoformat(),
        "balance": portfolio.get("balance", 20000.0),
        "equity": portfolio.get("equity", 20000.0),
        "portfolio_value": portfolio.get("equity", 20000.0),
        "num_positions": len(load_file(POSITIONS_FILE, []))
    }
    
    history.append(new_point)
    
    # Keep only the last 1000 points to prevent file growth
    if len(history) > 1000:
        history = history[-1000:]
    
    save_file(PORTFOLIO_HISTORY_FILE, history)
    logger.info(f"Updated portfolio history: {new_point['timestamp']}")

def simulated_trade_update():
    """
    Add simulated trade data for demonstration purposes
    In production, this would be replaced by actual trading bot updates
    """
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0, 
        "equity": 20000.0,
        "trades": [],
        "last_updated": datetime.now().isoformat()
    })
    
    # Get ML config to use realistic parameters
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    pairs = list(ml_config.get("pairs", {}).keys())
    
    # If no pairs, use defaults
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    
    # Choose a random pair
    pair = random.choice(pairs)
    
    # Get pair config for realistic parameters
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    # Choose direction
    direction = random.choice(["LONG", "SHORT"])
    
    # Define realistic entry and exit prices
    base_prices = {
        "SOL/USD": 150.0,
        "BTC/USD": 60000.0,
        "ETH/USD": 3500.0,
        "ADA/USD": 0.45,
        "DOT/USD": 7.50,
        "LINK/USD": 15.0
    }
    
    base_price = base_prices.get(pair, 100.0)
    price_variation = base_price * 0.05  # 5% variation
    
    entry_price = round(base_price + random.uniform(-price_variation, price_variation), 2)
    
    # Calculate exit price based on win rate
    win_rate = pair_config.get("win_rate", 0.85)
    leverage = pair_config.get("base_leverage", 38.0)
    
    is_win = random.random() < win_rate
    price_change_pct = random.uniform(0.005, 0.03)  # 0.5% to 3% change
    
    if is_win:
        exit_price = round(entry_price * (1 + price_change_pct) if direction == "LONG" else entry_price * (1 - price_change_pct), 2)
    else:
        exit_price = round(entry_price * (1 - price_change_pct) if direction == "LONG" else entry_price * (1 + price_change_pct), 2)
    
    # Calculate PnL
    if direction == "LONG":
        pnl_percentage = (exit_price / entry_price) - 1
    else:
        pnl_percentage = (entry_price / exit_price) - 1
    
    pnl_percentage *= leverage
    position_size = portfolio.get("balance", 20000.0) * 0.20  # 20% position size
    pnl_amount = position_size * pnl_percentage
    
    # Create trade
    now = datetime.now()
    entry_time = (now.replace(minute=now.minute - 5)).isoformat()
    exit_time = now.isoformat()
    
    trade = {
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl_percentage": pnl_percentage,
        "pnl_amount": pnl_amount,
        "exit_reason": "TP" if pnl_percentage > 0 else "SL",
        "position_size": position_size,
        "leverage": leverage,
        "confidence": random.uniform(0.70, 0.95)
    }
    
    # Add trade to portfolio
    trades = portfolio.get("trades", [])
    trades.append(trade)
    
    # Update portfolio balance
    portfolio["balance"] = portfolio.get("balance", 20000.0) + pnl_amount
    portfolio["equity"] = portfolio.get("equity", 20000.0) + pnl_amount
    portfolio["last_updated"] = now.isoformat()
    
    # Save updated portfolio
    portfolio["trades"] = trades
    save_file(PORTFOLIO_FILE, portfolio)
    
    logger.info(f"Added simulated trade: {pair} {direction}, PnL: ${pnl_amount:.2f} ({pnl_percentage*100:.2f}%)")

def auto_update(stop_event):
    """Automatically update dashboard data at regular intervals"""
    logger.info(f"Starting auto-update with {UPDATE_INTERVAL} second interval")
    
    try:
        while not stop_event.is_set():
            # Update portfolio history
            update_portfolio_history()
            
            # Simulate trade updates (replace with actual updates in production)
            if random.random() < 0.3:  # 30% chance of a trade every update
                simulated_trade_update()
            
            # Wait for next update
            for _ in range(UPDATE_INTERVAL):
                if stop_event.is_set():
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Auto-update stopped by user")
    except Exception as e:
        logger.error(f"Error in auto-update: {e}")
    finally:
        logger.info("Auto-update stopped")

def main():
    """Main function"""
    logger.info("Starting dashboard auto-update")
    
    # Create stop event
    stop_event = threading.Event()
    
    # Start auto-update in a separate thread
    update_thread = threading.Thread(target=auto_update, args=(stop_event,))
    update_thread.daemon = True
    update_thread.start()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping auto-update...")
        stop_event.set()
        update_thread.join(timeout=5)
        logger.info("Auto-update stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())