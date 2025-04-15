#!/usr/bin/env python3

"""
Add Simulated Trade

This script adds a single simulated trade to the portfolio
using current market prices from the Kraken API.

Usage:
    python add_simulated_trade.py [pair]
"""

import os
import sys
import json
import logging
import random
import requests
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"

# Environment variables
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

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

def get_current_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Get current prices from Kraken API for multiple pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary mapping pairs to current prices
    """
    prices = {}
    
    try:
        # Format pairs for Kraken API
        kraken_pairs = [pair.replace("/", "") for pair in pairs]
        pair_str = ",".join(kraken_pairs)
        
        # Make API request to Kraken
        url = f"https://api.kraken.com/0/public/Ticker?pair={pair_str}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if "result" in data:
                result = data["result"]
                
                # Extract prices from response
                for kraken_pair, info in result.items():
                    # Convert back to original format
                    original_pair = None
                    for p in pairs:
                        if p.replace("/", "") in kraken_pair:
                            original_pair = p
                            break
                    
                    if original_pair:
                        # Use last trade price (c[0])
                        prices[original_pair] = float(info["c"][0])
            
            logger.info(f"Retrieved current prices from Kraken API: {prices}")
        else:
            logger.warning(f"Failed to get prices from Kraken API: {response.status_code}")
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
    
    return prices

def add_simulated_trade(pair=None):
    """
    Add a simulated trade using current market prices.
    
    Args:
        pair: Optional specific pair to trade
    """
    # Load portfolio and ML config
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0, 
        "equity": 20000.0,
        "trades": [],
        "last_updated": datetime.now().isoformat()
    })
    
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    # Get available pairs
    pairs = list(ml_config.get("pairs", {}).keys())
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
                "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
    
    # If specific pair is provided, use it
    if pair and pair in pairs:
        pairs = [pair]
    else:
        # Choose a random pair
        pair = random.choice(pairs)
        pairs = [pair]
    
    # Get current prices
    current_prices = get_current_prices(pairs)
    
    # If we can't get the price, exit
    if not current_prices or pair not in current_prices:
        logger.error(f"Could not get current price for {pair}")
        return False
    
    # Get pair config
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    # Get parameters
    entry_price = current_prices[pair]
    direction = "LONG" if random.random() < 0.5 else "SHORT"
    confidence = random.uniform(0.70, 0.95)
    
    # Get leverage from config
    base_leverage = pair_config.get("base_leverage", 38.0)
    max_leverage = pair_config.get("max_leverage", 100.0)
    
    # Calculate leverage based on confidence
    leverage = base_leverage + (confidence * (max_leverage - base_leverage))
    leverage = min(max_leverage, max(base_leverage, leverage))
    
    # Calculate exit price
    win_rate = pair_config.get("win_rate", 0.85)
    is_win = random.random() < win_rate
    price_change_pct = random.uniform(0.005, 0.025)  # 0.5% to 2.5% change
    
    if is_win:
        exit_price = entry_price * (1 + price_change_pct) if direction == "LONG" else entry_price * (1 - price_change_pct)
    else:
        exit_price = entry_price * (1 - price_change_pct) if direction == "LONG" else entry_price * (1 + price_change_pct)
    
    exit_price = round(exit_price, 6)  # Round to 6 decimal places
    
    # Calculate PnL
    if direction == "LONG":
        pnl_percentage = (exit_price / entry_price) - 1
    else:
        pnl_percentage = (entry_price / exit_price) - 1
    
    pnl_percentage *= leverage
    
    # Calculate position size
    risk_percentage = pair_config.get("risk_percentage", 0.20)  # Default 20% risk
    position_size = portfolio.get("balance", 20000.0) * risk_percentage * confidence
    
    # Calculate PnL amount
    pnl_amount = position_size * pnl_percentage
    
    # Create realistic timestamps
    now = datetime.now()
    entry_time = (now.replace(minute=now.minute - random.randint(5, 30))).isoformat()
    exit_time = now.isoformat()
    
    # Create trade
    trade = {
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl_percentage": pnl_percentage,
        "pnl_amount": pnl_amount,
        "exit_reason": "TP" if is_win else "SL",
        "position_size": position_size,
        "leverage": leverage,
        "confidence": confidence,
        "strategy": "Adaptive" if random.random() < 0.5 else "ARIMA"
    }
    
    # Add to portfolio
    trades = portfolio.get("trades", [])
    trades.append(trade)
    
    # Update portfolio balance
    portfolio["balance"] = portfolio.get("balance", 20000.0) + pnl_amount
    portfolio["equity"] = portfolio.get("equity", 20000.0) + pnl_amount
    portfolio["last_updated"] = now.isoformat()
    
    # Save updated portfolio
    portfolio["trades"] = trades
    if save_file(PORTFOLIO_FILE, portfolio):
        logger.info(f"Added trade for {pair} {direction}, Entry: ${entry_price:.6f}, Exit: ${exit_price:.6f}, PnL: ${pnl_amount:.2f} ({pnl_percentage*100:.2f}%)")
        return True
    else:
        logger.error("Failed to save portfolio")
        return False

if __name__ == "__main__":
    # Get pair from command line arguments
    pair = sys.argv[1] if len(sys.argv) > 1 else None
    
    if add_simulated_trade(pair):
        print("Successfully added simulated trade.")
    else:
        print("Failed to add simulated trade.")
        sys.exit(1)