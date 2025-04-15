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
import requests
from typing import Dict, List, Any
from datetime import datetime
import threading
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

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
        # For sandbox mode, we'll check if real API access is available
        if KRAKEN_API_KEY and KRAKEN_API_SECRET:
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
                prices = _get_prices_from_local_data(pairs)
        else:
            # No API keys, use local data
            logger.info("No API keys, using local data for prices")
            prices = _get_prices_from_local_data(pairs)
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        prices = _get_prices_from_local_data(pairs)
    
    return prices

def _get_prices_from_local_data(pairs: List[str]) -> Dict[str, float]:
    """Get prices from local data files"""
    prices = {}
    
    try:
        # Try to load from position data first
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                position_data = json.load(f)
                
            for position in position_data:
                if position["pair"] in pairs and "last_price" in position:
                    prices[position["pair"]] = position["last_price"]
        
        # Fill in missing pairs with more realistic market prices 
        market_prices = {
            "SOL/USD": 130.89,
            "BTC/USD": 63872.0,
            "ETH/USD": 3123.18,
            "ADA/USD": 0.64,
            "DOT/USD": 3.67,
            "LINK/USD": 12.70,
            "AVAX/USD": 19.93,
            "MATIC/USD": 0.18,
            "UNI/USD": 5.41,
            "ATOM/USD": 4.12
        }
        
        for pair in pairs:
            if pair not in prices:
                prices[pair] = market_prices.get(pair, 100.0)
    except Exception as e:
        logger.error(f"Error loading local price data: {e}")
        # Use market prices as last resort
        prices = {
            "SOL/USD": 130.89,
            "BTC/USD": 63872.0,
            "ETH/USD": 3123.18,
            "ADA/USD": 0.64,
            "DOT/USD": 3.67,
            "LINK/USD": 12.70,
            "AVAX/USD": 19.93,
            "MATIC/USD": 0.18,
            "UNI/USD": 5.41,
            "ATOM/USD": 4.12
        }
    
    return {p: prices[p] for p in pairs if p in prices}

def update_positions_with_current_prices(positions: List[Dict[str, Any]], prices: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Update positions with current prices and calculate unrealized PnL
    
    Args:
        positions: List of position dictionaries
        prices: Current prices dictionary
        
    Returns:
        Updated positions
    """
    updated = False
    total_unrealized_pnl = 0.0
    
    for position in positions:
        pair = position["pair"]
        if pair in prices:
            # Update current price
            current_price = prices[pair]
            entry_price = position["entry_price"]
            size = position["size"]
            leverage = position["leverage"]
            direction = position["direction"]
            
            # Update current price in position
            position["current_price"] = current_price
            
            # Calculate margin (may not exist in older positions)
            if "margin" not in position:
                # Calculate margin from position size and entry price
                notional_value = size * entry_price
                position["margin"] = notional_value / leverage
            
            margin = position["margin"]
            
            # Calculate unrealized P&L
            if direction.lower() == "long":
                price_change_pct = (current_price / entry_price) - 1
            else:  # short
                price_change_pct = (entry_price / current_price) - 1
                
            pnl_pct = price_change_pct * leverage
            pnl_amount = margin * pnl_pct
            
            # Update unrealized PnL in position
            position["unrealized_pnl"] = pnl_amount
            position["unrealized_pnl_pct"] = pnl_pct * 100
            total_unrealized_pnl += pnl_amount
            
            # Update duration
            entry_time = datetime.fromisoformat(position["entry_time"])
            now = datetime.now()
            duration = now - entry_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            position["duration"] = f"{int(hours)}h {int(minutes)}m"
            
            updated = True
    
    return positions, updated, total_unrealized_pnl

def update_portfolio_from_positions(portfolio: Dict[str, Any], positions: List[Dict[str, Any]], total_unrealized_pnl: float) -> Dict[str, Any]:
    """
    Update portfolio data based on current positions
    
    Args:
        portfolio: Portfolio dictionary
        positions: List of position dictionaries
        total_unrealized_pnl: Total unrealized PnL
        
    Returns:
        Updated portfolio
    """
    # Calculate portfolio equity (balance + unrealized PnL)
    starting_capital = 20000.0  # Default starting capital
    current_balance = portfolio.get("balance", starting_capital)
    
    # Update portfolio
    portfolio["unrealized_pnl"] = total_unrealized_pnl
    portfolio["equity"] = current_balance + total_unrealized_pnl
    portfolio["unrealized_pnl_percentage"] = (total_unrealized_pnl / starting_capital) * 100
    portfolio["total_return_percentage"] = ((current_balance + total_unrealized_pnl) / starting_capital - 1) * 100
    portfolio["last_updated"] = datetime.now().isoformat()
    
    return portfolio

def update_portfolio_history():
    """Update portfolio history with current data"""
    # Load current data
    positions = load_file(POSITIONS_FILE, [])
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Get all pairs from positions
    pairs = list(set([p["pair"] for p in positions]))
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
                "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
    
    # Get current prices
    prices = get_current_prices(pairs)
    
    # Update positions with current prices
    positions, positions_updated, total_unrealized_pnl = update_positions_with_current_prices(positions, prices)
    
    # Save updated positions
    if positions_updated:
        save_file(POSITIONS_FILE, positions)
    
    # Update portfolio with position data
    portfolio = update_portfolio_from_positions(portfolio, positions, total_unrealized_pnl)
    save_file(PORTFOLIO_FILE, portfolio)
    
    # Create new history point
    new_point = {
        "timestamp": datetime.now().isoformat(),
        "balance": portfolio.get("balance", 20000.0),
        "equity": portfolio.get("equity", 20000.0),
        "portfolio_value": portfolio.get("equity", 20000.0),
        "unrealized_pnl": total_unrealized_pnl,
        "num_positions": len(positions)
    }
    
    history.append(new_point)
    
    # Keep only the last 1000 points to prevent file growth
    if len(history) > 1000:
        history = history[-1000:]
    
    save_file(PORTFOLIO_HISTORY_FILE, history)
    logger.info(f"Updated portfolio history: {new_point['timestamp']}, Unrealized PnL: ${total_unrealized_pnl:.2f}")

def simulated_trade_update():
    """
    Add simulated trade data for demonstration purposes using real-time market prices
    from Kraken API for accurate entry and exit prices.
    """
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0, 
        "equity": 20000.0,
        "trades": [],
        "last_updated": datetime.now().isoformat()
    })
    
    # Get ML config for trading parameters
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    pairs = list(ml_config.get("pairs", {}).keys())
    
    # If no pairs, use defaults
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
                "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
    
    # Choose a random pair
    pair = random.choice(pairs)
    
    # Get pair config for trading parameters
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    # Get current market prices for all pairs
    current_prices = get_current_prices(pairs)
    
    # If we can't get prices from API, don't create a simulated trade
    if not current_prices or pair not in current_prices:
        logger.warning(f"Could not get current price for {pair}, skipping trade simulation")
        return
    
    # Use actual market price as entry price
    entry_price = current_prices[pair]
    
    # Choose direction - use ML model win rate to determine if this is a good trade
    model_accuracy = pair_config.get("accuracy", 0.85)
    win_rate = pair_config.get("win_rate", 0.85)
    
    # Direction is chosen based on model accuracy
    if random.random() < model_accuracy:
        # Good prediction - choose direction that will likely win
        direction = "LONG" if random.random() < 0.5 else "SHORT"
        is_win = True if random.random() < win_rate else False
    else:
        # Bad prediction - choose direction that will likely lose
        direction = "LONG" if random.random() < 0.5 else "SHORT"
        is_win = False if random.random() < (1 - win_rate) else True
    
    # Get leverage from config or use default
    base_leverage = pair_config.get("base_leverage", 38.0)
    max_leverage = pair_config.get("max_leverage", 100.0)
    
    # Adjust leverage based on confidence
    confidence = random.uniform(0.70, 0.95)
    leverage = base_leverage + (confidence * (max_leverage - base_leverage))
    leverage = min(max_leverage, max(base_leverage, leverage))  # Ensure within range
    
    # Calculate a realistic price change based on win/loss
    price_change_pct = random.uniform(0.005, 0.025)  # 0.5% to 2.5% change
    
    # Calculate exit price based on win/loss and direction
    if is_win:
        exit_price = entry_price * (1 + price_change_pct) if direction == "LONG" else entry_price * (1 - price_change_pct)
    else:
        exit_price = entry_price * (1 - price_change_pct) if direction == "LONG" else entry_price * (1 + price_change_pct)
    
    exit_price = round(exit_price, 6)  # Round to 6 decimal places for accurate pricing
    
    # Calculate PnL
    if direction == "LONG":
        pnl_percentage = (exit_price / entry_price) - 1
    else:
        pnl_percentage = (entry_price / exit_price) - 1
    
    pnl_percentage *= leverage
    
    # Use dynamic position sizing based on confidence
    risk_percentage = pair_config.get("risk_percentage", 0.20)  # Default 20% risk
    position_size = portfolio.get("balance", 20000.0) * risk_percentage * confidence
    
    # Calculate actual PnL amount
    pnl_amount = position_size * pnl_percentage
    
    # Create trade with realistic timestamps
    now = datetime.now()
    entry_time = (now.replace(minute=now.minute - random.randint(5, 30))).isoformat()
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
        "exit_reason": "TP" if is_win else "SL",
        "position_size": position_size,
        "leverage": leverage,
        "confidence": confidence,
        "strategy": "Adaptive" if random.random() < 0.5 else "ARIMA"
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
    
    logger.info(f"Added simulated trade: {pair} {direction}, Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, PnL: ${pnl_amount:.2f} ({pnl_percentage*100:.2f}%)")

def auto_update(stop_event):
    """Automatically update dashboard data at regular intervals"""
    logger.info(f"Starting auto-update with {UPDATE_INTERVAL} second interval")
    
    try:
        while not stop_event.is_set():
            # Update portfolio history with real position data
            update_portfolio_history()
            
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