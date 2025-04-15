#!/usr/bin/env python3
"""
Reset Portfolio and Positions

This script resets the portfolio and positions to a clean state
with proper initial values and positions, using real-time prices.
"""
import os
import json
import logging
import random
import requests
from datetime import datetime, timedelta

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
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

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

def create_portfolio():
    """Create a fresh portfolio with initial values"""
    portfolio = {
        "initial_capital": 20000.0,
        "available_capital": 20000.0,
        "balance": 20000.0,
        "equity": 20000.0,
        "total_value": 20000.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "unrealized_pnl_usd": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "profitable_trades": 0,
        "updated_at": datetime.now().isoformat(),
        "open_positions_count": 0,
        "margin_used_pct": 0.0,
        "available_margin": 20000.0
    }
    
    logger.info(f"Created fresh portfolio with ${portfolio['balance']} balance")
    return portfolio

def create_positions():
    """Create realistic positions with proper entries and current prices"""
    pairs = ["BTC/USD", "ETH/USD", "ADA/USD", "LINK/USD", "MATIC/USD", "UNI/USD", "ATOM/USD", "SOL/USD", "DOT/USD"]
    positions = []
    trades = []
    used_capital = 0.0
    max_capital = 20000.0
    
    # Dictionary to track position counts by category
    categories_count = {
        "those dudes": 0,
        "him all along": 0
    }
    
    # Generate 3-5 positions
    num_positions = random.randint(3, 5)
    selected_pairs = random.sample(pairs, num_positions)
    
    for i, pair in enumerate(selected_pairs):
        current_price = get_kraken_price(pair)
        if not current_price:
            logger.warning(f"Skipping {pair} - couldn't get price")
            continue
        
        # Generate realistic position data
        direction = random.choice(["long", "short"])
        
        # Dynamic leverage based on confidence
        confidence = round(random.uniform(0.65, 0.95), 2)
        base_leverage = 5.0
        max_leverage = 125.0
        leverage = round(base_leverage + (confidence * (max_leverage - base_leverage)), 1)
        
        # Calculate reasonable position size (5-15% of portfolio)
        position_percentage = random.uniform(0.05, 0.15)
        position_value = position_percentage * max_capital
        
        # Check if we have enough capital left
        if used_capital + position_value > max_capital:
            position_value = max_capital - used_capital
            
        if position_value <= 0:
            logger.warning(f"Skipping {pair} - not enough capital left")
            continue
            
        used_capital += position_value
        
        # Create the position
        entry_price = current_price
        position_size = position_value
        stop_loss_pct = 4.0  # Fixed at 4%
        take_profit_pct = round(stop_loss_pct * random.uniform(2.5, 4.5), 1)  # 2.5-4.5x the stop loss
        model = random.choice(["Adaptive", "ARIMA"])
        category = random.choice(["those dudes", "him all along"])
        
        # Keep track of how many positions are in each category
        categories_count[category] += 1
        
        # Ensure we have at least one of each category
        if i == num_positions - 1:
            if categories_count["those dudes"] == 0:
                category = "those dudes"
            elif categories_count["him all along"] == 0:
                category = "him all along"
        
        trade_id = f"trade_{i+1}"
        
        position = {
            "pair": pair,
            "entry_price": entry_price,
            "current_price": current_price,
            "position_size": position_size,
            "direction": direction,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "unrealized_pnl_pct": 0.0,
            "unrealized_pnl_amount": 0.0,
            "current_value": position_size,
            "confidence": confidence,
            "model": model,
            "category": category,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "open_trade_id": trade_id
        }
        
        # Create matching trade record
        trade = {
            "id": trade_id,
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "status": "open",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "exit_price": None,
            "exit_time": None,
            "confidence": confidence,
            "model": model,
            "category": category,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct
        }
        
        positions.append(position)
        trades.append(trade)
        
        logger.info(f"Created {direction} position for {pair} at ${entry_price} with {leverage}x leverage")
    
    logger.info(f"Created {len(positions)} positions using ${used_capital:.2f} capital")
    return positions, trades, used_capital

def create_portfolio_history(portfolio):
    """Create initial portfolio history data"""
    history = []
    
    # Create 24 hourly points leading up to now
    current_value = portfolio["balance"]
    
    # Starting from 24 hours ago, slight variations in portfolio value
    for i in range(24, 0, -1):
        hours_ago = i
        # Use timedelta instead of replace to avoid hour out of range issues
        timestamp = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
        
        # Random variation of ±2%
        variation = random.uniform(-0.02, 0.02)
        value = current_value * (1 + variation)
        
        history.append({
            "timestamp": timestamp,
            "portfolio_value": round(value, 2)
        })
    
    # Add current value
    history.append({
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": current_value
    })
    
    logger.info(f"Created portfolio history with {len(history)} data points")
    return history

def update_portfolio_with_positions(portfolio, positions, used_capital):
    """Update portfolio with position information"""
    portfolio["available_capital"] = portfolio["balance"] - used_capital
    portfolio["open_positions_count"] = len(positions)
    portfolio["margin_used_pct"] = round((used_capital / portfolio["balance"]) * 100, 2)
    portfolio["available_margin"] = round(portfolio["balance"] - used_capital, 2)
    portfolio["updated_at"] = datetime.now().isoformat()
    
    return portfolio

def main():
    """Reset portfolio and create new positions"""
    logger.info("=" * 50)
    logger.info("Resetting portfolio and positions")
    logger.info("=" * 50)
    
    # Create fresh portfolio
    portfolio = create_portfolio()
    
    # Create new positions
    positions, trades, used_capital = create_positions()
    
    # Update portfolio with position information
    portfolio = update_portfolio_with_positions(portfolio, positions, used_capital)
    
    # Create portfolio history
    history = create_portfolio_history(portfolio)
    
    # Save everything
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)
    
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)
    
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)
    
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("Reset complete!")
    logger.info(f"Portfolio balance: ${portfolio['balance']}")
    logger.info(f"Open positions: {len(positions)}")
    logger.info(f"Available capital: ${portfolio['available_capital']}")

if __name__ == "__main__":
    main()