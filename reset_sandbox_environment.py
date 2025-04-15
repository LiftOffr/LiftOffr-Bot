#!/usr/bin/env python3
"""
Reset Sandbox Environment

This script resets the sandbox trading environment:
1. Resets portfolio to $20,000
2. Removes all current positions
3. Clears trade history
4. Initializes portfolio history with starting balance
"""
import os
import json
import logging
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
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Make sure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def reset_portfolio():
    """Reset portfolio to $20,000"""
    portfolio = {
        "total": 20000.0,
        "available": 20000.0,
        "balance": 20000.0,
        "equity": 20000.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "daily_pnl": 0.0,
        "weekly_pnl": 0.0,
        "monthly_pnl": 0.0,
        "open_positions_count": 0,
        "margin_used_pct": 0.0,
        "available_margin": 20000.0,
        "max_leverage": 125.0
    }
    
    # Save portfolio
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)
    
    logger.info("Reset portfolio to $20,000")
    return portfolio

def reset_positions():
    """Remove all positions"""
    # Create empty positions dictionary
    positions = {}
    
    # Save positions
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)
    
    logger.info("Removed all positions")
    return positions

def reset_portfolio_history():
    """Reset portfolio history with starting balance"""
    # Create initial portfolio history entry
    now = datetime.now().isoformat()
    
    # Create both formats for compatibility
    # Format 1: Array of objects
    portfolio_history_array = [
        {
            "timestamp": now,
            "portfolio_value": 20000.0
        }
    ]
    
    # Save array format
    with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
        json.dump(portfolio_history_array, f, indent=2)
    
    logger.info("Reset portfolio history with starting balance")
    return portfolio_history_array

def reset_trades():
    """Clear trade history"""
    # Create empty trades array
    trades = []
    
    # Save trades
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    
    logger.info("Cleared trade history")
    return trades

def main():
    """Reset the entire sandbox environment"""
    logger.info("Resetting sandbox environment...")
    
    # Reset everything
    portfolio = reset_portfolio()
    positions = reset_positions()
    portfolio_history = reset_portfolio_history()
    trades = reset_trades()
    
    logger.info("Sandbox environment reset complete")
    
    # Return summary
    return {
        "portfolio": portfolio,
        "positions": positions,
        "portfolio_history": portfolio_history,
        "trades": trades
    }

if __name__ == "__main__":
    main()