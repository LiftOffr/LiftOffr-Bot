#!/usr/bin/env python3
"""
Add Sandbox Trade

This script adds a simulated trade to the sandbox portfolio without relying on
the Kraken API. Useful for testing portfolio management and performance tracking.

Usage:
    python add_sandbox_trade.py --pair PAIR --side SIDE --profit PROFIT --amount AMOUNT

Example:
    python add_sandbox_trade.py --pair BTC/USD --side buy --profit 1000 --amount 5000
"""

import os
import json
import argparse
import logging
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PORTFOLIO_FILE = "config/sandbox_portfolio.json"
SANDBOX_MODE = True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Add a simulated trade to sandbox portfolio")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g., BTC/USD)")
    parser.add_argument("--side", type=str, choices=["buy", "sell"], required=True, help="Trade side (buy or sell)")
    parser.add_argument("--profit", type=float, required=True, help="Profit/loss for the trade")
    parser.add_argument("--amount", type=float, required=True, help="Trade amount in USD")
    return parser.parse_args()


def load_portfolio():
    """Load sandbox portfolio"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        else:
            logger.error(f"Portfolio file {PORTFOLIO_FILE} not found")
            return None
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return None


def save_portfolio(portfolio):
    """Save sandbox portfolio"""
    try:
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")
        return False


def add_trade(portfolio, pair, side, profit, amount):
    """Add a trade to the portfolio"""
    if not portfolio:
        return False

    # Update portfolio equity
    portfolio["equity"] += profit
    portfolio["current_capital"] += profit
    
    # Create trade details
    current_time = datetime.now()
    entry_time = current_time - timedelta(hours=1)
    exit_time = current_time
    
    trade_id = str(uuid.uuid4())
    
    # Calculate other trade metrics
    entry_price = 1.0  # Placeholder
    exit_price = 1.0 + (profit / amount) if side == "buy" else 1.0 - (profit / amount)
    
    if side == "buy":
        position_side = "long"
    else:
        position_side = "short"
    
    # Create trade object
    trade = {
        "id": trade_id,
        "pair": pair,
        "side": position_side,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "amount": amount,
        "leverage": 3,
        "profit_loss": profit,
        "profit_loss_percent": (profit / amount) * 100,
        "tags": ["simulated", "demo"],
        "strategy": "ml_ensemble"
    }
    
    # Add to completed trades
    portfolio["completed_trades"].append(trade)
    portfolio["last_updated"] = datetime.now().isoformat()
    
    return True


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load portfolio
    portfolio = load_portfolio()
    if not portfolio:
        return 1
    
    # Add trade
    if add_trade(portfolio, args.pair, args.side, args.profit, args.amount):
        if save_portfolio(portfolio):
            logger.info(f"Added simulated trade: {args.pair} {args.side} with profit/loss ${args.profit:.2f}")
            return 0
    
    logger.error("Failed to add simulated trade")
    return 1


if __name__ == "__main__":
    exit(main())