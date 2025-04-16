#!/usr/bin/env python3
"""
Add Active Position to Sandbox Portfolio

This script adds an active position to the sandbox portfolio without relying on
the Kraken API. Useful for testing portfolio management and position tracking.

Usage:
    python add_active_position.py --pair PAIR --side SIDE --amount AMOUNT --entry_price PRICE --leverage LEVERAGE

Example:
    python add_active_position.py --pair ETH/USD --side buy --amount 4000 --entry_price 3245.60 --leverage 2
"""

import os
import json
import argparse
import logging
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Add an active position to sandbox portfolio")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g., ETH/USD)")
    parser.add_argument("--side", type=str, choices=["buy", "sell"], required=True, help="Position side (buy or sell)")
    parser.add_argument("--amount", type=float, required=True, help="Position amount in USD")
    parser.add_argument("--entry_price", type=float, required=True, help="Entry price")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage used for the position")
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


def add_active_position(portfolio, pair, side, amount, entry_price, leverage):
    """Add an active position to the portfolio"""
    if not portfolio:
        return False

    # Determine the position side
    position_side = "long" if side == "buy" else "short"
    
    # Generate position ID
    position_id = str(uuid.uuid4())
    
    # Calculate size in units
    units = amount / entry_price
    
    # Create position details
    position = {
        "id": position_id,
        "pair": pair,
        "side": position_side,
        "entry_time": datetime.now().isoformat(),
        "entry_price": entry_price,
        "current_price": entry_price,  # Start with entry price as current price
        "amount": amount,
        "units": units,
        "leverage": leverage,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_percent": 0.0,
        "strategy": "ml_ensemble",
        "stop_loss": None,
        "take_profit": None
    }
    
    # Add to positions dict
    if "positions" not in portfolio:
        portfolio["positions"] = {}
    
    portfolio["positions"][position_id] = position
    
    # Update current capital to reflect the allocated amount
    portfolio["current_capital"] -= amount
    
    # Update last updated timestamp
    portfolio["last_updated"] = datetime.now().isoformat()
    
    return True


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load portfolio
    portfolio = load_portfolio()
    if not portfolio:
        return 1
    
    # Add active position
    if add_active_position(portfolio, args.pair, args.side, args.amount, args.entry_price, args.leverage):
        if save_portfolio(portfolio):
            logger.info(f"Added active position: {args.pair} {args.side} with amount ${args.amount:.2f} at price ${args.entry_price:.2f}")
            return 0
    
    logger.error("Failed to add active position")
    return 1


if __name__ == "__main__":
    exit(main())