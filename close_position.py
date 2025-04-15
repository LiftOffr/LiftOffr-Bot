#!/usr/bin/env python3
"""
Close Position

This script closes a specific position:
1. Loads position from the data file
2. Calculates profit/loss based on current market price
3. Updates portfolio balance
4. Removes position from positions file
5. Adds trade to trades file
"""
import os
import json
import logging
import argparse
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories and files
DATA_DIR = "data"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Close a specific position")
    parser.add_argument("--position-id", type=str, required=True, 
                        help="Position ID to close")
    parser.add_argument("--liquidation", action="store_true", default=False,
                        help="Whether this is a liquidation (default: False)")
    return parser.parse_args()

def load_positions():
    """Load positions from file"""
    try:
        if os.path.exists(POSITIONS_PATH):
            with open(POSITIONS_PATH, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions")
            return positions
        else:
            logger.warning(f"Positions file not found: {POSITIONS_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return {}

def save_positions(positions):
    """Save positions to file"""
    try:
        with open(POSITIONS_PATH, 'w') as f:
            json.dump(positions, f, indent=2)
        logger.info(f"Saved {len(positions)} positions")
    except Exception as e:
        logger.error(f"Error saving positions: {e}")

def load_trades():
    """Load trades from file"""
    try:
        if os.path.exists(TRADES_PATH):
            with open(TRADES_PATH, 'r') as f:
                trades = json.load(f)
                # Check if trades is a list (legacy format) or a dict (new format)
                if isinstance(trades, list):
                    # Convert to dict with index as keys
                    trades_dict = {}
                    for i, trade in enumerate(trades):
                        trades_dict[f'legacy_{i}'] = trade
                    logger.info(f"Converted {len(trades)} trades from list to dict format")
                    return trades_dict
                else:
                    logger.info(f"Loaded {len(trades)} trades")
                    return trades
        else:
            logger.warning(f"Trades file not found: {TRADES_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return {}

def save_trades(trades):
    """Save trades to file"""
    try:
        with open(TRADES_PATH, 'w') as f:
            json.dump(trades, f, indent=2)
        logger.info(f"Saved {len(trades)} trades")
    except Exception as e:
        logger.error(f"Error saving trades: {e}")

def load_portfolio():
    """Load portfolio from file"""
    try:
        if os.path.exists(PORTFOLIO_PATH):
            with open(PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            logger.info(f"Loaded portfolio with balance: ${portfolio.get('balance', 0):.2f}")
            return portfolio
        else:
            logger.warning(f"Portfolio file not found: {PORTFOLIO_PATH}")
            return {
                "balance": 20000.0,
                "initial_balance": 20000.0,
                "last_updated": datetime.datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return {
            "balance": 20000.0,
            "initial_balance": 20000.0,
            "last_updated": datetime.datetime.now().isoformat()
        }

def save_portfolio(portfolio):
    """Save portfolio to file"""
    try:
        portfolio["last_updated"] = datetime.datetime.now().isoformat()
        with open(PORTFOLIO_PATH, 'w') as f:
            json.dump(portfolio, f, indent=2)
        logger.info(f"Saved portfolio with balance: ${portfolio.get('balance', 0):.2f}")
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")

def close_position(position_id, liquidation=False):
    """
    Close a position
    
    Args:
        position_id: Position ID to close
        liquidation: Whether this is a liquidation
    """
    # Load data
    positions = load_positions()
    trades = load_trades()
    portfolio = load_portfolio()
    
    # Check if position exists
    if position_id not in positions:
        logger.error(f"Position {position_id} not found")
        return False
    
    # Get position data
    position = positions[position_id]
    
    # Get relevant data
    pair = position.get('symbol', '')
    entry_price = position.get('entry_price', 0)
    current_price = position.get('current_price', entry_price)
    size = position.get('size', 0)
    long = position.get('long', True)
    leverage = position.get('leverage', 1.0)
    liquidation_price = position.get('liquidation_price', 0)
    
    # Determine exit price
    if liquidation:
        exit_price = liquidation_price
    else:
        exit_price = current_price
    
    # Calculate profit/loss
    if long:
        profit_loss = (exit_price - entry_price) * size
    else:
        profit_loss = (entry_price - exit_price) * size
    
    # If liquidated, loss is fixed at the risk amount
    if liquidation:
        # In leveraged trading, we lose the margin amount
        risk_amount = (size * entry_price) / leverage
        profit_loss = -risk_amount
    
    # Update portfolio balance
    portfolio['balance'] = portfolio.get('balance', 0) + profit_loss
    
    # Create trade record
    trade_id = f"trade_{position_id}"
    trade = {
        'symbol': pair,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'size': size,
        'long': long,
        'leverage': leverage,
        'entry_time': position.get('entry_time', ''),
        'exit_time': datetime.datetime.now().isoformat(),
        'profit_loss': profit_loss,
        'status': 'liquidated' if liquidation else 'closed'
    }
    
    # Add to trades
    trades[trade_id] = trade
    
    # Remove from positions
    del positions[position_id]
    
    # Save data
    save_positions(positions)
    save_trades(trades)
    save_portfolio(portfolio)
    
    # Log the trade
    direction = "LONG" if long else "SHORT"
    status = "LIQUIDATED" if liquidation else "CLOSED"
    pnl_pct = (profit_loss / (entry_price * size)) * 100
    
    logger.info(f"{status} {direction} position {position_id} for {pair}")
    logger.info(f"Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
    logger.info(f"P/L: ${profit_loss:.2f} ({pnl_pct:.2f}%)")
    logger.info(f"New portfolio balance: ${portfolio.get('balance', 0):.2f}")
    
    return True

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    position_id = args.position_id
    liquidation = args.liquidation
    
    # Close position
    result = close_position(position_id, liquidation)
    
    if result:
        logger.info("Position closed successfully")
    else:
        logger.error("Failed to close position")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise