#!/usr/bin/env python3
"""
Update Positions

This script updates all open positions with current prices:
1. Loads positions from the data file
2. Updates prices with current market data
3. Saves updated positions to the data file
"""
import os
import json
import logging
import argparse
import datetime
import numpy as np

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Update all open positions with current prices")
    parser.add_argument("--volatility", type=float, default=0.01, 
                        help="Price volatility (standard deviation) as a percentage (default: 0.01)")
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

def get_current_price(pair, last_price, volatility):
    """
    Get current price for a pair (simulated)
    
    Args:
        pair: Trading pair
        last_price: Last price
        volatility: Price volatility as a percentage
        
    Returns:
        float: Current price
    """
    # Calculate new price with random walk
    # Price moves randomly with standard deviation based on volatility
    price_change = last_price * volatility * np.random.randn()
    new_price = last_price + price_change
    
    # Ensure price is positive
    if new_price <= 0:
        new_price = last_price
    
    return new_price

def update_positions(volatility):
    """Update all positions with current prices"""
    # Load positions
    positions = load_positions()
    if not positions:
        logger.warning("No positions to update")
        return
    
    # Update each position
    for position_id, position in positions.items():
        try:
            # Get pair and last price
            pair = position.get('symbol', 'BTC/USD')
            last_price = position.get('current_price', position.get('entry_price', 0))
            
            # Get new price
            new_price = get_current_price(pair, last_price, volatility)
            
            # Update position
            position['current_price'] = new_price
            position['last_updated'] = datetime.datetime.now().isoformat()
            
            # Calculate profit/loss
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            long = position.get('long', True)
            
            if long:
                pnl = (new_price - entry_price) * size
                pnl_pct = ((new_price / entry_price) - 1) * 100
            else:
                pnl = (entry_price - new_price) * size
                pnl_pct = ((entry_price / new_price) - 1) * 100
            
            # Log update
            direction = "LONG" if long else "SHORT"
            logger.info(f"Updated {direction} position {position_id} for {pair}: ${last_price:.2f} -> ${new_price:.2f}, P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Check for liquidation
            liquidation_price = position.get('liquidation_price', 0)
            if (long and new_price <= liquidation_price) or (not long and new_price >= liquidation_price):
                logger.warning(f"Position {position_id} would be liquidated at current price ${new_price:.2f}")
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
    
    # Save updated positions
    save_positions(positions)

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    volatility = args.volatility
    
    logger.info(f"Updating positions with volatility {volatility}")
    
    # Update positions
    update_positions(volatility)
    
    logger.info("Position update complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error updating positions: {e}")
        raise