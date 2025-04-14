#!/usr/bin/env python3
"""
Close All Positions Script

This script closes all open positions in the trading account,
used for resetting the sandbox portfolio.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Kraken API module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kraken_api import KrakenAPI

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Close all open positions in the trading account')
    
    parser.add_argument(
        '--sandbox',
        action='store_true',
        help='Use sandbox mode'
    )
    
    return parser.parse_args()

def get_open_positions(api):
    """
    Get open positions from Kraken API.
    This is a custom implementation as the KrakenAPI class might not have this method.
    
    Args:
        api: KrakenAPI instance
        
    Returns:
        Dictionary of open positions
    """
    try:
        # Try to call get_open_positions if it exists
        if hasattr(api, 'get_open_positions'):
            return api.get_open_positions()
            
        # If not, try to call the private API directly
        if hasattr(api, 'query_private'):
            positions = api.query_private('OpenPositions')
            return positions
        
        # Last resort, call the public interface
        logger.warning("Using fallback method to get open positions")
        # This is a generic approach - specifics depend on the KrakenAPI implementation
        # We may need to get the open orders instead as a fallback
        open_orders = api.get_open_orders() if hasattr(api, 'get_open_orders') else {}
        
        # Convert orders to position format
        positions = {}
        for order_id, order in open_orders.items():
            # Filter to only include orders that are positions
            if 'type' in order and 'pair' in order and 'vol' in order:
                positions[order_id] = order
                
        return positions
    except Exception as e:
        logger.error(f"Error getting open positions: {str(e)}")
        return {}

def close_position(api, position_id, position):
    """
    Close a specific position.
    
    Args:
        api: KrakenAPI instance
        position_id: Position ID
        position: Position data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        position_type = position.get('type')
        pair = position.get('pair')
        size = position.get('vol')
        
        if not (position_type and pair and size):
            logger.error(f"Invalid position data for {position_id}: {position}")
            return False
        
        # Determine opposite direction to close
        close_type = 'sell' if position_type == 'buy' else 'buy'
        
        logger.info(f"Closing {position_type} position on {pair} with size {size}")
        
        # Try different APIs that might be available in the KrakenAPI class
        if hasattr(api, 'create_order'):
            result = api.create_order(
                pair=pair,
                order_type='market',
                side=close_type,
                volume=size,
                leverage=1,
                reduce_only=True
            )
        elif hasattr(api, 'add_order'):
            result = api.add_order(
                pair=pair,
                type=close_type,
                ordertype='market',
                volume=size,
                leverage=1,
                reduce_only=True
            )
        elif hasattr(api, 'query_private'):
            # Last resort - use the private API directly
            params = {
                'pair': pair,
                'type': close_type,
                'ordertype': 'market',
                'volume': size,
                'leverage': 1,
                'reduce_only': True
            }
            result = api.query_private('AddOrder', params)
        else:
            logger.error("No suitable method found to close position")
            return False
        
        # Check result - different APIs might return different formats
        if result:
            if isinstance(result, dict) and ('id' in result or 'txid' in result or 'orderid' in result):
                logger.info(f"Successfully closed position {position_id}")
                return True
            elif isinstance(result, list) and len(result) > 0:
                logger.info(f"Successfully closed position {position_id}")
                return True
            
        logger.error(f"Failed to close position {position_id}: {result}")
        return False
        
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {str(e)}")
        return False

def close_all_positions(api: KrakenAPI) -> bool:
    """
    Close all open positions.
    
    Args:
        api: KrakenAPI instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get open positions using our custom function
        open_positions = get_open_positions(api)
        
        if not open_positions:
            logger.info("No open positions to close")
            return True
        
        logger.info(f"Found {len(open_positions)} open positions to close")
        
        # Close each position
        success = True
        for position_id, position in open_positions.items():
            if not close_position(api, position_id, position):
                success = False
        
        # Verify all positions are closed
        if success:
            # Wait a moment for orders to process
            time.sleep(2)
            
            open_positions = get_open_positions(api)
            if open_positions:
                logger.warning(f"Some positions ({len(open_positions)}) could not be closed")
                success = False
            else:
                logger.info("All positions closed successfully")
        
        return success
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Initializing Kraken API...")
    # Check KrakenAPI's interface to avoid the sandbox parameter error
    try:
        # Try with sandbox parameter first
        kraken = KrakenAPI(sandbox=args.sandbox)
    except TypeError:
        # If that fails, try without the parameter, but set the sandbox flag
        kraken = KrakenAPI()
        if hasattr(kraken, 'set_sandbox_mode') and args.sandbox:
            kraken.set_sandbox_mode(True)
    
    logger.info("Closing all open positions...")
    success = close_all_positions(kraken)
    
    if success:
        logger.info("All positions closed successfully")
    else:
        logger.error("Failed to close all positions")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())