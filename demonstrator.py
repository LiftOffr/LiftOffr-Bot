#!/usr/bin/env python3
"""
Trade Entry Demonstrator

This script demonstrates the proper trade entry mechanism that:
1. Uses only available capital after existing positions
2. Implements risk management for leverage and position size
3. Shows liquidation prices and risk parameters
"""
import os
import logging
import requests
import time
from datetime import datetime

# Import the trade entry manager
from trade_entry_manager import TradeEntryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def demonstrate_sequential_trades():
    """Demonstrate opening trades sequentially using available capital"""
    logger.info("=" * 50)
    logger.info("Demonstrating sequential trades using available capital")
    logger.info("=" * 50)
    
    # Create the trade manager
    trade_manager = TradeEntryManager()
    
    # Get initial available capital
    initial_capital = trade_manager.get_available_capital()
    logger.info(f"Initial available capital: ${initial_capital:.2f}")
    
    # List of pairs to demonstrate with
    pairs_to_trade = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]
    
    # Open trades sequentially, always using AVAILABLE capital
    for i, pair in enumerate(pairs_to_trade):
        # Get current price
        current_price = get_kraken_price(pair)
        if not current_price:
            logger.error(f"Could not get price for {pair}, skipping")
            continue
        
        # Generate confidence (decreasing slightly for each subsequent trade for demo)
        confidence = 0.90 - (i * 0.05)
        confidence = max(0.65, confidence)  # Ensure minimum confidence
        
        # Determine model and category (alternating for demo)
        model = "Adaptive" if i % 2 == 0 else "ARIMA"
        category = "those dudes" if i % 2 == 0 else "him all along"
        
        # Show available capital before trade
        available_before = trade_manager.get_available_capital()
        logger.info(f"Available capital before trade #{i+1}: ${available_before:.2f}")
        
        # Try to open trade
        if available_before < 100:
            logger.warning(f"Insufficient capital (${available_before:.2f}) to open trade for {pair}")
            break
            
        # Open the trade
        position = trade_manager.open_trade(
            pair=pair,
            entry_price=current_price,
            confidence=confidence,
            model=model,
            category=category
        )
        
        if position:
            # Show available capital after trade
            available_after = trade_manager.get_available_capital()
            logger.info(f"Available capital after trade #{i+1}: ${available_after:.2f}")
            logger.info(f"Capital used for trade: ${available_before - available_after:.2f}")
            
            # Show position details
            logger.info(f"Position details:")
            logger.info(f"  Pair: {position['pair']}")
            logger.info(f"  Direction: {position['direction']}")
            logger.info(f"  Entry price: ${position['entry_price']}")
            logger.info(f"  Position size: ${position['position_size']:.2f}")
            logger.info(f"  Leverage: {position['leverage']}x")
            logger.info(f"  Liquidation price: ${position['liquidation_price']:.2f}")
            logger.info(f"  Stop loss: {position['stop_loss_pct']}%")
            logger.info(f"  Take profit: {position['take_profit_pct']}%")
            logger.info("-" * 30)
        else:
            logger.warning(f"Failed to open trade for {pair}")
        
        # Pause between trades
        time.sleep(1)
    
    # Show final available capital
    final_capital = trade_manager.get_available_capital()
    logger.info(f"Final available capital: ${final_capital:.2f}")
    logger.info(f"Total capital used: ${initial_capital - final_capital:.2f}")

if __name__ == "__main__":
    demonstrate_sequential_trades()