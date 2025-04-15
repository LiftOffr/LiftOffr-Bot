#!/usr/bin/env python3

"""
Reset Sandbox Positions

This script resets the sandbox positions by:
1. Downloading current market prices from the Kraken API
2. Creating new positions with accurate entry prices based on current market data
3. Calculating proper PnL values and margins
4. Updating the portfolio files with the new data

Use this to reset the sandbox environment with realistic data.
"""

import os
import sys
import json
import logging
import random
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

# Environment variables
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

# Default pairs if no ML config available
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
    "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

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
                prices = _get_fallback_prices(pairs)
        else:
            # No API keys, use fallback prices
            logger.info("No API keys, using fallback prices")
            prices = _get_fallback_prices(pairs)
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        prices = _get_fallback_prices(pairs)
    
    return prices

def _get_fallback_prices(pairs: List[str]) -> Dict[str, float]:
    """Get fallback prices based on recent market data"""
    fallback_prices = {
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
    
    return {p: fallback_prices.get(p, 100.0) for p in pairs if p in fallback_prices}

def create_new_positions(pairs: List[str], ml_config: Dict[str, Any], current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Create a set of new positions using current market prices.
    
    Args:
        pairs: List of trading pairs
        ml_config: ML configuration data
        current_prices: Current market prices
        
    Returns:
        List of position dictionaries
    """
    positions = []
    starting_capital = 20000.0
    position_capital = starting_capital * 0.95  # Use 95% of capital for positions
    capital_per_pair = position_capital / len(pairs)
    
    # Create timestamp for all positions
    now = datetime.now()
    timestamp = now.isoformat()
    
    for pair in pairs:
        if pair not in current_prices:
            logger.warning(f"No price available for {pair}, skipping")
            continue
        
        # Get pair configuration
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Get parameters
        base_leverage = pair_config.get("base_leverage", 38.0)
        max_leverage = pair_config.get("max_leverage", 100.0)
        risk_percentage = pair_config.get("risk_percentage", 0.20)
        current_price = current_prices[pair]
        
        # Create two positions per pair (one for each strategy)
        for strategy in ["Adaptive", "ARIMA"]:
            # Generate position parameters
            direction = "LONG" if random.random() < 0.5 else "SHORT"
            confidence = random.uniform(0.70, 0.95)
            
            # Adjust leverage based on confidence
            leverage = base_leverage + (confidence * (max_leverage - base_leverage))
            leverage = min(max_leverage, max(base_leverage, leverage))
            
            # Calculate margin (capital per position)
            margin = capital_per_pair / 2  # Divide equally between two strategies
            
            # Calculate position size
            position_size = (margin * leverage) / current_price
            
            # Calculate stop loss (0.5% away from entry)
            stop_loss_pct = 0.005
            stop_loss = current_price * (1 - stop_loss_pct) if direction == "LONG" else current_price * (1 + stop_loss_pct)
            
            # Calculate take profit (1% away from entry)
            take_profit_pct = 0.01
            take_profit = current_price * (1 + take_profit_pct) if direction == "LONG" else current_price * (1 - take_profit_pct)
            
            # Calculate liquidation price (depends on leverage)
            # Higher leverage means liquidation price is closer to entry
            liquidation_pct = 0.01 * (100 / leverage)  # 1% movement divided by leverage
            liquidation_price = current_price * (1 - liquidation_pct) if direction == "LONG" else current_price * (1 + liquidation_pct)
            
            # Create position
            position = {
                "pair": pair,
                "strategy": strategy,
                "direction": direction,
                "entry_price": current_price,
                "current_price": current_price,
                "size": position_size,
                "leverage": leverage,
                "margin": margin,
                "liquidation_price": liquidation_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "entry_time": timestamp,
                "duration": "0h 0m",
                "confidence": confidence
            }
            
            positions.append(position)
    
    return positions

def reset_sandbox_data():
    """Reset sandbox data with new positions and portfolio"""
    try:
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Load ML config
        ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
        
        # Get list of pairs
        pairs = list(ml_config.get("pairs", {}).keys())
        if not pairs:
            pairs = DEFAULT_PAIRS
        
        # Get current prices
        current_prices = get_current_prices(pairs)
        
        # Make sure we have prices for at least some pairs
        if not current_prices:
            logger.error("Could not get prices for any pairs, aborting")
            return False
        
        # Only use pairs with available prices
        available_pairs = [p for p in pairs if p in current_prices]
        logger.info(f"Using {len(available_pairs)} pairs with available prices: {available_pairs}")
        
        # Create new positions
        positions = create_new_positions(available_pairs, ml_config, current_prices)
        logger.info(f"Created {len(positions)} new positions")
        
        # Save positions
        if save_file(POSITIONS_FILE, positions):
            logger.info(f"Saved {len(positions)} positions to {POSITIONS_FILE}")
        else:
            logger.error(f"Failed to save positions to {POSITIONS_FILE}")
            return False
        
        # Create portfolio
        starting_capital = 20000.0
        portfolio = {
            "balance": starting_capital,
            "equity": starting_capital,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "total_return_percentage": 0.0,
            "trades": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Save portfolio
        if save_file(PORTFOLIO_FILE, portfolio):
            logger.info(f"Saved portfolio to {PORTFOLIO_FILE}")
        else:
            logger.error(f"Failed to save portfolio to {PORTFOLIO_FILE}")
            return False
        
        # Create initial portfolio history point
        history_point = {
            "timestamp": datetime.now().isoformat(),
            "balance": starting_capital,
            "equity": starting_capital,
            "portfolio_value": starting_capital,
            "unrealized_pnl": 0.0,
            "num_positions": len(positions)
        }
        
        # Save portfolio history
        if save_file(PORTFOLIO_HISTORY_FILE, [history_point]):
            logger.info(f"Saved portfolio history to {PORTFOLIO_HISTORY_FILE}")
        else:
            logger.error(f"Failed to save portfolio history to {PORTFOLIO_HISTORY_FILE}")
            return False
        
        logger.info("Successfully reset sandbox data")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting sandbox data: {e}")
        return False

if __name__ == "__main__":
    print("Resetting sandbox positions and portfolio data...")
    if reset_sandbox_data():
        print("Successfully reset sandbox data with current market prices")
    else:
        print("Failed to reset sandbox data")
        sys.exit(1)