#!/usr/bin/env python3
"""
Setup ML Trading System

This script configures the ML trading system with lighter-weight models:
1. Sets up configuration for all trading pairs
2. Sets maximum leverage to 75x (reduced from 125x)
3. Resets the portfolio to $20,000
"""
import os
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories
MODEL_WEIGHTS_DIR = "model_weights"
DATA_DIR = "data"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Default pairs to configure
DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "ADA/USD",
    "DOT/USD",
    "LINK/USD",
    "AVAX/USD",
    "MATIC/USD",
    "UNI/USD",
    "ATOM/USD"
]

# Maximum leverage setting (reduced from 125x to 75x)
MAX_LEVERAGE = 75.0
MIN_LEVERAGE = 5.0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Setup ML trading system")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to configure")
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Confidence threshold for trading signals")
    parser.add_argument("--risk-percentage", type=float, default=0.20,
                        help="Risk percentage for each trade")
    return parser.parse_args()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [MODEL_WEIGHTS_DIR, DATA_DIR, CONFIG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def update_ml_config(pairs, confidence_threshold=0.65, risk_percentage=0.20):
    """
    Update ML configuration for all pairs
    
    Args:
        pairs: List of trading pairs to configure
        confidence_threshold: Confidence threshold for trading signals
        risk_percentage: Risk percentage for each trade
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create new ML configuration
        ml_config = {
            "models": {},
            "global_settings": {
                "confidence_threshold": confidence_threshold,
                "dynamic_leverage_range": {
                    "min": MIN_LEVERAGE,
                    "max": MAX_LEVERAGE
                },
                "risk_percentage": risk_percentage,
                "max_positions_per_pair": 1
            }
        }
        
        # Configure each pair with a simple model
        for pair in pairs:
            symbol = pair.split('/')[0].lower()
            ml_config["models"][pair] = {
                "model_type": "lstm",  # Using LSTM as a simpler model type
                "confidence_threshold": confidence_threshold,
                "min_leverage": MIN_LEVERAGE,
                "max_leverage": MAX_LEVERAGE,
                "risk_percentage": risk_percentage
            }
        
        # Save configuration
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML configuration with {len(pairs)} pairs")
        logger.info(f"Max leverage set to {MAX_LEVERAGE}x (reduced from 125x)")
        return True
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False

def reset_portfolio():
    """Reset portfolio to initial state with $20,000"""
    try:
        # Create portfolio data
        now = datetime.now().isoformat()
        portfolio_data = {
            "balance": 20000.0,
            "initial_balance": 20000.0,
            "last_updated": now
        }
        
        # Save portfolio data
        portfolio_path = f"{DATA_DIR}/sandbox_portfolio.json"
        with open(portfolio_path, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        # Reset positions
        positions_path = f"{DATA_DIR}/sandbox_positions.json"
        with open(positions_path, 'w') as f:
            json.dump({}, f, indent=2)
        
        # Reset trades
        trades_path = f"{DATA_DIR}/sandbox_trades.json"
        with open(trades_path, 'w') as f:
            json.dump({}, f, indent=2)
        
        # Reset portfolio history
        history_path = f"{DATA_DIR}/sandbox_portfolio_history.json"
        history_data = [{
            "timestamp": now,
            "balance": 20000.0,
            "unrealized_pnl": 0.0,
            "equity": 20000.0
        }]
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info("Reset portfolio to $20,000")
        logger.info("Cleared all positions, trades, and history")
        return True
    except Exception as e:
        logger.error(f"Error resetting portfolio: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    pairs = args.pairs
    confidence_threshold = args.confidence_threshold
    risk_percentage = args.risk_percentage
    
    logger.info(f"Setting up ML trading system for {len(pairs)} pairs")
    logger.info(f"Pairs: {', '.join(pairs)}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Update ML configuration
    update_ml_config(pairs, confidence_threshold, risk_percentage)
    
    # Reset portfolio
    reset_portfolio()
    
    logger.info("ML trading system setup complete")
    logger.info("Remember to activate trading with 'python activate_ml_trading.py --sandbox'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error setting up ML trading system: {e}")
        raise