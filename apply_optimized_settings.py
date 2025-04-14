#!/usr/bin/env python3
"""
Apply Optimized Settings to Trading Bot

This script applies the optimized ML ensemble weights and trading parameters
to the trading bot for maximum performance. It extracts the optimized settings
from the optimization results and configures the bot accordingly.

Usage:
    python apply_optimized_settings.py [--pair PAIR] [--sandbox]
"""

import os
import sys
import json
import logging
import argparse
import shutil
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/apply_settings.log")
    ]
)
logger = logging.getLogger(__name__)

# Paths
OPTIMIZATION_DIR = "optimization_results/enhanced"
BACKTEST_DIR = "backtest_results/maximized"
CONFIG_DIR = "config"
ML_CONFIG_PATH = os.path.join(CONFIG_DIR, "ml_config.json")
ENSEMBLE_DIR = "models/ensemble"

def load_optimization_results(pair: str) -> Optional[Dict[str, Any]]:
    """
    Load optimization results for a trading pair
    
    Args:
        pair: Trading pair (e.g., "SOL/USD")
        
    Returns:
        Dict: Optimization results or None if not found
    """
    pair_code = pair.replace("/", "")
    path = os.path.join(OPTIMIZATION_DIR, f"{pair_code}_results.json")
    
    if not os.path.exists(path):
        logger.error(f"Optimization results not found: {path}")
        return None
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading optimization results: {e}")
        return None

def load_backtest_results() -> Optional[Dict[str, Any]]:
    """
    Load backtest results
    
    Returns:
        Dict: Backtest results or None if not found
    """
    path = os.path.join(BACKTEST_DIR, "maximized_returns_backtest.json")
    
    if not os.path.exists(path):
        logger.error(f"Backtest results not found: {path}")
        return None
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return None

def create_ensemble_config(pair: str, optimization_results: Dict[str, Any]) -> bool:
    """
    Create ensemble configuration from optimization results
    
    Args:
        pair: Trading pair
        optimization_results: Optimization results
        
    Returns:
        bool: True if successful, False otherwise
    """
    pair_code = pair.replace("/", "")
    
    # Ensure ensemble directory exists
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    
    # Extract model weights
    weights = optimization_results.get("optimal_weights", {})
    regime_weights = optimization_results.get("market_regime_weights", {})
    
    if not weights:
        logger.error("No weights found in optimization results")
        return False
        
    # Create weights configuration
    weights_config = {
        "trading_pair": pair,
        "timeframe": optimization_results.get("timeframe", "1h"),
        "base_weights": weights,
        "market_regime_weights": regime_weights,
        "optimization_results": {
            "max_accuracy": optimization_results.get("achieved_accuracy", 0),
            "max_profit_factor": optimization_results.get("achieved_profit_factor", 0),
            "best_weights": weights,
            "best_regime_weights": regime_weights
        },
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "version": "2.0.0"
    }
    
    # Save weights configuration
    weights_path = os.path.join(ENSEMBLE_DIR, f"{pair_code}_weights.json")
    with open(weights_path, 'w') as f:
        json.dump(weights_config, f, indent=2)
        
    logger.info(f"Saved ensemble weights to {weights_path}")
    
    # Create ensemble configuration
    ensemble_config = {
        "pair": pair,
        "timeframe": optimization_results.get("timeframe", "1h"),
        "base_models": {
            model_type: f"{model_type}/{pair_code}_{optimization_results.get('timeframe', '1h')}.h5"
            for model_type in weights.keys()
        },
        "weights": weights,
        "market_regime_weights": regime_weights,
        "prediction_threshold": 0.55,  # Higher threshold for increased confidence
        "features": {
            "use_price": True,
            "use_volume": True,
            "use_technical": True,
            "use_sentiment": False,  # Could be enabled if sentiment data available
            "lookback_periods": 24
        }
    }
    
    # Save ensemble configuration
    ensemble_path = os.path.join(ENSEMBLE_DIR, f"{pair_code}_ensemble.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
        
    logger.info(f"Saved ensemble configuration to {ensemble_path}")
    
    return True

def update_ml_config(pairs: list, backtest_results: Dict[str, Any]) -> bool:
    """
    Update ML configuration with optimized settings
    
    Args:
        pairs: List of trading pairs
        backtest_results: Backtest results
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Load existing ML config if available
    ml_config = {}
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                ml_config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading existing ML config: {e}")
            
    # Get trading parameters from backtest results
    trading_params = backtest_results.get("backtest_config", {})
    
    # Update ML config
    ml_config.update({
        "ml_enabled": True,
        "use_ensemble": True,
        "use_market_regime": True,
        "pairs": pairs,
        "base_leverage": trading_params.get("base_leverage", 75.0),
        "max_leverage": trading_params.get("max_leverage", 200.0),
        "risk_per_trade": trading_params.get("risk_per_trade", 0.4),
        "confidence_threshold": trading_params.get("confidence_threshold", 0.55),
        "position_sizing": {
            "dynamic": True,
            "confidence_scaling": True,
            "max_allocation_per_pair": 0.5,  # 50% of available capital per pair
            "base_risk": trading_params.get("risk_per_trade", 0.4)
        },
        "exit_rules": {
            "use_trailing_stop": True,
            "trailing_stop_activation": 0.02,  # Activate trailing stop after 2% profit
            "trailing_stop_distance": 0.01,    # 1% trailing stop distance
            "take_profit": 0.1,                # 10% take-profit
            "stop_loss": 0.04,                 # 4% stop-loss
            "use_prediction_reversal": True    # Exit on prediction reversal
        },
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Save updated ML config
    with open(ML_CONFIG_PATH, 'w') as f:
        json.dump(ml_config, f, indent=2)
        
    logger.info(f"Updated ML configuration in {ML_CONFIG_PATH}")
    
    return True

def apply_settings(pair: str, sandbox: bool = True) -> bool:
    """
    Apply optimized settings for a trading pair
    
    Args:
        pair: Trading pair
        sandbox: Whether to run in sandbox mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Applying optimized settings for {pair} (sandbox: {sandbox})")
    
    # Load optimization results
    optimization_results = load_optimization_results(pair)
    if not optimization_results:
        return False
        
    # Load backtest results
    backtest_results = load_backtest_results()
    if not backtest_results:
        return False
        
    # Create ensemble configuration
    if not create_ensemble_config(pair, optimization_results):
        return False
        
    # Update ML configuration
    if not update_ml_config([pair], backtest_results):
        return False
        
    logger.info(f"Successfully applied optimized settings for {pair}")
    
    # Display command to start the trading bot with optimized settings
    sandbox_flag = "--sandbox" if sandbox else ""
    cmd = f"python main.py --pair {pair} {sandbox_flag} --live-ml"
    
    print("\n" + "=" * 80)
    print("OPTIMIZED SETTINGS APPLIED SUCCESSFULLY")
    print("=" * 80)
    print(f"Trading Pair: {pair}")
    print(f"ML Accuracy: {optimization_results.get('achieved_accuracy', 0) * 100:.2f}%")
    print(f"Profit Factor: {optimization_results.get('achieved_profit_factor', 0):.2f}")
    print(f"Win Rate: {backtest_results.get('win_rate', 0) * 100:.2f}%")
    print(f"Expected Return: {backtest_results.get('total_return_pct', 0) * 100:.2f}%")
    print("=" * 80)
    print("\nTo start the trading bot with optimized settings, run:")
    print(f"  {cmd}")
    print("=" * 80)
    
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apply Optimized Settings to Trading Bot")
    
    parser.add_argument("--pair", type=str, default="SOL/USD",
                        help="Trading pair to apply settings for")
    
    parser.add_argument("--sandbox", action="store_true",
                        help="Run in sandbox mode (no real trades)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Apply optimized settings
    if not apply_settings(args.pair, args.sandbox):
        logger.error("Failed to apply optimized settings")
        sys.exit(1)
        
if __name__ == "__main__":
    main()