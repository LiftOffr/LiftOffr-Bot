#!/usr/bin/env python3
"""
Apply Optimized Trade Settings to ML Trading System

This script applies the optimized trade settings from the trade optimizer 
to the ML trading system, enhancing trading performance.
"""
import os
import json
import logging
import argparse
import time
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
DATA_DIR = "data"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
OPTIMIZATION_FILE = f"{DATA_DIR}/trade_optimization.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apply optimized trade settings to ML trading system")
    
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Trading pairs to apply settings to"
    )
    
    parser.add_argument(
        "--apply-only",
        choices=["market_states", "optimal_hours", "position_params", "all"],
        default="all",
        help="Apply only specific settings"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all settings to default before applying"
    )
    
    return parser.parse_args()

def load_optimization_data() -> Dict[str, Any]:
    """
    Load optimization data from file
    
    Returns:
        Optimization data
    """
    try:
        if os.path.exists(OPTIMIZATION_FILE):
            with open(OPTIMIZATION_FILE, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded optimization data from {OPTIMIZATION_FILE}")
            return data
    except Exception as e:
        logger.error(f"Error loading optimization data: {e}")
    
    return {
        'market_states': {},
        'optimal_hours': {},
        'volatility_data': {},
        'pair_specific_params': {},
        'last_updated': datetime.now().isoformat()
    }

def load_ml_config() -> Dict[str, Any]:
    """
    Load ML config from file
    
    Returns:
        ML config
    """
    try:
        if os.path.exists(ML_CONFIG_FILE):
            with open(ML_CONFIG_FILE, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded ML config from {ML_CONFIG_FILE}")
            return data
    except Exception as e:
        logger.error(f"Error loading ML config: {e}")
    
    return {
        'pairs': DEFAULT_PAIRS,
        'models': {},
        'strategies': [],
        'trading_params': {}
    }

def load_positions() -> Dict[str, Any]:
    """
    Load positions from file
    
    Returns:
        Positions dictionary
    """
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                positions_data = json.load(f)
            
            # Check if positions is a list or dictionary
            if isinstance(positions_data, list):
                # Convert list of positions to dictionary by pair
                positions_dict = {}
                for position in positions_data:
                    pair = position.get('pair')
                    if pair:
                        positions_dict[pair] = position
                return positions_dict
            elif isinstance(positions_data, dict):
                return positions_data
            else:
                logger.error(f"Unexpected positions data type: {type(positions_data)}")
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
    
    return {}

def save_ml_config(config: Dict[str, Any]):
    """
    Save ML config to file
    
    Args:
        config: ML config
    """
    try:
        with open(ML_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved ML config to {ML_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")

def save_positions(positions: Dict[str, Any]):
    """
    Save positions to file
    
    Args:
        positions: Positions dictionary
    """
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=2)
        logger.info(f"Saved positions to {POSITIONS_FILE}")
    except Exception as e:
        logger.error(f"Error saving positions: {e}")

def apply_market_states(
    optimization_data: Dict[str, Any],
    ml_config: Dict[str, Any],
    pairs: List[str]
):
    """
    Apply market states to ML config
    
    Args:
        optimization_data: Optimization data
        ml_config: ML config
        pairs: Trading pairs
    """
    market_states = optimization_data.get('market_states', {})
    
    # Create trading params if it doesn't exist
    if 'trading_params' not in ml_config:
        ml_config['trading_params'] = {}
    
    # Apply market states
    for pair in pairs:
        if pair in market_states:
            if 'market_state' not in ml_config['trading_params']:
                ml_config['trading_params']['market_state'] = {}
            
            ml_config['trading_params']['market_state'][pair] = market_states[pair]
            logger.info(f"Applied market state for {pair}: {market_states[pair]}")

def apply_optimal_hours(
    optimization_data: Dict[str, Any],
    ml_config: Dict[str, Any],
    pairs: List[str]
):
    """
    Apply optimal hours to ML config
    
    Args:
        optimization_data: Optimization data
        ml_config: ML config
        pairs: Trading pairs
    """
    optimal_hours = optimization_data.get('optimal_hours', {})
    
    # Create trading params if it doesn't exist
    if 'trading_params' not in ml_config:
        ml_config['trading_params'] = {}
    
    # Apply optimal hours
    for pair in pairs:
        if pair in optimal_hours:
            if 'optimal_hours' not in ml_config['trading_params']:
                ml_config['trading_params']['optimal_hours'] = {}
            
            ml_config['trading_params']['optimal_hours'][pair] = optimal_hours[pair]
            logger.info(f"Applied optimal hours for {pair}")

def apply_position_params(
    optimization_data: Dict[str, Any],
    ml_config: Dict[str, Any],
    positions: Dict[str, Any],
    pairs: List[str]
):
    """
    Apply position parameters to ML config and positions
    
    Args:
        optimization_data: Optimization data
        ml_config: ML config
        positions: Positions dictionary
        pairs: Trading pairs
    """
    pair_params = optimization_data.get('pair_specific_params', {})
    
    # Create trading params if it doesn't exist
    if 'trading_params' not in ml_config:
        ml_config['trading_params'] = {}
    
    # Apply position parameters to ML config
    for pair in pairs:
        if pair in pair_params:
            if 'pair_params' not in ml_config['trading_params']:
                ml_config['trading_params']['pair_params'] = {}
            
            ml_config['trading_params']['pair_params'][pair] = pair_params[pair]
            logger.info(f"Applied position parameters for {pair} to ML config")
    
    # Apply position parameters to active positions
    modified_positions = False
    for pair, position in positions.items():
        if pair in pair_params and pair in pairs:
            params = pair_params[pair]
            
            # Update stop loss
            if 'optimal_stop_loss' in params:
                old_sl = position.get('stop_loss_pct', 4.0)
                new_sl = params['optimal_stop_loss']
                
                # Only update if significant difference
                if abs(new_sl - old_sl) / old_sl > 0.1:  # >10% difference
                    position['stop_loss_pct'] = new_sl
                    modified_positions = True
                    logger.info(f"Updated stop loss for {pair} position: {old_sl:.2f}% -> {new_sl:.2f}%")
            
            # Update take profit
            if 'optimal_take_profit' in params:
                old_tp = position.get('take_profit_pct', 15.0)
                new_tp = params['optimal_take_profit']
                
                # Only update if significant difference
                if abs(new_tp - old_tp) / old_tp > 0.1:  # >10% difference
                    position['take_profit_pct'] = new_tp
                    modified_positions = True
                    logger.info(f"Updated take profit for {pair} position: {old_tp:.2f}% -> {new_tp:.2f}%")
            
            # Update trailing stop parameters
            if 'trailing_stop_activation' in params and 'trailing_stop_distance' in params:
                old_act = position.get('trailing_stop_activation', 5.0)
                new_act = params['trailing_stop_activation']
                
                old_dist = position.get('trailing_stop_distance', 2.5)
                new_dist = params['trailing_stop_distance']
                
                # Only update if significant difference
                if (abs(new_act - old_act) / old_act > 0.1 or  # >10% difference
                        abs(new_dist - old_dist) / old_dist > 0.1):
                    position['trailing_stop_activation'] = new_act
                    position['trailing_stop_distance'] = new_dist
                    modified_positions = True
                    logger.info(f"Updated trailing stop for {pair} position: "
                               f"Activation {old_act:.2f}% -> {new_act:.2f}%, "
                               f"Distance {old_dist:.2f}% -> {new_dist:.2f}%")
    
    # Save positions if modified
    if modified_positions:
        save_positions(positions)

def apply_all_settings(
    optimization_data: Dict[str, Any],
    ml_config: Dict[str, Any],
    positions: Dict[str, Any],
    pairs: List[str]
):
    """
    Apply all optimization settings
    
    Args:
        optimization_data: Optimization data
        ml_config: ML config
        positions: Positions dictionary
        pairs: Trading pairs
    """
    apply_market_states(optimization_data, ml_config, pairs)
    apply_optimal_hours(optimization_data, ml_config, pairs)
    apply_position_params(optimization_data, ml_config, positions, pairs)

def reset_settings(ml_config: Dict[str, Any], pairs: List[str]):
    """
    Reset all settings to default
    
    Args:
        ml_config: ML config
        pairs: Trading pairs
    """
    # Reset market states
    if 'trading_params' in ml_config and 'market_state' in ml_config['trading_params']:
        for pair in pairs:
            if pair in ml_config['trading_params']['market_state']:
                ml_config['trading_params']['market_state'][pair] = 'normal'
                logger.info(f"Reset market state for {pair} to normal")
    
    # Reset optimal hours
    if 'trading_params' in ml_config and 'optimal_hours' in ml_config['trading_params']:
        for pair in pairs:
            if pair in ml_config['trading_params']['optimal_hours']:
                ml_config['trading_params']['optimal_hours'][pair] = {
                    'entry': list(range(24)),
                    'exit': list(range(24))
                }
                logger.info(f"Reset optimal hours for {pair} to all hours")
    
    # Reset pair params
    if 'trading_params' in ml_config and 'pair_params' in ml_config['trading_params']:
        for pair in pairs:
            if pair in ml_config['trading_params']['pair_params']:
                ml_config['trading_params']['pair_params'][pair] = {
                    'min_confidence': 0.65,
                    'ideal_entry_volatility': 'medium',
                    'min_momentum_confirmation': 2,
                    'optimal_take_profit': 15.0,
                    'optimal_stop_loss': 4.0,
                    'trailing_stop_activation': 5.0,
                    'trailing_stop_distance': 2.5
                }
                logger.info(f"Reset pair parameters for {pair} to default")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load data
    optimization_data = load_optimization_data()
    ml_config = load_ml_config()
    positions = load_positions()
    
    # Check if optimization data is available
    if not optimization_data or not optimization_data.get('last_updated'):
        logger.error("No optimization data available. Run the trade optimizer first.")
        return 1
    
    # Check last updated
    last_updated = optimization_data.get('last_updated')
    try:
        last_updated_dt = datetime.fromisoformat(last_updated)
        time_delta = datetime.now() - last_updated_dt
        
        if time_delta.days > 1:
            logger.warning(f"Optimization data is {time_delta.days} days old. Consider running the trade optimizer.")
    except (ValueError, TypeError):
        logger.warning("Could not determine age of optimization data.")
    
    # Reset settings if requested
    if args.reset:
        reset_settings(ml_config, args.pairs)
    
    # Apply settings
    if args.apply_only == "market_states":
        apply_market_states(optimization_data, ml_config, args.pairs)
    elif args.apply_only == "optimal_hours":
        apply_optimal_hours(optimization_data, ml_config, args.pairs)
    elif args.apply_only == "position_params":
        apply_position_params(optimization_data, ml_config, positions, args.pairs)
    else:  # "all"
        apply_all_settings(optimization_data, ml_config, positions, args.pairs)
    
    # Save ML config
    save_ml_config(ml_config)
    
    logger.info("Applied optimized trade settings to ML trading system")
    logger.info("To activate these settings, restart the trading bot")
    
    return 0

if __name__ == "__main__":
    exit(main())