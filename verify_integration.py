#!/usr/bin/env python3
"""
Verify Integration of Trade Optimizer

This script verifies that the trade optimizer is properly integrated with the ML trading system
by checking that optimized settings are being applied to live trades.
"""
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
OPTIMIZATION_FILE = f"{DATA_DIR}/trade_optimization.json"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"

def load_file(filename: str) -> Any:
    """Load JSON data from file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
    
    return None

def verify_optimization_data():
    """Verify optimization data"""
    optimization_data = load_file(OPTIMIZATION_FILE)
    
    if not optimization_data:
        logger.error("No optimization data found")
        return False
    
    # Check for required keys
    required_keys = ['market_states', 'optimal_hours', 'volatility_data', 'pair_specific_params']
    missing_keys = [k for k in required_keys if k not in optimization_data]
    
    if missing_keys:
        logger.error(f"Missing keys in optimization data: {missing_keys}")
        return False
    
    # Check last updated timestamp
    last_updated = optimization_data.get('last_updated')
    if not last_updated:
        logger.warning("No last_updated timestamp in optimization data")
    else:
        try:
            last_updated_dt = datetime.fromisoformat(last_updated)
            time_delta = datetime.now() - last_updated_dt
            
            if time_delta.days > 0:
                logger.warning(f"Optimization data is {time_delta.days} days old")
            else:
                logger.info(f"Optimization data was updated {time_delta.seconds // 3600} hours ago")
        except ValueError:
            logger.warning(f"Invalid last_updated timestamp: {last_updated}")
    
    # Check market states for all trading pairs
    market_states = optimization_data.get('market_states', {})
    for pair, state in market_states.items():
        logger.info(f"Market state for {pair}: {state}")
    
    # Log volatility regimes
    volatility_data = optimization_data.get('volatility_data', {})
    for pair, data in volatility_data.items():
        regime = data.get('regime', 'unknown')
        current = data.get('current', 0.0)
        logger.info(f"Volatility for {pair}: {regime} ({current:.2f}%)")
    
    # Check pair specific parameters
    pair_params = optimization_data.get('pair_specific_params', {})
    for pair, params in pair_params.items():
        if params:
            tp = params.get('optimal_take_profit', 15.0)
            sl = params.get('optimal_stop_loss', 4.0)
            logger.info(f"Parameters for {pair}: TP={tp:.2f}%, SL={sl:.2f}%")
    
    return True

def verify_ml_config():
    """Verify ML config for optimizer integration"""
    ml_config = load_file(ML_CONFIG_FILE)
    
    if not ml_config:
        logger.error("No ML config found")
        return False
    
    # Check for trading parameters
    if 'trading_params' not in ml_config:
        logger.error("No trading_params in ML config")
        return False
    
    trading_params = ml_config.get('trading_params', {})
    
    # Check for market states
    market_states = trading_params.get('market_state', {})
    if market_states:
        for pair, state in market_states.items():
            logger.info(f"ML config market state for {pair}: {state}")
    else:
        logger.warning("No market states in ML config")
    
    # Check for optimal hours
    optimal_hours = trading_params.get('optimal_hours', {})
    if optimal_hours:
        for pair, hours in optimal_hours.items():
            entry_hours = hours.get('entry', [])
            exit_hours = hours.get('exit', [])
            logger.info(f"ML config optimal hours for {pair}: "
                        f"Entry: {len(entry_hours)}, Exit: {len(exit_hours)}")
    else:
        logger.warning("No optimal hours in ML config")
    
    # Check for pair parameters
    pair_params = trading_params.get('pair_params', {})
    if pair_params:
        for pair, params in pair_params.items():
            if params:
                tp = params.get('optimal_take_profit', 15.0)
                sl = params.get('optimal_stop_loss', 4.0)
                logger.info(f"ML config parameters for {pair}: TP={tp:.2f}%, SL={sl:.2f}%")
    else:
        logger.warning("No pair parameters in ML config")
    
    return True

def verify_positions():
    """Verify positions for optimizer integration"""
    positions = load_file(POSITIONS_FILE)
    
    if not positions:
        logger.error("No positions found")
        return False
    
    # Convert positions to dictionary if it's a list
    if isinstance(positions, list):
        positions_dict = {}
        for position in positions:
            pair = position.get('pair')
            if pair:
                positions_dict[pair] = position
        positions = positions_dict
    
    if not positions:
        logger.info("No open positions")
        return True
    
    # Get optimization data for comparison
    optimization_data = load_file(OPTIMIZATION_FILE)
    pair_params = optimization_data.get('pair_specific_params', {}) if optimization_data else {}
    
    # Check positions
    for pair, position in positions.items():
        logger.info(f"Position for {pair}:")
        logger.info(f"  Direction: {position.get('direction', 'unknown')}")
        logger.info(f"  Entry price: {position.get('entry_price', 0.0)}")
        logger.info(f"  Leverage: {position.get('leverage', 0.0)}")
        logger.info(f"  Take profit: {position.get('take_profit_pct', 0.0):.2f}%")
        logger.info(f"  Stop loss: {position.get('stop_loss_pct', 0.0):.2f}%")
        
        # Compare with optimization data
        if pair in pair_params:
            opt_tp = pair_params[pair].get('optimal_take_profit', 15.0)
            opt_sl = pair_params[pair].get('optimal_stop_loss', 4.0)
            
            tp_match = abs(position.get('take_profit_pct', 0.0) - opt_tp) < 0.01
            sl_match = abs(position.get('stop_loss_pct', 0.0) - opt_sl) < 0.01
            
            if tp_match and sl_match:
                logger.info(f"  Position parameters match optimization data")
            elif tp_match:
                logger.warning(f"  Take profit matches but stop loss doesn't")
            elif sl_match:
                logger.warning(f"  Stop loss matches but take profit doesn't")
            else:
                logger.warning(f"  Position parameters don't match optimization data")
    
    return True

def verify_portfolio():
    """Verify portfolio for optimizer integration"""
    portfolio = load_file(PORTFOLIO_FILE)
    
    if not portfolio:
        logger.error("No portfolio found")
        return False
    
    # Check portfolio
    initial_capital = portfolio.get('initial_capital', 0.0)
    current_capital = portfolio.get('current_capital', 0.0)
    
    logger.info(f"Portfolio:")
    logger.info(f"  Initial capital: ${initial_capital:.2f}")
    logger.info(f"  Current capital: ${current_capital:.2f}")
    logger.info(f"  Change: {((current_capital / initial_capital) - 1) * 100:.2f}%")
    
    # Check allocation
    allocation = portfolio.get('allocation', {})
    if allocation:
        for pair, amount in allocation.items():
            logger.info(f"  Allocation for {pair}: ${amount:.2f}")
    
    return True

def main():
    """Main function"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Verify integration
    logger.info("Verifying trade optimizer integration")
    
    # Verify optimization data
    logger.info("\n--- Verification of optimization data ---")
    if not verify_optimization_data():
        logger.error("Optimization data verification failed")
    
    # Verify ML config
    logger.info("\n--- Verification of ML config ---")
    if not verify_ml_config():
        logger.error("ML config verification failed")
    
    # Verify positions
    logger.info("\n--- Verification of positions ---")
    if not verify_positions():
        logger.error("Positions verification failed")
    
    # Verify portfolio
    logger.info("\n--- Verification of portfolio ---")
    if not verify_portfolio():
        logger.error("Portfolio verification failed")
    
    logger.info("\nVerification complete")

if __name__ == "__main__":
    main()