#!/usr/bin/env python3
"""
Activate Multi-Timeframe Models for Trading

This script activates the trained multi-timeframe models for trading:
1. Verifies that all required models exist for each trading pair
2. Updates the ML configuration to use the ensemble models
3. Starts the trading system in sandbox mode

Usage:
    python activate_multi_timeframe_models.py [--pairs PAIR1,PAIR2,...] [--sandbox]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('activate_multi_timeframe.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
MODEL_TYPES = ['15m', '1h', '4h', '1d', 'unified', 'ensemble']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Activate multi-timeframe models for trading')
    parser.add_argument('--pairs', type=str, default=None,
                        help='Comma-separated list of pairs to activate (e.g., BTC/USD,ETH/USD)')
    parser.add_argument('--sandbox', action='store_true', default=True,
                        help='Run in sandbox mode (default: True)')
    return parser.parse_args()


def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """
    Run a shell command and log output
    
    Args:
        cmd: Command to run
        description: Description of the command
        
    Returns:
        CompletedProcess if successful, None otherwise
    """
    if description:
        logging.info(f"{description}")
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Log the output
        for line in result.stdout.splitlines():
            logging.info(line)
        
        return result
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Command output:")
        
        for line in e.output.splitlines():
            logging.error(line)
        
        return None


def check_model_files(pairs: List[str]) -> bool:
    """
    Check if model files exist for all pairs
    
    Args:
        pairs: List of trading pairs to check
        
    Returns:
        True if all models exist, False otherwise
    """
    logging.info("Checking model files...")
    
    all_models_exist = True
    
    for pair in pairs:
        pair_symbol = pair.replace('/', '_')
        
        # Check ensemble model (required)
        ensemble_model_path = f"ml_models/{pair_symbol}_ensemble.h5"
        ensemble_info_path = f"ml_models/{pair_symbol}_ensemble_info.json"
        
        if not os.path.exists(ensemble_model_path) or not os.path.exists(ensemble_info_path):
            logging.error(f"Missing ensemble model for {pair}")
            all_models_exist = False
            continue
        
        # Check individual timeframe models
        for model_type in MODEL_TYPES[:-1]:  # Exclude 'ensemble'
            model_path = f"ml_models/{pair_symbol}_{model_type}.h5"
            info_path = f"ml_models/{pair_symbol}_{model_type}_info.json"
            
            if not os.path.exists(model_path) or not os.path.exists(info_path):
                logging.warning(f"Missing {model_type} model for {pair}")
    
    return all_models_exist


def update_ml_config(pairs: List[str]) -> bool:
    """
    Update ML configuration to use ensemble models
    
    Args:
        pairs: List of trading pairs to configure
        
    Returns:
        True if successful, False otherwise
    """
    logging.info("Updating ML configuration...")
    
    config_path = "config/ml_config.json"
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Load existing config if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load existing ML config: {e}")
            config = {}
    else:
        config = {}
    
    # Initialize or update sections
    if 'models' not in config:
        config['models'] = {}
    
    if 'enabled_pairs' not in config:
        config['enabled_pairs'] = []
    
    if 'global_settings' not in config:
        config['global_settings'] = {}
    
    # Update global settings
    config['global_settings'].update({
        'use_ml_predictions': True,
        'confidence_threshold': 0.65,
        'max_trades_per_pair': 1,
        'max_portfolio_allocation': 0.25,
        'dynamic_position_sizing': True,
        'use_ensemble_models': True
    })
    
    # Update enabled pairs
    config['enabled_pairs'] = pairs
    
    # Update model configurations
    for pair in pairs:
        pair_symbol = pair.replace('/', '_')
        
        # Load ensemble model info
        ensemble_info_path = f"ml_models/{pair_symbol}_ensemble_info.json"
        try:
            with open(ensemble_info_path, 'r') as f:
                ensemble_info = json.load(f)
            
            # Get directional accuracy
            directional_accuracy = ensemble_info.get('metrics', {}).get('directional_accuracy', 0.5)
            
            # Calculate leverage based on model accuracy
            # Higher accuracy models get higher leverage
            if directional_accuracy >= 0.8:
                max_leverage = 25.0
            elif directional_accuracy >= 0.75:
                max_leverage = 10.0
            elif directional_accuracy >= 0.7:
                max_leverage = 5.0
            else:
                max_leverage = 3.0
            
            # Configure model settings
            config['models'][pair] = {
                'model_type': 'ensemble',
                'model_path': f"ml_models/{pair_symbol}_ensemble.h5",
                'prediction_horizon': 1,  # Hours
                'directional_accuracy': directional_accuracy,
                'max_leverage': max_leverage,
                'confidence_scaling': True,
                'component_models': [
                    {'timeframe': '15m', 'path': f"ml_models/{pair_symbol}_15m.h5", 'weight': 0.3},
                    {'timeframe': '1h', 'path': f"ml_models/{pair_symbol}_1h.h5", 'weight': 0.2},
                    {'timeframe': '4h', 'path': f"ml_models/{pair_symbol}_4h.h5", 'weight': 0.2},
                    {'timeframe': '1d', 'path': f"ml_models/{pair_symbol}_1d.h5", 'weight': 0.1},
                    {'timeframe': 'unified', 'path': f"ml_models/{pair_symbol}_unified.h5", 'weight': 0.2}
                ]
            }
            
            logging.info(f"Configured {pair} with directional accuracy: {directional_accuracy:.2f}, max leverage: {max_leverage}")
            
        except Exception as e:
            logging.error(f"Failed to load ensemble info for {pair}: {e}")
            return False
    
    # Save updated config
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Updated ML config saved to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save ML config: {e}")
        return False


def reset_sandbox_portfolio(starting_capital=20000.0) -> bool:
    """
    Reset sandbox portfolio to starting capital
    
    Args:
        starting_capital: Starting capital amount
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Resetting sandbox portfolio to ${starting_capital:.2f}...")
    
    # Run reset script
    result = run_command(
        ['python', 'reset_sandbox_portfolio.py', f'--balance={starting_capital}'],
        "Resetting sandbox portfolio"
    )
    
    return result is not None


def restart_trading_bot(sandbox=True) -> bool:
    """
    Restart the trading bot
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Restarting trading bot (sandbox={sandbox})...")
    
    # Update restart trigger file
    try:
        with open('.trading_bot_restart_trigger', 'w') as f:
            f.write(f"{time.time()}")
    except Exception as e:
        logging.error(f"Failed to update restart trigger: {e}")
    
    # Start trading bot
    cmd = ['python', 'main.py']
    if sandbox:
        cmd.append('--sandbox')
    
    result = run_command(cmd, "Starting trading bot")
    
    return result is not None


def main():
    """Main function"""
    args = parse_arguments()
    
    # Determine which pairs to process
    if args.pairs is not None:
        pairs_to_process = args.pairs.split(',')
        # Validate pairs
        for pair in pairs_to_process:
            if pair not in SUPPORTED_PAIRS:
                logging.warning(f"Unsupported pair: {pair}")
                pairs_to_process.remove(pair)
    else:
        pairs_to_process = SUPPORTED_PAIRS.copy()
    
    # Check model files
    if not check_model_files(pairs_to_process):
        logging.error("Some models are missing. Please train them before activating.")
        return False
    
    # Update ML config
    if not update_ml_config(pairs_to_process):
        logging.error("Failed to update ML config")
        return False
    
    # Reset sandbox portfolio if in sandbox mode
    if args.sandbox:
        if not reset_sandbox_portfolio():
            logging.warning("Failed to reset sandbox portfolio")
    
    # Restart trading bot
    if not restart_trading_bot(args.sandbox):
        logging.error("Failed to restart trading bot")
        return False
    
    logging.info(f"Multi-timeframe models activated for {len(pairs_to_process)} pairs")
    return True


if __name__ == "__main__":
    main()