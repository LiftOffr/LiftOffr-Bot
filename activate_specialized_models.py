#!/usr/bin/env python3
"""
Activate Specialized Models for Trading

This script configures the trading bot to use the trained specialized models.
It sets up the appropriate configuration for the ensemble models that combine
entry timing, exit timing, position sizing, and cancellation models.

Usage:
    python activate_specialized_models.py [--pair PAIR] [--timeframe TIMEFRAME] [--sandbox]

Example:
    python activate_specialized_models.py --pair SOL/USD --timeframe 1h --sandbox
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('activate_models.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MODEL_TYPES = ['entry', 'exit', 'cancel', 'sizing', 'ensemble']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Activate specialized models for trading')
    parser.add_argument('--pair', type=str, choices=SUPPORTED_PAIRS,
                      help='Trading pair to activate (e.g., SOL/USD). If not specified, activates all available pairs.')
    parser.add_argument('--timeframe', type=str, choices=TIMEFRAMES,
                      help='Timeframe to activate (e.g., 1h). If not specified, chooses the best timeframe for each pair.')
    parser.add_argument('--sandbox', action='store_true',
                      help='Run in sandbox mode')
    parser.add_argument('--force', action='store_true',
                      help='Force model activation even if some models are missing')
    return parser.parse_args()


def run_command(command: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """
    Run a shell command and log output
    
    Args:
        command: Command to run
        description: Description of the command
        
    Returns:
        CompletedProcess if successful, None otherwise
    """
    if description:
        logging.info(f"{description}")
    
    logging.info(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=True
        )
        logging.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(e.stdout)
        return None
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return None


def check_model_files(pair: str, timeframe: str) -> bool:
    """
    Check if model files exist for a specific pair and timeframe
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 1h)
        
    Returns:
        True if all required models exist, False otherwise
    """
    pair_formatted = pair.replace('/', '_')
    required_models = ['entry', 'exit', 'cancel', 'sizing']
    
    # Check for each required model
    for model_type in required_models:
        model_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_model.h5"
        info_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_info.json"
        
        if not os.path.exists(model_path) or not os.path.exists(info_path):
            logging.warning(f"Missing {model_type} model for {pair} ({timeframe})")
            return False
    
    # Check for ensemble info
    ensemble_info_path = f"ml_models/{pair_formatted}_{timeframe}_ensemble_info.json"
    if not os.path.exists(ensemble_info_path):
        logging.warning(f"Missing ensemble info for {pair} ({timeframe})")
        return False
    
    logging.info(f"All models found for {pair} ({timeframe})")
    return True


def find_best_timeframe(pair: str) -> Optional[str]:
    """
    Find the best timeframe for a pair based on performance metrics
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        
    Returns:
        Best timeframe or None if no models available
    """
    pair_formatted = pair.replace('/', '_')
    best_score = -1
    best_timeframe = None
    
    for timeframe in TIMEFRAMES:
        ensemble_info_path = f"ml_models/{pair_formatted}_{timeframe}_ensemble_info.json"
        
        if not os.path.exists(ensemble_info_path):
            continue
        
        try:
            with open(ensemble_info_path, 'r') as f:
                info = json.load(f)
            
            if 'metrics' not in info:
                continue
            
            metrics = info['metrics']
            
            # Extract key metrics
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            
            # Calculate combined score (custom formula)
            combined_score = win_rate * 0.4 + min(profit_factor / 5, 1) * 0.4 + min(sharpe_ratio / 3, 1) * 0.2
            
            if combined_score > best_score:
                best_score = combined_score
                best_timeframe = timeframe
        except Exception as e:
            logging.error(f"Error processing {ensemble_info_path}: {e}")
    
    if best_timeframe:
        logging.info(f"Best timeframe for {pair}: {best_timeframe} (score: {best_score:.2f})")
    else:
        logging.warning(f"No valid models found for {pair}")
    
    return best_timeframe


def get_available_pairs() -> List[str]:
    """
    Get list of pairs with available models
    
    Returns:
        List of trading pairs
    """
    available_pairs = []
    
    for pair in SUPPORTED_PAIRS:
        pair_formatted = pair.replace('/', '_')
        has_models = False
        
        for timeframe in TIMEFRAMES:
            ensemble_info_path = f"ml_models/{pair_formatted}_{timeframe}_ensemble_info.json"
            if os.path.exists(ensemble_info_path):
                has_models = True
                break
        
        if has_models:
            available_pairs.append(pair)
    
    return available_pairs


def update_ml_config(
    pairs: List[Tuple[str, str]],
    sandbox: bool = True,
    force: bool = False
) -> bool:
    """
    Update ML configuration for the trading bot
    
    Args:
        pairs: List of (pair, timeframe) tuples
        sandbox: Whether to run in sandbox mode
        force: Whether to force update even if some models are missing
        
    Returns:
        True if successful, False otherwise
    """
    if not pairs:
        logging.error("No pairs specified for ML configuration")
        return False
    
    # Check models
    valid_pairs = []
    for pair, timeframe in pairs:
        if check_model_files(pair, timeframe) or force:
            valid_pairs.append((pair, timeframe))
    
    if not valid_pairs:
        logging.error("No valid pairs with complete models")
        return False
    
    # Create ML config
    ml_config = {
        "enabled": True,
        "sandbox": sandbox,
        "models": {}
    }
    
    for pair, timeframe in valid_pairs:
        pair_formatted = pair.replace('/', '_')
        
        # Add model configuration
        ml_config["models"][pair] = {
            "enabled": True,
            "timeframe": timeframe,
            "model_path": f"ml_models/{pair_formatted}_{timeframe}",
            "use_ensemble": True,
            "specialized_models": {
                "entry": {
                    "enabled": True,
                    "model_name": f"{pair_formatted}_{timeframe}_entry_model.h5",
                    "threshold": 0.6  # Adjust threshold for precision
                },
                "exit": {
                    "enabled": True,
                    "model_name": f"{pair_formatted}_{timeframe}_exit_model.h5",
                    "threshold": 0.5  # Lower threshold for better recall
                },
                "cancel": {
                    "enabled": True,
                    "model_name": f"{pair_formatted}_{timeframe}_cancel_model.h5",
                    "threshold": 0.7  # Higher threshold to avoid unnecessary cancellations
                },
                "sizing": {
                    "enabled": True,
                    "model_name": f"{pair_formatted}_{timeframe}_sizing_model.h5",
                    "min_size": 0.25,  # Minimum position size (fraction of max)
                    "max_size": 1.0    # Maximum position size
                }
            },
            "risk_management": {
                "max_position_size": 0.2,       # Max 20% of portfolio per position
                "max_positions": 5,             # Max 5 open positions
                "confidence_scaling": True,     # Scale position size based on confidence
                "max_leverage": 75,             # Maximum leverage for highest confidence trades
                "min_leverage": 1,              # Minimum leverage for lowest confidence trades
                "confidence_thresholds": {
                    "low": 0.6,                 # Minimum threshold to consider a signal
                    "medium": 0.75,
                    "high": 0.85,
                    "very_high": 0.95
                },
                "leverage_tiers": {
                    "low": 1,                   # 1x leverage for low confidence
                    "medium": 3,                # 3x leverage for medium confidence
                    "high": 15,                 # 15x leverage for high confidence
                    "very_high": 75             # 75x leverage for very high confidence
                }
            }
        }
    
    # Write config file
    config_path = "config/ml_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(ml_config, f, indent=2)
        logging.info(f"ML configuration written to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error writing ML configuration: {e}")
        return False


def reset_sandbox_portfolio(starting_capital: float = 20000.0) -> bool:
    """
    Reset sandbox portfolio to starting capital
    
    Args:
        starting_capital: Starting capital amount
        
    Returns:
        True if successful, False otherwise
    """
    config_path = "config/sandbox_portfolio.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    portfolio = {
        "base_currency": "USD",
        "starting_capital": starting_capital,
        "current_capital": starting_capital,
        "equity": starting_capital,
        "positions": {},
        "completed_trades": [],
        "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(portfolio, f, indent=2)
        logging.info(f"Sandbox portfolio reset to {starting_capital} USD")
        return True
    except Exception as e:
        logging.error(f"Error resetting sandbox portfolio: {e}")
        return False


def restart_trading_bot(sandbox: bool = True) -> bool:
    """
    Restart the trading bot
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    # Update trading mode in config
    config_path = "config/trading_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        # Check if config exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update mode
        config["sandbox_mode"] = sandbox
        config["use_ml_models"] = True
        
        # Write config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.error(f"Error updating trading config: {e}")
        return False
    
    # Create restart trigger file
    try:
        with open(".trading_bot_restart_trigger", "w") as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        logging.info("Created restart trigger for trading bot")
    except Exception as e:
        logging.error(f"Error creating restart trigger: {e}")
        return False
    
    # Use the bot manager to restart the trading bot
    try:
        result = run_command(
            ["python", "restart_trading_bot.py", "--sandbox" if sandbox else ""],
            "Restarting trading bot"
        )
        return result is not None
    except Exception as e:
        logging.error(f"Error restarting trading bot: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    # Determine which pairs to activate
    if args.pair:
        pairs_to_activate = [args.pair]
    else:
        pairs_to_activate = get_available_pairs()
    
    if not pairs_to_activate:
        logging.error("No pairs with available models found")
        return False
    
    # Prepare pair-timeframe tuples
    pair_timeframes = []
    
    for pair in pairs_to_activate:
        if args.timeframe:
            # Use specified timeframe
            timeframe = args.timeframe
        else:
            # Find best timeframe
            timeframe = find_best_timeframe(pair)
        
        if timeframe:
            pair_timeframes.append((pair, timeframe))
    
    if not pair_timeframes:
        logging.error("No valid pair-timeframe combinations found")
        return False
    
    # Reset sandbox portfolio if in sandbox mode
    if args.sandbox:
        if not reset_sandbox_portfolio(20000.0):
            logging.warning("Failed to reset sandbox portfolio")
    
    # Update ML configuration
    if not update_ml_config(pair_timeframes, args.sandbox, args.force):
        logging.error("Failed to update ML configuration")
        return False
    
    # Restart trading bot
    if not restart_trading_bot(args.sandbox):
        logging.error("Failed to restart trading bot")
        return False
    
    # Print activated models
    logging.info("=== Activated Models ===")
    for pair, timeframe in pair_timeframes:
        logging.info(f"{pair} ({timeframe})")
    
    logging.info(f"Trading bot started in {'sandbox' if args.sandbox else 'live'} mode")
    return True


if __name__ == "__main__":
    main()