#!/usr/bin/env python3
"""
Model Integration with Trading System

This script integrates trained models with the trading system by:
1. Updating the ML configuration file
2. Registering new models in the system
3. Setting up trading parameters
4. Activating trading for new pairs

The integration process ensures that trained models are properly
configured for live trading with appropriate risk parameters.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("integrate_models")

# Constants
CONFIG_PATH = "config/new_coins_training_config.json"
ML_CONFIG_PATH = "config/ml_config.json"
ENSEMBLE_DIR = "ensemble_models"
BACKTEST_RESULTS_DIR = "backtest_results"
INITIAL_CAPITAL = 20000.0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Integrate trained models with trading system.')
    parser.add_argument('--pairs', type=str, required=True,
                      help='Comma-separated list of pairs to integrate')
    parser.add_argument('--models', type=str, default='ensemble',
                      help='Model type to use for trading (default: ensemble)')
    parser.add_argument('--risk-percentage', type=float, default=None,
                      help='Risk percentage per trade (default: from config)')
    parser.add_argument('--base-leverage', type=float, default=None,
                      help='Base leverage for trading (default: from config)')
    parser.add_argument('--max-leverage', type=float, default=None,
                      help='Maximum leverage for trading (default: from config)')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                      help='Confidence threshold for trading (default: from config)')
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                      help=f'Initial capital for trading (default: {INITIAL_CAPITAL})')
    parser.add_argument('--sandbox', action='store_true',
                      help='Use sandbox mode for trading')
    parser.add_argument('--force', action='store_true',
                      help='Force updates even if configuration exists')
    
    return parser.parse_args()


def load_config():
    """Load configuration from config file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def load_ml_config():
    """Load ML configuration file."""
    try:
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                ml_config = json.load(f)
            logger.info(f"Loaded ML configuration from {ML_CONFIG_PATH}")
            return ml_config
        else:
            logger.warning(f"ML configuration file not found: {ML_CONFIG_PATH}")
            # Create default configuration
            ml_config = {
                "pairs": {},
                "global_settings": {
                    "default_risk_percentage": 0.20,
                    "default_confidence_threshold": 0.65,
                    "default_base_leverage": 20.0,
                    "default_max_leverage": 125.0,
                    "enable_dynamic_leverage": True,
                    "enable_market_regime_detection": True
                }
            }
            return ml_config
    except Exception as e:
        logger.error(f"Error loading ML configuration: {e}")
        return None


def load_backtest_results(pair: str, model_type: str) -> Optional[Dict]:
    """
    Load backtest results for a pair.
    
    Args:
        pair: Trading pair
        model_type: Model type
        
    Returns:
        results: Backtest results or None if not available
    """
    pair_filename = pair.replace("/", "_")
    results_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_{model_type}_metrics.json"
    
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded backtest results for {pair} from {results_file}")
            return results
        else:
            logger.warning(f"Backtest results not found: {results_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return None


def check_model_files(pair: str, model_type: str) -> bool:
    """
    Check if model files exist for a pair.
    
    Args:
        pair: Trading pair
        model_type: Model type
        
    Returns:
        exists: Whether the model files exist
    """
    pair_filename = pair.replace("/", "_")
    
    if model_type == 'ensemble':
        config_file = f"{ENSEMBLE_DIR}/ensemble_{pair_filename}_config.json"
        if not os.path.exists(config_file):
            logger.warning(f"Ensemble configuration file not found: {config_file}")
            return False
    else:
        # Check specific model type
        model_dir = "ml_models"
        model_path = f"{model_dir}/{model_type}_{pair_filename}_model"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
    
    return True


def update_ml_config(pairs: List[str], model_type: str, args, config: Dict) -> bool:
    """
    Update ML configuration for new pairs.
    
    Args:
        pairs: List of pairs to integrate
        model_type: Model type to use
        args: Command line arguments
        config: Training configuration
        
    Returns:
        success: Whether the update succeeded
    """
    # Load ML configuration
    ml_config = load_ml_config()
    if ml_config is None:
        return False
    
    # Update configuration for each pair
    for pair in pairs:
        # Check if model files exist
        if not check_model_files(pair, model_type):
            logger.error(f"Cannot update ML config for {pair}, model files not found")
            continue
        
        # Get pair-specific settings
        pair_settings = config.get("pair_specific_settings", {}).get(pair, {})
        
        # Get risk parameters from args or config
        risk_percentage = args.risk_percentage or pair_settings.get("risk_percentage", 0.20)
        base_leverage = args.base_leverage or pair_settings.get("base_leverage", 20.0)
        max_leverage = args.max_leverage or pair_settings.get("max_leverage", 125.0)
        confidence_threshold = args.confidence_threshold or pair_settings.get("confidence_threshold", 0.65)
        
        # Load backtest results if available
        backtest_results = load_backtest_results(pair, model_type)
        
        # Set expected metrics from backtest results or defaults
        if backtest_results:
            expected_accuracy = backtest_results.get("accuracy", 0.90)
            expected_win_rate = backtest_results.get("win_rate", 0.90)
            expected_return = backtest_results.get("total_return", 2.0)
        else:
            expected_accuracy = pair_settings.get("target_accuracy", 0.90)
            expected_win_rate = pair_settings.get("expected_win_rate", 0.90)
            expected_return = pair_settings.get("target_return", 2.0)
        
        # Update or create configuration for this pair
        if pair not in ml_config["pairs"] or args.force:
            ml_config["pairs"][pair] = {
                "active": True,
                "model_type": model_type,
                "risk_percentage": risk_percentage,
                "base_leverage": base_leverage,
                "max_leverage": max_leverage,
                "confidence_threshold": confidence_threshold,
                "expected_accuracy": expected_accuracy,
                "expected_win_rate": expected_win_rate,
                "expected_return": expected_return,
                "trained": True,
                "ensemble": model_type == "ensemble",
                "sandbox_only": args.sandbox
            }
            logger.info(f"Added configuration for {pair}")
        else:
            logger.info(f"Configuration for {pair} already exists, use --force to update")
    
    # Save updated configuration
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=2)
        logger.info(f"Saved updated ML configuration to {ML_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML configuration: {e}")
        return False


def run_command(cmd: List[str], description: str = None) -> bool:
    """
    Run a shell command.
    
    Args:
        cmd: Command to run
        description: Command description
        
    Returns:
        success: Whether the command succeeded
    """
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def activate_trading_bot(pairs: List[str], args) -> bool:
    """
    Activate trading bot for specified pairs.
    
    Args:
        pairs: List of pairs to activate
        args: Command line arguments
        
    Returns:
        success: Whether the activation succeeded
    """
    pairs_arg = ",".join(pairs)
    sandbox_arg = "--sandbox" if args.sandbox else ""
    
    cmd = [
        sys.executable, "start_trading_bot.py",
        "--pairs", pairs_arg,
        "--capital", str(args.initial_capital)
    ]
    
    if sandbox_arg:
        cmd.append(sandbox_arg)
    
    return run_command(
        cmd,
        f"Activating trading bot for {pairs_arg}"
    )


def verify_trading_status(pairs: List[str], args) -> bool:
    """
    Verify trading status for pairs.
    
    Args:
        pairs: List of pairs to verify
        args: Command line arguments
        
    Returns:
        success: Whether all pairs are trading
    """
    sandbox_arg = "--sandbox" if args.sandbox else ""
    
    cmd = [
        sys.executable, "check_trading_status.py",
        "--pairs", ",".join(pairs)
    ]
    
    if sandbox_arg:
        cmd.append(sandbox_arg)
    
    return run_command(
        cmd,
        f"Verifying trading status for {', '.join(pairs)}"
    )


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse pairs
    pairs = args.pairs.split(',')
    
    logger.info(f"Integrating models for pairs: {', '.join(pairs)}")
    logger.info(f"Model type: {args.models}")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return 1
    
    # Update ML configuration
    success = update_ml_config(pairs, args.models, args, config)
    if not success:
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # Activate trading bot
    if args.sandbox:
        logger.info("Activating trading bot in sandbox mode")
    else:
        logger.info("Activating trading bot in live mode")
    
    success = activate_trading_bot(pairs, args)
    if not success:
        logger.error("Failed to activate trading bot. Exiting.")
        return 1
    
    # Verify trading status
    success = verify_trading_status(pairs, args)
    if not success:
        logger.warning("Some pairs may not be trading correctly.")
    
    logger.info("Model integration completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())