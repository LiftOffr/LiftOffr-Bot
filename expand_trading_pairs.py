#!/usr/bin/env python3
"""
Expand Trading Pairs Script

This script expands the bot's trading capabilities to additional cryptocurrency pairs:
1. Downloads historical data for new pairs
2. Prepares datasets for ML training
3. Trains models for each new pair
4. Creates ensemble models for improved accuracy
5. Updates ML configuration to include new pairs
6. Activates trading for all pairs

Usage:
    python expand_trading_pairs.py [--pairs DOT/USD,ADA/USD,LINK/USD] [--sandbox]
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
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ['DOT/USD', 'ADA/USD', 'LINK/USD']  # Additional pairs beyond SOL/USD, BTC/USD, ETH/USD
DEFAULT_CAPITAL = 20000.0
DEFAULT_TIMEFRAME = '1h'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Expand trading to additional pairs')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs to add (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=DEFAULT_CAPITAL,
        help=f'Starting capital for trading (default: {DEFAULT_CAPITAL})'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f'Timeframe for historical data (default: {DEFAULT_TIMEFRAME})'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data to fetch (default: 365)'
    )
    
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching historical data'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training models'
    )
    
    # Add both sandbox and live mode options
    trading_mode = parser.add_mutually_exclusive_group()
    trading_mode.add_argument(
        '--sandbox',
        action='store_true',
        help='Use sandbox trading mode (default)'
    )
    trading_mode.add_argument(
        '--live',
        action='store_true',
        help='Use live trading mode instead of sandbox'
    )
    
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """
    Run a shell command and log output.
    
    Args:
        cmd: List of command and arguments
        description: Description of the command
        
    Returns:
        CompletedProcess if successful, None if failed
    """
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log stdout if available
        if process.stdout:
            for line in process.stdout.strip().split('\n'):
                if line:
                    logger.info(f"Output: {line}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        
        # Log stderr if available
        if e.stderr:
            for line in e.stderr.strip().split('\n'):
                if line:
                    logger.error(f"Error: {line}")
        
        return None

def fetch_historical_data(pairs: List[str], timeframe: str = '1h', days: int = 365) -> bool:
    """
    Fetch historical data for the specified pairs.
    
    Args:
        pairs: List of trading pairs
        timeframe: Timeframe for historical data
        days: Number of days of historical data to fetch
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Fetching historical data for {len(pairs)} pairs...")
    
    # Check if fetch_extended_historical_data.py exists
    if os.path.exists("fetch_extended_historical_data.py"):
        cmd = [
            "python",
            "fetch_extended_historical_data.py",
            "--pairs", ",".join(pairs),
            "--timeframe", timeframe,
            "--days", str(days)
        ]
        
        result = run_command(cmd, f"Fetching {days} days of historical data for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to other scripts if fetch_extended_historical_data.py doesn't exist
    for pair in pairs:
        # Try historical_data_fetcher.py
        if os.path.exists("historical_data_fetcher.py"):
            cmd = [
                "python",
                "historical_data_fetcher.py",
                "--pair", pair,
                "--timeframe", timeframe,
                "--days", str(days)
            ]
            
            result = run_command(cmd, f"Fetching historical data for {pair}")
            if result is None:
                logger.error(f"Failed to fetch historical data for {pair}")
                return False
        
        # Try fetch_1d_data.py (specific for 1-day timeframe)
        elif os.path.exists("fetch_1d_data.py") and timeframe.lower() in ['1d', 'daily', 'd']:
            cmd = [
                "python",
                "fetch_1d_data.py",
                "--pair", pair
            ]
            
            result = run_command(cmd, f"Fetching daily historical data for {pair}")
            if result is None:
                logger.error(f"Failed to fetch historical data for {pair}")
                return False
        
        # No suitable script found
        else:
            logger.error("No suitable script found for fetching historical data")
            return False
    
    return True

def prepare_datasets(pairs: List[str], timeframe: str = '1h') -> bool:
    """
    Prepare datasets for ML training.
    
    Args:
        pairs: List of trading pairs
        timeframe: Timeframe for historical data
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Preparing datasets for {len(pairs)} pairs...")
    
    # Check if prepare_all_datasets.py exists
    if os.path.exists("prepare_all_datasets.py"):
        cmd = [
            "python",
            "prepare_all_datasets.py",
            "--pairs", ",".join(pairs),
            "--timeframe", timeframe
        ]
        
        result = run_command(cmd, f"Preparing datasets for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to prepare_enhanced_dataset.py
    elif os.path.exists("prepare_enhanced_dataset.py"):
        # Prepare datasets for each pair individually
        for pair in pairs:
            cmd = [
                "python",
                "prepare_enhanced_dataset.py",
                "--pair", pair,
                "--timeframe", timeframe
            ]
            
            result = run_command(cmd, f"Preparing enhanced dataset for {pair}")
            if result is None:
                logger.error(f"Failed to prepare dataset for {pair}")
                return False
        
        return True
    
    # No suitable script found
    else:
        logger.error("No suitable script found for preparing datasets")
        return False

def train_models(pairs: List[str]) -> bool:
    """
    Train models for each pair.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Training models for {len(pairs)} pairs...")
    
    # Check if enhance_ml_model_accuracy.py exists (the script we just created)
    if os.path.exists("enhance_ml_model_accuracy.py"):
        cmd = [
            "python",
            "enhance_ml_model_accuracy.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, f"Training enhanced models for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to comprehensive_ml_training.py
    elif os.path.exists("comprehensive_ml_training.py"):
        cmd = [
            "python",
            "comprehensive_ml_training.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, f"Comprehensive ML training for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to enhanced_dual_strategy_trainer.py
    elif os.path.exists("enhanced_dual_strategy_trainer.py"):
        cmd = [
            "python",
            "enhanced_dual_strategy_trainer.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, f"Enhanced dual strategy training for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # No suitable script found
    else:
        logger.error("No suitable script found for training models")
        return False

def create_ensemble_models(pairs: List[str]) -> bool:
    """
    Create ensemble models for each pair.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating ensemble models for {len(pairs)} pairs...")
    
    # Check if create_ensemble_model.py exists
    if os.path.exists("create_ensemble_model.py"):
        # Create ensemble models for each pair individually
        for pair in pairs:
            cmd = [
                "python",
                "create_ensemble_model.py",
                "--pair", pair
            ]
            
            result = run_command(cmd, f"Creating ensemble model for {pair}")
            if result is None:
                # Try alternative argument format
                cmd = [
                    "python",
                    "create_ensemble_model.py",
                    "--trading-pair", pair
                ]
                
                result = run_command(cmd, f"Creating ensemble model for {pair} (alternative format)")
                if result is None:
                    logger.error(f"Failed to create ensemble model for {pair}")
                    return False
        
        return True
    
    # No suitable script found
    else:
        logger.error("create_ensemble_model.py not found")
        return False

def update_ml_config(all_pairs: List[str]) -> bool:
    """
    Update ML configuration to include new pairs.
    
    Args:
        all_pairs: List of all trading pairs (existing + new)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Updating ML configuration for {len(all_pairs)} pairs...")
    
    # Check if update_ml_config.py exists
    if os.path.exists("update_ml_config.py"):
        cmd = [
            "python",
            "update_ml_config.py",
            "--pairs", ",".join(all_pairs),
            "--update-strategy",
            "--update-ensemble"
        ]
        
        result = run_command(cmd, f"Updating ML configuration for {len(all_pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to manual configuration update
    elif os.path.exists("ml_config.json"):
        try:
            # Load existing configuration
            with open("ml_config.json", "r") as f:
                config = json.load(f)
            
            # Update configuration
            config["trading_pairs"] = all_pairs
            
            # Ensure all pairs are in the pair_configs section
            if "pair_configs" not in config:
                config["pair_configs"] = {}
            
            for pair in all_pairs:
                pair_key = pair.replace("/", "")
                if pair_key not in config["pair_configs"]:
                    config["pair_configs"][pair_key] = {
                        "confidence_threshold": 0.65,
                        "base_leverage": 20.0,
                        "max_leverage": 125.0,
                        "max_risk": 0.2
                    }
            
            # Save updated configuration
            with open("ml_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"ML configuration updated for {len(all_pairs)} pairs")
            return True
        
        except Exception as e:
            logger.error(f"Error updating ML configuration: {str(e)}")
            return False
    
    # No suitable configuration file found
    else:
        logger.error("ML configuration file not found")
        return False

def activate_trading(all_pairs: List[str], capital: float, sandbox: bool = True) -> bool:
    """
    Activate trading for all pairs.
    
    Args:
        all_pairs: List of all trading pairs
        capital: Starting capital
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating trading for {len(all_pairs)} pairs...")
    
    # Check if quick_activate_ml.py exists
    if os.path.exists("quick_activate_ml.py"):
        cmd = [
            "python",
            "quick_activate_ml.py",
            "--pairs", ",".join(all_pairs),
            "--capital", str(capital)
        ]
        
        if not sandbox:
            cmd.append("--live")
        
        result = run_command(cmd, f"Activating ML trading for {len(all_pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to activate_ml_with_ensembles.py
    elif os.path.exists("activate_ml_with_ensembles.py"):
        cmd = [
            "python",
            "activate_ml_with_ensembles.py",
            "--pairs", ",".join(all_pairs)
        ]
        
        if not sandbox:
            cmd.append("--live")
        
        result = run_command(cmd, f"Activating ML trading with ensembles for {len(all_pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to integrated_strategy.py
    elif os.path.exists("integrated_strategy.py"):
        cmd = [
            "python",
            "integrated_strategy.py",
            "--pairs", ",".join(all_pairs),
            "--capital", str(capital),
            "--strategy", "ml",
            "--multi-strategy"
        ]
        
        if not sandbox:
            cmd.append("--live")
        else:
            cmd.append("--sandbox")
        
        result = run_command(cmd, f"Starting ML trading with integrated strategy for {len(all_pairs)} pairs")
        if result is not None:
            return True
    
    # No suitable script found
    else:
        logger.error("No suitable script found for activating trading")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse new pairs list
    new_pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Existing pairs (we assume these are already set up)
    existing_pairs = ['SOL/USD', 'BTC/USD', 'ETH/USD']
    
    # All pairs (existing + new)
    all_pairs = existing_pairs + [p for p in new_pairs if p not in existing_pairs]
    
    # Determine if in sandbox or live mode
    use_sandbox = not args.live  # Default to sandbox unless live is specified
    
    logger.info("=" * 80)
    logger.info("EXPAND TRADING PAIRS")
    logger.info("=" * 80)
    logger.info(f"Existing pairs: {', '.join(existing_pairs)}")
    logger.info(f"New pairs to add: {', '.join(new_pairs)}")
    logger.info(f"Total pairs after expansion: {', '.join(all_pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"Historical data timeframe: {args.timeframe}")
    logger.info(f"Days of historical data: {args.days}")
    logger.info(f"Trading mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info(f"Skip fetch: {args.skip_fetch}")
    logger.info(f"Skip train: {args.skip_train}")
    logger.info("=" * 80)
    
    # Confirmation for live mode
    if args.live:
        confirm = input("WARNING: You are about to expand trading to new pairs in LIVE mode. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live mode expansion not confirmed. Exiting.")
            return 1
    
    # Step 1: Fetch historical data for new pairs
    if not args.skip_fetch:
        logger.info("Step 1: Fetching historical data...")
        if not fetch_historical_data(new_pairs, args.timeframe, args.days):
            logger.error("Failed to fetch historical data. Exiting.")
            return 1
    else:
        logger.info("Step 1: Skipping historical data fetching as requested")
    
    # Step 2: Prepare datasets for ML training
    if not args.skip_fetch:
        logger.info("Step 2: Preparing datasets...")
        if not prepare_datasets(new_pairs, args.timeframe):
            logger.error("Failed to prepare datasets. Exiting.")
            return 1
    else:
        logger.info("Step 2: Skipping dataset preparation as requested")
    
    # Step 3: Train models for new pairs
    if not args.skip_train:
        logger.info("Step 3: Training models...")
        if not train_models(new_pairs):
            logger.error("Failed to train models. Continuing anyway...")
    else:
        logger.info("Step 3: Skipping model training as requested")
    
    # Step 4: Create ensemble models for new pairs
    if not args.skip_train:
        logger.info("Step 4: Creating ensemble models...")
        if not create_ensemble_models(new_pairs):
            logger.error("Failed to create ensemble models. Continuing anyway...")
    else:
        logger.info("Step 4: Skipping ensemble model creation as requested")
    
    # Step 5: Update ML configuration
    logger.info("Step 5: Updating ML configuration...")
    if not update_ml_config(all_pairs):
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # Step 6: Activate trading for all pairs
    logger.info("Step 6: Activating trading...")
    if not activate_trading(all_pairs, args.capital, use_sandbox):
        logger.error("Failed to activate trading. Exiting.")
        return 1
    
    logger.info("=" * 80)
    logger.info("TRADING EXPANSION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Now trading on {len(all_pairs)} pairs: {', '.join(all_pairs)}")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())