#!/usr/bin/env python3
"""
Dual Strategy ML Training Runner

This script runs the enhanced training process for all trading pairs,
specifically integrating ARIMA and Adaptive strategies with advanced ML models
to achieve 90% win rate and 1000% returns.

The script:
1. Updates the ML configuration for hyper-performance
2. Prepares training data for all supported pairs
3. Launches enhanced training for each pair
4. Orchestrates the pruning and optimization process
5. Applies the trained models to the live trading environment

Usage:
    python run_dual_strategy_ml_training.py [--pairs PAIRS] [--max-leverage MAX_LEVERAGE]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dual_strategy_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "ml_enhanced_config.json"
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
DEFAULT_MAX_LEVERAGE = 125
DEFAULT_TARGET_WIN_RATE = 0.9
DEFAULT_TARGET_RETURN = 1000.0
DATA_DIR = "training_data"
MODELS_DIR = "models"
HISTORICAL_DATA_DIR = "historical_data"

def run_command(command: str, description: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and log output
    
    Args:
        command: Shell command to run
        description: Optional description of the command
        check: Whether to check the return code
        
    Returns:
        CompletedProcess object
    """
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                logger.info(f"[STDOUT] {line}")
        
        if result.stderr.strip():
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"[STDERR] {line}")
        
        if description and result.returncode == 0:
            logger.info(f"{description} completed successfully")
        elif description:
            logger.error(f"{description} failed with code {result.returncode}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}: {e}")
        if check:
            raise
        return e
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        if check:
            raise
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=str(e))

def load_config() -> Dict[str, Any]:
    """
    Load ML configuration from file
    
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {CONFIG_PATH}")
            return config
        else:
            logger.warning(f"Configuration file {CONFIG_PATH} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def ensure_directories() -> None:
    """
    Ensure all required directories exist
    """
    directories = [DATA_DIR, MODELS_DIR, HISTORICAL_DATA_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def prepare_training_data(trading_pairs: List[str]) -> None:
    """
    Prepare training data for all trading pairs
    
    Args:
        trading_pairs: List of trading pairs
    """
    logger.info("Preparing training data for all pairs...")
    
    for pair in trading_pairs:
        pair_filename = pair.replace("/", "")
        
        # Check if we have historical data for this pair
        pair_data_path = os.path.join(HISTORICAL_DATA_DIR, f"{pair_filename}_1h.csv")
        if not os.path.exists(pair_data_path):
            logger.warning(f"Historical data not found for {pair}, fetching now...")
            fetch_command = f"python enhanced_historical_data_fetcher.py --pair {pair} --timeframe 1h"
            run_command(fetch_command, f"Fetching historical data for {pair}")
        
        # Prepare enhanced dataset with both ARIMA and Adaptive features
        prepare_command = f"python prepare_enhanced_dataset.py --pair {pair} --timeframe 1h"
        run_command(prepare_command, f"Preparing enhanced dataset for {pair}", check=False)
    
    logger.info("Finished preparing training data for all pairs")

def update_ml_config(max_leverage: int) -> None:
    """
    Update ML configuration for hyperperformance
    
    Args:
        max_leverage: Maximum leverage setting
    """
    logger.info(f"Updating ML configuration for hyperperformance (max leverage: {max_leverage})...")
    
    update_command = f"python update_ml_config_for_hyperperformance.py --max-leverage {max_leverage}"
    run_command(update_command, "Updating ML configuration")

def train_integrated_models(trading_pairs: List[str], target_win_rate: float, target_return: float, epochs: int = 300) -> None:
    """
    Train integrated models for all trading pairs
    
    Args:
        trading_pairs: List of trading pairs
        target_win_rate: Target win rate (0.0-1.0)
        target_return: Target return percentage (e.g., 1000.0 for 1000%)
        epochs: Number of training epochs
    """
    logger.info(f"Training integrated models for {len(trading_pairs)} pairs...")
    
    # Convert pairs list to command-line argument
    pairs_arg = " ".join([f"--pairs {pair}" for pair in trading_pairs])
    
    # Run the enhanced strategy training
    train_command = f"python enhanced_strategy_training.py {pairs_arg} --epochs {epochs} --target-win-rate {target_win_rate} --target-return {target_return}"
    run_command(train_command, "Training integrated models")

def deploy_models_to_live_trading(trading_pairs: List[str], sandbox: bool = True) -> None:
    """
    Deploy trained models to live trading
    
    Args:
        trading_pairs: List of trading pairs
        sandbox: Whether to use sandbox mode
    """
    sandbox_arg = "--sandbox" if sandbox else "--live"
    
    logger.info(f"Deploying models to {'sandbox' if sandbox else 'live'} trading...")
    
    # Restart trading with new models
    deploy_command = f"python run_optimized_ml_trading.py --reset {sandbox_arg} --pairs {' '.join(trading_pairs)}"
    run_command(deploy_command, "Deploying models to trading environment")

def run_model_auto_pruning(trading_pairs: List[str]) -> None:
    """
    Run automatic model pruning to remove underperforming models
    
    Args:
        trading_pairs: List of trading pairs
    """
    logger.info("Running automatic model pruning...")
    
    # First, analyze model performance
    analyze_command = "python analyze_model_performance.py"
    run_command(analyze_command, "Analyzing model performance")
    
    # Then prune underperforming models
    prune_command = "python auto_prune_ml_models.py --performance-threshold 0.6 --min-samples 10"
    run_command(prune_command, "Pruning underperforming models")

def optimize_model_hyperparameters(trading_pairs: List[str]) -> None:
    """
    Optimize model hyperparameters for all trading pairs
    
    Args:
        trading_pairs: List of trading pairs
    """
    logger.info("Optimizing model hyperparameters...")
    
    for pair in trading_pairs:
        optimize_command = f"python adaptive_hyperparameter_tuning.py --pair {pair} --max-trials 50"
        run_command(optimize_command, f"Optimizing hyperparameters for {pair}", check=False)

def main():
    """Main function to run the dual strategy ML training"""
    parser = argparse.ArgumentParser(description="Dual Strategy ML Training Runner")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_TRADING_PAIRS,
                        help=f"Trading pairs to train on (default: {' '.join(DEFAULT_TRADING_PAIRS)})")
    parser.add_argument("--max-leverage", type=int, default=DEFAULT_MAX_LEVERAGE,
                        help=f"Maximum leverage setting (default: {DEFAULT_MAX_LEVERAGE})")
    parser.add_argument("--target-win-rate", type=float, default=DEFAULT_TARGET_WIN_RATE,
                        help=f"Target win rate (default: {DEFAULT_TARGET_WIN_RATE})")
    parser.add_argument("--target-return", type=float, default=DEFAULT_TARGET_RETURN,
                        help=f"Target return percentage (default: {DEFAULT_TARGET_RETURN}%)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs (default: 300)")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip-config-update", action="store_true",
                        help="Skip ML configuration update step")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Skip deploying models to trading environment")
    args = parser.parse_args()
    
    # Ensure all required directories exist
    ensure_directories()
    
    # Start time for tracking total execution time
    start_time = time.time()
    
    # Step 1: Update ML configuration for hyperperformance
    if not args.skip_config_update:
        update_ml_config(args.max_leverage)
    
    # Step 2: Prepare training data for all pairs
    if not args.skip_data_prep:
        prepare_training_data(args.pairs)
    
    # Step 3: Train integrated models
    train_integrated_models(
        trading_pairs=args.pairs,
        target_win_rate=args.target_win_rate,
        target_return=args.target_return,
        epochs=args.epochs
    )
    
    # Step 4: Run model auto-pruning
    run_model_auto_pruning(args.pairs)
    
    # Step 5: Optimize model hyperparameters
    optimize_model_hyperparameters(args.pairs)
    
    # Step 6: Deploy models to live trading (in sandbox mode)
    if not args.no_deploy:
        deploy_models_to_live_trading(args.pairs, sandbox=True)
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Dual strategy ML training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Trained {len(args.pairs)} trading pairs")
    logger.info(f"Target win rate: {args.target_win_rate*100:.1f}%, Target return: {args.target_return:.1f}%")
    logger.info("Models are now active in the trading environment")

if __name__ == "__main__":
    main()