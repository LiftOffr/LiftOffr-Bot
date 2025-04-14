#!/usr/bin/env python3
"""
Train Enhanced Dual Strategy AI for Multiple Pairs

This script automates the process of training the enhanced dual strategy AI models
for multiple cryptocurrency pairs. It integrates the ARIMA and Adaptive strategies
with advanced ML techniques to achieve 90% win rate and 1000% returns.

Usage:
    python train_all_dual_strategy_ai.py [--pairs PAIR1 PAIR2 ...] [--timeframes TIMEFRAME1 TIMEFRAME2 ...]
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import List, Optional, Dict, Any

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
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
EPOCHS = 500
TARGET_WIN_RATE = 0.9
TARGET_RETURN = 1000.0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train enhanced dual strategy AI for multiple trading pairs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help=f"Trading pairs to train models for (default: {DEFAULT_PAIRS})"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help=f"Timeframes to train models for (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=TARGET_WIN_RATE,
        help=f"Target win rate (default: {TARGET_WIN_RATE})"
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=TARGET_RETURN,
        help=f"Target return percentage (default: {TARGET_RETURN})"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update ML configuration for hyperperformance"
    )
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="Skip auto-pruning of models"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip hyperparameter optimization"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy trained models to live trading (sandbox mode)"
    )
    
    return parser.parse_args()


def run_command(command: List[str], description: Optional[str] = None) -> bool:
    """
    Run a shell command
    
    Args:
        command: Command to run as a list of strings
        description: Optional description of the command
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        logger.info(description)
    
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command: {' '.join(command)}")
        return False


def update_ml_config_for_hyperperformance() -> bool:
    """
    Update ML configuration for hyperperformance
    
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "update_ml_config_for_hyperperformance.py"
    ]
    
    return run_command(command, "Updating ML configuration for hyperperformance")


def prepare_training_data(pair: str, timeframe: str) -> bool:
    """
    Prepare training data for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "prepare_enhanced_dataset.py",
        "--pair", pair,
        "--timeframe", timeframe
    ]
    
    return run_command(command, f"Preparing training data for {pair} ({timeframe})")


def train_dual_strategy_model(
    pair: str,
    timeframe: str,
    epochs: int = EPOCHS,
    target_win_rate: float = TARGET_WIN_RATE,
    target_return: float = TARGET_RETURN
) -> bool:
    """
    Train dual strategy model for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        epochs: Number of training epochs
        target_win_rate: Target win rate
        target_return: Target return percentage
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "enhanced_dual_strategy_trainer.py",
        "--pairs", pair,
        "--timeframes", timeframe,
        "--epochs", str(epochs),
        "--target-win-rate", str(target_win_rate),
        "--target-return", str(target_return)
    ]
    
    return run_command(command, f"Training dual strategy model for {pair} ({timeframe})")


def auto_prune_models(pair: str, timeframe: str) -> bool:
    """
    Auto-prune models for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "auto_prune_ml_models.py",
        "--pair", pair,
        "--timeframe", timeframe
    ]
    
    return run_command(command, f"Auto-pruning models for {pair} ({timeframe})")


def optimize_hyperparameters(pair: str, timeframe: str) -> bool:
    """
    Optimize hyperparameters for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "adaptive_hyperparameter_tuning.py",
        "--pair", pair,
        "--timeframe", timeframe
    ]
    
    return run_command(command, f"Optimizing hyperparameters for {pair} ({timeframe})")


def deploy_to_live_trading(pairs: List[str], sandbox: bool = True) -> bool:
    """
    Deploy trained models to live trading
    
    Args:
        pairs: List of trading pairs to deploy
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "deploy_ml_models.py",
        "--pairs"
    ] + pairs
    
    if sandbox:
        command.append("--sandbox")
    
    return run_command(command, f"Deploying trained models to {'sandbox' if sandbox else 'live'} trading")


def main() -> None:
    """Main function"""
    args = parse_arguments()
    
    # Update ML configuration if requested
    if args.update_config:
        if not update_ml_config_for_hyperperformance():
            logger.error("Failed to update ML configuration for hyperperformance")
    
    # Track successful and failed pairs
    successful_pairs: Dict[str, List[str]] = {}
    failed_pairs: Dict[str, List[str]] = {}
    
    for pair in args.pairs:
        if pair not in successful_pairs:
            successful_pairs[pair] = []
        if pair not in failed_pairs:
            failed_pairs[pair] = []
        
        for timeframe in args.timeframes:
            # Step 1: Prepare training data
            if not prepare_training_data(pair, timeframe):
                logger.error(f"Failed to prepare training data for {pair} ({timeframe})")
                failed_pairs[pair].append(timeframe)
                continue
            
            # Step 2: Train dual strategy model
            if not train_dual_strategy_model(
                pair,
                timeframe,
                args.epochs,
                args.target_win_rate,
                args.target_return
            ):
                logger.error(f"Failed to train dual strategy model for {pair} ({timeframe})")
                failed_pairs[pair].append(timeframe)
                continue
            
            # Step 3: Auto-prune models (optional)
            if not args.skip_prune:
                if not auto_prune_models(pair, timeframe):
                    logger.warning(f"Auto-pruning models for {pair} ({timeframe}) failed but continuing")
            
            # Step 4: Optimize hyperparameters (optional)
            if not args.skip_optimization:
                if not optimize_hyperparameters(pair, timeframe):
                    logger.warning(f"Hyperparameter optimization for {pair} ({timeframe}) failed but continuing")
            
            successful_pairs[pair].append(timeframe)
    
    # Step 5: Deploy trained models to live trading (optional)
    if args.deploy:
        # Only deploy pairs with successful training
        deploy_pairs = [pair for pair, timeframes in successful_pairs.items() if timeframes]
        
        if deploy_pairs:
            if not deploy_to_live_trading(deploy_pairs, sandbox=True):
                logger.error("Failed to deploy trained models to live trading")
        else:
            logger.warning("No successfully trained pairs to deploy")
    
    # Show summary
    logger.info("\n=== Training Summary ===")
    for pair in args.pairs:
        if successful_pairs[pair]:
            logger.info(f"{pair}: Successfully trained for timeframes: {', '.join(successful_pairs[pair])}")
        if failed_pairs[pair]:
            logger.info(f"{pair}: Failed to train for timeframes: {', '.join(failed_pairs[pair])}")


if __name__ == "__main__":
    main()