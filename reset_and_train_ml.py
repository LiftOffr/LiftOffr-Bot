#!/usr/bin/env python3
"""
Reset and Train ML

A simplified script to reset the sandbox portfolio and activate ML trading.
This streamlined version focuses on the core functionality needed to:
1. Reset the sandbox portfolio
2. Apply optimized ML configuration
3. Activate ML trading for selected pairs

Usage:
    python reset_and_train_ml.py --pairs SOL/USD,BTC/USD,ETH/USD
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_reset_train.log')
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_CAPITAL = 20000.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_BASE_LEVERAGE = 20.0
DEFAULT_MAX_LEVERAGE = 125.0
DEFAULT_MAX_RISK_PER_TRADE = 0.20  # 20% of available capital

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reset portfolio and activate ML trading')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=DEFAULT_CAPITAL,
        help=f'Starting capital for trading (default: {DEFAULT_CAPITAL})'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for ML signals (default: {DEFAULT_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--base-leverage',
        type=float,
        default=DEFAULT_BASE_LEVERAGE,
        help=f'Base leverage for trades (default: {DEFAULT_BASE_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-leverage',
        type=float,
        default=DEFAULT_MAX_LEVERAGE,
        help=f'Maximum leverage for high-confidence trades (default: {DEFAULT_MAX_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-risk',
        type=float,
        default=DEFAULT_MAX_RISK_PER_TRADE,
        help=f'Maximum risk per trade as fraction of capital (default: {DEFAULT_MAX_RISK_PER_TRADE})'
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
    
    parser.add_argument(
        '--skip-reset',
        action='store_true',
        help='Skip portfolio reset step'
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

def reset_portfolio(sandbox: bool = True) -> bool:
    """
    Reset the portfolio by closing all open positions.
    
    Args:
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Resetting {'sandbox' if sandbox else 'live'} portfolio...")
    
    # Build command
    cmd = ["python", "close_all_positions.py"]
    
    if not sandbox:
        cmd.append("--live")
    else:
        cmd.append("--sandbox")
    
    # Run command
    result = run_command(cmd, "Closing all open positions")
    
    return result is not None

def update_ml_configuration(
    pairs: List[str],
    confidence_threshold: float,
    base_leverage: float,
    max_leverage: float,
    max_risk: float
) -> bool:
    """
    Update ML configuration with optimized settings.
    
    Args:
        pairs: List of trading pairs
        confidence_threshold: Confidence threshold for ML signals
        base_leverage: Base leverage for trades
        max_leverage: Maximum leverage for high-confidence trades
        max_risk: Maximum risk per trade as fraction of capital
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Updating ML configuration...")
    
    # Build command
    cmd = [
        "python",
        "update_ml_config.py",
        "--pairs", ",".join(pairs),
        "--confidence", str(confidence_threshold),
        "--base-leverage", str(base_leverage),
        "--max-leverage", str(max_leverage),
        "--max-risk", str(max_risk),
        "--update-strategy",
        "--update-ensemble"
    ]
    
    # Run command
    result = run_command(cmd, "Updating ML configuration")
    
    return result is not None

def activate_ml_trading(
    pairs: List[str],
    capital: float = DEFAULT_CAPITAL,
    sandbox: bool = True
) -> bool:
    """
    Activate ML trading for specified pairs.
    
    Args:
        pairs: List of trading pairs
        capital: Starting capital
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating ML trading for {', '.join(pairs)}...")
    
    # First check if integrated_strategy.py exists
    if not os.path.exists("integrated_strategy.py"):
        logger.error("integrated_strategy.py not found!")
        return False
    
    # Build command to run the strategy
    cmd = [
        "python",
        "integrated_strategy.py",
        "--pairs", ",".join(pairs),
        "--capital", str(capital),
        "--strategy", "ml",
        "--multi-strategy"
    ]
    
    if not sandbox:
        cmd.append("--live")
    else:
        cmd.append("--sandbox")
    
    # Run command
    result = run_command(cmd, "Starting ML trading")
    
    return result is not None

def create_ensemble_models(pairs: List[str]) -> bool:
    """
    Create ensemble models for all pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists("create_ensemble_model.py"):
        logger.warning("create_ensemble_model.py not found, skipping ensemble model creation")
        return True
        
    logger.info("Creating ensemble models for all pairs...")
    
    # Track overall success
    success = True
    
    # Create ensemble models one pair at a time
    for pair in pairs:
        logger.info(f"Creating ensemble model for {pair}...")
        
        # Some scripts might use different argument formats, try both common formats
        try:
            # First try with --pair
            result = run_command(
                ["python", "create_ensemble_model.py", "--pair", pair],
                f"Creating ensemble model for {pair}"
            )
            
            if result is None:
                # If that failed, try with --trading-pair
                result = run_command(
                    ["python", "create_ensemble_model.py", "--trading-pair", pair],
                    f"Creating ensemble model for {pair} (alternative format)"
                )
                
            if result is None:
                logger.warning(f"Failed to create ensemble model for {pair}")
                success = False
                
        except Exception as e:
            logger.error(f"Error creating ensemble model for {pair}: {str(e)}")
            success = False
    
    return success

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Determine if in sandbox or live mode
    use_sandbox = not args.live  # Default to sandbox unless live is specified
    
    logger.info("=" * 80)
    logger.info("RESET AND TRAIN ML")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"ML confidence threshold: {args.confidence:.2f}")
    logger.info(f"Leverage settings: Base={args.base_leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Max risk per trade: {args.max_risk * 100:.1f}%")
    logger.info(f"Trading mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info(f"Skip reset: {args.skip_reset}")
    logger.info("=" * 80)
    
    # Confirmation for live mode
    if args.live:
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return 1
    
    # Reset portfolio if not skipped
    if not args.skip_reset:
        logger.info("Step 1: Resetting portfolio...")
        if not reset_portfolio(use_sandbox):
            logger.error("Failed to reset portfolio. Exiting.")
            return 1
    else:
        logger.info("Step 1: Skipping portfolio reset as requested")
    
    # Update ML configuration
    logger.info("Step 2: Updating ML configuration...")
    if not update_ml_configuration(
        pairs,
        args.confidence,
        args.base_leverage,
        args.max_leverage,
        args.max_risk
    ):
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # Check if training data exists before trying to create ensemble models
    training_data_available = False
    for pair in pairs:
        pair_path = pair.replace('/', '')
        if os.path.exists(f"training_data/{pair_path}_1h_enhanced.csv") or \
           os.path.exists(f"training_data/{pair.split('/')[0]}/{pair.split('/')[1]}_1h_enhanced.csv"):
            training_data_available = True
            break
    
    if training_data_available:
        # Create ensemble models only if training data is available
        logger.info("Step 3: Creating ensemble models...")
        if not create_ensemble_models(pairs):
            logger.warning("Failed to create ensemble models. Continuing anyway...")
    else:
        logger.warning("Training data not found. Skipping ensemble model creation.")
        # Try to copy and use pre-trained models if available
        if os.path.exists("models/ensemble"):
            logger.info("Using pre-trained ensemble models if available.")
    
    # Activate ML trading
    logger.info("Step 4: Activating ML trading...")
    if not activate_ml_trading(pairs, args.capital, use_sandbox):
        logger.error("Failed to activate ML trading. Exiting.")
        return 1
    
    logger.info("=" * 80)
    logger.info("RESET AND TRAIN ML COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("The bot is now training with ML models and should begin optimizing.")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())