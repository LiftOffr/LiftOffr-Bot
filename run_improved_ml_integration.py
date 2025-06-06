#!/usr/bin/env python3
"""
Improved ML Integration Script

This script:
1. Updates the trading bot with improved ML models
2. Retrains all models with correct input/output shapes
3. Creates and configures ensemble models
4. Updates ML configuration with optimized settings
5. Activates ML trading across all supported pairs

Usage:
    python run_improved_ml_integration.py --pairs SOL/USD,BTC/USD,ETH/USD --sandbox
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_integration.log')
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
    parser = argparse.ArgumentParser(description='Run improved ML integration')
    
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
    
    parser.add_argument(
        '--sandbox',
        action='store_true',
        help='Use sandbox mode for trading'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training/retraining step'
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

def reset_sandbox_portfolio() -> bool:
    """
    Reset sandbox portfolio to initial state.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Resetting sandbox portfolio...")
    
    # Close all open positions
    result = run_command(
        ["python", "close_all_positions.py", "--sandbox"],
        "Closing all open positions"
    )
    
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
    
    # Update ML configuration
    result = run_command(
        [
            "python", "update_ml_config.py",
            "--pairs", ",".join(pairs),
            "--confidence", str(confidence_threshold),
            "--base-leverage", str(base_leverage),
            "--max-leverage", str(max_leverage),
            "--max-risk", str(max_risk),
            "--update-strategy", "--update-ensemble"
        ],
        "Updating ML configuration"
    )
    
    return result is not None

def retrain_models(pairs: List[str]) -> bool:
    """
    Retrain models for all pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Retraining models for all pairs...")
    
    success = True
    
    for pair in pairs:
        logger.info(f"Retraining models for {pair}...")
        
        # Retrain LSTM model
        result = run_command(
            ["python", "improved_model_retraining.py", "--pair", pair, "--model", "lstm,tcn,transformer"],
            f"Retraining models for {pair}"
        )
        
        if result is None:
            logger.error(f"Failed to retrain models for {pair}")
            success = False
    
    return success

def create_ensemble_models(pairs: List[str]) -> bool:
    """
    Create ensemble models for all pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating ensemble models for all pairs...")
    
    # Create ensemble models
    result = run_command(
        ["python", "create_ensemble_model.py", "--pairs", ",".join(pairs)],
        "Creating ensemble models"
    )
    
    return result is not None

def activate_ml_trading(pairs: List[str], capital: float, sandbox: bool = True) -> bool:
    """
    Activate ML trading for all pairs.
    
    Args:
        pairs: List of trading pairs
        capital: Starting capital for trading
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Activating ML trading...")
    
    # Build command
    cmd = [
        "python", "reset_and_activate_ml_trading.py",
        "--pairs", ",".join(pairs),
        "--capital", str(capital),
        "--confidence", str(DEFAULT_CONFIDENCE_THRESHOLD),
        "--base-leverage", str(DEFAULT_BASE_LEVERAGE),
        "--max-leverage", str(DEFAULT_MAX_LEVERAGE),
        "--max-risk", str(DEFAULT_MAX_RISK_PER_TRADE)
    ]
    
    if sandbox:
        # No need to add flag as it's the default
        pass
    else:
        cmd.append("--live")
    
    # Activate ML trading
    result = run_command(cmd, "Activating ML trading")
    
    return result is not None

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    logger.info("=" * 80)
    logger.info("RUNNING IMPROVED ML INTEGRATION")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"ML confidence threshold: {args.confidence:.2f}")
    logger.info(f"Leverage settings: Base={args.base_leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Max risk per trade: {args.max_risk * 100:.1f}%")
    logger.info(f"Trading mode: {'SANDBOX' if args.sandbox else 'LIVE'}")
    logger.info(f"Skip training: {args.skip_training}")
    logger.info(f"Skip reset: {args.skip_reset}")
    logger.info("=" * 80)
    
    # Confirmation for live mode
    if not args.sandbox:
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return 1
    
    # Reset sandbox portfolio if not skipped
    if not args.skip_reset:
        logger.info("Step 1: Resetting sandbox portfolio...")
        if not reset_sandbox_portfolio():
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
    
    # Retrain models if not skipped
    if not args.skip_training:
        logger.info("Step 3: Retraining models...")
        if not retrain_models(pairs):
            logger.error("Failed to retrain models. Exiting.")
            return 1
        
        logger.info("Step 4: Creating ensemble models...")
        if not create_ensemble_models(pairs):
            logger.error("Failed to create ensemble models. Exiting.")
            return 1
    else:
        logger.info("Step 3/4: Skipping model training as requested")
    
    # Activate ML trading
    logger.info("Step 5: Activating ML trading...")
    if not activate_ml_trading(pairs, args.capital, args.sandbox):
        logger.error("Failed to activate ML trading. Exiting.")
        return 1
    
    logger.info("=" * 80)
    logger.info("IMPROVED ML INTEGRATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("The bot is now trading with improved ML models and self-optimization.")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())