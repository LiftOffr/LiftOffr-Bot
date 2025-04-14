#!/usr/bin/env python3
"""
Quick ML Activation Script

This streamlined script activates ML trading without the overhead of ensemble model creation,
which can timeout or fail when training data is unavailable.

Usage:
    python quick_activate_ml.py --pairs SOL/USD,BTC/USD,ETH/USD
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Any

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
DEFAULT_PAIRS = ['SOL/USD']
DEFAULT_CAPITAL = 20000.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_BASE_LEVERAGE = 20.0
DEFAULT_MAX_LEVERAGE = 125.0
DEFAULT_MAX_RISK_PER_TRADE = 0.20  # 20% of available capital

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quickly activate ML trading with optimized parameters')
    
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
    
    return parser.parse_args()

def update_ml_config(pairs: List[str], confidence: float, 
                    base_leverage: float, max_leverage: float, max_risk: float) -> bool:
    """
    Update ML configuration directly by writing to ml_config.json
    
    Args:
        pairs: List of trading pairs
        confidence: Confidence threshold for ML signals
        base_leverage: Base leverage for trades
        max_leverage: Maximum leverage for high-confidence trades
        max_risk: Maximum risk per trade as fraction of capital
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Default config
        config = {
            "global": {
                "confidence_threshold": confidence,
                "base_leverage": base_leverage,
                "max_leverage": max_leverage, 
                "max_risk_per_trade": max_risk,
                "trading_pairs": pairs
            },
            "strategy_integration": {
                "signal_weight_multiplier": 1.5,
                "max_signal_age_seconds": 300,
                "use_ensemble_models": True,
                "feature_scaling": "standard"
            },
            "ensemble": {}
        }
        
        # Add pair-specific configs
        for pair in pairs:
            pair_key = pair.replace('/', '')
            config["ensemble"][pair_key] = {
                "model_weights": {
                    "lstm": 0.35,
                    "cnn": 0.35,
                    "transformer": 0.30
                },
                "confidence_scaling": "sigmoid"
            }
        
        # Write to file
        os.makedirs('config', exist_ok=True)
        with open('config/ml_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"ML configuration updated with {len(pairs)} trading pairs")
        return True
            
    except Exception as e:
        logger.error(f"Failed to update ML configuration: {str(e)}")
        return False

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

def activate_ml_trading(pairs: List[str], capital: float = DEFAULT_CAPITAL, 
                       sandbox: bool = True) -> bool:
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

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Determine if in sandbox or live mode
    use_sandbox = not args.live  # Default to sandbox unless live is specified
    
    logger.info("=" * 80)
    logger.info("QUICK ML ACTIVATION")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"ML confidence threshold: {args.confidence:.2f}")
    logger.info(f"Leverage settings: Base={args.base_leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Max risk per trade: {args.max_risk * 100:.1f}%")
    logger.info(f"Trading mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info("=" * 80)
    
    # Confirmation for live mode
    if args.live:
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return 1
    
    # Update ML configuration
    logger.info("Step 1: Updating ML configuration...")
    if not update_ml_config(
        pairs,
        args.confidence,
        args.base_leverage,
        args.max_leverage,
        args.max_risk
    ):
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # Activate ML trading
    logger.info("Step 2: Activating ML trading...")
    if not activate_ml_trading(pairs, args.capital, use_sandbox):
        logger.error("Failed to activate ML trading. Exiting.")
        return 1
    
    logger.info("=" * 80)
    logger.info("ML ACTIVATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("The bot is now trading with ML configuration.")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())