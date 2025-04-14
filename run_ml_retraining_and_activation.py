#!/usr/bin/env python3
"""
ML Retraining and Activation Script

This comprehensive script orchestrates the complete ML training and activation workflow:
1. Resets the sandbox portfolio to start with a clean slate
2. Retrains all ML models to fix input shape mismatches
3. Updates ML configuration with optimized parameters
4. Activates ML trading across all specified pairs

Usage:
    python run_ml_retraining_and_activation.py --pairs SOL/USD,BTC/USD,ETH/USD [--live]
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
DEFAULT_PAIRS = ['SOL/USD']
DEFAULT_CAPITAL = 20000.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_BASE_LEVERAGE = 20.0
DEFAULT_MAX_LEVERAGE = 125.0
DEFAULT_MAX_RISK_PER_TRADE = 0.20  # 20% of available capital

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retrain ML models and activate ML trading')
    
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
        '--skip-reset',
        action='store_true',
        help='Skip portfolio reset step'
    )
    
    parser.add_argument(
        '--skip-retrain',
        action='store_true',
        help='Skip model retraining step'
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

def reset_portfolio(sandbox: bool = True) -> bool:
    """
    Reset portfolio by closing all open positions.
    
    Args:
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Resetting {'sandbox' if sandbox else 'live'} portfolio...")
    
    # Close all positions
    cmd = ["python", "close_all_positions.py"]
    if sandbox:
        cmd.append("--sandbox")
    
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
    
    # Update configuration using the utility script
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
    
    result = run_command(cmd, "Updating ML configuration")
    
    return result is not None

def retrain_models(pairs: List[str]) -> bool:
    """
    Retrain ML models to fix input shape mismatch.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Retraining ML models...")
    
    # Try to use improved_model_retraining.py first
    if os.path.exists("improved_model_retraining.py"):
        cmd = [
            "python",
            "improved_model_retraining.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, "Retraining models using improved retraining script")
        if result is not None:
            return True
    
    # Fall back to other retraining scripts if improved version fails or doesn't exist
    if os.path.exists("retrain_models_with_correct_shape.py"):
        cmd = [
            "python",
            "retrain_models_with_correct_shape.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, "Retraining models with correct shape")
        if result is not None:
            return True
    
    # If we reached here, try the enhanced training scripts as a last resort
    if os.path.exists("enhanced_dual_strategy_trainer.py"):
        cmd = [
            "python",
            "enhanced_dual_strategy_trainer.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, "Training with enhanced dual strategy trainer")
        if result is not None:
            return True
    
    logger.error("All model retraining attempts failed")
    return False

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
        
    logger.info("Creating ensemble models...")
    
    # Track overall success
    success = True
    
    # Create ensemble models one pair at a time
    for pair in pairs:
        logger.info(f"Creating ensemble model for {pair}...")
        
        # Check if necessary training data exists
        pair_path = pair.replace('/', '')
        training_data_exists = (
            os.path.exists(f"training_data/{pair_path}_1h_enhanced.csv") or
            os.path.exists(f"training_data/{pair.split('/')[0]}/{pair.split('/')[1]}_1h_enhanced.csv")
        )
        
        if not training_data_exists:
            logger.warning(f"Training data not found for {pair}, skipping ensemble model creation")
            continue
        
        # Try different argument formats
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

def activate_ml_trading(
    pairs: List[str],
    capital: float = DEFAULT_CAPITAL,
    sandbox: bool = True
) -> bool:
    """
    Activate ML trading for all pairs.
    
    Args:
        pairs: List of trading pairs
        capital: Starting capital
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating ML trading for {', '.join(pairs)}...")
    
    # Try activating with quick_activate_ml.py if it exists
    if os.path.exists("quick_activate_ml.py"):
        cmd = [
            "python",
            "quick_activate_ml.py",
            "--pairs", ",".join(pairs),
            "--capital", str(capital)
        ]
        
        if not sandbox:
            cmd.append("--live")
        else:
            cmd.append("--sandbox")
            
        result = run_command(cmd, "Starting ML trading with quick activation")
        if result is not None:
            return True
    
    # Fall back to integrated_strategy.py if quick activation script doesn't exist
    if os.path.exists("integrated_strategy.py"):
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
            
        result = run_command(cmd, "Starting ML trading with integrated strategy")
        if result is not None:
            return True
    
    logger.error("Failed to activate ML trading - no suitable script found")
    return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Determine if in sandbox or live mode
    use_sandbox = not args.live  # Default to sandbox unless live is specified
    
    logger.info("=" * 80)
    logger.info("ML RETRAINING AND ACTIVATION")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"ML confidence threshold: {args.confidence:.2f}")
    logger.info(f"Leverage settings: Base={args.base_leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Max risk per trade: {args.max_risk * 100:.1f}%")
    logger.info(f"Trading mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info(f"Skip reset: {args.skip_reset}")
    logger.info(f"Skip retrain: {args.skip_retrain}")
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
    
    # Retrain models if not skipped
    if not args.skip_retrain:
        logger.info("Step 2: Retraining ML models...")
        if not retrain_models(pairs):
            logger.warning("Failed to retrain ML models. Continuing anyway...")
    else:
        logger.info("Step 2: Skipping model retraining as requested")
    
    # Update ML configuration
    logger.info("Step 3: Updating ML configuration...")
    if not update_ml_configuration(
        pairs,
        args.confidence,
        args.base_leverage,
        args.max_leverage,
        args.max_risk
    ):
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # Create ensemble models
    logger.info("Step 4: Creating ensemble models...")
    if not create_ensemble_models(pairs):
        logger.warning("Failed to create ensemble models. Continuing anyway...")
    
    # Activate ML trading
    logger.info("Step 5: Activating ML trading...")
    if not activate_ml_trading(pairs, args.capital, use_sandbox):
        logger.error("Failed to activate ML trading. Exiting.")
        return 1
    
    logger.info("=" * 80)
    logger.info("ML RETRAINING AND ACTIVATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("The bot is now training with ML models and should begin optimizing.")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())