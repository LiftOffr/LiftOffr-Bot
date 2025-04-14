#!/usr/bin/env python3
"""
Run ML Training and Live Integration

This module provides the main entry point for training ML models and running
the ML-enhanced trading bot with integrated ML predictions.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_deployment.log')
    ]
)
logger = logging.getLogger(__name__)

def run_ml_training(
    assets: List[str],
    optimize: bool = False,
    extreme_leverage: bool = False,
    days: int = 90,
    force_retrain: bool = False,
    visualize: bool = False
):
    """
    Run ML training for specified assets
    
    Args:
        assets: List of trading pairs to train models for
        optimize: Whether to enable hyperparameter optimization
        extreme_leverage: Whether to use extreme leverage settings
        days: Number of days of historical data to use
        force_retrain: Whether to force retrain models that already exist
        visualize: Whether to generate visualizations
    """
    logger.info(f"Starting ML training for assets: {assets}")
    
    # Build command
    command = ["python", "train_ml_live_integration.py", "--assets"] + assets
    
    if optimize:
        command.append("--optimize")
    
    if extreme_leverage:
        command.append("--extreme-leverage")
    
    if force_retrain:
        command.append("--force-retrain")
    
    if visualize:
        command.append("--visualize")
    
    command += ["--days", str(days)]
    
    # Run training
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info("ML training completed successfully")
        
        # Log output
        if result.stdout:
            logger.info(f"Training output:\n{result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ML training failed with error: {e}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr}")
        return False

def run_ensemble_training():
    """Run strategy ensemble training"""
    logger.info("Starting strategy ensemble training")
    
    # Build command
    command = ["python", "strategy_ensemble_trainer.py"]
    
    # Run training
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info("Strategy ensemble training completed successfully")
        
        # Log output
        if result.stdout:
            logger.info(f"Ensemble training output:\n{result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Strategy ensemble training failed with error: {e}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr}")
        return False

def run_ml_live_bot(
    assets: List[str],
    sandbox: bool = True,
    extreme_leverage: bool = False,
    ml_position_sizing: bool = False,
    interval: int = 60,
    max_iterations: Optional[int] = None
):
    """
    Run ML live trading bot
    
    Args:
        assets: List of trading pairs to trade
        sandbox: Whether to use sandbox mode
        extreme_leverage: Whether to use extreme leverage settings
        ml_position_sizing: Whether to use ML-enhanced position sizing
        interval: Seconds between trading iterations
        max_iterations: Maximum number of iterations (None for infinite)
    """
    logger.info(f"Starting ML live trading bot for assets: {assets}")
    
    # Build command
    command = ["python", "run_ml_live_bot.py", "--pairs"] + assets
    
    if not sandbox:
        command.append("--live")
    
    if extreme_leverage:
        command.append("--extreme-leverage")
    
    if ml_position_sizing:
        command.append("--ml-position-sizing")
    
    command += ["--interval", str(interval)]
    
    if max_iterations is not None:
        command += ["--max-iterations", str(max_iterations)]
    
    # Run bot
    try:
        logger.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ML live trading bot failed with error: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("ML live trading bot stopped by user")
        return True

def main():
    """Run ML training and live integration"""
    parser = argparse.ArgumentParser(description='Run ML training and live integration')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading pairs to train and trade')
    
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip ML training and just run the bot')
    
    parser.add_argument('--skip-ensemble', action='store_true',
                      help='Skip strategy ensemble training')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Enable hyperparameter optimization')
    
    parser.add_argument('--days', type=int, default=90,
                      help='Number of days of historical data to use')
    
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retrain models that already exist')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations during training')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Use sandbox mode (no real trades)')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (not sandbox)')
    
    parser.add_argument('--extreme-leverage', action='store_true',
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--ml-position-sizing', action='store_true',
                      help='Use ML-enhanced position sizing')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                      help='Maximum number of iterations (None for infinite)')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    # Run ML training
    if not args.skip_training:
        success = run_ml_training(
            assets=args.assets,
            optimize=args.optimize,
            extreme_leverage=args.extreme_leverage,
            days=args.days,
            force_retrain=args.force_retrain,
            visualize=args.visualize
        )
        
        if not success:
            logger.error("ML training failed, cannot continue")
            return 1
    
    # Run strategy ensemble training
    if not args.skip_ensemble:
        success = run_ensemble_training()
        
        if not success:
            logger.error("Strategy ensemble training failed, cannot continue")
            return 1
    
    # Run ML live trading bot
    run_ml_live_bot(
        assets=args.assets,
        sandbox=args.sandbox,
        extreme_leverage=args.extreme_leverage,
        ml_position_sizing=args.ml_position_sizing,
        interval=args.interval,
        max_iterations=args.max_iterations
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())