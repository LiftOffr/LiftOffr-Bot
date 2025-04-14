#!/usr/bin/env python3
"""
Run ML Training and Live Integration

This script provides an end-to-end solution for:
1. Training ML models for all supported assets
2. Training the strategy ensemble for collaborative trading
3. Starting the ML-enhanced trading bot with the trained models

It combines multiple components into a single command for easy deployment.
"""

import os
import sys
import time
import json
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

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

def run_command(command: List[str], cwd: Optional[str] = None) -> int:
    """
    Run a shell command
    
    Args:
        command: Command to run (as list of arguments)
        cwd: Working directory
        
    Returns:
        int: Return code
    """
    logger.info(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    return process.returncode

def train_ml_models(
    assets: List[str],
    days: int,
    optimize: bool,
    force_retrain: bool,
    extreme_leverage: bool
) -> bool:
    """
    Train ML models for all assets
    
    Args:
        assets: Trading pairs to train
        days: Days of data for training
        optimize: Whether to optimize models
        force_retrain: Force retraining even if models exist
        extreme_leverage: Train for extreme leverage settings
        
    Returns:
        bool: Success flag
    """
    logger.info(f"Training ML models for assets: {assets}")
    
    # Build command
    command = ["./train_ml_live_integration.py"]
    
    # Add assets
    command.extend(["--assets"] + assets)
    
    # Add days
    command.extend(["--days", str(days)])
    
    # Add flags
    if optimize:
        command.append("--optimize")
    
    if force_retrain:
        command.append("--force-retrain")
    
    if extreme_leverage:
        command.append("--extreme-leverage")
    
    # Run command
    return run_command(command) == 0

def train_strategy_ensemble(
    assets: List[str],
    visualize: bool
) -> bool:
    """
    Train strategy ensemble for collaborative trading
    
    Args:
        assets: Trading pairs to train
        visualize: Whether to generate visualizations
        
    Returns:
        bool: Success flag
    """
    logger.info(f"Training strategy ensemble for assets: {assets}")
    
    # Prepare command
    command = ["./strategy_ensemble_trainer.py"]
    
    # Run command
    return run_command(command) == 0

def start_ml_trading_bot(
    assets: List[str],
    sandbox: bool,
    extreme_leverage: bool,
    ml_position_sizing: bool,
    interval: int
) -> bool:
    """
    Start ML-enhanced trading bot
    
    Args:
        assets: Trading pairs to trade
        sandbox: Whether to run in sandbox mode
        extreme_leverage: Whether to use extreme leverage settings
        ml_position_sizing: Whether to use ML-based position sizing
        interval: Seconds between trading iterations
        
    Returns:
        bool: Success flag
    """
    logger.info(f"Starting ML trading bot for assets: {assets}")
    
    # Build command
    command = ["./run_ml_live_bot.py"]
    
    # Add assets
    command.extend(["--pairs"] + assets)
    
    # Add flags
    if not sandbox:
        command.append("--live")
    
    if extreme_leverage:
        command.append("--extreme-leverage")
    
    if ml_position_sizing:
        command.append("--ml-position-sizing")
    
    # Add interval
    command.extend(["--interval", str(interval)])
    
    # Run command
    return run_command(command) == 0

def main():
    """Run the full ML training and live integration pipeline"""
    parser = argparse.ArgumentParser(description='Train and deploy ML-enhanced trading bot')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading pairs to trade')
    
    parser.add_argument('--days', type=int, default=90,
                      help='Days of data for training')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize models during training')
    
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retraining even if models exist')
    
    parser.add_argument('--extreme-leverage', action='store_true',
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--ml-position-sizing', action='store_true',
                      help='Use ML-based position sizing')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Run in sandbox mode (no real trades)')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (not sandbox)')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip ML and ensemble training')
    
    parser.add_argument('--skip-bot', action='store_true',
                      help='Skip starting trading bot')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    # Log starting configuration
    logger.info(f"Starting ML training and deployment with configuration:")
    logger.info(f"  Assets: {args.assets}")
    logger.info(f"  Training days: {args.days}")
    logger.info(f"  Optimize: {args.optimize}")
    logger.info(f"  Force retrain: {args.force_retrain}")
    logger.info(f"  Extreme leverage: {args.extreme_leverage}")
    logger.info(f"  ML position sizing: {args.ml_position_sizing}")
    logger.info(f"  Trading mode: {'sandbox' if args.sandbox else 'LIVE'}")
    logger.info(f"  Trading interval: {args.interval} seconds")
    logger.info(f"  Skip training: {args.skip_training}")
    logger.info(f"  Skip bot: {args.skip_bot}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/ensemble", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Run pipeline steps
    if not args.skip_training:
        # Train ML models
        logger.info("Step 1: Training ML models")
        if not train_ml_models(
            assets=args.assets,
            days=args.days,
            optimize=args.optimize,
            force_retrain=args.force_retrain,
            extreme_leverage=args.extreme_leverage
        ):
            logger.error("ML model training failed")
            return 1
        
        # Train strategy ensemble
        logger.info("Step 2: Training strategy ensemble")
        if not train_strategy_ensemble(
            assets=args.assets,
            visualize=True
        ):
            logger.error("Strategy ensemble training failed")
            return 1
    else:
        logger.info("Skipping training steps as requested")
    
    # Start trading bot
    if not args.skip_bot:
        logger.info("Step 3: Starting ML trading bot")
        if not start_ml_trading_bot(
            assets=args.assets,
            sandbox=args.sandbox,
            extreme_leverage=args.extreme_leverage,
            ml_position_sizing=args.ml_position_sizing,
            interval=args.interval
        ):
            logger.error("ML trading bot failed to start")
            return 1
    else:
        logger.info("Skipping bot start as requested")
    
    logger.info("ML training and deployment pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())