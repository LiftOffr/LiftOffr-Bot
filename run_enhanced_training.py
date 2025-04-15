#!/usr/bin/env python3

"""
Run Enhanced Ultra Training

This script orchestrates the ultra-enhanced training process to achieve higher
accuracy and returns by:

1. Preparing and checking dependencies
2. Validating data availability
3. Training and optimizing models with ultra parameters
4. Applying optimized settings to the trading system
5. Starting the trading bot with improved models

Usage:
    python run_enhanced_training.py [--pairs PAIRS] [--sandbox] [--aggressive]
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default pairs
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Enhanced Ultra Training")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use more aggressive optimization settings")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of hyperparameter optimization trials")
    parser.add_argument("--target-accuracy", type=float, default=0.999,
                        help="Target accuracy (0.0-1.0)")
    parser.add_argument("--target-return", type=float, default=10.0,
                        help="Target return (1.0 = 100%)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and just apply settings")
    parser.add_argument("--reset-portfolio", action="store_true",
                        help="Reset portfolio to initial balance")
    parser.add_argument("--initial-balance", type=float, default=20000.0,
                        help="Initial portfolio balance if resetting")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(description)
        
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning(result.stderr)
            
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

def check_dependencies():
    """Check and install required dependencies"""
    logger.info("Checking dependencies...")
    
    # Check for required ML libraries
    try:
        import tensorflow
        import numpy
        import pandas
        import optuna
        logger.info("Core ML dependencies are already installed")
    except ImportError:
        logger.info("Installing missing ML dependencies...")
        result = run_command(
            ["python", "ensure_ml_dependencies.py"],
            "Installing required ML dependencies"
        )
        if not result:
            logger.error("Failed to install ML dependencies")
            return False
    
    return True

def check_historical_data(pairs):
    """Check if historical data is available for all pairs"""
    logger.info("Checking historical data availability...")
    
    missing_pairs = []
    for pair in pairs:
        pair_filename = pair.replace('/', '_')
        data_file = f"training_data/{pair_filename}_data.csv"
        
        if not os.path.exists(data_file):
            missing_pairs.append(pair)
    
    if missing_pairs:
        logger.warning(f"Missing historical data for {len(missing_pairs)} pairs: {missing_pairs}")
        
        # Fetch missing data
        for pair in missing_pairs:
            logger.info(f"Fetching historical data for {pair}...")
            run_command(
                ["python", "fetch_extended_historical_data.py", "--pair", pair],
                f"Fetching historical data for {pair}"
            )
    
    return True

def train_and_optimize_models(args):
    """Train and optimize models with ultra parameters"""
    logger.info("Starting ultra-enhanced training process...")
    
    # Prepare command
    cmd = [
        "python", "enhanced_ultra_training.py",
        "--pairs", args.pairs,
        "--epochs", str(args.epochs),
        "--trials", str(args.trials),
        "--target-accuracy", str(args.target_accuracy),
        "--target-return", str(args.target_return)
    ]
    
    if args.aggressive:
        cmd.append("--aggressive")
    
    # Run training
    result = run_command(
        cmd,
        "Training and optimizing ML models"
    )
    
    if not result:
        logger.error("Training and optimization failed")
        return False
    
    logger.info("Training and optimization completed successfully")
    return True

def apply_optimized_settings(args):
    """Apply optimized settings to the trading system"""
    logger.info("Applying optimized settings...")
    
    # Prepare command
    cmd = [
        "python", "apply_optimized_settings.py",
        "--pairs", args.pairs
    ]
    
    if args.reset_portfolio:
        cmd.append("--reset-portfolio")
        cmd.extend(["--initial-balance", str(args.initial_balance)])
    
    if args.aggressive:
        cmd.append("--aggressive")
    
    # Apply settings
    result = run_command(
        cmd,
        "Applying optimized settings to trading configuration"
    )
    
    if not result:
        logger.error("Failed to apply optimized settings")
        return False
    
    logger.info("Optimized settings applied successfully")
    return True

def start_trading_bot(args):
    """Start the trading bot with improved models"""
    if not args.sandbox:
        logger.warning("CAUTION: Starting trading bot in LIVE mode!")
        time.sleep(5)  # Give user time to cancel if needed
    
    logger.info(f"Starting trading bot in {'sandbox' if args.sandbox else 'LIVE'} mode...")
    
    # Prepare command
    cmd = [
        "python", 
        "run_risk_aware_sandbox_trader.py" if args.sandbox else "run_risk_aware_trader.py",
        "--pairs", args.pairs
    ]
    
    # Run in background
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Trading bot started with PID {process.pid}")
        logger.info("Trading bot is now running in the background")
        
        # Write PID to file for later management
        with open("bot_pid.txt", "w") as f:
            f.write(str(process.pid))
        
        return True
    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        return False

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    logger.info(f"Starting enhanced ultra training process for {len(pairs)} pairs")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Failed to check or install dependencies")
        return 1
    
    # Check historical data
    if not check_historical_data(pairs):
        logger.error("Failed to check or fetch historical data")
        return 1
    
    # Train and optimize models
    if not args.skip_training:
        if not train_and_optimize_models(args):
            logger.error("Training and optimization failed")
            return 1
    else:
        logger.info("Skipping training as requested")
    
    # Apply optimized settings
    if not apply_optimized_settings(args):
        logger.error("Failed to apply optimized settings")
        return 1
    
    # Start trading bot
    if not start_trading_bot(args):
        logger.error("Failed to start trading bot")
        return 1
    
    logger.info("Enhanced ultra training process completed successfully")
    logger.info("Trading bot is now running with ultra-optimized models and settings")
    return 0

if __name__ == "__main__":
    sys.exit(main())