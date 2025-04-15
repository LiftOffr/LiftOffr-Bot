#!/usr/bin/env python3
"""
Execute New Coins Training

This script orchestrates the complete process of adding 4 new trading pairs:
1. AVAX/USD (Avalanche)
2. MATIC/USD (Polygon)
3. UNI/USD (Uniswap)
4. ATOM/USD (Cosmos)

The process includes:
1. Data collection
2. Feature engineering
3. Model training with optimized parameters
4. Ensemble creation
5. Comprehensive backtesting
6. Integration with the trading system

The goal is to achieve 95%+ accuracy and 1000%+ backtest returns for each pair.

Usage:
    python execute_new_coins_training.py [--skip-download] [--skip-training] [--parallel]
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("new_coins_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("new_coins_training")

# Constants
NEW_COINS = ["AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
TIMEFRAMES = ["1h", "4h", "1d"]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Execute training pipeline for new coins.')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading historical data')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature preparation')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--parallel', action='store_true', help='Run training in parallel')
    parser.add_argument('--sandbox', action='store_true', help='Use sandbox mode for trading')
    return parser.parse_args()


def run_command(cmd, description=None, check=True):
    """Run a shell command and log output."""
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Process output in real-time
        stdout_lines = []
        stderr_lines = []
        
        def read_pipe(pipe, lines):
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                lines.append(line)
                logger.info(line.strip())
        
        # Read stdout and stderr
        while process.poll() is None:
            read_pipe(process.stdout, stdout_lines)
            read_pipe(process.stderr, stderr_lines)
            time.sleep(0.1)
        
        # Read any remaining output
        read_pipe(process.stdout, stdout_lines)
        read_pipe(process.stderr, stderr_lines)
        
        # Check return code
        if check and process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        "historical_data",
        "training_data",
        "ml_models",
        "config",
        "logs",
        "backtest_results",
        "optimization_results"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_name}")
    
    return True


def download_historical_data():
    """Download historical data for new coins."""
    logger.info("Downloading historical data for new coins")
    
    success = True
    for pair in NEW_COINS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Downloading {pair} ({timeframe}) data")
            
            cmd = [
                "python", "fetch_historical_data.py",
                "--pair", pair,
                "--timeframe", timeframe,
                "--days", "365"
            ]
            
            if not run_command(cmd, f"Downloading {pair} ({timeframe}) data"):
                logger.error(f"Failed to download {pair} ({timeframe}) data")
                success = False
            
            # Sleep briefly to avoid rate limits
            time.sleep(1)
    
    return success


def prepare_features():
    """Prepare features for model training."""
    logger.info("Preparing features for new coins")
    
    success = True
    for pair in NEW_COINS:
        logger.info(f"Preparing features for {pair}")
        
        cmd = [
            "python", "prepare_training_data.py",
            "--pair", pair,
            "--timeframes", ",".join(TIMEFRAMES),
            "--include-cross-asset"
        ]
        
        if not run_command(cmd, f"Preparing features for {pair}"):
            logger.error(f"Failed to prepare features for {pair}")
            success = False
    
    return success


def train_models(parallel=False):
    """Train models for new coins."""
    logger.info("Training models for new coins")
    
    cmd = [
        "python", "train_new_coins.py",
        "--pairs", ",".join(NEW_COINS)
    ]
    
    if parallel:
        cmd.append("--parallel")
    
    return run_command(cmd, "Training models for new coins")


def integrate_new_coins(sandbox=True):
    """Integrate new coins into the trading system."""
    logger.info("Integrating new coins into the trading system")
    
    cmd = [
        "python", "add_four_trading_pairs.py"
    ]
    
    if sandbox:
        cmd.append("--sandbox")
    
    return run_command(cmd, "Integrating new coins")


def update_dashboard():
    """Update the dashboard to include new coins."""
    logger.info("Updating dashboard for new coins")
    
    cmd = [
        "python", "auto_update_dashboard.py"
    ]
    
    return run_command(cmd, "Updating dashboard")


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting training pipeline for new coins")
    
    # Ensure directories exist
    ensure_directories()
    
    # 1. Download historical data
    if not args.skip_download:
        if not download_historical_data():
            logger.error("Failed to download historical data")
            return 1
    else:
        logger.info("Skipping historical data download")
    
    # 2. Prepare features
    if not args.skip_download and not args.skip_features:
        if not prepare_features():
            logger.error("Failed to prepare features")
            return 1
    else:
        logger.info("Skipping feature preparation")
    
    # 3. Train models
    if not args.skip_training:
        if not train_models(args.parallel):
            logger.error("Failed to train models")
            return 1
    else:
        logger.info("Skipping model training")
    
    # 4. Integrate new coins
    if not integrate_new_coins(args.sandbox):
        logger.error("Failed to integrate new coins")
        return 1
    
    # 5. Update dashboard
    if not update_dashboard():
        logger.warning("Failed to update dashboard")
        # Continue anyway as this is not critical
    
    logger.info("Training pipeline completed successfully")
    
    # Print summary
    logger.info("\n===== Training Summary =====")
    
    # Try to load ML config to get accuracy metrics
    try:
        import json
        
        with open("config/ml_config.json", 'r') as f:
            ml_config = json.load(f)
        
        logger.info("Model Performance:")
        for pair in NEW_COINS:
            if pair in ml_config.get("pairs", {}):
                pair_config = ml_config["pairs"][pair]
                accuracy = pair_config.get("accuracy", "N/A")
                win_rate = pair_config.get("win_rate", "N/A")
                backtest_return = pair_config.get("backtest_return", "N/A")
                base_leverage = pair_config.get("base_leverage", "N/A")
                
                logger.info(f"{pair}:")
                logger.info(f"  Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"  Accuracy: {accuracy}")
                logger.info(f"  Win Rate: {win_rate:.4f}" if isinstance(win_rate, float) else f"  Win Rate: {win_rate}")
                logger.info(f"  Backtest Return: {backtest_return:.2f}x" if isinstance(backtest_return, float) else f"  Backtest Return: {backtest_return}")
                logger.info(f"  Base Leverage: {base_leverage:.2f}x" if isinstance(base_leverage, float) else f"  Base Leverage: {base_leverage}")
                logger.info("")
    except Exception as e:
        logger.error(f"Error loading ML config: {e}")
    
    logger.info("To see the results in the dashboard, start the application workflow")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())