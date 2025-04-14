#!/usr/bin/env python3
"""
Run Extended ML Training Pipeline

This script automates the complete ML training pipeline with extended historical data:
1. Fetches extended historical data (365 days)
2. Prepares enhanced datasets for all pairs
3. Trains all ML model architectures
4. Creates ensemble models
5. Validates model performance

Usage:
    python run_extended_ml_training.py [--pairs PAIR1 PAIR2 ...] [--days DAYS] [--no-fetch] [--no-train]
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_DAYS = 365
MIN_SAMPLES = 200

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run extended ML training pipeline")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to train on (default: {', '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes to train on (default: {', '.join(DEFAULT_TIMEFRAMES)})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Number of days of historical data to fetch (default: {DEFAULT_DAYS})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help=f"Minimum number of samples required (default: {MIN_SAMPLES})")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip fetching historical data")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training models")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    return parser.parse_args()

def run_command(command, description=None):
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(line)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}")
        if e.stdout:
            for line in e.stdout.strip().split('\n'):
                logger.info(line)
        if e.stderr:
            for line in e.stderr.strip().split('\n'):
                logger.error(line)
        return False

def fetch_extended_data(pairs, timeframes, days):
    """Fetch extended historical data for all pairs"""
    pairs_str = " ".join(pairs)
    command = f"python fetch_extended_historical_data.py --days {days} --pairs {pairs_str}"
    return run_command(command, f"Fetching {days} days of historical data for {', '.join(pairs)}")

def prepare_datasets(pairs, timeframes, min_samples):
    """Prepare enhanced datasets for all pairs"""
    success = True
    
    for pair in pairs:
        for timeframe in timeframes:
            pair_name = pair.replace("/", "")
            command = f"python prepare_enhanced_dataset.py --pair {pair} --timeframe {timeframe} --min-samples {min_samples}"
            if not run_command(command, f"Preparing enhanced dataset for {pair} ({timeframe})"):
                success = False
    
    return success

def train_models(pairs, timeframes, epochs, batch_size):
    """Train ML models for all pairs"""
    pairs_str = " ".join([p.replace("/", "") for p in pairs])
    timeframes_str = " ".join(timeframes)
    command = f"python train_on_enhanced_datasets.py --pairs {pairs_str} --timeframes {timeframes_str} --epochs {epochs} --batch-size {batch_size} --ensemble"
    return run_command(command, f"Training ML models for {', '.join(pairs)} on {', '.join(timeframes)}")

def update_ml_config():
    """Update ML config to use new models"""
    command = "python update_ml_config.py"
    return run_command(command, "Updating ML configuration")

def main():
    """Main function"""
    args = parse_arguments()
    success = True
    
    # 1. Fetch extended historical data
    if not args.no_fetch:
        if not fetch_extended_data(args.pairs, args.timeframes, args.days):
            logger.error("Failed to fetch historical data")
            success = False
    
    # 2. Prepare enhanced datasets
    if success:
        if not prepare_datasets(args.pairs, args.timeframes, args.min_samples):
            logger.error("Failed to prepare enhanced datasets")
            success = False
    
    # 3. Train ML models
    if success and not args.no_train:
        if not train_models(args.pairs, args.timeframes, args.epochs, args.batch_size):
            logger.error("Failed to train ML models")
            success = False
    
    # 4. Update ML config
    if success and not args.no_train:
        if not update_ml_config():
            logger.error("Failed to update ML configuration")
            success = False
    
    if success:
        logger.info("ML training pipeline completed successfully")
        return 0
    else:
        logger.error("ML training pipeline failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())