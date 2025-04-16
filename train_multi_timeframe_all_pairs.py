#!/usr/bin/env python3
"""
Train Multi-Timeframe Models for All Pairs

This script orchestrates the training of models across all timeframes (1m, 5m, 15m, 1h, 4h, 1d)
for all supported trading pairs. It ensures proper training with anti-overfitting measures
and creates ensemble models that combine predictions from all timeframes.

Usage:
    python train_multi_timeframe_all_pairs.py [--pairs PAIR1,PAIR2,...] [--force] [--skip-small-tf]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_all_timeframes.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
SMALL_TIMEFRAMES = ['1m', '5m']
STANDARD_TIMEFRAMES = ['15m', '1h', '4h', '1d']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multi-timeframe models for all pairs')
    parser.add_argument('--pairs', type=str, default=None,
                      help='Comma-separated list of pairs to train (e.g., BTC/USD,ETH/USD)')
    parser.add_argument('--force', action='store_true',
                      help='Force retraining of existing models')
    parser.add_argument('--skip-small-tf', action='store_true',
                      help='Skip training of small timeframe (1m, 5m) models')
    parser.add_argument('--skip-ensemble', action='store_true',
                      help='Skip training of ensemble models')
    parser.add_argument('--days', type=int, default=14,
                      help='Number of days of data to use for small timeframe models')
    return parser.parse_args()


def run_command(cmd: List[str], description: str = None) -> bool:
    """
    Run a shell command and log output
    
    Args:
        cmd: Command to run
        description: Description of the command
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        logging.info(f"{description}")
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to log
        for line in process.stdout:
            line = line.strip()
            if line:
                logging.info(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logging.error(f"Command failed with exit code {return_code}")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False


def ensure_data_exists(pairs: List[str]) -> bool:
    """
    Ensure historical data exists for all timeframes
    
    Args:
        pairs: List of trading pairs to check
        
    Returns:
        True if all data exists, False otherwise
    """
    logging.info("Checking for historical data...")
    
    # Check standard timeframes (15m, 1h, 4h, 1d)
    for pair in pairs:
        for timeframe in STANDARD_TIMEFRAMES:
            pair_symbol = pair.replace('/', '_')
            file_path = f"historical_data/{pair_symbol}_{timeframe}.csv"
            
            if not os.path.exists(file_path):
                logging.warning(f"Missing data for {pair} ({timeframe})")
                # Fetch historical data
                cmd = ["python", "fetch_kraken_historical_data.py", "--pair", pair, "--timeframe", timeframe]
                if not run_command(cmd, f"Fetching {timeframe} data for {pair}"):
                    logging.error(f"Failed to fetch {timeframe} data for {pair}")
                    return False
    
    # Check small timeframes (1m, 5m)
    for pair in pairs:
        for timeframe in SMALL_TIMEFRAMES:
            pair_symbol = pair.replace('/', '_')
            file_path = f"historical_data/{pair_symbol}_{timeframe}.csv"
            
            if not os.path.exists(file_path):
                logging.warning(f"Missing data for {pair} ({timeframe})")
                # Fetch small timeframe data
                cmd = ["python", "fetch_kraken_small_timeframes.py", "--pairs", pair, "--days", "14"]
                if not run_command(cmd, f"Fetching small timeframe data for {pair}"):
                    logging.error(f"Failed to fetch small timeframe data for {pair}")
                    return False
    
    return True


def train_small_timeframe_models(pairs: List[str], force: bool = False, days: int = 14) -> bool:
    """
    Train small timeframe models (1m, 5m)
    
    Args:
        pairs: List of trading pairs to train
        force: Whether to force retraining
        days: Number of days of data to use
        
    Returns:
        True if successful, False otherwise
    """
    logging.info("Training small timeframe models...")
    
    for pair in pairs:
        logging.info(f"Training small timeframe models for {pair}...")
        
        cmd = ["python", "train_with_small_timeframes.py", "--pair", pair]
        if force:
            cmd.append("--force")
        if days:
            cmd.extend(["--days", str(days)])
        
        if not run_command(cmd, f"Training small timeframe models for {pair}"):
            logging.error(f"Failed to train small timeframe models for {pair}")
            return False
    
    return True


def train_standard_timeframe_models(pairs: List[str], force: bool = False) -> bool:
    """
    Train standard timeframe models (15m, 1h, 4h, 1d)
    
    Args:
        pairs: List of trading pairs to train
        force: Whether to force retraining
        
    Returns:
        True if successful, False otherwise
    """
    logging.info("Training standard timeframe models...")
    
    for pair in pairs:
        logging.info(f"Training standard timeframe models for {pair}...")
        
        cmd = ["python", "multi_timeframe_trainer.py", "--pair", pair]
        if force:
            cmd.append("--force")
        
        if not run_command(cmd, f"Training standard timeframe models for {pair}"):
            logging.error(f"Failed to train standard timeframe models for {pair}")
            return False
    
    return True


def create_ensemble_for_all_timeframes(pairs: List[str], force: bool = False) -> bool:
    """
    Create ensemble models that combine all timeframes (1m to 1d)
    
    Args:
        pairs: List of trading pairs to process
        force: Whether to force retraining
        
    Returns:
        True if successful, False otherwise
    """
    logging.info("Creating all-timeframe ensemble models...")
    
    for pair in pairs:
        logging.info(f"Creating all-timeframe ensemble for {pair}...")
        
        cmd = ["python", "create_timeframe_ensemble.py", "--pair", pair]
        if force:
            cmd.append("--force")
        
        if not run_command(cmd, f"Creating all-timeframe ensemble for {pair}"):
            logging.error(f"Failed to create all-timeframe ensemble for {pair}")
            return False
    
    return True


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create required directories
    os.makedirs("historical_data", exist_ok=True)
    os.makedirs("ml_models", exist_ok=True)
    
    # Determine which pairs to process
    if args.pairs is not None:
        pairs_to_process = args.pairs.split(',')
        # Validate pairs
        for pair in list(pairs_to_process):
            if pair not in SUPPORTED_PAIRS:
                logging.warning(f"Unsupported pair: {pair}")
                pairs_to_process.remove(pair)
    else:
        pairs_to_process = SUPPORTED_PAIRS.copy()
    
    if not pairs_to_process:
        logging.error("No valid trading pairs to process")
        return False
    
    logging.info(f"Processing {len(pairs_to_process)} trading pairs: {', '.join(pairs_to_process)}")
    
    # Ensure historical data exists
    if not ensure_data_exists(pairs_to_process):
        logging.error("Failed to ensure historical data exists")
        return False
    
    # Train small timeframe models (1m, 5m)
    if not args.skip_small_tf:
        if not train_small_timeframe_models(pairs_to_process, args.force, args.days):
            logging.error("Failed to train small timeframe models")
            return False
    else:
        logging.info("Skipping small timeframe models as requested")
    
    # Train standard timeframe models (15m, 1h, 4h, 1d)
    if not train_standard_timeframe_models(pairs_to_process, args.force):
        logging.error("Failed to train standard timeframe models")
        return False
    
    # Create ensemble models for all timeframes
    if not args.skip_ensemble:
        if not create_ensemble_for_all_timeframes(pairs_to_process, args.force):
            logging.error("Failed to create ensemble models")
            return False
    else:
        logging.info("Skipping ensemble models as requested")
    
    logging.info("=== Training Summary ===")
    logging.info(f"Successfully trained models for {len(pairs_to_process)} trading pairs")
    logging.info(f"Trained pairs: {', '.join(pairs_to_process)}")
    
    return True


if __name__ == "__main__":
    main()