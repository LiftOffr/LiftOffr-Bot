#!/usr/bin/env python3
"""
Train Models for All Trading Pairs

This script coordinates the training process for all trading pairs:
1. Fetches historical data for all timeframes
2. Trains individual models for each timeframe
3. Creates ensemble models
4. Activates the trading system

Usage:
    python train_all_trading_pairs.py [--pairs PAIR1,PAIR2,...] [--mode full|minimal]
                                      [--days DAYS] [--activate]
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_all_pairs.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for all trading pairs')
    parser.add_argument('--pairs', type=str, default=None,
                      help='Comma-separated list of pairs to train (e.g., BTC/USD,ETH/USD)')
    parser.add_argument('--mode', type=str, choices=['full', 'minimal'], default='full',
                      help='Training mode: full=all timeframes, minimal=standard timeframes only')
    parser.add_argument('--days', type=int, default=14,
                      help='Number of days of data to fetch for small timeframes')
    parser.add_argument('--activate', action='store_true',
                      help='Activate trading system after training')
    parser.add_argument('--force', action='store_true',
                      help='Force retraining of existing models')
    return parser.parse_args()


def run_command(cmd: List[str], description: str = None, timeout: int = None) -> bool:
    """
    Run a shell command and log output
    
    Args:
        cmd: Command to run
        description: Description of the command
        timeout: Timeout in seconds
        
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
        return_code = process.wait(timeout=timeout)
        
        if return_code != 0:
            logging.error(f"Command failed with exit code {return_code}")
            return False
        
        return True
    
    except subprocess.TimeoutExpired:
        process.kill()
        logging.warning(f"Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False


def fetch_historical_data(pairs: List[str], days: int) -> bool:
    """
    Fetch historical data for all timeframes
    
    Args:
        pairs: List of trading pairs to fetch
        days: Number of days of data to fetch for small timeframes
        
    Returns:
        True if successful, False otherwise
    """
    logging.info("Fetching historical data...")
    
    # Fetch standard timeframe data
    for pair in pairs:
        cmd = ["python", "fetch_kraken_historical_data.py", "--pair", pair]
        if not run_command(cmd, f"Fetching standard timeframe data for {pair}"):
            logging.error(f"Failed to fetch standard timeframe data for {pair}")
            return False
    
    # Fetch small timeframe data
    pairs_arg = ",".join(pairs)
    cmd = ["python", "fetch_kraken_small_timeframes.py", "--pairs", pairs_arg, "--days", str(days)]
    if not run_command(cmd, f"Fetching small timeframe data for {len(pairs)} pairs"):
        logging.error(f"Failed to fetch small timeframe data")
        return False
    
    return True


def train_models(pairs: List[str], mode: str, force: bool) -> bool:
    """
    Train models for all pairs
    
    Args:
        pairs: List of trading pairs to train
        mode: Training mode (full or minimal)
        force: Whether to force retraining
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Training models for {len(pairs)} pairs (mode: {mode})...")
    
    # Prepare command
    pairs_arg = ",".join(pairs)
    cmd = ["python", "train_multi_timeframe_all_pairs.py", "--pairs", pairs_arg]
    
    if force:
        cmd.append("--force")
    
    if mode == 'minimal':
        cmd.append("--skip-small-tf")
    
    # Run training
    if not run_command(cmd, f"Training models for {len(pairs)} pairs"):
        logging.error(f"Training failed")
        return False
    
    return True


def activate_trading(sandbox: bool = True) -> bool:
    """
    Activate the trading system
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Activating trading system (sandbox={sandbox})...")
    
    cmd = ["python", "activate_multi_timeframe_models.py"]
    if sandbox:
        cmd.append("--sandbox")
    
    if not run_command(cmd, "Activating trading system"):
        logging.error("Failed to activate trading system")
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
        return 1
    
    logging.info(f"Processing {len(pairs_to_process)} trading pairs: {', '.join(pairs_to_process)}")
    
    # Fetch historical data
    if not fetch_historical_data(pairs_to_process, args.days):
        logging.error("Failed to fetch historical data")
        return 1
    
    # Train models
    if not train_models(pairs_to_process, args.mode, args.force):
        logging.error("Failed to train models")
        return 1
    
    # Activate trading system if requested
    if args.activate:
        if not activate_trading(True):  # Always use sandbox mode for safety
            logging.error("Failed to activate trading system")
            return 1
    
    logging.info("=== Training Process Complete ===")
    logging.info(f"Trained models for {len(pairs_to_process)} trading pairs")
    logging.info(f"Trained pairs: {', '.join(pairs_to_process)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())