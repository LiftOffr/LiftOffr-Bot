#!/usr/bin/env python3
"""
Prepare Historical Data for Multiple Pairs

This script automates the process of fetching and preparing historical data
for multiple cryptocurrency pairs for the enhanced ML trading system.

Usage:
    python prepare_all_historical_data.py [--pairs PAIR1 PAIR2 ...] [--timeframes TIMEFRAME1 TIMEFRAME2 ...]
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
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DATA_DIR = "historical_data"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Prepare historical data for multiple trading pairs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help=f"Trading pairs to prepare data for (default: {DEFAULT_PAIRS})"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help=f"Timeframes to prepare data for (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force downloading data even if files already exist"
    )
    
    return parser.parse_args()


def prepare_directories() -> None:
    """Prepare directories for historical data"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create pair-specific directories
    for pair in DEFAULT_PAIRS:
        pair_dir = os.path.join(DATA_DIR, pair.replace("/", ""))
        os.makedirs(pair_dir, exist_ok=True)


def run_command(command: List[str], description: Optional[str] = None) -> bool:
    """
    Run a shell command
    
    Args:
        command: Command to run as a list of strings
        description: Optional description of the command
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        logger.info(description)
    
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command: {' '.join(command)}")
        return False


def fetch_historical_data(pair: str, timeframe: str, force: bool = False) -> bool:
    """
    Fetch historical data for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        force: Whether to force download even if files exist
        
    Returns:
        True if successful, False otherwise
    """
    # Check if data already exists
    pair_code = pair.replace("/", "")
    data_path = os.path.join(DATA_DIR, f"{pair_code}", f"{pair_code}_{timeframe}.csv")
    
    if os.path.exists(data_path) and not force:
        logger.info(f"Data for {pair} ({timeframe}) already exists at {data_path}")
        return True
    
    # Fetch data
    command = [
        "python",
        "enhanced_historical_data_fetcher.py",
        "--pair", pair,
        "--timeframe", timeframe
    ]
    
    return run_command(command, f"Fetching historical data for {pair} ({timeframe})")


def prepare_enhanced_dataset(pair: str, timeframe: str) -> bool:
    """
    Prepare enhanced dataset for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python",
        "prepare_enhanced_dataset.py",
        "--pair", pair,
        "--timeframe", timeframe
    ]
    
    return run_command(command, f"Preparing enhanced dataset for {pair} ({timeframe})")


def main() -> None:
    """Main function"""
    args = parse_arguments()
    prepare_directories()
    
    # Track successful and failed pairs
    successful_pairs = []
    failed_pairs = []
    
    for pair in args.pairs:
        pair_success = True
        
        for timeframe in args.timeframes:
            if not fetch_historical_data(pair, timeframe, args.force):
                logger.error(f"Failed to fetch historical data for {pair} ({timeframe})")
                pair_success = False
                continue
            
            # Allow some time between requests to avoid rate limiting
            time.sleep(1)
            
            if not prepare_enhanced_dataset(pair, timeframe):
                logger.error(f"Failed to prepare enhanced dataset for {pair} ({timeframe})")
                pair_success = False
        
        if pair_success:
            successful_pairs.append(pair)
        else:
            failed_pairs.append(pair)
    
    # Show summary
    logger.info("\n=== Summary ===")
    if successful_pairs:
        logger.info(f"Successfully prepared data for: {', '.join(successful_pairs)}")
    if failed_pairs:
        logger.info(f"Failed to prepare data for: {', '.join(failed_pairs)}")


if __name__ == "__main__":
    main()