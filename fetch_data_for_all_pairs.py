#!/usr/bin/env python3
"""
Fetch Historical Data for All Trading Pairs

This script orchestrates the fetching of historical data for all 10 cryptocurrency pairs
across all timeframes (1m, 5m, 15m, 1h, 4h, 1d).

Usage:
    python fetch_data_for_all_pairs.py [--pairs PAIR1,PAIR2,...] [--small-only]
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
        logging.FileHandler('data_fetch.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
STANDARD_TIMEFRAMES = ['15m', '1h', '4h', '1d']
SMALL_TIMEFRAMES = ['1m', '5m']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fetch historical data for all trading pairs')
    parser.add_argument('--pairs', type=str,
                      help='Comma-separated list of pairs to fetch (default: all)')
    parser.add_argument('--small-only', action='store_true',
                      help='Only fetch small timeframes (1m, 5m)')
    return parser.parse_args()


def run_command(command: List[str], description: str = None, timeout: int = 300) -> bool:
    """
    Run a shell command and log output
    
    Args:
        command: Command to run
        description: Description of the command
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        logging.info(f"{description}")
    
    logging.info(f"Running: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Process output
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
        logging.error(f"Command timed out after {timeout} seconds")
        try:
            process.kill()
        except:
            pass
        return False
    
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False


def fetch_standard_timeframes(pair: str) -> bool:
    """
    Fetch historical data for standard timeframes (15m, 1h, 4h, 1d)
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        
    Returns:
        True if successful, False otherwise
    """
    command = ["python", "fetch_kraken_historical_data.py", "--pair", pair]
    return run_command(command, f"Fetching standard timeframes for {pair}")


def fetch_small_timeframes(pair: str, days: int = 14) -> bool:
    """
    Fetch historical data for small timeframes (1m, 5m)
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        days: Number of days of historical data to fetch
        
    Returns:
        True if successful, False otherwise
    """
    command = ["python", "fetch_kraken_small_timeframes.py", "--pairs", pair, "--days", str(days)]
    return run_command(command, f"Fetching small timeframes for {pair}")


def check_data_files(pair: str) -> dict:
    """
    Check which data files exist for a pair
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        
    Returns:
        Dictionary with timeframes as keys and True/False as values
    """
    pair_formatted = pair.replace('/', '_')
    data_files = {}
    
    # Check standard timeframes
    for timeframe in STANDARD_TIMEFRAMES + SMALL_TIMEFRAMES:
        file_path = f"historical_data/{pair_formatted}_{timeframe}.csv"
        data_files[timeframe] = os.path.exists(file_path)
    
    return data_files


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create required directories
    os.makedirs("historical_data", exist_ok=True)
    
    # Determine which pairs to process
    if args.pairs:
        pairs_to_process = args.pairs.split(',')
        # Validate pairs
        for pair in pairs_to_process:
            if pair not in SUPPORTED_PAIRS:
                logging.error(f"Unsupported pair: {pair}")
                pairs_to_process.remove(pair)
    else:
        pairs_to_process = SUPPORTED_PAIRS
    
    # Process each pair
    for i, pair in enumerate(pairs_to_process, 1):
        logging.info(f"=== Processing pair {i}/{len(pairs_to_process)}: {pair} ===")
        
        # Check existing data files
        existing_files = check_data_files(pair)
        
        # If small-only flag is set, only fetch small timeframes
        if not args.small_only:
            # Check if any standard timeframe is missing
            if not all(existing_files.get(tf, False) for tf in STANDARD_TIMEFRAMES):
                if fetch_standard_timeframes(pair):
                    logging.info(f"Successfully fetched standard timeframes for {pair}")
                else:
                    logging.error(f"Failed to fetch standard timeframes for {pair}")
            else:
                logging.info(f"Standard timeframes already exist for {pair}")
        
        # Check if any small timeframe is missing
        if not all(existing_files.get(tf, False) for tf in SMALL_TIMEFRAMES):
            if fetch_small_timeframes(pair):
                logging.info(f"Successfully fetched small timeframes for {pair}")
            else:
                logging.error(f"Failed to fetch small timeframes for {pair}")
        else:
            logging.info(f"Small timeframes already exist for {pair}")
        
        # Update data file status
        updated_files = check_data_files(pair)
        for timeframe, exists in updated_files.items():
            status = "✓" if exists else "✗"
            logging.info(f"{pair} ({timeframe}): {status}")
    
    logging.info("=== Data fetch complete ===")
    
    # Check final status for all pairs
    missing_files = []
    for pair in pairs_to_process:
        file_status = check_data_files(pair)
        for timeframe, exists in file_status.items():
            if not exists:
                missing_files.append(f"{pair} ({timeframe})")
    
    if missing_files:
        logging.warning("The following data files are still missing:")
        for missing in missing_files:
            logging.warning(f"  {missing}")
    else:
        logging.info("All data files have been successfully fetched")
    
    return True


if __name__ == "__main__":
    main()