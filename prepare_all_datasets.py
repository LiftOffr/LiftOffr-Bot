#!/usr/bin/env python3
"""
Prepare All Enhanced Datasets

This script prepares enhanced datasets for all specified trading pairs and timeframes.
It automatically adjusts the minimum sample size requirement based on available data.
"""

import os
import sys
import logging
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prepare_datasets.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "MATICUSD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
MIN_SAMPLES = 200

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare enhanced datasets for all pairs")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to process (default: {', '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes to process (default: {', '.join(DEFAULT_TIMEFRAMES)})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help=f"Minimum number of samples required (default: {MIN_SAMPLES})")
    parser.add_argument("--parallel", action="store_true",
                        help="Process pairs in parallel")
    return parser.parse_args()

def prepare_dataset(pair, timeframe, min_samples):
    """Prepare enhanced dataset for a single pair and timeframe"""
    logger.info(f"Preparing enhanced dataset for {pair} ({timeframe})")
    
    try:
        # Build command
        command = f"python prepare_enhanced_dataset.py --pair {pair} --timeframe {timeframe} --min-samples {min_samples}"
        
        # Run command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error preparing dataset for {pair} ({timeframe})")
            logger.error(f"STDERR: {stderr}")
            return False
        
        logger.info(f"Successfully prepared dataset for {pair} ({timeframe})")
        return True
    
    except Exception as e:
        logger.error(f"Exception preparing dataset for {pair} ({timeframe}): {e}")
        return False

def prepare_datasets_sequential(pairs, timeframes, min_samples):
    """Prepare datasets for all pairs sequentially"""
    success_count = 0
    total_count = len(pairs) * len(timeframes)
    
    for pair in pairs:
        for timeframe in timeframes:
            if prepare_dataset(pair, timeframe, min_samples):
                success_count += 1
    
    logger.info(f"Successfully prepared {success_count}/{total_count} datasets")
    return success_count

def prepare_datasets_parallel(pairs, timeframes, min_samples):
    """Prepare datasets for all pairs in parallel"""
    success_count = 0
    total_count = len(pairs) * len(timeframes)
    
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(pairs))) as executor:
        # Submit all tasks
        futures = []
        for pair in pairs:
            for timeframe in timeframes:
                future = executor.submit(prepare_dataset, pair, timeframe, min_samples)
                futures.append(future)
        
        # Process results as they complete
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    logger.info(f"Successfully prepared {success_count}/{total_count} datasets")
    return success_count

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info(f"Preparing enhanced datasets for {len(args.pairs)} pairs and {len(args.timeframes)} timeframes")
    
    if args.parallel:
        success_count = prepare_datasets_parallel(args.pairs, args.timeframes, args.min_samples)
    else:
        success_count = prepare_datasets_sequential(args.pairs, args.timeframes, args.min_samples)
    
    if success_count > 0:
        logger.info("Enhanced datasets prepared successfully")
    else:
        logger.error("Failed to prepare any enhanced datasets")
        sys.exit(1)

if __name__ == "__main__":
    main()