#!/usr/bin/env python3
"""
Train Multi-Timeframe Models for All Pairs

This script runs the multi-timeframe trainer for all supported trading pairs.
It implements the three-phase approach:
1. Train individual models for each timeframe (15m, 1h, 4h, 1d)
2. Train unified models that combine data from all timeframes
3. Train ensemble meta-models that integrate predictions from all timeframes

Usage:
    python train_multi_timeframe_all_pairs.py [--force] [--pairs PAIR1,PAIR2,...]
                                            [--skip-individual] [--skip-unified]
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

# Constants
SUPPORTED_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multi-timeframe models for all pairs')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining of existing models')
    parser.add_argument('--pairs', type=str, default=None,
                        help='Comma-separated list of pairs to train (e.g., BTC/USD,ETH/USD)')
    parser.add_argument('--skip-individual', action='store_true',
                        help='Skip training individual timeframe models')
    parser.add_argument('--skip-unified', action='store_true',
                        help='Skip training unified models')
    parser.add_argument('--skip-ensemble', action='store_true',
                        help='Skip training ensemble models')
    return parser.parse_args()


def run_trainer(
    pair: str,
    force: bool = False,
    skip_individual: bool = False,
    skip_unified: bool = False,
    skip_ensemble: bool = False
) -> bool:
    """
    Run multi-timeframe trainer for a single pair
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        force: Whether to force retraining
        skip_individual: Whether to skip individual timeframe models
        skip_unified: Whether to skip unified models
        skip_ensemble: Whether to skip ensemble models
        
    Returns:
        True if successful, False otherwise
    """
    cmd = ['python', 'multi_timeframe_trainer.py', '--pair', pair]
    
    if force:
        cmd.append('--force')
    
    if skip_individual:
        cmd.append('--skip-individual')
    
    if skip_unified:
        cmd.append('--skip-unified')
    
    if skip_ensemble:
        cmd.append('--skip-ensemble')
    
    logging.info(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Log the output
        for line in result.stdout.splitlines():
            logging.info(line)
        
        logging.info(f"Training completed for {pair} in {elapsed_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        logging.error(f"Training failed for {pair} after {elapsed_time:.2f} seconds")
        logging.error(f"Command output:")
        
        for line in e.output.splitlines():
            logging.error(line)
        
        return False
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        logging.error(f"Error running trainer for {pair} after {elapsed_time:.2f} seconds: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create directory for models if it doesn't exist
    os.makedirs('ml_models', exist_ok=True)
    
    # Determine which pairs to process
    if args.pairs is not None:
        pairs_to_process = args.pairs.split(',')
        # Validate pairs
        for pair in pairs_to_process:
            if pair not in SUPPORTED_PAIRS:
                logging.warning(f"Unsupported pair: {pair}")
    else:
        pairs_to_process = SUPPORTED_PAIRS
    
    # Process each pair
    results = {}
    for pair in pairs_to_process:
        success = run_trainer(
            pair,
            force=args.force,
            skip_individual=args.skip_individual,
            skip_unified=args.skip_unified,
            skip_ensemble=args.skip_ensemble
        )
        results[pair] = success
    
    # Print summary
    logging.info("=== Training Summary ===")
    total_success = 0
    for pair, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"{pair}: {status}")
        if success:
            total_success += 1
    
    # Overall success
    logging.info(f"Successfully trained {total_success}/{len(pairs_to_process)} pairs")
    
    return total_success == len(pairs_to_process)


if __name__ == "__main__":
    main()