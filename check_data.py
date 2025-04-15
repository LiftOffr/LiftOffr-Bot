#!/usr/bin/env python3

"""
Simple script to check if we have historical data files for our trading pairs.
"""

import os
import sys

TRAINING_DATA_DIR = "training_data"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def main():
    """Check for data files and print status"""
    print("\nChecking for historical data files:")
    print("=" * 60)
    print(f"{'PAIR':<10} | {'EXISTS':<8} | {'FILE PATH'}")
    print("-" * 60)
    
    missing_pairs = []
    
    for pair in DEFAULT_PAIRS:
        pair_filename = pair.replace('/', '_')
        data_file = f"{TRAINING_DATA_DIR}/{pair_filename}_data.csv"
        exists = os.path.exists(data_file)
        
        print(f"{pair:<10} | {str(exists):<8} | {data_file}")
        
        if not exists:
            missing_pairs.append(pair)
    
    print("=" * 60)
    
    if missing_pairs:
        print(f"\nMissing data for {len(missing_pairs)} pairs:")
        for pair in missing_pairs:
            print(f"  {pair}")
        
        print("\nTo fetch missing data, run:")
        for pair in missing_pairs:
            print(f"  python fetch_extended_historical_data.py --pair {pair}")
    else:
        print("\nAll data files exist!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())