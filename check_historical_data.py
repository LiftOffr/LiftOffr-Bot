#!/usr/bin/env python3

"""
Check Historical Data

This script checks if historical data is available for all trading pairs
and fetches any missing data.

Usage:
    python check_historical_data.py [--pairs PAIRS]
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TRAINING_DATA_DIR = "training_data"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check Historical Data")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch missing data")
    parser.add_argument("--force", action="store_true",
                        help="Force fetch data even if it exists")
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

def check_data_file(pair) -> Dict[str, Any]:
    """
    Check if data file exists for a specific pair and analyze its contents
    
    Returns:
        dict: Data file status
    """
    pair_filename = pair.replace('/', '_')
    data_file = f"{TRAINING_DATA_DIR}/{pair_filename}_data.csv"
    
    result = {
        "exists": os.path.exists(data_file),
        "file_path": data_file,
        "rows": 0,
        "date_range": None,
        "timeframe": None,
        "needs_update": True
    }
    
    if result["exists"]:
        try:
            # Load data file
            df = pd.read_csv(data_file)
            result["rows"] = len(df)
            
            # Check date range
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                start_date = df["datetime"].min()
                end_date = df["datetime"].max()
                result["date_range"] = {
                    "start": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": end_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "days": (end_date - start_date).days
                }
                
                # Determine if data needs update (if last data point is more than 1 day old)
                now = datetime.now()
                result["needs_update"] = (now - end_date).days > 1
            
            # Determine timeframe
            if len(df) > 1 and "datetime" in df.columns:
                time_diffs = []
                for i in range(1, min(100, len(df))):
                    time_diff = (df["datetime"].iloc[i] - df["datetime"].iloc[i-1]).total_seconds() / 60
                    time_diffs.append(time_diff)
                avg_diff = sum(time_diffs) / len(time_diffs)
                
                if avg_diff < 5:
                    result["timeframe"] = "1m"
                elif avg_diff < 20:
                    result["timeframe"] = "5m"
                elif avg_diff < 70:
                    result["timeframe"] = "15m"
                elif avg_diff < 150:
                    result["timeframe"] = "1h"
                elif avg_diff < 300:
                    result["timeframe"] = "4h"
                else:
                    result["timeframe"] = "1d"
        
        except Exception as e:
            logger.error(f"Error analyzing data file for {pair}: {e}")
    
    return result

def fetch_historical_data(pair, force=False) -> bool:
    """
    Fetch historical data for a specific pair
    
    Args:
        pair (str): Trading pair
        force (bool): Force fetch even if data exists
        
    Returns:
        bool: Success/failure
    """
    logger.info(f"Fetching historical data for {pair}...")
    
    # Check if data already exists
    if not force:
        data_status = check_data_file(pair)
        if data_status["exists"] and not data_status["needs_update"]:
            logger.info(f"Historical data for {pair} is already up-to-date")
            return True
    
    # Ensure directory exists
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    # Run fetch command
    cmd = ["python", "fetch_extended_historical_data.py", "--pair", pair]
    
    if force:
        cmd.append("--force")
    
    result = run_command(cmd, f"Fetching historical data for {pair}")
    
    if not result:
        logger.error(f"Failed to fetch historical data for {pair}")
        return False
    
    # Verify data was fetched
    data_status = check_data_file(pair)
    if not data_status["exists"]:
        logger.error(f"Data file for {pair} still doesn't exist after fetch")
        return False
    
    logger.info(f"Successfully fetched historical data for {pair}")
    return True

def check_all_pairs(pairs) -> Dict[str, Dict[str, Any]]:
    """
    Check data status for all pairs
    
    Returns:
        dict: Status for each pair
    """
    results = {}
    
    for pair in pairs:
        results[pair] = check_data_file(pair)
    
    return results

def main():
    """Main function"""
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    logger.info(f"Checking historical data for {len(pairs)} pairs...")
    
    # Check status for all pairs
    results = check_all_pairs(pairs)
    
    # Display summary
    print("\nHistorical Data Summary:")
    print("=" * 100)
    print(f"{'PAIR':<10} | {'EXISTS':<8} | {'ROWS':<8} | {'TIMEFRAME':<10} | {'START DATE':<20} | {'END DATE':<20} | {'DAYS':<8} | {'NEEDS UPDATE':<12}")
    print("-" * 100)
    
    missing_pairs = []
    outdated_pairs = []
    
    for pair, status in results.items():
        exists = status.get("exists", False)
        rows = status.get("rows", 0)
        timeframe = status.get("timeframe", "N/A")
        date_range = status.get("date_range", {})
        
        if date_range is None:
            date_range = {}
            
        start_date = date_range.get("start", "N/A") if date_range else "N/A"
        end_date = date_range.get("end", "N/A") if date_range else "N/A"
        days = date_range.get("days", 0) if date_range else 0
        needs_update = status.get("needs_update", True)
        
        print(f"{pair:<10} | {str(exists):<8} | {rows:<8} | {timeframe:<10} | {start_date:<20} | {end_date:<20} | {days:<8} | {str(needs_update):<12}")
        
        if not exists:
            missing_pairs.append(pair)
        elif needs_update:
            outdated_pairs.append(pair)
    
    print("=" * 100)
    
    # Fetch missing or outdated data if requested
    if args.fetch:
        pairs_to_fetch = missing_pairs + outdated_pairs
        if args.force:
            pairs_to_fetch = pairs
        
        if pairs_to_fetch:
            print(f"\nFetching data for {len(pairs_to_fetch)} pairs...")
            
            for pair in pairs_to_fetch:
                fetch_historical_data(pair, args.force)
            
            # Check status again after fetching
            results = check_all_pairs(pairs)
            
            # Display updated summary
            print("\nUpdated Historical Data Summary:")
            print("=" * 100)
            print(f"{'PAIR':<10} | {'EXISTS':<8} | {'ROWS':<8} | {'TIMEFRAME':<10} | {'START DATE':<20} | {'END DATE':<20} | {'DAYS':<8} | {'NEEDS UPDATE':<12}")
            print("-" * 100)
            
            for pair, status in results.items():
                exists = status.get("exists", False)
                rows = status.get("rows", 0)
                timeframe = status.get("timeframe", "N/A")
                date_range = status.get("date_range", {})
                
                if date_range is None:
                    date_range = {}
                
                start_date = date_range.get("start", "N/A") if date_range else "N/A"
                end_date = date_range.get("end", "N/A") if date_range else "N/A"
                days = date_range.get("days", 0) if date_range else 0
                needs_update = status.get("needs_update", True)
                
                print(f"{pair:<10} | {str(exists):<8} | {rows:<8} | {timeframe:<10} | {start_date:<20} | {end_date:<20} | {days:<8} | {str(needs_update):<12}")
            
            print("=" * 100)
        else:
            print("\nAll data is up-to-date. No need to fetch.")
    else:
        # Show recommendations
        if missing_pairs or outdated_pairs:
            print("\nRecommendations:")
            if missing_pairs:
                missing_str = ",".join(missing_pairs)
                print(f"  Fetch missing data for {len(missing_pairs)} pairs:")
                print(f"  python check_historical_data.py --pairs {missing_str} --fetch")
            
            if outdated_pairs:
                outdated_str = ",".join(outdated_pairs)
                print(f"  Update outdated data for {len(outdated_pairs)} pairs:")
                print(f"  python check_historical_data.py --pairs {outdated_str} --fetch")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())