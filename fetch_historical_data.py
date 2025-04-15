#!/usr/bin/env python3
"""
Fetch Historical Data

This script fetches historical data for specified trading pairs from Kraken API.
The data is saved in CSV format for later use in model training.

Usage:
    python fetch_historical_data.py --pair PAIR --timeframe TIMEFRAME [--days DAYS] [--output OUTPUT]

Example:
    python fetch_historical_data.py --pair BTC/USD --timeframe 1h --days 365 --output historical_data/BTC_USD_1h.csv
"""
import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("historical_data_fetch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("historical_data_fetch")

# Constants
KRAKEN_API_URL = "https://api.kraken.com/0/public"
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "7d": 604800,
    "15d": 1296000
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch historical data from Kraken API.')
    parser.add_argument('--pair', type=str, required=True, help='Trading pair (e.g., BTC/USD)')
    parser.add_argument('--timeframe', type=str, required=True, help='Timeframe (e.g., 1m, 5m, 15m, 30m, 1h, 4h, 1d, 7d, 15d)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to fetch')
    parser.add_argument('--output', type=str, help='Output file path (default: historical_data/{PAIR}_{TIMEFRAME}.csv)')
    return parser.parse_args()


def format_pair_for_kraken(pair: str) -> str:
    """Format trading pair for Kraken API."""
    # Replace / with nothing, e.g., BTC/USD -> BTCUSD
    return pair.replace("/", "")


def fetch_ohlc_data(pair: str, timeframe: str, since: int = None) -> Tuple[Optional[List[List]], Optional[int]]:
    """
    Fetch OHLC data from Kraken API.
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1h, 4h, 1d)
        since: Unix timestamp for start time (optional)
        
    Returns:
        Tuple of (data, last_timestamp) or (None, None) if error
    """
    kraken_pair = format_pair_for_kraken(pair)
    url = f"{KRAKEN_API_URL}/OHLC"
    
    params = {
        "pair": kraken_pair,
        "interval": TIMEFRAME_SECONDS.get(timeframe, 3600) // 60  # Convert seconds to minutes
    }
    
    if since is not None:
        params["since"] = since
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data and data["error"]:
            logger.error(f"API error: {data['error']}")
            return None, None
        
        if "result" not in data:
            logger.error("No result in API response")
            return None, None
        
        # Extract result for the pair
        result_key = list(data["result"].keys())[0] if isinstance(data["result"], dict) else None
        
        if result_key and result_key in data["result"]:
            ohlc_data = data["result"][result_key]
            last = data["result"].get("last", None)
            return ohlc_data, last
        
        logger.error(f"Unexpected API response format: {data}")
        return None, None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, None


def fetch_all_historical_data(pair: str, timeframe: str, days: int = 365) -> Optional[List[List]]:
    """
    Fetch all historical data for a trading pair and timeframe.
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1h, 4h, 1d)
        days: Number of days of historical data to fetch
        
    Returns:
        List of OHLC data or None if error
    """
    # Calculate start time (current time - days)
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())
    
    all_data = []
    last = start_time
    
    while True:
        logger.info(f"Fetching data for {pair} ({timeframe}) since {last}")
        
        data, new_last = fetch_ohlc_data(pair, timeframe, last)
        
        if data is None:
            if all_data:
                logger.warning("Failed to fetch more data, returning what we have so far")
                break
            else:
                logger.error("Failed to fetch any data")
                return None
        
        # Add new data
        all_data.extend(data)
        
        # Check if we got all data
        if new_last is None or new_last <= last:
            logger.info("No more data to fetch")
            break
        
        # Update last
        last = new_last
        
        # Sleep to avoid rate limits
        time.sleep(2)
    
    # Sort data by timestamp
    all_data.sort(key=lambda x: x[0])
    
    # Remove duplicates
    seen = set()
    unique_data = []
    
    for item in all_data:
        if item[0] not in seen:
            seen.add(item[0])
            unique_data.append(item)
    
    logger.info(f"Fetched {len(unique_data)} unique data points")
    
    return unique_data


def save_data_to_csv(data: List[List], output_file: str) -> bool:
    """
    Save data to CSV file.
    
    Args:
        data: List of OHLC data
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
        )
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(data)} data points to {output_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate timeframe
    if args.timeframe not in TIMEFRAME_SECONDS:
        logger.error(f"Invalid timeframe: {args.timeframe}. Supported timeframes: {list(TIMEFRAME_SECONDS.keys())}")
        return 1
    
    # Set default output file if not provided
    if not args.output:
        pair_filename = args.pair.replace("/", "_")
        args.output = f"historical_data/{pair_filename}_{args.timeframe}.csv"
    
    # Create historical_data directory if it doesn't exist
    os.makedirs("historical_data", exist_ok=True)
    
    # Fetch data
    data = fetch_all_historical_data(args.pair, args.timeframe, args.days)
    if data is None:
        return 1
    
    # Save data
    if not save_data_to_csv(data, args.output):
        return 1
    
    logger.info(f"Successfully fetched historical data for {args.pair} ({args.timeframe})")
    return 0


if __name__ == "__main__":
    sys.exit(main())