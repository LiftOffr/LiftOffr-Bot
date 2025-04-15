#!/usr/bin/env python3

"""
Fetch Extended Historical Data

This script fetches extended historical data for cryptocurrency trading pairs
from Kraken's API. It saves the data in CSV format for use in training and
backtesting ML models.

Usage:
    python fetch_extended_historical_data.py --pair SOL/USD [--interval 1h] [--days 365]
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

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
KRAKEN_REST_URL = "https://api.kraken.com/0/public"
INTERVALS = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
    "2w": 21600
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fetch Extended Historical Data")
    parser.add_argument("--pair", type=str, required=True,
                        help="Trading pair (e.g., 'SOL/USD')")
    parser.add_argument("--interval", type=str, default="1h",
                        choices=list(INTERVALS.keys()),
                        help="Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 2w)")
    parser.add_argument("--days", type=int, default=365,
                        help="Number of days of historical data to fetch")
    parser.add_argument("--start", type=str,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing file")
    return parser.parse_args()

def get_kraken_symbol(pair: str) -> str:
    """Convert trading pair to Kraken symbol format"""
    base, quote = pair.split("/")
    
    # Kraken uses X as prefix for most crypto and Z for fiat currencies
    base_prefix = "X" if base not in ["DOT", "ADA", "SOL", "LINK"] else ""
    quote_prefix = "Z" if quote == "USD" else ""
    
    return f"{base_prefix}{base}{quote_prefix}{quote}"

def fetch_ohlc_data(pair: str, interval: str, since=None) -> Tuple[List[List[float]], int]:
    """
    Fetch OHLC data from Kraken API
    
    Args:
        pair (str): Trading pair in Kraken format
        interval (str): Time interval
        since (int): Unix timestamp to fetch data after
        
    Returns:
        Tuple[List[List[float]], int]: OHLC data and last timestamp
    """
    # Convert interval to minutes
    interval_minutes = INTERVALS[interval]
    
    # Construct URL
    url = f"{KRAKEN_REST_URL}/OHLC"
    
    # Construct payload
    payload = {
        "pair": pair,
        "interval": interval_minutes
    }
    
    if since:
        payload["since"] = since
    
    # Send request
    try:
        response = requests.get(url, params=payload)
        data = response.json()
        
        if "error" in data and data["error"]:
            logger.error(f"Error fetching OHLC data: {data['error']}")
            return [], None
        
        # Extract results
        result = data["result"]
        ohlc_data = result[pair]
        last = result["last"]
        
        return ohlc_data, last
    except Exception as e:
        logger.error(f"Error fetching OHLC data: {e}")
        return [], None

def process_ohlc_data(ohlc_data: List[List[float]]) -> pd.DataFrame:
    """
    Process OHLC data into a pandas DataFrame
    
    Args:
        ohlc_data (List[List[float]]): OHLC data from Kraken API
        
    Returns:
        pd.DataFrame: Processed data
    """
    # Create DataFrame
    columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(ohlc_data, columns=columns)
    
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    
    # Convert price columns to float
    for col in ["open", "high", "low", "close", "vwap"]:
        df[col] = df[col].astype(float)
    
    # Convert volume to float
    df["volume"] = df["volume"].astype(float)
    
    # Calculate additional features
    df["price_range"] = df["high"] - df["low"]
    df["price_range_pct"] = df["price_range"] / df["low"]
    
    # Add percentage change
    df["pct_change"] = df["close"].pct_change()
    
    # Sort by datetime
    df = df.sort_values("datetime")
    
    # Drop duplicates if any
    df = df.drop_duplicates(subset=["datetime"])
    
    return df

def fetch_all_historical_data(pair: str, interval: str, days: int = 365, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetch all historical data for a pair within the given date range
    
    Args:
        pair (str): Trading pair
        interval (str): Time interval
        days (int): Number of days (used if start/end not provided)
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        pd.DataFrame: Historical data
    """
    # Convert pair to Kraken format
    kraken_pair = get_kraken_symbol(pair)
    
    # Calculate start and end timestamps
    if not end_date:
        end_date = datetime.now()
    
    if not start_date:
        start_date = end_date - timedelta(days=days)
    
    start_timestamp = int(start_date.timestamp())
    
    # Fetch data in chunks
    all_data = []
    last_timestamp = start_timestamp
    chunk_days = 30  # Fetch in chunks of 30 days
    
    while True:
        logger.info(f"Fetching data for {pair} from {datetime.fromtimestamp(last_timestamp)}")
        
        chunk_data, last = fetch_ohlc_data(kraken_pair, interval, last_timestamp)
        
        if not chunk_data or not last:
            logger.warning(f"No more data available or error occurred")
            break
        
        all_data.extend(chunk_data)
        
        # Break if we've reached the end date or there's no more data
        if last <= last_timestamp or datetime.fromtimestamp(last) >= end_date:
            break
        
        last_timestamp = last
        
        # Sleep to avoid rate limits
        time.sleep(1)
    
    # Process data
    if all_data:
        df = process_ohlc_data(all_data)
        
        # Filter to the requested date range
        df = df[(df["datetime"] >= pd.Timestamp(start_date)) & 
                (df["datetime"] <= pd.Timestamp(end_date))]
        
        return df
    else:
        return pd.DataFrame()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create the training data directory if it doesn't exist
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    # Parse start and end dates if provided
    start_date = None
    end_date = None
    
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Prepare output file
    pair_filename = args.pair.replace('/', '_')
    output_file = f"{TRAINING_DATA_DIR}/{pair_filename}_data.csv"
    
    # Check if file exists
    if os.path.exists(output_file) and not args.force:
        logger.warning(f"File {output_file} already exists. Use --force to overwrite.")
        return 1
    
    # Fetch data
    logger.info(f"Fetching {args.days} days of {args.interval} data for {args.pair}...")
    df = fetch_all_historical_data(args.pair, args.interval, args.days, start_date, end_date)
    
    if df.empty:
        logger.error(f"No data found for {args.pair}")
        return 1
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(df)} rows of {args.interval} data for {args.pair} to {output_file}")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())