#!/usr/bin/env python3
"""
Fetch 15-minute Historical Data from Kraken API

This script fetches 15-minute OHLCV data from the Kraken API for multiple
cryptocurrency pairs and saves it to the historical_data directory.

Kraken API documentation: https://docs.kraken.com/rest/#operation/getOHLCData

Usage:
    python fetch_kraken_15m_data.py --pair BTC/USD --days 30
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kraken_data_fetch.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)

# Kraken API endpoints
KRAKEN_API_URL = "https://api.kraken.com/0/public"
OHLC_ENDPOINT = "/OHLC"

# Default pairs to fetch (can be modified)
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Timeframe intervals (in minutes)
INTERVALS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1w': 10080
}

# Kraken asset pair mapping (Kraken uses different symbols)
KRAKEN_PAIR_MAPPING = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
    "ADA/USD": "ADAUSD",
    "DOT/USD": "DOTUSD",
    "LINK/USD": "LINKUSD",
    "AVAX/USD": "AVAXUSD",
    "MATIC/USD": "MATICUSD",
    "UNI/USD": "UNIUSD",
    "ATOM/USD": "ATOMUSD"
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fetch 15-minute historical data from Kraken API")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help=f"Trading pair to fetch data for (default: BTC/USD)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated list of trading pairs to fetch data for")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help=f"Timeframe (default: 15m, options: {', '.join(INTERVALS.keys())})")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days of historical data to fetch (default: 30)")
    parser.add_argument("--batch", type=int, default=720,
                        help="Number of candles to fetch per request (default: 720, max: 720)")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Fetch data for all supported pairs")
    return parser.parse_args()

def get_kraken_pair_name(pair: str) -> str:
    """Convert standard pair name to Kraken pair name"""
    return KRAKEN_PAIR_MAPPING.get(pair, pair.replace("/", ""))

def fetch_ohlcv_data(
    pair: str, 
    timeframe: str = '15m', 
    since: Optional[int] = None, 
    count: int = 720
) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    """
    Fetch OHLCV data from Kraken API
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '15m', '1h')
        since: Unix timestamp to start from
        count: Number of candles to fetch (max 720)
        
    Returns:
        Tuple of (DataFrame with OHLCV data, Unix timestamp of last candle)
    """
    # Get interval in minutes
    interval = INTERVALS.get(timeframe, 15)
    
    # Get Kraken pair name
    kraken_pair = get_kraken_pair_name(pair)
    
    # Build URL and parameters
    url = f"{KRAKEN_API_URL}{OHLC_ENDPOINT}"
    params = {
        "pair": kraken_pair,
        "interval": interval
    }
    
    if since is not None:
        params["since"] = since
        
    # Log full request for debugging
    logger.info(f"API Request: {url} with params {params}")
    
    # Make request with retries
    max_retries = 3
    retry_delay = 5  # seconds
    
    for retry in range(max_retries):
        try:
            logger.info(f"Fetching {timeframe} data for {pair} (Kraken pair: {kraken_pair})")
            response = requests.get(url, params=params)
            
            # Log response status and headers for debugging
            logger.info(f"Response status: {response.status_code}")
            
            # Check if response is successful
            response.raise_for_status()
            
            # Log raw response content for debugging
            logger.debug(f"Response content: {response.text[:200]}...")
            
            # Parse response
            data = response.json()
            
            # Check for errors
            if data.get("error") and len(data["error"]) > 0:
                logger.error(f"Kraken API error: {data['error']}")
                return None, None
            
            # Get result
            result = data.get("result", {})
            
            # Log all keys in the result to identify the correct structure
            logger.info(f"Result keys: {list(result.keys())}")
            
            # Extract OHLCV data and last timestamp
            if kraken_pair in result:
                ohlcv_data = result.get(kraken_pair, [])
            else:
                # Try to find the first key that contains OHLCV data
                found_key = None
                for key in result.keys():
                    if isinstance(result[key], list) and len(result[key]) > 0:
                        found_key = key
                        break
                
                if found_key:
                    logger.info(f"Using key {found_key} for OHLCV data instead of {kraken_pair}")
                    ohlcv_data = result.get(found_key, [])
                else:
                    ohlcv_data = []
            
            last = result.get("last", None)
            
            # Check if data is available
            if not ohlcv_data:
                logger.warning(f"No OHLCV data returned for {pair} ({timeframe})")
                return None, None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=[
                "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Convert numeric columns
            for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"Fetched {len(df)} candles for {pair} ({timeframe})")
            
            return df, last
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            
            if retry < max_retries - 1:
                retry_delay_actual = retry_delay * (retry + 1)
                logger.info(f"Retrying in {retry_delay_actual} seconds... (attempt {retry + 1}/{max_retries})")
                time.sleep(retry_delay_actual)
            else:
                logger.error("Max retries reached")
                return None, None
        
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            logger.exception("Exception details:")
            return None, None
    
    return None, None

def fetch_complete_history(pair: str, timeframe: str, days: int, batch_size: int = 720) -> Optional[pd.DataFrame]:
    """
    Fetch complete history for a trading pair
    
    Args:
        pair: Trading pair (e.g., 'BTC/USD')
        timeframe: Timeframe (e.g., '15m', '1h')
        days: Number of days of historical data to fetch
        batch_size: Number of candles per request (max 720)
        
    Returns:
        DataFrame with complete OHLCV history
    """
    # Calculate start time
    interval_minutes = INTERVALS.get(timeframe, 15)
    total_intervals = int((days * 24 * 60) / interval_minutes)
    
    logger.info(f"Fetching {total_intervals} {timeframe} intervals for {pair} ({days} days)")
    
    # Initialize variables
    all_data = []
    last_timestamp = None
    
    # Fetch data in batches
    while True:
        # Fetch batch
        df_batch, last = fetch_ohlcv_data(pair, timeframe, last_timestamp, batch_size)
        
        if df_batch is None or df_batch.empty:
            if all_data:
                # We already have some data, so continue with what we have
                logger.warning(f"Couldn't fetch more data for {pair} ({timeframe}), continuing with {len(all_data)} batches")
                break
            else:
                # No data at all
                logger.error(f"Failed to fetch any data for {pair} ({timeframe})")
                return None
        
        # Append batch to all data
        all_data.append(df_batch)
        
        # Set last timestamp for next batch
        last_timestamp = last
        
        # Check if we have enough data
        total_rows = sum(len(df) for df in all_data)
        if total_rows >= total_intervals:
            logger.info(f"Fetched enough data ({total_rows} candles) for {pair} ({timeframe})")
            break
        
        # Check if we've reached the beginning of data
        if last is None:
            logger.info(f"Reached the beginning of available data for {pair} ({timeframe})")
            break
        
        # Add delay to avoid rate limiting
        time.sleep(2)
    
    # Combine all batches
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    combined_df.drop_duplicates(subset=["timestamp"], inplace=True)
    
    # Sort by timestamp
    combined_df.sort_values("timestamp", inplace=True)
    
    # Reset index
    combined_df.reset_index(drop=True, inplace=True)
    
    # Truncate to requested number of days
    if len(combined_df) > total_intervals:
        combined_df = combined_df.tail(total_intervals)
    
    logger.info(f"Final dataset for {pair} ({timeframe}) has {len(combined_df)} candles")
    
    return combined_df

def save_to_file(df: pd.DataFrame, pair: str, timeframe: str) -> str:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        pair: Trading pair
        timeframe: Timeframe
        
    Returns:
        Path to saved file
    """
    # Create filename
    pair_clean = pair.replace("/", "_")
    filename = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}.csv"
    
    # Save to file
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(df)} candles to {filename}")
    
    return filename

def fetch_and_save_data(pair: str, timeframe: str, days: int, batch_size: int) -> Optional[str]:
    """
    Fetch and save data for a trading pair
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        days: Number of days of historical data
        batch_size: Number of candles per request
        
    Returns:
        Path to saved file
    """
    # Fetch data
    df = fetch_complete_history(pair, timeframe, days, batch_size)
    
    if df is None or df.empty:
        logger.error(f"No data fetched for {pair} ({timeframe})")
        return None
    
    # Save to file
    return save_to_file(df, pair, timeframe)

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine pairs to fetch
    if args.all:
        pairs = DEFAULT_PAIRS
    elif args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",")]
    else:
        pairs = [args.pair]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("FETCH 15-MINUTE HISTORICAL DATA FROM KRAKEN API")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Batch Size: {args.batch}")
    logger.info("=" * 80 + "\n")
    
    # Fetch data for each pair
    successful_fetches = 0
    failed_fetches = 0
    
    for i, pair in enumerate(pairs):
        logger.info(f"Processing pair {i+1}/{len(pairs)}: {pair}")
        
        # Fetch and save data
        file_path = fetch_and_save_data(pair, args.timeframe, args.days, args.batch)
        
        if file_path:
            successful_fetches += 1
        else:
            failed_fetches += 1
        
        # Add delay between pairs to avoid rate limiting
        if i < len(pairs) - 1:
            delay = 5
            logger.info(f"Waiting {delay} seconds before next pair...")
            time.sleep(delay)
    
    # Print summary
    logger.info("\nFetch Summary:")
    logger.info(f"  Successfully fetched: {successful_fetches}/{len(pairs)} pairs")
    
    if failed_fetches > 0:
        logger.warning(f"  Failed fetches: {failed_fetches}/{len(pairs)} pairs")
    
    return 0 if successful_fetches > 0 else 1

if __name__ == "__main__":
    sys.exit(main())