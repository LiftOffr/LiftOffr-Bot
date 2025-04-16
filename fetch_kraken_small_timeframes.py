#!/usr/bin/env python3
"""
Fetch 1m and 5m Timeframe Data from Kraken

This script fetches historical price data in 1-minute and 5-minute timeframes
for all supported trading pairs from the Kraken API.

Usage:
    python fetch_kraken_small_timeframes.py [--days DAYS] [--pairs PAIR1,PAIR2,...]
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

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
TIMEFRAMES = ['1m', '5m']
BASE_URL = 'https://api.kraken.com/0/public'
INTERVALS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}
KRAKEN_PAIR_MAPPING = {
    'BTC/USD': 'XBTUSD',
    'ETH/USD': 'ETHUSD',
    'SOL/USD': 'SOLUSD',
    'ADA/USD': 'ADAUSD',
    'DOT/USD': 'DOTUSD',
    'LINK/USD': 'LINKUSD',
    'AVAX/USD': 'AVAXUSD',
    'MATIC/USD': 'MATICUSD',
    'UNI/USD': 'UNIUSD',
    'ATOM/USD': 'ATOMUSD'
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fetch Kraken historical data')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data to fetch')
    parser.add_argument('--pairs', type=str, default=None,
                        help='Comma-separated list of pairs to fetch (e.g., BTC/USD,ETH/USD)')
    return parser.parse_args()


def get_kraken_data(
    pair: str,
    timeframe: str,
    since: Optional[int] = None,
    until: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from Kraken API
    
    Args:
        pair: Trading pair (e.g., XBTUSD)
        timeframe: Timeframe (e.g., 1m, 5m)
        since: Start time in Unix timestamp (optional)
        until: End time in Unix timestamp (optional)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    # Convert timeframe to interval
    interval = INTERVALS.get(timeframe)
    if interval is None:
        logging.error(f"Invalid timeframe: {timeframe}")
        return None
    
    # Construct request URL
    url = f"{BASE_URL}/OHLC"
    
    params = {
        'pair': pair,
        'interval': interval
    }
    
    if since is not None:
        params['since'] = since
    
    # Make API request
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'error' in data and data['error']:
            logging.error(f"Kraken API error: {data['error']}")
            return None
        
        # Extract result data
        result = None
        for key in data['result']:
            if key != 'last':
                result = data['result'][key]
                break
        
        if result is None:
            logging.error(f"No data returned for {pair} ({timeframe})")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(
            result,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            df[col] = df[col].astype(float)
        
        # Convert count to int
        df['count'] = df['count'].astype(int)
        
        # Filter by until timestamp if provided
        if until is not None:
            df = df[df['timestamp'].astype(int) / 10**9 <= until]
        
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {pair} ({timeframe}): {e}")
        return None


def fetch_historical_data(
    pair: str,
    timeframe: str,
    days: int = 30
) -> bool:
    """
    Fetch historical data for a pair and timeframe
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1m, 5m)
        days: Number of days of historical data to fetch
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Fetching {timeframe} data for {pair} (last {days} days)...")
    
    # Map standard pair to Kraken pair format
    kraken_pair = KRAKEN_PAIR_MAPPING.get(pair)
    if kraken_pair is None:
        logging.error(f"Unknown pair: {pair}")
        return False
    
    # Calculate start and end times
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    
    # Fetch data
    df = get_kraken_data(kraken_pair, timeframe, start_time, end_time)
    if df is None:
        return False
    
    # Save to CSV
    pair_symbol = pair.replace('/', '_')
    file_path = f"historical_data/{pair_symbol}_{timeframe}.csv"
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {len(df)} {timeframe} candles for {pair} to {file_path}")
    
    return True


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create directory for historical data if it doesn't exist
    os.makedirs('historical_data', exist_ok=True)
    
    # Determine which pairs to process
    if args.pairs is not None:
        pairs_to_process = args.pairs.split(',')
        # Validate pairs
        for pair in pairs_to_process:
            if pair not in SUPPORTED_PAIRS:
                logging.warning(f"Unsupported pair: {pair}")
                pairs_to_process.remove(pair)
    else:
        pairs_to_process = SUPPORTED_PAIRS.copy()
    
    # Fetch data for each pair and timeframe
    results = {}
    for pair in pairs_to_process:
        pair_results = {}
        for timeframe in TIMEFRAMES:
            success = fetch_historical_data(pair, timeframe, args.days)
            pair_results[timeframe] = success
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        results[pair] = pair_results
    
    # Print summary
    logging.info("=== Fetch Summary ===")
    all_success = True
    for pair, pair_results in results.items():
        for timeframe, success in pair_results.items():
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"{pair} ({timeframe}): {status}")
            if not success:
                all_success = False
    
    return all_success


if __name__ == "__main__":
    main()