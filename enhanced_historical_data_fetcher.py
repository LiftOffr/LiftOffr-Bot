#!/usr/bin/env python3
"""
Enhanced Historical Data Fetcher for Multiple Crypto Assets

This script fetches historical OHLCV data for multiple cryptocurrencies
and timeframes from Kraken to use in backtesting and ML model training.
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
KRAKEN_API_URL = "https://api.kraken.com/0/public"
DATA_DIR = "historical_data"
PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
TIMEFRAMES = {
    "1h": 60,     # 1 hour in minutes
    "4h": 240,    # 4 hours in minutes
    "1d": 1440    # 1 day in minutes
}
# Maximum number of candles per request (Kraken limit)
MAX_CANDLES_PER_REQUEST = 720


def ensure_data_directories():
    """Create data directories if they don't exist"""
    for pair in PAIRS:
        pair_dir = os.path.join(DATA_DIR, pair)
        os.makedirs(pair_dir, exist_ok=True)


def fetch_ohlc_data(pair, interval_minutes, since=None):
    """
    Fetch OHLC data from Kraken API
    
    Args:
        pair (str): Trading pair
        interval_minutes (int): Candle interval in minutes
        since (int, optional): Timestamp to start from
        
    Returns:
        list: OHLC data
    """
    endpoint = f"{KRAKEN_API_URL}/OHLC"
    
    params = {
        "pair": pair,
        "interval": interval_minutes
    }
    
    if since:
        params["since"] = since
    
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        if 'error' in data and data['error']:
            logger.error(f"Kraken API error: {data['error']}")
            return []
        
        if 'result' in data:
            # Extract the actual data (the first key is the pair name)
            pair_data = list(data['result'].values())[0]
            return pair_data
        
        return []
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return []


def process_ohlc_data(ohlc_data):
    """
    Process OHLC data into a pandas DataFrame
    
    Args:
        ohlc_data (list): OHLC data from Kraken
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if not ohlc_data:
        return pd.DataFrame()
    
    # Create DataFrame from the data
    df = pd.DataFrame(ohlc_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        df[col] = df[col].astype(float)
    
    return df


def fetch_historical_data(pair, interval_name, days_back=30):
    """
    Fetch and save historical data for a specific pair and interval
    
    Args:
        pair (str): Trading pair
        interval_name (str): Interval name (e.g., '1h', '4h', '1d')
        days_back (int): Number of days to fetch
    """
    interval_minutes = TIMEFRAMES[interval_name]
    
    # Calculate the start timestamp (days_back days ago from now)
    start_time = datetime.now() - timedelta(days=days_back)
    start_timestamp = int(start_time.timestamp())
    
    logger.info(f"Fetching {interval_name} data for {pair} (last {days_back} days)")
    
    all_data = []
    last_timestamp = start_timestamp
    
    # Fetch data in batches due to API limits
    while True:
        ohlc_data = fetch_ohlc_data(pair, interval_minutes, since=last_timestamp)
        
        if not ohlc_data:
            break
            
        all_data.extend(ohlc_data)
        logger.info(f"Fetched {len(ohlc_data)} {interval_name} candles for {pair}")
        
        # Update last timestamp to get the next batch
        last_timestamp = int(ohlc_data[-1][0]) + interval_minutes * 60
        
        # Check if we've reached current time
        if last_timestamp >= int(datetime.now().timestamp()):
            break
            
        # Respect API rate limits
        time.sleep(2)
    
    # Process and save the data
    if all_data:
        df = process_ohlc_data(all_data)
        
        if not df.empty:
            # Sort by timestamp and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            # Save to CSV
            output_file = os.path.join(DATA_DIR, pair, f"{pair}_{interval_name}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} rows to {output_file}")
            
            return df
    
    logger.warning(f"No data fetched for {pair} {interval_name}")
    return pd.DataFrame()


def merge_timeframes(pair):
    """
    Merge different timeframes for a pair to create a multi-timeframe dataset
    
    Args:
        pair (str): Trading pair
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    logger.info(f"Creating multi-timeframe dataset for {pair}")
    
    # Base it on the 1h timeframe
    base_file = os.path.join(DATA_DIR, pair, f"{pair}_1h.csv")
    if not os.path.exists(base_file):
        logger.error(f"Base file {base_file} not found")
        return pd.DataFrame()
    
    base_df = pd.read_csv(base_file)
    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
    base_df.set_index('timestamp', inplace=True)
    
    # Add higher timeframes
    for timeframe in ['4h', '1d']:
        tf_file = os.path.join(DATA_DIR, pair, f"{pair}_{timeframe}.csv")
        if not os.path.exists(tf_file):
            continue
        
        tf_df = pd.read_csv(tf_file)
        tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])
        tf_df.set_index('timestamp', inplace=True)
        
        # Rename columns to avoid conflicts
        tf_df = tf_df.add_prefix(f"{timeframe}_")
        
        # Forward fill the higher timeframe data
        tf_df = tf_df.resample('1H').ffill()
        
        # Merge with base dataframe
        base_df = base_df.join(tf_df, how='left')
    
    # Reset index and save
    base_df = base_df.reset_index()
    
    # Save multi-timeframe data
    output_file = os.path.join(DATA_DIR, pair, f"{pair}_multi_timeframe.csv")
    base_df.to_csv(output_file, index=False)
    logger.info(f"Saved multi-timeframe dataset to {output_file}")
    
    return base_df


def main():
    """Main function to fetch historical data for all pairs and timeframes"""
    # Ensure data directories exist
    ensure_data_directories()
    
    # Fetch data for each pair and timeframe
    for pair in PAIRS:
        for interval_name in TIMEFRAMES:
            # Different historical windows for different timeframes
            if interval_name == '1h':
                days_back = 30  # 1 month of hourly data
            elif interval_name == '4h':
                days_back = 90  # 3 months of 4h data
            else:  # 1d
                days_back = 365  # 1 year of daily data
                
            fetch_historical_data(pair, interval_name, days_back)
        
        # Create multi-timeframe dataset
        merge_timeframes(pair)
    
    logger.info("All historical data fetched and processed successfully")


if __name__ == "__main__":
    main()