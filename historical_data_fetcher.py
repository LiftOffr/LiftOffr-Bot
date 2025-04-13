#!/usr/bin/env python3
"""
Historical Data Fetcher for Kraken Trading Bot

This module fetches extended historical price data from Kraken API
for training ML models with more comprehensive data.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback
from config import TRADING_PAIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}
MAX_REQUESTS_PER_TIMEFRAME = 10  # To avoid rate limiting
RETRY_DELAY = 5  # seconds
MAX_RETRIES = 3

def ensure_data_directory():
    """Ensure the data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")

def get_unix_timestamp(dt):
    """Convert datetime to Unix timestamp"""
    return int(dt.timestamp())

def get_start_timestamp(days_back=365):
    """Get start timestamp for data fetching"""
    start_date = datetime.now() - timedelta(days=days_back)
    return get_unix_timestamp(start_date)

def fetch_ohlc_data(trading_pair, interval, since=None):
    """
    Fetch OHLC data from Kraken API
    
    Args:
        trading_pair (str): Trading pair (e.g., 'SOLUSD')
        interval (int): Interval in seconds
        since (int, optional): Start timestamp
        
    Returns:
        list: OHLC data
    """
    # Format trading pair for Kraken API (e.g., SOLUSD -> SOL/USD)
    if "/" not in trading_pair:
        base = trading_pair[:-3]
        quote = trading_pair[-3:]
        pair = f"{base}/{quote}"
    else:
        pair = trading_pair
    
    # Remove slash for the API request
    kraken_pair = pair.replace("/", "")
    
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": kraken_pair,
        "interval": interval // 60  # Convert seconds to minutes for Kraken API
    }
    
    if since:
        params["since"] = since
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Fetching {pair} data for {interval}s timeframe since {since}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data and data["error"]:
                logger.error(f"API Error: {data['error']}")
                time.sleep(RETRY_DELAY)
                continue
                
            result = data["result"]
            # Get first key (should be the pair name)
            pair_key = next(iter(result.keys() - ["last"]))
            return result[pair_key], result.get("last")
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            traceback.print_exc()
            time.sleep(RETRY_DELAY)
    
    logger.error(f"Failed to fetch data after {MAX_RETRIES} attempts")
    return [], None

def process_ohlc_data(ohlc_data):
    """
    Process OHLC data into a pandas DataFrame
    
    Args:
        ohlc_data (list): List of OHLC data from Kraken
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(ohlc_data, columns=columns)
    
    # Convert timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")
    
    # Convert price columns to float
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    
    # Sort by timestamp
    df = df.sort_index()
    
    return df

def fetch_historical_data(trading_pair=TRADING_PAIR, timeframes=None, days_back=365):
    """
    Fetch historical data for multiple timeframes
    
    Args:
        trading_pair (str): Trading pair to fetch data for
        timeframes (list, optional): List of timeframes to fetch
        days_back (int, optional): Number of days to fetch data for
        
    Returns:
        dict: Dictionary of DataFrame for each timeframe
    """
    ensure_data_directory()
    
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    
    result = {}
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            logger.warning(f"Invalid timeframe: {tf}")
            continue
        
        interval = TIMEFRAMES[tf]
        filename = f"{DATA_DIR}/{trading_pair.replace('/', '')}-{tf}.csv"
        
        # Check if we already have this data file
        if os.path.exists(filename):
            logger.info(f"Loading existing data for {tf} timeframe")
            try:
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                result[tf] = df
                continue
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                # If we fail to load, we'll fetch new data
        
        # Fetch new data
        start_ts = get_start_timestamp(days_back)
        all_data = []
        last_ts = start_ts
        
        for _ in range(MAX_REQUESTS_PER_TIMEFRAME):
            ohlc_data, last = fetch_ohlc_data(trading_pair, interval, last_ts)
            
            if not ohlc_data:
                break
                
            all_data.extend(ohlc_data)
            
            # If we've reached current time, break
            if not last or last == last_ts:
                break
                
            last_ts = last
            # Sleep to avoid rate limiting
            time.sleep(0.5)
        
        if all_data:
            df = process_ohlc_data(all_data)
            df.to_csv(filename)
            result[tf] = df
            logger.info(f"Fetched and saved {len(df)} records for {tf} timeframe")
        else:
            logger.warning(f"No data fetched for {tf} timeframe")
    
    return result

def add_technical_indicators(df):
    """
    Add technical indicators to DataFrame
    
    Args:
        df (pd.DataFrame): OHLC DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Calculate EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Calculate Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Calculate price volatility (standard deviation of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def fetch_and_prepare_data(trading_pair=TRADING_PAIR, days_back=365):
    """
    Fetch historical data and prepare it for ML model training
    
    Args:
        trading_pair (str): Trading pair to fetch data for
        days_back (int): Number of days to fetch data for
        
    Returns:
        dict: Dictionary of prepared DataFrames for each timeframe
    """
    logger.info(f"Fetching historical data for {trading_pair} going back {days_back} days")
    
    # Fetch data for all timeframes
    historical_data = fetch_historical_data(trading_pair, days_back=days_back)
    
    # Add technical indicators to each timeframe
    for tf, df in historical_data.items():
        historical_data[tf] = add_technical_indicators(df)
        logger.info(f"Added technical indicators to {tf} timeframe data")
    
    return historical_data

def prepare_datasets_for_ml(data_dict, sequence_length=60):
    """
    Prepare datasets for ML model training
    
    Args:
        data_dict (dict): Dictionary of DataFrames for each timeframe
        sequence_length (int): Sequence length for time series data
        
    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val for each timeframe
    """
    ml_datasets = {}
    
    for tf, df in data_dict.items():
        logger.info(f"Preparing ML dataset for {tf} timeframe")
        
        # Select features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ema9', 'ema21', 'ema50', 'ema100',
            'rsi', 'macd', 'signal', 'macd_hist',
            'upper_band', 'middle_band', 'lower_band',
            'atr', 'volatility'
        ]
        
        features = df[feature_columns].values
        target = df['close'].pct_change(5).shift(-5).values  # 5-period future returns
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - sequence_length - 5):
            X.append(features[i:i+sequence_length])
            y.append(np.sign(target[i+sequence_length]))  # Direction of future returns
        
        X = np.array(X)
        y = np.array(y)
        
        # Remove NaN values
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split into training and validation sets (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        ml_datasets[tf] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_columns': feature_columns
        }
        
        logger.info(f"{tf} dataset prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    
    return ml_datasets

def save_datasets(ml_datasets, trading_pair=TRADING_PAIR):
    """
    Save prepared datasets to disk
    
    Args:
        ml_datasets (dict): Dictionary of prepared datasets
        trading_pair (str): Trading pair
    """
    dataset_dir = f"{DATA_DIR}/ml_datasets"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    for tf, datasets in ml_datasets.items():
        filename = f"{dataset_dir}/{trading_pair.replace('/', '')}-{tf}.npz"
        feature_columns = datasets.pop('feature_columns')
        
        np.savez(
            filename,
            X_train=datasets['X_train'],
            y_train=datasets['y_train'],
            X_val=datasets['X_val'],
            y_val=datasets['y_val']
        )
        
        # Save feature columns separately
        with open(f"{dataset_dir}/{trading_pair.replace('/', '')}-{tf}-features.json", 'w') as f:
            json.dump(feature_columns, f)
        
        logger.info(f"Saved ML dataset for {tf} timeframe to {filename}")

def load_datasets(trading_pair=TRADING_PAIR, timeframe="1h"):
    """
    Load prepared datasets from disk
    
    Args:
        trading_pair (str): Trading pair
        timeframe (str): Timeframe to load
        
    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val
    """
    dataset_dir = f"{DATA_DIR}/ml_datasets"
    filename = f"{dataset_dir}/{trading_pair.replace('/', '')}-{timeframe}.npz"
    
    if not os.path.exists(filename):
        logger.error(f"Dataset file not found: {filename}")
        return None
    
    data = np.load(filename)
    
    # Load feature columns
    feature_file = f"{dataset_dir}/{trading_pair.replace('/', '')}-{timeframe}-features.json"
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            feature_columns = json.load(f)
    else:
        feature_columns = None
    
    return {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
        'feature_columns': feature_columns
    }

def main():
    """Main function to fetch and prepare data"""
    logger.info("Starting historical data fetcher")
    
    # Fetch and prepare data for ML training
    historical_data = fetch_and_prepare_data(days_back=365*2)  # 2 years of data
    
    # Prepare datasets for ML training
    ml_datasets = prepare_datasets_for_ml(historical_data)
    
    # Save datasets
    save_datasets(ml_datasets)
    
    logger.info("Historical data fetching and preparation complete")

if __name__ == "__main__":
    main()