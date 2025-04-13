#!/usr/bin/env python3
"""
Create Sample Historical Data for Testing the Backtest System

This script generates sample historical data for SOL/USD, BTC/USD, and ETH/USD
to use for testing the backtesting system when real API data isn't available.

Note: This is only for development and testing purposes. In production,
use real historical data from Kraken API.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants
DATA_DIR = "historical_data"
PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
TIMEFRAMES = {
    "1h": 60,   # minutes
    "4h": 240,  # minutes
    "1d": 1440  # minutes
}
NUM_DAYS = {
    "1h": 30,   # 30 days of hourly data
    "4h": 90,   # 90 days of 4h data
    "1d": 365   # 365 days of daily data
}
INITIAL_PRICES = {
    "SOLUSD": 100.0,
    "BTCUSD": 40000.0,
    "ETHUSD": 2500.0
}
VOLATILITY = {
    "SOLUSD": 0.03,
    "BTCUSD": 0.02,
    "ETHUSD": 0.025
}
TREND_FACTORS = {
    "SOLUSD": 0.001,
    "BTCUSD": 0.0005,
    "ETHUSD": 0.0008
}
CORRELATION_MATRIX = {
    "SOLUSD": {"SOLUSD": 1.0, "BTCUSD": 0.7, "ETHUSD": 0.8},
    "BTCUSD": {"SOLUSD": 0.7, "BTCUSD": 1.0, "ETHUSD": 0.9},
    "ETHUSD": {"SOLUSD": 0.8, "BTCUSD": 0.9, "ETHUSD": 1.0}
}


def ensure_directories():
    """Create necessary directories if they don't exist"""
    for pair in PAIRS:
        pair_dir = os.path.join(DATA_DIR, pair)
        os.makedirs(pair_dir, exist_ok=True)


def generate_price_data(pair, timeframe, num_candles):
    """
    Generate synthetic price data for a pair and timeframe
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe identifier
        num_candles (int): Number of candles to generate
        
    Returns:
        pd.DataFrame: DataFrame with synthetic OHLCV data
    """
    # Set the random seed for reproducibility
    # Create a more reliable seed from the pair name and timeframe
    if pair == "SOLUSD":
        pair_seed = 1
    elif pair == "BTCUSD":
        pair_seed = 2
    else:  # ETHUSD
        pair_seed = 3
    
    np.random.seed(pair_seed * 100 + int(TIMEFRAMES[timeframe]))
    
    # Calculate time interval in minutes
    interval_minutes = TIMEFRAMES[timeframe]
    
    # Generate timestamps
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [end_time - timedelta(minutes=i * interval_minutes) for i in range(num_candles)]
    timestamps.reverse()  # Oldest first
    
    # Initial price
    initial_price = INITIAL_PRICES[pair]
    
    # Volatility and trend parameters
    volatility = VOLATILITY[pair]
    trend = TREND_FACTORS[pair]
    
    # Generate price data with random walk + trend
    close_prices = []
    current_price = initial_price
    
    for i in range(num_candles):
        # Random component with trend
        random_change = np.random.normal(0, volatility)
        trend_change = trend * i  # Simple upward trend
        
        # Apply changes
        price_change = current_price * (random_change + trend_change)
        current_price += price_change
        
        # Ensure price stays positive
        current_price = max(current_price, initial_price * 0.5)
        
        close_prices.append(current_price)
    
    # Generate OHLC data
    data = []
    for i in range(num_candles):
        close = close_prices[i]
        
        # Generate random high/low/open around close
        high_factor = 1 + abs(np.random.normal(0, volatility / 2))
        low_factor = 1 - abs(np.random.normal(0, volatility / 2))
        open_factor = 1 + np.random.normal(0, volatility / 3)
        
        high = close * high_factor
        low = close * low_factor
        open_price = close * open_factor
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, max(open_price, close))
        low = min(low, min(open_price, close))
        
        # Generate random volume
        volume = np.random.gamma(2.0, initial_price / 100) * (1 + np.random.normal(0, 0.2))
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'vwap': (high + low + close) / 3,
            'count': int(volume / 10)  # Fake trade count
        })
    
    return pd.DataFrame(data)


def save_data(df, pair, timeframe):
    """
    Save DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        pair (str): Trading pair
        timeframe (str): Timeframe identifier
    """
    output_file = os.path.join(DATA_DIR, pair, f"{pair}_{timeframe}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows of {timeframe} data for {pair} to {output_file}")


def create_multi_timeframe_dataset(pair):
    """
    Create a multi-timeframe dataset by combining different timeframes
    
    Args:
        pair (str): Trading pair
    """
    base_file = os.path.join(DATA_DIR, pair, f"{pair}_1h.csv")
    if not os.path.exists(base_file):
        print(f"Base file {base_file} not found, skipping multi-timeframe dataset creation")
        return
    
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
    print(f"Saved multi-timeframe dataset for {pair} to {output_file}")


def main():
    """Main function to generate sample data"""
    print("Creating sample historical data for backtesting...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Generate and save data for each pair and timeframe
    for pair in PAIRS:
        for timeframe, interval_minutes in TIMEFRAMES.items():
            # Calculate number of candles
            num_days = NUM_DAYS[timeframe]
            num_candles = (num_days * 24 * 60) // interval_minutes
            
            # Generate data
            df = generate_price_data(pair, timeframe, num_candles)
            
            # Save data
            save_data(df, pair, timeframe)
        
        # Create multi-timeframe dataset
        create_multi_timeframe_dataset(pair)
    
    print("Sample data generation complete!")


if __name__ == "__main__":
    main()