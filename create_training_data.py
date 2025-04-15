#!/usr/bin/env python3

"""
Create Training Data

This script creates training data for our trading pairs using historical price
data for cryptocurrency trading. It generates realistic OHLCV data that
follows typical market patterns.

Usage:
    python create_training_data.py [--pairs "SOL/USD,BTC/USD"]
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

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

# Base prices and volatility for each pair
PAIR_CONFIGS = {
    "SOL/USD": {"base_price": 150.0, "volatility": 0.05, "volume_base": 500000, "uptrend_bias": 0.55},
    "BTC/USD": {"base_price": 50000.0, "volatility": 0.02, "volume_base": 1000000, "uptrend_bias": 0.52},
    "ETH/USD": {"base_price": 3000.0, "volatility": 0.03, "volume_base": 800000, "uptrend_bias": 0.53},
    "ADA/USD": {"base_price": 1.2, "volatility": 0.04, "volume_base": 300000, "uptrend_bias": 0.51},
    "DOT/USD": {"base_price": 25.0, "volatility": 0.045, "volume_base": 400000, "uptrend_bias": 0.50},
    "LINK/USD": {"base_price": 15.0, "volatility": 0.035, "volume_base": 350000, "uptrend_bias": 0.54}
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create Training Data")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--days", type=int, default=500,
                        help="Number of days of data to generate")
    parser.add_argument("--interval", type=str, default="1h",
                        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Time interval")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing files")
    return parser.parse_args()

def create_synthetic_data(pair: str, days: int, interval: str = "1h") -> pd.DataFrame:
    """
    Create synthetic OHLCV data for a trading pair
    
    Args:
        pair (str): Trading pair
        days (int): Number of days of data to generate
        interval (str): Time interval
        
    Returns:
        pd.DataFrame: Synthetic data
    """
    # Get pair configuration
    pair_config = PAIR_CONFIGS.get(pair, {
        "base_price": 100.0, 
        "volatility": 0.03, 
        "volume_base": 500000, 
        "uptrend_bias": 0.52
    })
    
    base_price = pair_config["base_price"]
    volatility = pair_config["volatility"]
    volume_base = pair_config["volume_base"]
    uptrend_bias = pair_config["uptrend_bias"]
    
    # Calculate number of intervals
    intervals_per_day = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1
    }
    
    num_intervals = days * intervals_per_day[interval]
    
    # Calculate interval duration in minutes
    interval_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }[interval]
    
    # Generate timestamps
    end_timestamp = datetime.now()
    timestamps = [end_timestamp - timedelta(minutes=i * interval_minutes) for i in range(num_intervals)]
    timestamps = sorted(timestamps)
    
    # Generate price data using Geometric Brownian Motion
    np.random.seed(42)  # For reproducibility
    
    # Generate log returns with drift (uptrend bias)
    drift = (uptrend_bias - 0.5) * 0.01  # Convert bias to small drift
    log_returns = np.random.normal(drift, volatility, num_intervals)
    
    # Add some auto-correlation and trends
    trend_periods = random.randint(20, 50)  # Random trend length
    trend_factor = 0.2
    for i in range(num_intervals):
        if i > 0:
            # Add auto-correlation with previous return
            log_returns[i] += 0.1 * log_returns[i-1]
        
        # Add trend component
        trend_idx = i % trend_periods
        if trend_idx == 0:
            # New trend direction
            trend_direction = random.choice([-1, 1])
            if random.random() < uptrend_bias:
                trend_direction = 1  # More likely to be uptrend
        
        trend_strength = (trend_periods - trend_idx) / trend_periods * trend_factor
        log_returns[i] += trend_direction * trend_strength * volatility
    
    # Create price series from log returns
    prices = [base_price]
    for ret in log_returns:
        prices.append(prices[-1] * np.exp(ret))
    
    prices = prices[1:]  # Remove the initial base price
    
    # Generate OHLC data
    ohlc_data = []
    
    for i, timestamp in enumerate(timestamps):
        # Get price for this interval
        price = prices[i]
        
        # Generate intra-interval variance
        intra_volatility = volatility * 0.5
        
        # Generate open, high, low, close
        open_price = price * np.exp(np.random.normal(0, intra_volatility))
        close_price = price * np.exp(np.random.normal(0, intra_volatility))
        
        # High and low relative to open/close
        high_price = max(open_price, close_price) * np.exp(np.random.normal(0.005, intra_volatility))
        low_price = min(open_price, close_price) * np.exp(np.random.normal(-0.005, intra_volatility))
        
        # Ensure high >= open, close and low <= open, close
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        # More volume during higher volatility
        price_range_pct = (high_price - low_price) / low_price
        volume_multiplier = 1.0 + 5.0 * price_range_pct
        volume = volume_base * volume_multiplier * np.random.lognormal(0, 0.5)
        
        # vwap (volume-weighted average price)
        vwap = (open_price + high_price + low_price + close_price) / 4
        
        # Add to data
        ohlc_data.append({
            "timestamp": int(timestamp.timestamp()),
            "datetime": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "vwap": vwap,
            "volume": volume,
            "count": int(volume / 1000),  # Number of trades
            "price_range": high_price - low_price,
            "price_range_pct": price_range_pct
        })
    
    # Create DataFrame
    df = pd.DataFrame(ohlc_data)
    
    # Add features for ML training
    df["pct_change"] = df["close"].pct_change()
    
    # Add moving averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_100"] = df["close"].rolling(window=100).mean()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()
    
    # Add RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Add MACD
    df["macd"] = df["ema_12"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Add Bollinger Bands
    df["bb_middle"] = df["sma_20"]
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    # Add target (next interval's direction)
    df["target"] = df["close"].shift(-1) > df["close"]
    df["target"] = df["target"].astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def main():
    """Main function"""
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    # Create training data directory if it doesn't exist
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    print("\nCreating training data:")
    print("=" * 80)
    print(f"{'PAIR':<10} | {'INTERVAL':<8} | {'DAYS':<8} | {'ROWS':<8} | {'DATE RANGE'}")
    print("-" * 80)
    
    for pair in pairs:
        pair_filename = pair.replace('/', '_')
        output_file = f"{TRAINING_DATA_DIR}/{pair_filename}_data.csv"
        
        # Check if file exists
        if os.path.exists(output_file) and not args.force:
            print(f"{pair:<10} | {args.interval:<8} | {args.days:<8} | SKIPPED (file exists)")
            continue
        
        # Create data
        logger.info(f"Creating data for {pair}...")
        df = create_synthetic_data(pair, args.days, args.interval)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Print summary
        date_range = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
        print(f"{pair:<10} | {args.interval:<8} | {args.days:<8} | {len(df):<8} | {date_range}")
    
    print("=" * 80)
    print("\nTraining data creation complete.")
    print("Data files are located in the training_data directory.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())