#!/usr/bin/env python3
"""
Fetch 1-day (1d) timeframe data for SOLUSD
This script downloads daily historical data and formats it to match the existing timeframe data files.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

def fetch_ohlc_data(symbol="XSOLUSD", interval=1440, count=720):
    """
    Fetch OHLC data from Kraken
    
    Args:
        symbol (str): Symbol pair (e.g., "XSOLUSD")
        interval (int): Timeframe in minutes (1440 = 1 day)
        count (int): Number of candles to fetch
        
    Returns:
        pd.DataFrame: DataFrame with OHLC data
    """
    print(f"Fetching {symbol} data for {interval} minute timeframe...")
    
    # Kraken API endpoint for OHLC data
    url = f"https://api.kraken.com/0/public/OHLC"
    
    # API parameters
    params = {
        "pair": symbol,
        "interval": 1440 // 60,  # 24 hours = 1 day (Kraken uses hours)
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params)
        data = response.json()
        
        if "error" in data and data["error"]:
            print(f"API Error: {data['error']}")
            return None
        
        # Extract result
        result = data["result"]
        ohlc_data = list(result.values())[0]
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        for col in ["open", "high", "low", "close", "vwap", "volume"]:
            df[col] = df[col].astype(float)
        
        print(f"Successfully fetched {len(df)} candles for {symbol} {interval}min")
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def process_and_save_data(df, symbol="SOLUSD", timeframe="1d"):
    """
    Process and save OHLC data to CSV
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        symbol (str): Symbol pair
        timeframe (str): Timeframe string
    """
    if df is None or len(df) == 0:
        print("No data to process")
        return
    
    # Ensure directory exists
    os.makedirs("historical_data", exist_ok=True)
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    
    # Format timestamp string to match other files
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to CSV
    output_file = f"historical_data/{symbol}_{timeframe}.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def main():
    """Main function to fetch and save 1d timeframe data"""
    # Fetch 1d (1440 min) data
    df_1d = fetch_ohlc_data(symbol="XSOLUSD", interval=1440, count=720)
    
    # Process and save
    if df_1d is not None:
        process_and_save_data(df_1d, symbol="SOLUSD", timeframe="1d")

if __name__ == "__main__":
    main()