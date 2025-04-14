#!/usr/bin/env python3
"""
Extended Historical Data Fetcher for Kraken Trading Bot

This script fetches an extended history (365 days) of data for all trading pairs
to significantly increase the sample size for ML model training.
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
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_historical_data.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "MATICUSD"]
DEFAULT_DAYS_BACK = 365  # Fetch a full year of data
HISTORICAL_DATA_DIR = "historical_data"

# Timeframe intervals in seconds
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800
}

class ExtendedHistoricalDataFetcher:
    """
    Fetches extended historical data from Kraken API
    """
    
    def __init__(self, pair: str, timeframe: str = "1h", days_back: int = DEFAULT_DAYS_BACK):
        """
        Initialize the fetcher
        
        Args:
            pair: Trading pair (e.g., "SOL/USD")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            days_back: Number of days of historical data to fetch
        """
        self.pair = pair
        self.timeframe = timeframe
        self.days_back = days_back
        
        # Create filename versions of the pair
        self.pair_filename = pair.replace("/", "")
        
        # Create necessary directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
        # Also create pair-specific directory
        pair_dir = os.path.join(HISTORICAL_DATA_DIR, self.pair_filename)
        os.makedirs(pair_dir, exist_ok=True)
    
    def fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Kraken API
        
        Returns:
            DataFrame with historical data or None if error
        """
        logger.info(f"Fetching {self.days_back} days of historical data for {self.pair} ({self.timeframe})")
        
        try:
            # Calculate time interval
            end_time = int(time.time())
            start_time = end_time - (self.days_back * 24 * 60 * 60)
            
            # Get interval in seconds
            interval_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 3600)  # Default to 1h if not found
            
            # Adjust pair format for Kraken API
            kraken_pair = self.pair.replace("/", "")
            
            # Construct API URL
            url = f"https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": kraken_pair,
                "interval": interval_seconds // 60,  # Kraken API uses minutes
                "since": start_time
            }
            
            # Fetch data
            response = requests.get(url, params=params)
            data = response.json()
            
            if "error" in data and data["error"]:
                logger.error(f"Kraken API error: {data['error']}")
                return None
            
            if "result" not in data:
                logger.error(f"Unexpected API response: {data}")
                return None
            
            # Extract result
            result = data["result"]
            if kraken_pair not in result:
                available_pairs = list(result.keys())
                if available_pairs:
                    # If the pair name is not exact, try to find a match
                    kraken_pair = available_pairs[0]
                else:
                    logger.error(f"Pair {self.pair} not found in API response")
                    return None
            
            # Get OHLC data
            ohlc_data = result[kraken_pair]
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=[
                "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            for col in ["open", "high", "low", "close", "vwap", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"Fetched {len(df)} candles for {self.pair} ({self.timeframe})")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {self.pair}: {e}")
            return None
    
    def save_to_csv(self, df: pd.DataFrame) -> bool:
        """
        Save historical data to CSV file
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create pair directory if it doesn't exist
            pair_dir = os.path.join(HISTORICAL_DATA_DIR, self.pair_filename)
            os.makedirs(pair_dir, exist_ok=True)
            
            # Define output path
            output_path = os.path.join(pair_dir, f"{self.pair_filename}_{self.timeframe}.csv")
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(df)} candles to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data for {self.pair}: {e}")
            return False
    
    def merge_timeframes(self, dfs: Dict[str, pd.DataFrame]) -> bool:
        """
        Merge dataframes from different timeframes into a single file
        
        Args:
            dfs: Dictionary of timeframes to dataframes
            
        Returns:
            True if merged successfully, False otherwise
        """
        try:
            # Merge all timeframes into a single dataframe
            merged_data = {}
            
            for timeframe, df in dfs.items():
                # Skip if dataframe is empty
                if df is None or df.empty:
                    continue
                
                # Process each row
                for _, row in df.iterrows():
                    timestamp = row["timestamp"]
                    
                    # Initialize if needed
                    if timestamp not in merged_data:
                        merged_data[timestamp] = {
                            "timestamp": timestamp,
                            "open": {},
                            "high": {},
                            "low": {},
                            "close": {},
                            "volume": {}
                        }
                    
                    # Add data for this timeframe
                    merged_data[timestamp]["open"][timeframe] = row["open"]
                    merged_data[timestamp]["high"][timeframe] = row["high"]
                    merged_data[timestamp]["low"][timeframe] = row["low"]
                    merged_data[timestamp]["close"][timeframe] = row["close"]
                    merged_data[timestamp]["volume"][timeframe] = row["volume"]
            
            # Sort by timestamp
            sorted_data = sorted(merged_data.values(), key=lambda x: x["timestamp"])
            
            # Convert to dataframe format
            rows = []
            for item in sorted_data:
                row = {
                    "timestamp": item["timestamp"]
                }
                
                # Add data for each timeframe
                for timeframe in dfs.keys():
                    for metric in ["open", "high", "low", "close", "volume"]:
                        if timeframe in item[metric]:
                            row[f"{metric}_{timeframe}"] = item[metric][timeframe]
                
                rows.append(row)
            
            # Create dataframe
            merged_df = pd.DataFrame(rows)
            
            # Save to CSV
            output_path = os.path.join(HISTORICAL_DATA_DIR, f"{self.pair_filename}_merged.csv")
            merged_df.to_csv(output_path, index=False)
            
            logger.info(f"Merged {len(merged_df)} timestamps to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error merging timeframes for {self.pair}: {e}")
            return False

def fetch_data_for_pair(pair: str, timeframes: List[str], days_back: int = DEFAULT_DAYS_BACK) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for a single pair across multiple timeframes
    
    Args:
        pair: Trading pair (e.g., "SOL/USD")
        timeframes: List of timeframes to fetch
        days_back: Number of days of historical data to fetch
        
    Returns:
        Dictionary of timeframes to dataframes
    """
    results = {}
    
    for timeframe in timeframes:
        # Initialize fetcher
        fetcher = ExtendedHistoricalDataFetcher(pair, timeframe, days_back)
        
        # Fetch data
        df = fetcher.fetch_historical_data()
        
        if df is not None and not df.empty:
            # Save to CSV
            fetcher.save_to_csv(df)
            results[timeframe] = df
        else:
            logger.warning(f"No data fetched for {pair} ({timeframe})")
    
    # Merge timeframes if requested
    if len(results) > 1:
        fetcher = ExtendedHistoricalDataFetcher(pair, list(results.keys())[0], days_back)
        fetcher.merge_timeframes(results)
    
    return results

def fetch_data_for_all_pairs(pairs: List[str], timeframes: List[str], days_back: int = DEFAULT_DAYS_BACK, merge: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch historical data for all specified pairs and timeframes
    
    Args:
        pairs: List of trading pairs
        timeframes: List of timeframe names
        days_back: Number of days of historical data to fetch
        merge: Whether to merge timeframes for each pair
        
    Returns:
        Dictionary of pairs to timeframes to dataframes
    """
    results = {}
    
    for pair in pairs:
        logger.info(f"Fetching data for {pair}")
        pair_results = fetch_data_for_pair(pair, timeframes, days_back)
        results[pair] = pair_results
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fetch extended historical data for ML model training")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK,
                        help=f"Number of days of historical data to fetch (default: {DEFAULT_DAYS_BACK})")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to fetch (default: {', '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes to fetch (default: {', '.join(DEFAULT_TIMEFRAMES)})")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't merge timeframes for each pair")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Ensure historical data directory exists
    os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
    
    # Fetch data for all pairs
    fetch_data_for_all_pairs(args.pairs, args.timeframes, args.days, not args.no_merge)
    
    logger.info("Data fetch completed!")

if __name__ == "__main__":
    main()