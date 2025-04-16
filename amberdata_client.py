#!/usr/bin/env python3
"""
Amberdata API Client

This module provides functionality to fetch OHLCV data from Amberdata API
for training cryptocurrency trading models at various timeframes.

APIs used:
- OHLCV Historical API: https://api.amberdata.com/markets/spot/ohlcv/{instrument}
- OHLCV Latest API per exchange: https://api.amberdata.com/market/spot/ohlcv/exchange/{exchange}/latest

Usage:
    from amberdata_client import AmberdataClient
    
    client = AmberdataClient(api_key='YOUR_API_KEY')
    df = client.get_historical_ohlcv('BTC_USD', '1h', days=30)
    print(df.head())
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)

class AmberdataClient:
    """Client for interacting with Amberdata API"""
    
    BASE_URL = "https://api.amberdata.com"
    HISTORICAL_ENDPOINT = "/markets/spot/ohlcv/{instrument}"
    LATEST_ENDPOINT = "/market/spot/ohlcv/exchange/{exchange}/latest"
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client with API key"""
        self.api_key = api_key or os.environ.get('AMBERDATA_API_KEY')
        if not self.api_key:
            logger.warning("No Amberdata API key provided. API requests will fail.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            'x-api-key': self.api_key,
            'Accept': 'application/json'
        }
    
    def _format_pair(self, pair: str) -> str:
        """Format trading pair for Amberdata API"""
        # Convert e.g., BTC/USD to BTC_USD
        return pair.replace('/', '_')
    
    def _parse_ohlcv_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """Parse OHLCV API response into DataFrame"""
        try:
            if 'payload' not in response_data or 'data' not in response_data['payload']:
                logger.error(f"Unexpected API response format: {response_data}")
                return pd.DataFrame()
            
            data = response_data['payload']['data']
            if not data:
                logger.warning("No data returned from API")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure expected columns and types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df:
                    df[col] = pd.to_numeric(df[col])
            
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error parsing OHLCV response: {e}")
            return pd.DataFrame()
    
    def get_historical_ohlcv(
        self, 
        pair: str, 
        timeframe: str = '1h', 
        days: int = 30,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a trading pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '15m', '1h', '4h', '1d')
            days: Number of days of historical data to fetch
            start_time: Start time for data (overrides days)
            end_time: End time for data (default: current time)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check if API key is available
        if not self.api_key:
            logger.error("Amberdata API key not provided")
            return pd.DataFrame()
        
        # Map timeframe
        api_timeframe = self.TIMEFRAME_MAP.get(timeframe, '1h')
        
        # Format pair
        instrument = self._format_pair(pair)
        
        # Calculate timestamps
        end_time = end_time or datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
        
        # Convert to milliseconds
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # Build URL
        url = f"{self.BASE_URL}{self.HISTORICAL_ENDPOINT.format(instrument=instrument)}"
        
        # Build params
        params = {
            'timeframe': api_timeframe,
            'startDate': start_timestamp,
            'endDate': end_timestamp
        }
        
        # Make request
        try:
            logger.info(f"Fetching historical OHLCV data for {pair} ({timeframe}) from {start_time} to {end_time}")
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            df = self._parse_ohlcv_response(data)
            
            logger.info(f"Got {len(df)} OHLCV records for {pair} ({timeframe})")
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical OHLCV data: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return pd.DataFrame()
    
    def get_latest_ohlcv(self, pair: str, exchange: str = 'kraken', timeframe: str = '1h') -> pd.DataFrame:
        """
        Get latest OHLCV data for a trading pair from a specific exchange
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            exchange: Exchange name (e.g., 'kraken')
            timeframe: Timeframe (e.g., '15m', '1h', '4h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check if API key is available
        if not self.api_key:
            logger.error("Amberdata API key not provided")
            return pd.DataFrame()
        
        # Map timeframe
        api_timeframe = self.TIMEFRAME_MAP.get(timeframe, '1h')
        
        # Format pair
        instrument = self._format_pair(pair)
        
        # Build URL
        url = f"{self.BASE_URL}{self.LATEST_ENDPOINT.format(exchange=exchange)}"
        
        # Build params
        params = {
            'timeframe': api_timeframe,
            'pair': instrument
        }
        
        # Make request
        try:
            logger.info(f"Fetching latest OHLCV data for {pair} ({timeframe}) from {exchange}")
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            df = self._parse_ohlcv_response(data)
            
            logger.info(f"Got {len(df)} latest OHLCV records for {pair} ({timeframe}) from {exchange}")
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching latest OHLCV data: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return pd.DataFrame()
    
    def fetch_and_save_data(
        self,
        pair: str,
        timeframe: str = '1h',
        days: int = 365,
        use_exchange: bool = False,
        exchange: str = 'kraken'
    ) -> str:
        """
        Fetch OHLCV data and save to file
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '15m', '1h', '4h', '1d')
            days: Number of days of historical data to fetch
            use_exchange: Whether to use exchange-specific API
            exchange: Exchange name (e.g., 'kraken')
            
        Returns:
            Path to saved file
        """
        # Format pair for filename
        pair_clean = pair.replace('/', '_').lower()
        
        # Define filename
        filename = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}.csv"
        
        # Fetch data
        if use_exchange:
            df = self.get_latest_ohlcv(pair, exchange=exchange, timeframe=timeframe)
        else:
            df = self.get_historical_ohlcv(pair, timeframe=timeframe, days=days)
        
        # Check if data was fetched
        if df.empty:
            logger.error(f"No data fetched for {pair} ({timeframe})")
            return ""
        
        # Save to file
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} OHLCV records to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error saving data to file: {e}")
            return ""
    
    def fetch_multiple_timeframes(
        self,
        pair: str,
        timeframes: List[str] = ['15m', '1h', '4h', '1d'],
        days: int = 365
    ) -> Dict[str, str]:
        """
        Fetch OHLCV data for multiple timeframes
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            timeframes: List of timeframes to fetch
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping timeframe to file path
        """
        results = {}
        
        for timeframe in timeframes:
            logger.info(f"Fetching {timeframe} data for {pair}")
            file_path = self.fetch_and_save_data(pair, timeframe=timeframe, days=days)
            
            if file_path:
                results[timeframe] = file_path
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def fetch_multiple_pairs(
        self,
        pairs: List[str],
        timeframe: str = '1h',
        days: int = 365
    ) -> Dict[str, str]:
        """
        Fetch OHLCV data for multiple trading pairs
        
        Args:
            pairs: List of trading pairs (e.g., ['BTC/USD', 'ETH/USD'])
            timeframe: Timeframe (e.g., '15m', '1h', '4h', '1d')
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping pair to file path
        """
        results = {}
        
        for pair in pairs:
            logger.info(f"Fetching {pair} data ({timeframe})")
            file_path = self.fetch_and_save_data(pair, timeframe=timeframe, days=days)
            
            if file_path:
                results[pair] = file_path
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return results

def fetch_data_for_training(
    api_key: str,
    pairs: List[str] = ['BTC/USD', 'ETH/USD', 'SOL/USD'],
    timeframes: List[str] = ['1h'],
    days: int = 365
) -> Dict[str, Dict[str, str]]:
    """
    Fetch data for multiple pairs and timeframes for training
    
    Args:
        api_key: Amberdata API key
        pairs: List of trading pairs
        timeframes: List of timeframes
        days: Number of days of historical data
        
    Returns:
        Nested dictionary mapping pair to timeframe to file path
    """
    client = AmberdataClient(api_key=api_key)
    results = {}
    
    for pair in pairs:
        pair_results = {}
        
        for timeframe in timeframes:
            logger.info(f"Fetching {pair} data ({timeframe})")
            file_path = client.fetch_and_save_data(pair, timeframe=timeframe, days=days)
            
            if file_path:
                pair_results[timeframe] = file_path
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        results[pair] = pair_results
    
    return results

def main():
    """Main function for testing"""
    # Get API key from environment or argument
    api_key = os.environ.get('AMBERDATA_API_KEY')
    
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        logger.error("Please provide Amberdata API key as environment variable AMBERDATA_API_KEY or as argument")
        return 1
    
    # Create client
    client = AmberdataClient(api_key=api_key)
    
    # Test fetching historical data
    pair = 'BTC/USD'
    timeframe = '1h'
    days = 7
    
    logger.info(f"Testing historical data fetch for {pair} ({timeframe}) for {days} days")
    df = client.get_historical_ohlcv(pair, timeframe=timeframe, days=days)
    
    if not df.empty:
        logger.info(f"Successfully fetched {len(df)} records:")
        logger.info(df.head())
        
        # Save to file
        pair_clean = pair.replace('/', '_').lower()
        filename = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}_test.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved test data to {filename}")
    else:
        logger.error("Failed to fetch historical data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())