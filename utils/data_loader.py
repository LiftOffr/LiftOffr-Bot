#!/usr/bin/env python3
"""
Historical Data Loader

This module provides functionality to load and preprocess historical data for
backtesting and analysis. It includes:

1. Loading historical data from files or APIs
2. Calculating technical indicators
3. Creating sample data for testing
4. Preprocessing data for ML models

It serves as a common data access layer for all backtesting and optimization modules.
"""

import os
import csv
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]

class HistoricalDataLoader:
    """
    Loads and preprocesses historical data for backtesting and analysis.
    """
    
    def __init__(self, data_dir: str = HISTORICAL_DATA_DIR):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = data_dir
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Initialized data loader with data directory: {data_dir}")
    
    def get_available_pairs(self) -> List[str]:
        """
        Get list of trading pairs with available historical data.
        
        Returns:
            List of trading pair symbols
        """
        pairs = set()
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                # Extract pair symbol from filename (e.g., "BTC_USD_1h.csv" -> "BTC/USD")
                parts = filename.split("_")
                if len(parts) >= 2:
                    pair = f"{parts[0]}/{parts[1]}"
                    pairs.add(pair)
        
        return list(pairs)
    
    def fetch_historical_data(self, pair: str, timeframe: str = "1h", 
                             days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data for a trading pair.
        
        Args:
            pair: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        # Format pair for filename (e.g., "BTC/USD" -> "BTC_USD")
        pair_filename = pair.replace("/", "_")
        filename = f"{self.data_dir}/{pair_filename}_{timeframe}.csv"
        
        try:
            # Check if file exists
            if os.path.exists(filename):
                # Load data from file
                data = pd.read_csv(filename, parse_dates=["timestamp"])
                
                # Set timestamp as index
                data.set_index("timestamp", inplace=True)
                
                # Filter for requested days
                if days > 0:
                    start_date = datetime.now() - timedelta(days=days)
                    data = data[data.index >= start_date]
                
                logger.info(f"Loaded {len(data)} rows from {filename}")
                return data
            else:
                # Generate sample data for testing
                logger.warning(f"Historical data file not found: {filename}. Generating sample data.")
                return self._generate_sample_data(pair, timeframe, days)
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    def _generate_sample_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Generate sample historical data for testing.
        
        Args:
            pair: Trading pair symbol
            timeframe: Timeframe
            days: Number of days to generate
            
        Returns:
            DataFrame with generated data
        """
        # Determine number of rows based on timeframe and days
        periods_per_day = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24,
            "4h": 6,
            "1d": 1
        }
        
        periods = periods_per_day.get(timeframe, 24) * days
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - timedelta(minutes=i * self._get_minutes_per_period(timeframe)) 
                     for i in range(periods)]
        timestamps.reverse()  # Sort chronologically
        
        # Generate price data
        # Start with a reasonable price based on the pair
        if pair.startswith("BTC"):
            base_price = 40000.0
        elif pair.startswith("ETH"):
            base_price = 2000.0
        elif pair.startswith("SOL"):
            base_price = 100.0
        else:
            base_price = 10.0
        
        # Generate price series with random walk and occasional jumps
        price_data = []
        current_price = base_price
        volatility = base_price * 0.02  # 2% base volatility
        
        for i in range(periods):
            # Random price change with occasional jumps
            if random.random() < 0.05:  # 5% chance of a larger move
                price_change = random.normalvariate(0, volatility * 5)
            else:
                price_change = random.normalvariate(0, volatility)
            
            # Apply some trend
            trend = 0.0001 * base_price * np.sin(i / (periods / 8))
            
            current_price += price_change + trend
            current_price = max(0.01, current_price)  # Ensure price is positive
            
            # Generate candle data
            candle_range = current_price * random.uniform(0.005, 0.03)  # 0.5-3% range
            high = current_price + candle_range * random.uniform(0.3, 1.0)
            low = current_price - candle_range * random.uniform(0.3, 1.0)
            open_price = low + (high - low) * random.uniform(0, 1)
            close_price = low + (high - low) * random.uniform(0, 1)
            
            # Ensure OHLC relationship
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume
            volume = base_price * 10 * random.uniform(0.5, 5.0)
            
            price_data.append({
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(price_data)
        df.set_index("timestamp", inplace=True)
        
        # Save generated data to file
        pair_filename = pair.replace("/", "_")
        filename = f"{self.data_dir}/{pair_filename}_{timeframe}.csv"
        df.reset_index().to_csv(filename, index=False)
        logger.info(f"Generated and saved {len(df)} rows of sample data to {filename}")
        
        return df
    
    def _get_minutes_per_period(self, timeframe: str) -> int:
        """
        Get number of minutes per period for a timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            
        Returns:
            Number of minutes
        """
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 60 * 24
        else:
            return 60  # Default to 1 hour
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added indicators
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # Moving Averages
        df["ema9"] = self._calculate_ema(df["close"], 9)
        df["ema21"] = self._calculate_ema(df["close"], 21)
        df["ema50"] = self._calculate_ema(df["close"], 50)
        df["ema100"] = self._calculate_ema(df["close"], 100)
        df["ema200"] = self._calculate_ema(df["close"], 200)
        
        # RSI
        df["rsi"] = self._calculate_rsi(df["close"], 14)
        
        # MACD
        macd, signal = self._calculate_macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = macd - signal
        
        # Bollinger Bands
        df["bb_middle"], df["bb_upper"], df["bb_lower"] = self._calculate_bollinger_bands(df["close"])
        
        # ATR
        df["atr"] = self._calculate_atr(df)
        
        # ADX
        df["adx"] = self._calculate_adx(df)
        
        # Volatility
        df["volatility"] = self._calculate_volatility(df)
        
        return df
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = series.diff()
        
        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA
        roll_up = up.ewm(span=period).mean()
        roll_down = down.ewm(span=period).mean()
        
        # Calculate RS
        rs = roll_up / roll_down
        
        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MACD and Signal Line.
        
        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal Line)
        """
        # Calculate EMAs
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd = fast_ema - slow_ema
        
        # Calculate Signal line
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Middle Band, Upper Band, Lower Band)
        """
        # Calculate middle band
        middle_band = series.rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = series.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return middle_band, upper_band, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with price data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            df: DataFrame with price data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        # This is a simplified ADX calculation
        # For a complete implementation, refer to technical analysis libraries
        
        # Calculate +DI and -DI
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(period).sum() / atr.rolling(period).sum())
        minus_di = 100 * (minus_dm.rolling(period).sum() / atr.rolling(period).sum())
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate price volatility.
        
        Args:
            df: DataFrame with price data
            period: Volatility period
            
        Returns:
            Series with volatility values
        """
        # Calculate returns
        returns = df["close"].pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(period).std()
        
        return volatility
    
    def save_dataframe(self, df: pd.DataFrame, pair: str, timeframe: str):
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            pair: Trading pair symbol
            timeframe: Timeframe
        """
        pair_filename = pair.replace("/", "_")
        filename = f"{self.data_dir}/{pair_filename}_{timeframe}.csv"
        
        # Reset index to include timestamp column
        df.reset_index().to_csv(filename, index=False)
        
        logger.info(f"Saved {len(df)} rows to {filename}")
    
    def merge_historical_data(self, pair: str, timeframes: List[str] = DEFAULT_TIMEFRAMES):
        """
        Merge historical data from multiple timeframes.
        
        Args:
            pair: Trading pair symbol
            timeframes: List of timeframes to merge
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        result = {}
        
        for timeframe in timeframes:
            data = self.fetch_historical_data(pair, timeframe)
            result[timeframe] = data
        
        return result
    
    def prepare_ml_dataset(self, df: pd.DataFrame, target_column: str = "close", 
                         window_size: int = 60, forecast_horizon: int = 24, 
                         features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for ML model training.
        
        Args:
            df: DataFrame with price and indicator data
            target_column: Target column to predict
            window_size: Size of the input window
            forecast_horizon: Number of periods to forecast
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        if df.empty:
            return np.array([]), np.array([])
        
        # Default features if not specified
        if features is None:
            features = ["open", "high", "low", "close", "volume", 
                      "ema9", "ema21", "rsi", "macd", "atr"]
        
        # Filter out features that don't exist in df
        features = [f for f in features if f in df.columns]
        
        # Create feature matrix and target vector
        X, y = [], []
        
        for i in range(window_size, len(df) - forecast_horizon):
            # Create feature window
            feature_window = df[features].iloc[i - window_size:i].values
            
            # Create target
            target = df[target_column].iloc[i + forecast_horizon]
            
            X.append(feature_window)
            y.append(target)
        
        return np.array(X), np.array(y)