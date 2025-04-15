#!/usr/bin/env python3
"""
Prepare Training Data

This script processes historical data and creates feature-rich datasets for ML training.
It includes technical indicators, cross-asset correlations, and other advanced features.

Usage:
    python prepare_training_data.py --pair PAIR [--timeframes TIMEFRAMES] [--output-dir OUTPUT_DIR]

Example:
    python prepare_training_data.py --pair BTC/USD --timeframes 1h,4h,1d --output-dir training_data
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prepare_training_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prepare_training_data")

# Constants
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
OUTPUT_DIR = "training_data"
CONFIG_PATH = "config/new_coins_training_config.json"
CORRELATION_LOOKBACK = 120  # Number of periods for correlation calculation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare training data for ML models.')
    parser.add_argument('--pair', type=str, required=True, help='Trading pair (e.g., BTC/USD)')
    parser.add_argument('--timeframes', type=str, default=None, help='Comma-separated list of timeframes (default: 1h,4h,1d)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--include-cross-asset', action='store_true', help='Include cross-asset correlations')
    parser.add_argument('--target-horizon', type=int, default=None, help='Prediction horizon in hours (default: from config or 12)')
    parser.add_argument('--lookback-window', type=int, default=None, help='Lookback window in periods (default: from config or 120)')
    return parser.parse_args()


def load_config() -> Dict:
    """Load configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {
            "feature_engineering": {
                "technical_indicators": [
                    "rsi", "macd", "bbands", "atr", "adx", "cci", "obv",
                    "mfi", "cmf", "stochastic", "williams_r"
                ],
                "derived_features": [
                    "price_momentum", "volatility_indicator", "trend_strength"
                ],
                "feature_importance": True,
                "feature_selection": True,
                "normalization": "standardization",
                "outlier_removal": "winsorization",
                "cross_asset_correlations": True,
                "timeframes": DEFAULT_TIMEFRAMES
            },
            "pair_specific_settings": {}
        }


def load_historical_data(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load historical data for a trading pair and timeframe.
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1h, 4h, 1d)
        
    Returns:
        DataFrame of historical data or None if error
    """
    pair_filename = pair.replace("/", "_")
    file_path = f"historical_data/{pair_filename}_{timeframe}.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Historical data file not found: {file_path}")
            return None
        
        data = pd.read_csv(file_path)
        
        # Parse timestamp if needed
        if "timestamp" in data.columns and not pd.api.types.is_datetime64_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None


def add_technical_indicators(
    df: pd.DataFrame, 
    indicators: List[str] = None
) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicator names to add
        
    Returns:
        DataFrame with added indicators
    """
    if indicators is None:
        indicators = [
            "rsi", "macd", "bbands", "atr", "adx", "cci", "obv",
            "mfi", "cmf", "stochastic", "williams_r"
        ]
    
    # Create a copy to avoid modifying the original
    df_with_indicators = df.copy()
    
    # Extract required columns
    open_prices = df_with_indicators["open"].values
    high_prices = df_with_indicators["high"].values
    low_prices = df_with_indicators["low"].values
    close_prices = df_with_indicators["close"].values
    volumes = df_with_indicators["volume"].values if "volume" in df_with_indicators.columns else None
    
    # Add indicators
    try:
        # RSI (Relative Strength Index)
        if "rsi" in indicators:
            df_with_indicators["rsi_14"] = talib.RSI(close_prices, timeperiod=14)
        
        # MACD (Moving Average Convergence Divergence)
        if "macd" in indicators:
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, fastperiod=12, slowperiod=26, signalperiod=9
            )
            df_with_indicators["macd"] = macd
            df_with_indicators["macd_signal"] = macd_signal
            df_with_indicators["macd_hist"] = macd_hist
        
        # Bollinger Bands
        if "bbands" in indicators:
            upper, middle, lower = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df_with_indicators["bbands_upper"] = upper
            df_with_indicators["bbands_middle"] = middle
            df_with_indicators["bbands_lower"] = lower
            
            # Bollinger Band Width
            df_with_indicators["bbands_width"] = (upper - lower) / middle
            
            # Bollinger Band %B
            df_with_indicators["bbands_pct_b"] = (close_prices - lower) / (upper - lower)
        
        # ATR (Average True Range)
        if "atr" in indicators:
            df_with_indicators["atr_14"] = talib.ATR(
                high_prices, low_prices, close_prices, timeperiod=14
            )
        
        # ADX (Average Directional Index)
        if "adx" in indicators:
            df_with_indicators["adx_14"] = talib.ADX(
                high_prices, low_prices, close_prices, timeperiod=14
            )
            
            # +DI and -DI
            df_with_indicators["plus_di_14"] = talib.PLUS_DI(
                high_prices, low_prices, close_prices, timeperiod=14
            )
            df_with_indicators["minus_di_14"] = talib.MINUS_DI(
                high_prices, low_prices, close_prices, timeperiod=14
            )
        
        # CCI (Commodity Channel Index)
        if "cci" in indicators:
            df_with_indicators["cci_14"] = talib.CCI(
                high_prices, low_prices, close_prices, timeperiod=14
            )
        
        # OBV (On-Balance Volume)
        if "obv" in indicators and volumes is not None:
            df_with_indicators["obv"] = talib.OBV(close_prices, volumes)
        
        # MFI (Money Flow Index)
        if "mfi" in indicators and volumes is not None:
            df_with_indicators["mfi_14"] = talib.MFI(
                high_prices, low_prices, close_prices, volumes, timeperiod=14
            )
        
        # CMF (Chaikin Money Flow)
        if "cmf" in indicators and volumes is not None:
            # Calculate CMF manually
            money_flow_multiplier = ((close_prices - low_prices) - (high_prices - close_prices)) / (high_prices - low_prices)
            money_flow_volume = money_flow_multiplier * volumes
            
            # 20-period CMF
            period = 20
            cmf = pd.Series(money_flow_volume).rolling(period).sum() / pd.Series(volumes).rolling(period).sum()
            df_with_indicators["cmf_20"] = cmf.values
        
        # Stochastic Oscillator
        if "stochastic" in indicators:
            stoch_k, stoch_d = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=14, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            df_with_indicators["stoch_k"] = stoch_k
            df_with_indicators["stoch_d"] = stoch_d
        
        # Williams %R
        if "williams_r" in indicators:
            df_with_indicators["williams_r_14"] = talib.WILLR(
                high_prices, low_prices, close_prices, timeperiod=14
            )
        
        # Parabolic SAR
        if "parabolic_sar" in indicators:
            df_with_indicators["psar"] = talib.SAR(
                high_prices, low_prices, acceleration=0.02, maximum=0.2
            )
        
        # TRIX (Triple Exponential Moving Average Oscillator)
        if "trix" in indicators:
            df_with_indicators["trix_14"] = talib.TRIX(close_prices, timeperiod=14)
        
        # Ichimoku Cloud
        if "ichimoku" in indicators:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            nine_period_high = pd.Series(high_prices).rolling(window=9).max()
            nine_period_low = pd.Series(low_prices).rolling(window=9).min()
            df_with_indicators["ichimoku_tenkan_sen"] = (nine_period_high + nine_period_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            twenty_six_period_high = pd.Series(high_prices).rolling(window=26).max()
            twenty_six_period_low = pd.Series(low_prices).rolling(window=26).min()
            df_with_indicators["ichimoku_kijun_sen"] = (twenty_six_period_high + twenty_six_period_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df_with_indicators["ichimoku_senkou_span_a"] = (df_with_indicators["ichimoku_tenkan_sen"] + df_with_indicators["ichimoku_kijun_sen"]) / 2
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            fifty_two_period_high = pd.Series(high_prices).rolling(window=52).max()
            fifty_two_period_low = pd.Series(low_prices).rolling(window=52).min()
            df_with_indicators["ichimoku_senkou_span_b"] = (fifty_two_period_high + fifty_two_period_low) / 2
        
        # Supertrend
        if "supertrend" in indicators:
            # ATR calculation
            atr_period = 10
            multiplier = 3
            
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
            
            # Basic Upper Band = (High + Low) / 2 + Multiplier * ATR
            # Basic Lower Band = (High + Low) / 2 - Multiplier * ATR
            basic_upper_band = (high_prices + low_prices) / 2 + multiplier * atr
            basic_lower_band = (high_prices + low_prices) / 2 - multiplier * atr
            
            # Initialize final bands
            final_upper_band = np.zeros_like(close_prices)
            final_lower_band = np.zeros_like(close_prices)
            supertrend = np.zeros_like(close_prices)
            
            for i in range(1, len(close_prices)):
                # Final Upper Band
                if basic_upper_band[i] < final_upper_band[i-1] or close_prices[i-1] > final_upper_band[i-1]:
                    final_upper_band[i] = basic_upper_band[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]
                
                # Final Lower Band
                if basic_lower_band[i] > final_lower_band[i-1] or close_prices[i-1] < final_lower_band[i-1]:
                    final_lower_band[i] = basic_lower_band[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]
                
                # Supertrend
                if supertrend[i-1] == final_upper_band[i-1] and close_prices[i] <= final_upper_band[i]:
                    supertrend[i] = final_upper_band[i]
                elif supertrend[i-1] == final_upper_band[i-1] and close_prices[i] > final_upper_band[i]:
                    supertrend[i] = final_lower_band[i]
                elif supertrend[i-1] == final_lower_band[i-1] and close_prices[i] >= final_lower_band[i]:
                    supertrend[i] = final_lower_band[i]
                elif supertrend[i-1] == final_lower_band[i-1] and close_prices[i] < final_lower_band[i]:
                    supertrend[i] = final_upper_band[i]
                else:
                    supertrend[i] = 0
            
            df_with_indicators["supertrend"] = supertrend
            df_with_indicators["supertrend_upper"] = final_upper_band
            df_with_indicators["supertrend_lower"] = final_lower_band
        
        # Vortex Indicator
        if "vortex" in indicators:
            period = 14
            # True Range
            tr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=1)
            
            # +VM and -VM
            vm_plus = abs(high_prices - np.roll(low_prices, 1))
            vm_minus = abs(low_prices - np.roll(high_prices, 1))
            
            # Rolling sums
            tr_sum = pd.Series(tr).rolling(window=period).sum()
            vm_plus_sum = pd.Series(vm_plus).rolling(window=period).sum()
            vm_minus_sum = pd.Series(vm_minus).rolling(window=period).sum()
            
            # +VI and -VI
            vi_plus = vm_plus_sum / tr_sum
            vi_minus = vm_minus_sum / tr_sum
            
            df_with_indicators["vortex_plus_14"] = vi_plus.values
            df_with_indicators["vortex_minus_14"] = vi_minus.values
            df_with_indicators["vortex_diff_14"] = (vi_plus - vi_minus).values
        
        # VWAP (Volume Weighted Average Price)
        if "vwap" in indicators and volumes is not None:
            # Group by day
            df_temp = df_with_indicators.copy()
            df_temp["date"] = df_temp["timestamp"].dt.date
            
            # Calculate VWAP for each day
            df_temp["vwap"] = np.nan
            
            for date, group in df_temp.groupby("date"):
                cumulative_vol = group["volume"].cumsum()
                cumulative_vol_price = (group["volume"] * ((group["high"] + group["low"] + group["close"]) / 3)).cumsum()
                group_vwap = cumulative_vol_price / cumulative_vol
                df_temp.loc[group.index, "vwap"] = group_vwap
            
            df_with_indicators["vwap"] = df_temp["vwap"]
        
        # Moving Averages
        df_with_indicators["sma_20"] = talib.SMA(close_prices, timeperiod=20)
        df_with_indicators["sma_50"] = talib.SMA(close_prices, timeperiod=50)
        df_with_indicators["sma_100"] = talib.SMA(close_prices, timeperiod=100)
        df_with_indicators["sma_200"] = talib.SMA(close_prices, timeperiod=200)
        
        df_with_indicators["ema_20"] = talib.EMA(close_prices, timeperiod=20)
        df_with_indicators["ema_50"] = talib.EMA(close_prices, timeperiod=50)
        df_with_indicators["ema_100"] = talib.EMA(close_prices, timeperiod=100)
        df_with_indicators["ema_200"] = talib.EMA(close_prices, timeperiod=200)
        
        # Pivot Points (Simple)
        if "pivot_points" in indicators:
            # Group by day
            df_temp = df_with_indicators.copy()
            df_temp["date"] = df_temp["timestamp"].dt.date
            
            # Calculate pivot points
            df_temp["pp"] = np.nan
            df_temp["r1"] = np.nan
            df_temp["r2"] = np.nan
            df_temp["s1"] = np.nan
            df_temp["s2"] = np.nan
            
            for date, group in df_temp.groupby("date"):
                # Get previous day high, low, close
                prev_day = (pd.to_datetime(date) - pd.Timedelta(days=1)).date()
                prev_group = df_temp[df_temp["date"] == prev_day]
                
                if len(prev_group) > 0:
                    prev_high = prev_group["high"].max()
                    prev_low = prev_group["low"].min()
                    prev_close = prev_group.iloc[-1]["close"]
                    
                    # Calculate pivot points
                    pp = (prev_high + prev_low + prev_close) / 3
                    r1 = (2 * pp) - prev_low
                    r2 = pp + (prev_high - prev_low)
                    s1 = (2 * pp) - prev_high
                    s2 = pp - (prev_high - prev_low)
                    
                    df_temp.loc[group.index, "pp"] = pp
                    df_temp.loc[group.index, "r1"] = r1
                    df_temp.loc[group.index, "r2"] = r2
                    df_temp.loc[group.index, "s1"] = s1
                    df_temp.loc[group.index, "s2"] = s2
            
            df_with_indicators["pivot_pp"] = df_temp["pp"]
            df_with_indicators["pivot_r1"] = df_temp["r1"]
            df_with_indicators["pivot_r2"] = df_temp["r2"]
            df_with_indicators["pivot_s1"] = df_temp["s1"]
            df_with_indicators["pivot_s2"] = df_temp["s2"]
        
        # Fibonacci Retracement
        if "fibonacci_retracement" in indicators:
            # Identify trends
            df_with_indicators["uptrend"] = df_with_indicators["close"] > df_with_indicators["sma_50"]
            
            # Calculate swing high and swing low
            window_size = 20
            high_roll_max = df_with_indicators["high"].rolling(window=window_size, center=True).max()
            low_roll_min = df_with_indicators["low"].rolling(window=window_size, center=True).min()
            
            # For uptrend, calculate retracements from swing low to swing high
            retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            
            for level in retracement_levels:
                level_str = str(int(level * 1000))
                
                # Uptrend retracement
                retracement_up = low_roll_min + (high_roll_max - low_roll_min) * level
                df_with_indicators[f"fib_up_{level_str}"] = retracement_up
                
                # Downtrend retracement
                retracement_down = high_roll_max - (high_roll_max - low_roll_min) * level
                df_with_indicators[f"fib_down_{level_str}"] = retracement_down
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
    
    return df_with_indicators


def add_derived_features(df: pd.DataFrame, derived_features: List[str] = None) -> pd.DataFrame:
    """
    Add derived features to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        derived_features: List of derived feature names to add
        
    Returns:
        DataFrame with added derived features
    """
    if derived_features is None:
        derived_features = [
            "price_momentum", "volatility_indicator", "trend_strength",
            "range_breakout", "volume_momentum"
        ]
    
    # Create a copy to avoid modifying the original
    df_with_features = df.copy()
    
    try:
        # Price momentum
        if "price_momentum" in derived_features:
            # Price change over different periods
            for period in [1, 3, 5, 10, 20, 50]:
                df_with_features[f"price_change_{period}"] = df_with_features["close"].pct_change(periods=period)
            
            # Rate of change
            for period in [10, 20, 50]:
                df_with_features[f"roc_{period}"] = talib.ROC(df_with_features["close"].values, timeperiod=period)
        
        # Volatility indicator
        if "volatility_indicator" in derived_features:
            # Price volatility over different periods
            for period in [10, 20, 50]:
                df_with_features[f"volatility_{period}"] = df_with_features["close"].rolling(window=period).std() / df_with_features["close"].rolling(window=period).mean()
            
            # ATR relative to price
            if "atr_14" in df_with_features.columns:
                df_with_features["atr_pct"] = df_with_features["atr_14"] / df_with_features["close"]
        
        # Trend strength
        if "trend_strength" in derived_features:
            # ADX trend strength
            if "adx_14" in df_with_features.columns:
                # Categorize ADX
                df_with_features["adx_trend_strength"] = pd.cut(
                    df_with_features["adx_14"],
                    bins=[0, 20, 40, 60, 100],
                    labels=[0, 1, 2, 3]
                ).astype(float)
            
            # Moving average trend
            for period in [20, 50, 100]:
                if f"sma_{period}" in df_with_features.columns:
                    df_with_features[f"ma_trend_{period}"] = (df_with_features["close"] - df_with_features[f"sma_{period}"]) / df_with_features[f"sma_{period}"]
            
            # Moving average crossovers
            if "sma_20" in df_with_features.columns and "sma_50" in df_with_features.columns:
                df_with_features["ma_crossover_20_50"] = df_with_features["sma_20"] > df_with_features["sma_50"]
            
            if "sma_50" in df_with_features.columns and "sma_200" in df_with_features.columns:
                df_with_features["ma_crossover_50_200"] = df_with_features["sma_50"] > df_with_features["sma_200"]
        
        # Range breakout
        if "range_breakout" in derived_features:
            # Bollinger Band breakouts
            if "bbands_upper" in df_with_features.columns:
                df_with_features["bb_upper_breakout"] = df_with_features["close"] > df_with_features["bbands_upper"]
                df_with_features["bb_lower_breakout"] = df_with_features["close"] < df_with_features["bbands_lower"]
                
                # Distance from bands
                df_with_features["bb_upper_distance"] = (df_with_features["close"] - df_with_features["bbands_upper"]) / df_with_features["close"]
                df_with_features["bb_lower_distance"] = (df_with_features["close"] - df_with_features["bbands_lower"]) / df_with_features["close"]
            
            # ATR-based breakouts
            if "atr_14" in df_with_features.columns:
                for period in [10, 20]:
                    # Rolling high and low
                    high_period = df_with_features["high"].rolling(window=period).max()
                    low_period = df_with_features["low"].rolling(window=period).min()
                    
                    # Breakout indicators
                    df_with_features[f"high_breakout_{period}"] = df_with_features["close"] > high_period
                    df_with_features[f"low_breakout_{period}"] = df_with_features["close"] < low_period
                    
                    # Distance from breakout levels in ATR units
                    df_with_features[f"high_distance_atr_{period}"] = (df_with_features["close"] - high_period) / df_with_features["atr_14"]
                    df_with_features[f"low_distance_atr_{period}"] = (df_with_features["close"] - low_period) / df_with_features["atr_14"]
        
        # Volume momentum
        if "volume_momentum" in derived_features and "volume" in df_with_features.columns:
            # Volume change over different periods
            for period in [1, 3, 5, 10]:
                df_with_features[f"volume_change_{period}"] = df_with_features["volume"].pct_change(periods=period)
            
            # Volume relative to moving average
            for period in [10, 20, 50]:
                df_with_features[f"volume_sma_{period}"] = df_with_features["volume"].rolling(window=period).mean()
                df_with_features[f"volume_ratio_{period}"] = df_with_features["volume"] / df_with_features[f"volume_sma_{period}"]
            
            # Volume and price divergence
            df_with_features["price_volume_divergence"] = df_with_features["price_change_1"] * df_with_features["volume_change_1"]
            
            # On-balance volume (OBV) momentum
            if "obv" in df_with_features.columns:
                for period in [10, 20]:
                    df_with_features[f"obv_sma_{period}"] = pd.Series(df_with_features["obv"]).rolling(window=period).mean().values
                    df_with_features[f"obv_ratio_{period}"] = df_with_features["obv"] / df_with_features[f"obv_sma_{period}"]
        
        # Support/Resistance levels
        if "support_resistance" in derived_features:
            # Identify potential support/resistance levels
            window_size = 20
            
            # Find local highs and lows
            df_with_features["local_high"] = df_with_features["high"].rolling(window=window_size, center=True).max() == df_with_features["high"]
            df_with_features["local_low"] = df_with_features["low"].rolling(window=window_size, center=True).min() == df_with_features["low"]
            
            # Distance from recent support/resistance
            recent_high = df_with_features["high"].rolling(window=50).max()
            recent_low = df_with_features["low"].rolling(window=50).min()
            
            df_with_features["distance_to_resistance"] = (recent_high - df_with_features["close"]) / df_with_features["close"]
            df_with_features["distance_to_support"] = (df_with_features["close"] - recent_low) / df_with_features["close"]
            
            # Relative position within range
            df_with_features["range_position"] = (df_with_features["close"] - recent_low) / (recent_high - recent_low)
        
        # Market regime features
        # Identify market regimes: trending, ranging, volatile
        if "cci_14" in df_with_features.columns and "adx_14" in df_with_features.columns:
            # Trending: high ADX (>25)
            trending = df_with_features["adx_14"] > 25
            
            # Ranging: low ADX (<20) and low volatility
            ranging = (df_with_features["adx_14"] < 20) & (df_with_features["bbands_width"] < df_with_features["bbands_width"].rolling(window=50).mean())
            
            # Volatile: high ATR and wideninng Bollinger Bands
            if "atr_14" in df_with_features.columns:
                volatile = (df_with_features["atr_14"] > df_with_features["atr_14"].rolling(window=20).mean() * 1.5) & \
                           (df_with_features["bbands_width"] > df_with_features["bbands_width"].rolling(window=20).mean() * 1.2)
                
                # Market regime indicator (one-hot encoded)
                df_with_features["market_regime_trending"] = trending.astype(int)
                df_with_features["market_regime_ranging"] = ranging.astype(int)
                df_with_features["market_regime_volatile"] = volatile.astype(int)
        
        # Cyclical features
        # Time-based features
        df_with_features["hour"] = df_with_features["timestamp"].dt.hour
        df_with_features["day_of_week"] = df_with_features["timestamp"].dt.dayofweek
        df_with_features["day_of_month"] = df_with_features["timestamp"].dt.day
        df_with_features["month"] = df_with_features["timestamp"].dt.month
        df_with_features["quarter"] = df_with_features["timestamp"].dt.quarter
        
        # Convert cyclical features to sine/cosine components
        for col in ["hour", "day_of_week", "day_of_month", "month"]:
            max_val = {"hour": 24, "day_of_week": 7, "day_of_month": 31, "month": 12}[col]
            df_with_features[f"{col}_sin"] = np.sin(2 * np.pi * df_with_features[col] / max_val)
            df_with_features[f"{col}_cos"] = np.cos(2 * np.pi * df_with_features[col] / max_val)
        
        # Weekend indicator
        df_with_features["is_weekend"] = df_with_features["day_of_week"].isin([5, 6]).astype(int)
        
    except Exception as e:
        logger.error(f"Error adding derived features: {e}")
    
    return df_with_features


def add_cross_asset_correlations(df: pd.DataFrame, target_pair: str) -> pd.DataFrame:
    """
    Add cross-asset correlations to the dataframe.
    
    Args:
        df: DataFrame with features
        target_pair: Target trading pair (e.g., BTC/USD)
        
    Returns:
        DataFrame with added cross-asset correlations
    """
    # Create a copy to avoid modifying the original
    df_with_correlations = df.copy()
    
    # Major reference pairs to consider for correlations
    reference_pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    # Remove target pair from reference pairs if present
    if target_pair in reference_pairs:
        reference_pairs.remove(target_pair)
    
    try:
        # Load data for reference pairs
        reference_data = {}
        
        for pair in reference_pairs:
            pair_data = load_historical_data(pair, "1h")
            if pair_data is not None:
                reference_data[pair] = pair_data
        
        # Calculate correlations
        for pair, pair_data in reference_data.items():
            # Resample if needed to match target dataframe timestamps
            pair_data = pair_data.set_index("timestamp")
            pair_data = pair_data.reindex(df_with_correlations["timestamp"])
            
            # Calculate rolling correlation for price
            corr_col_name = f"{pair.split('/')[0].lower()}_correlation"
            df_with_correlations[corr_col_name] = df_with_correlations["close"].rolling(CORRELATION_LOOKBACK).corr(pair_data["close"])
            
            # Price ratios
            price_ratio_col = f"{pair.split('/')[0].lower()}_price_ratio"
            df_with_correlations[price_ratio_col] = df_with_correlations["close"] / pair_data["close"]
            
            # Relative strength
            rel_strength_col = f"{pair.split('/')[0].lower()}_relative_strength"
            df_with_correlations[rel_strength_col] = (df_with_correlations["close"].pct_change(7) - pair_data["close"].pct_change(7))
        
        # Add correlation change
        for pair in reference_pairs:
            corr_col_name = f"{pair.split('/')[0].lower()}_correlation"
            if corr_col_name in df_with_correlations.columns:
                # Correlation momentum (change in correlation)
                df_with_correlations[f"{corr_col_name}_change"] = df_with_correlations[corr_col_name].diff(5)
        
        # Sector performance (if applicable)
        # For crypto, we can use basic categorization
        if target_pair in ["AVAX/USD", "SOL/USD", "MATIC/USD"]:
            # Layer 1 category
            if "sol_correlation" in df_with_correlations.columns and "eth_correlation" in df_with_correlations.columns:
                df_with_correlations["layer1_correlation"] = (df_with_correlations["sol_correlation"] + df_with_correlations["eth_correlation"]) / 2
        
        if target_pair in ["UNI/USD"]:
            # DeFi category
            if "eth_correlation" in df_with_correlations.columns:
                df_with_correlations["defi_correlation"] = df_with_correlations["eth_correlation"]
        
    except Exception as e:
        logger.error(f"Error adding cross-asset correlations: {e}")
    
    return df_with_correlations


def prepare_target_variable(
    df: pd.DataFrame, 
    prediction_horizon: int = 12, 
    threshold: float = 0.01,
    use_atr: bool = True
) -> pd.DataFrame:
    """
    Prepare target variable for ML training.
    
    Args:
        df: DataFrame with features
        prediction_horizon: Prediction horizon in periods
        threshold: Price change threshold for classification
        use_atr: Whether to use ATR for adaptive thresholds
        
    Returns:
        DataFrame with added target variable
    """
    # Create a copy to avoid modifying the original
    df_with_target = df.copy()
    
    try:
        # Calculate future price
        df_with_target["future_price"] = df_with_target["close"].shift(-prediction_horizon)
        
        # Calculate price change
        df_with_target["price_change"] = (df_with_target["future_price"] / df_with_target["close"]) - 1
        
        # Adaptive threshold based on ATR
        if use_atr and "atr_14" in df_with_target.columns:
            # Use ATR to scale threshold (higher volatility -> higher threshold)
            atr_scaled = df_with_target["atr_14"] / df_with_target["close"]
            adaptive_threshold = atr_scaled * 0.5  # Scale factor
            
            # Ensure minimum threshold
            adaptive_threshold = np.maximum(threshold, adaptive_threshold)
            
            # Apply adaptive threshold
            df_with_target["target"] = ((df_with_target["price_change"] > adaptive_threshold) | 
                                      ((df_with_target["price_change"] < -adaptive_threshold) & 
                                       (df_with_target["price_change"] < 0))).astype(int)
        else:
            # Fixed threshold
            df_with_target["target"] = ((df_with_target["price_change"] > threshold) | 
                                      ((df_with_target["price_change"] < -threshold) & 
                                       (df_with_target["price_change"] < 0))).astype(int)
        
        # Additional target variables for multi-task learning
        # Direction only
        df_with_target["target_direction"] = (df_with_target["price_change"] > 0).astype(int)
        
        # Price change regression target
        df_with_target["target_regression"] = df_with_target["price_change"]
        
        # Categorized price change
        bins = [-float('inf'), -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, float('inf')]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df_with_target["target_categorical"] = pd.cut(df_with_target["price_change"], bins=bins, labels=labels).astype(float)
        
    except Exception as e:
        logger.error(f"Error preparing target variable: {e}")
    
    return df_with_target


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with handled missing values
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    try:
        # Forward fill missing values
        df_clean = df_clean.fillna(method="ffill")
        
        # For remaining NaN values, use backward fill
        df_clean = df_clean.fillna(method="bfill")
        
        # For any still remaining NaN values, use column median
        df_clean = df_clean.fillna(df_clean.median())
        
        # Check if any NaN values remain
        if df_clean.isnull().sum().sum() > 0:
            logger.warning("Some NaN values remain after cleaning")
            
            # Remove rows with NaN values
            df_clean = df_clean.dropna()
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with NaN values")
        
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
    
    return df_clean


def prepare_features_for_pair(pair: str, config: Dict, args) -> bool:
    """
    Prepare training features for a specific pair.
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Preparing training features for {pair}")
    
    # Get timeframes from args or config
    timeframes = args.timeframes.split(",") if args.timeframes else config["feature_engineering"]["timeframes"]
    
    # Load pair-specific settings
    pair_config = config["pair_specific_settings"].get(pair, {})
    indicators = config["feature_engineering"]["technical_indicators"]
    derived_features = config["feature_engineering"]["derived_features"]
    
    # Get prediction horizon
    prediction_horizon = args.target_horizon or pair_config.get("prediction_horizon", 12)
    
    # Get output directory
    output_dir = args.output_dir
    
    # Process each timeframe
    for timeframe in timeframes:
        try:
            # Load historical data
            df = load_historical_data(pair, timeframe)
            if df is None:
                logger.error(f"Could not load historical data for {pair} ({timeframe})")
                continue
            
            # Add technical indicators
            df = add_technical_indicators(df, indicators)
            
            # Add derived features
            df = add_derived_features(df, derived_features)
            
            # Add cross-asset correlations if requested
            if args.include_cross_asset or config["feature_engineering"].get("cross_asset_correlations", False):
                df = add_cross_asset_correlations(df, pair)
            
            # Prepare target variable
            prediction_periods = prediction_horizon
            if timeframe == "1h":
                prediction_periods = prediction_horizon
            elif timeframe == "4h":
                prediction_periods = prediction_horizon // 4
            elif timeframe == "1d":
                prediction_periods = prediction_horizon // 24
            
            df = prepare_target_variable(df, prediction_periods)
            
            # Handle missing values
            df = handle_missing_values(df)
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{pair.replace('/', '_')}_{timeframe}_features.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"Saved processed data to {output_file} ({len(df)} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            logger.error(f"Error processing {pair} ({timeframe}): {e}")
            return False
    
    return True


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Prepare features for the specified pair
    success = prepare_features_for_pair(args.pair, config, args)
    
    if success:
        logger.info(f"Successfully prepared training features for {args.pair}")
        return 0
    else:
        logger.error(f"Failed to prepare training features for {args.pair}")
        return 1


if __name__ == "__main__":
    sys.exit(main())