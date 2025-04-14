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
import numpy as np
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
PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "MATICUSD"]
TIMEFRAMES = {
    "1m": 1,      # 1 minute (lowest available)
    "5m": 5,      # 5 minutes
    "15m": 15,    # 15 minutes
    "30m": 30,    # 30 minutes
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


def calculate_technical_indicators(df, prefix=''):
    """
    Calculate standard technical indicators used by most strategies
    
    Args:
        df (pd.DataFrame): Dataframe with OHLCV data
        prefix (str): Prefix to add to column names to avoid conflicts
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100]:
        df_copy[f'{prefix}sma_{period}'] = df_copy['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [9, 21, 50, 100]:
        df_copy[f'{prefix}ema_{period}'] = df_copy['close'].ewm(span=period, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df_copy['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    df_copy[f'{prefix}rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df_copy['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_copy['close'].ewm(span=26, adjust=False).mean()
    df_copy[f'{prefix}macd'] = ema_12 - ema_26
    df_copy[f'{prefix}macd_signal'] = df_copy[f'{prefix}macd'].ewm(span=9, adjust=False).mean()
    df_copy[f'{prefix}macd_histogram'] = df_copy[f'{prefix}macd'] - df_copy[f'{prefix}macd_signal']
    
    # Bollinger Bands
    sma_20 = df_copy['close'].rolling(window=20).mean()
    std_20 = df_copy['close'].rolling(window=20).std()
    df_copy[f'{prefix}bb_upper'] = sma_20 + (std_20 * 2)
    df_copy[f'{prefix}bb_middle'] = sma_20
    df_copy[f'{prefix}bb_lower'] = sma_20 - (std_20 * 2)
    
    # Bollinger Band Width (measure of volatility)
    df_copy[f'{prefix}bb_width'] = (df_copy[f'{prefix}bb_upper'] - df_copy[f'{prefix}bb_lower']) / df_copy[f'{prefix}bb_middle']
    
    # Stochastic Oscillator
    low_14 = df_copy['low'].rolling(window=14).min()
    high_14 = df_copy['high'].rolling(window=14).max()
    df_copy[f'{prefix}stoch_k'] = 100 * ((df_copy['close'] - low_14) / (high_14 - low_14 + 1e-9))
    df_copy[f'{prefix}stoch_d'] = df_copy[f'{prefix}stoch_k'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df_copy['high'] - df_copy['low']
    high_close = (df_copy['high'] - df_copy['close'].shift()).abs()
    low_close = (df_copy['low'] - df_copy['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_copy[f'{prefix}atr_14'] = true_range.rolling(window=14).mean()
    
    # ADX (Average Directional Index)
    plus_dm = df_copy['high'].diff()
    minus_dm = df_copy['low'].diff(-1).abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth directional movements and calculate DI
    tr_14 = true_range.rolling(window=14).sum()
    plus_di_14 = 100 * plus_dm.rolling(window=14).sum() / tr_14
    minus_di_14 = 100 * minus_dm.rolling(window=14).sum() / tr_14
    
    # Calculate DX and ADX
    dx = 100 * (plus_di_14 - minus_di_14).abs() / (plus_di_14 + minus_di_14 + 1e-9)
    df_copy[f'{prefix}adx_14'] = dx.rolling(window=14).mean()
    
    # Calculate some useful ratios
    df_copy[f'{prefix}close_to_sma_20'] = df_copy['close'] / df_copy[f'{prefix}sma_20']
    df_copy[f'{prefix}close_to_sma_50'] = df_copy['close'] / df_copy[f'{prefix}sma_50']
    df_copy[f'{prefix}close_to_ema_9'] = df_copy['close'] / df_copy[f'{prefix}ema_9']
    
    # Price momentum and rate of change
    for period in [1, 3, 5, 10, 20]:
        df_copy[f'{prefix}momentum_{period}'] = df_copy['close'] - df_copy['close'].shift(period)
        df_copy[f'{prefix}roc_{period}'] = df_copy['close'].pct_change(periods=period) * 100
    
    # Volume indicators
    df_copy[f'{prefix}volume_sma_20'] = df_copy['volume'].rolling(window=20).mean()
    df_copy[f'{prefix}volume_ratio'] = df_copy['volume'] / df_copy[f'{prefix}volume_sma_20']
    
    # On-Balance Volume (OBV)
    obv = (df_copy['volume'] * ((df_copy['close'].diff() > 0).astype(int) - 
                              (df_copy['close'].diff() < 0).astype(int))).cumsum()
    df_copy[f'{prefix}obv'] = obv
    
    # Ichimoku Cloud indicators (simplified)
    high_9 = df_copy['high'].rolling(window=9).max()
    low_9 = df_copy['low'].rolling(window=9).min()
    df_copy[f'{prefix}ichimoku_conversion'] = (high_9 + low_9) / 2
    
    high_26 = df_copy['high'].rolling(window=26).max()
    low_26 = df_copy['low'].rolling(window=26).min()
    df_copy[f'{prefix}ichimoku_base'] = (high_26 + low_26) / 2
    
    return df_copy


def merge_low_timeframe_data(pair):
    """
    Merge low timeframe data (1m, 5m, 15m, 30m) for a pair to create a high-resolution dataset
    optimized for ML model training with enhanced feature extraction.
    
    Args:
        pair (str): Trading pair
        
    Returns:
        pd.DataFrame: Merged dataset with high-resolution data and enhanced features
    """
    logger.info(f"Creating high-resolution dataset for {pair} from low timeframes")
    
    # Base it on the 1m timeframe if available, otherwise use the lowest available
    for base_tf in ['1m', '5m', '15m', '30m']:
        base_file = os.path.join(DATA_DIR, pair, f"{pair}_{base_tf}.csv")
        if os.path.exists(base_file):
            break
    else:
        logger.error(f"No low timeframe data found for {pair}")
        return pd.DataFrame()
    
    logger.info(f"Using {base_tf} as base timeframe for high-resolution dataset")
    
    base_df = pd.read_csv(base_file)
    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
    base_df.set_index('timestamp', inplace=True)
    
    # Calculate technical indicators for base timeframe
    logger.info(f"Calculating technical indicators for {base_tf} data")
    base_df = calculate_technical_indicators(base_df, prefix='base_')
    
    # Calculate additional features on base timeframe
    base_df['base_log_return'] = (base_df['close'] / base_df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else np.log(x))
    base_df['base_volume_change'] = base_df['volume'].pct_change()
    base_df['base_price_range'] = (base_df['high'] - base_df['low']) / base_df['close']
    
    # Add direction feature
    base_df['base_direction'] = 0  # Initialize
    base_df.loc[base_df['close'] > base_df['close'].shift(1), 'base_direction'] = 1  # Up
    base_df.loc[base_df['close'] < base_df['close'].shift(1), 'base_direction'] = -1  # Down
    
    # Fractal indicators (patterns within patterns)
    # Identify potential reversal points
    for i in range(2, 6):  # Look for patterns of different sizes
        # Bullish fractal (local low)
        condition_bull = (base_df['low'].shift(i) > base_df['low'].shift(i-1)) & \
                        (base_df['low'].shift(i-1) > base_df['low']) & \
                        (base_df['low'] < base_df['low'].shift(1)) & \
                        (base_df['low'].shift(1) < base_df['low'].shift(2))
        base_df[f'base_fractal_bull_{i}'] = condition_bull.astype(int)
        
        # Bearish fractal (local high)
        condition_bear = (base_df['high'].shift(i) < base_df['high'].shift(i-1)) & \
                        (base_df['high'].shift(i-1) < base_df['high']) & \
                        (base_df['high'] > base_df['high'].shift(1)) & \
                        (base_df['high'].shift(1) > base_df['high'].shift(2))
        base_df[f'base_fractal_bear_{i}'] = condition_bear.astype(int)
    
    # Add other low timeframes as features
    low_timeframes = ['1m', '5m', '15m', '30m']
    low_timeframes.remove(base_tf)  # Remove the base timeframe
    
    for timeframe in low_timeframes:
        tf_file = os.path.join(DATA_DIR, pair, f"{pair}_{timeframe}.csv")
        if not os.path.exists(tf_file):
            continue
        
        logger.info(f"Processing {timeframe} data for {pair}")
        tf_df = pd.read_csv(tf_file)
        tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])
        tf_df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators for this timeframe
        logger.info(f"Calculating technical indicators for {timeframe} data")
        tf_df = calculate_technical_indicators(tf_df, prefix=f'{timeframe}_')
        
        # Calculate advanced indicators for this timeframe
        # Log returns
        tf_df[f'{timeframe}_log_return'] = (tf_df['close'] / tf_df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else np.log(x))
        
        # Volatility (based on Garman-Klass range)
        high_low = np.log(tf_df['high'] / tf_df['low']).replace([np.inf, -np.inf], 0)
        close_open = np.log(tf_df['close'] / tf_df['open']).replace([np.inf, -np.inf], 0)
        tf_df[f'{timeframe}_volatility'] = 0.5 * high_low**2 - (2*np.log(2)-1) * close_open**2
        
        # Volume profile
        tf_df[f'{timeframe}_volume_change'] = tf_df['volume'].pct_change()
        
        # Price movement characteristics
        tf_df[f'{timeframe}_price_range'] = (tf_df['high'] - tf_df['low']) / tf_df['close']
        
        # Calculate close to high ratio, avoiding division by zero
        high_low_diff = tf_df['high'] - tf_df['low']
        tf_df[f'{timeframe}_close_to_high'] = (tf_df['close'] - tf_df['low']) / high_low_diff.replace(0, 1)
        
        # Momentum indicators
        tf_df[f'{timeframe}_close_shift'] = tf_df['close'].shift(1)
        tf_df[f'{timeframe}_momentum'] = tf_df['close'] - tf_df[f'{timeframe}_close_shift']
        
        # Pattern recognition: Calculate indicators for candlestick patterns
        # Doji
        tf_df[f'{timeframe}_doji'] = ((tf_df['close'] - tf_df['open']).abs() / (tf_df['high'] - tf_df['low']) < 0.1).astype(int)
        
        # Engulfing patterns
        bullish_engulfing = (tf_df['open'].shift(1) > tf_df['close'].shift(1)) & \
                           (tf_df['close'] > tf_df['open']) & \
                           (tf_df['close'] >= tf_df['open'].shift(1)) & \
                           (tf_df['open'] <= tf_df['close'].shift(1))
                           
        bearish_engulfing = (tf_df['close'].shift(1) > tf_df['open'].shift(1)) & \
                           (tf_df['open'] > tf_df['close']) & \
                           (tf_df['open'] >= tf_df['close'].shift(1)) & \
                           (tf_df['close'] <= tf_df['open'].shift(1))
                           
        tf_df[f'{timeframe}_bullish_engulfing'] = bullish_engulfing.astype(int)
        tf_df[f'{timeframe}_bearish_engulfing'] = bearish_engulfing.astype(int)
        
        # Remove original OHLCV columns to keep only derived features
        tf_df = tf_df.drop(['open', 'high', 'low', 'close', 'volume', f'{timeframe}_close_shift'], axis=1, errors='ignore')
        
        # Resample to match the base timeframe
        if timeframe == '1m':
            # No resampling needed if base is higher
            pass
        elif base_tf == '1m':
            # Forward fill the higher timeframe data to 1m
            minutes = int(timeframe[:-1])
            tf_df = tf_df.resample(f"{minutes}min").ffill()
        else:
            # More complex resampling between different timeframes
            base_minutes = int(base_tf[:-1])
            tf_minutes = int(timeframe[:-1])
            
            if tf_minutes < base_minutes:  # e.g., 5m into 15m
                # Aggregate to base timeframe
                tf_df = tf_df.resample(f"{base_minutes}min").last()
            else:  # e.g., 15m into 5m
                # Forward fill to match base timeframe
                tf_df = tf_df.resample(f"{base_minutes}min").ffill()
        
        # First rename columns to avoid duplicates
        tf_cols = tf_df.columns.tolist()
        rename_dict = {col: f"{col}_{timeframe}" for col in tf_cols}
        tf_df = tf_df.rename(columns=rename_dict)
        
        # Reset index for join
        tf_df = tf_df.reset_index()
        
        # Join with base dataframe
        base_df = pd.merge(base_df, tf_df, on='timestamp', how='left')
    
    # Forward fill any NaN values from the joins
    base_df = base_df.fillna(method='ffill')
    
    # Calculate cross-timeframe correlation features
    for tf1 in ['base'] + [tf for tf in low_timeframes if tf != base_tf]:
        for tf2 in ['base'] + [tf for tf in low_timeframes if tf != base_tf]:
            if tf1 != tf2:
                # Create correlation column between different timeframe indicators
                if f'{tf1}_rsi_14' in base_df.columns and f'{tf2}_rsi_14' in base_df.columns:
                    col_name = f'correlation_{tf1}_{tf2}_rsi'
                    # Correlation over a rolling window of 10 periods
                    base_df[col_name] = base_df[f'{tf1}_rsi_14'].rolling(10).corr(base_df[f'{tf2}_rsi_14'])
    
    # Add target labels (next candle direction)
    # 1 for up, 0 for down over the next candle
    base_df['next_candle_up'] = (base_df['close'].shift(-1) > base_df['close']).astype(int)
    
    # Add target percentage moves for regression models
    base_df['next_candle_pct_change'] = base_df['close'].pct_change(-1)  # Next candle percent change
    
    # Create larger targets for multi-period prediction (3, 5, 10 periods)
    for periods in [3, 5, 10, 20]:
        # Classification targets
        future_price = base_df['close'].shift(-periods)
        base_df[f'target_{periods}_up'] = (future_price > base_df['close']).astype(int)
        
        # Regression targets
        base_df[f'target_{periods}_pct_change'] = (future_price - base_df['close']) / base_df['close']
        
        # Target volatility (how much the price might move in either direction)
        future_high = base_df['high'].rolling(periods).max().shift(-periods)
        future_low = base_df['low'].rolling(periods).min().shift(-periods)
        base_df[f'target_{periods}_volatility'] = (future_high - future_low) / base_df['close']
    
    # Clean any remaining NaN values
    base_df = base_df.fillna(0)
    
    # Save merged dataset
    output_file = os.path.join(DATA_DIR, pair, f"{pair}_enhanced_low_tf.csv")
    base_df.reset_index().to_csv(output_file, index=False)
    logger.info(f"Saved enhanced high-resolution dataset with {len(base_df)} rows to {output_file}")
    
    return base_df


def main():
    """Main function to fetch historical data for all pairs and timeframes"""
    # Ensure data directories exist
    ensure_data_directories()
    
    # Fetch data for each pair and timeframe
    for pair in PAIRS:
        # Fetch low timeframe data first (more recent history)
        for interval_name in ['1m', '5m', '15m', '30m']:
            if interval_name == '1m':
                days_back = 7  # 1 week of minute data
            elif interval_name == '5m':
                days_back = 14  # 2 weeks of 5m data
            elif interval_name == '15m':
                days_back = 21  # 3 weeks of 15m data
            else:  # 30m
                days_back = 30  # 1 month of 30m data
                
            fetch_historical_data(pair, interval_name, days_back)
        
        # Create high-resolution dataset from low timeframes
        merge_low_timeframe_data(pair)
        
        # Fetch standard timeframe data (more historical coverage)
        for interval_name in ['1h', '4h', '1d']:
            if interval_name == '1h':
                days_back = 60  # 2 months of hourly data
            elif interval_name == '4h':
                days_back = 180  # 6 months of 4h data
            else:  # 1d
                days_back = 730  # 2 years of daily data
                
            fetch_historical_data(pair, interval_name, days_back)
        
        # Create multi-timeframe dataset (standard timeframes)
        merge_timeframes(pair)
    
    logger.info("All historical data fetched and processed successfully")


if __name__ == "__main__":
    main()