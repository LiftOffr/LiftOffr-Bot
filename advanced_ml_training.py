#!/usr/bin/env python3
"""
Advanced ML Training for Kraken Trading Bot

This script implements advanced training techniques to significantly improve model accuracy:
1. Feature engineering with more sophisticated technical indicators
2. Multi-timeframe data integration
3. Advanced hyperparameter tuning
4. Market regime-specific models
5. Ensemble stacking approaches
6. Sentiment data integration (optional)

Targeting 90% directional accuracy.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, SpatialDropout1D
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tcn import TCN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
MODELS_DIR = "models"
MODEL_TYPES = ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid"]
TIMEFRAMES = ["1h", "4h", "1d"]  # Multiple timeframes for context
SEED = 42
TRAIN_EPOCHS = 100  # Increase training epochs
PATIENCE = 15  # More patience for early stopping
BATCH_SIZE = 32

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure model directories exist
for model_type in MODEL_TYPES:
    os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)

os.makedirs(os.path.join(MODELS_DIR, "ensemble"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "regime_specific"), exist_ok=True)


def calculate_advanced_indicators(df):
    """
    Calculate advanced technical indicators for improved model performance.
    
    Args:
        df (pd.DataFrame): Price data DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Basic indicators
    # Moving averages
    for window in [5, 9, 20, 50, 100, 200]:
        df[f'sma{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # Price relative to moving averages
    for window in [9, 20, 50, 100]:
        df[f'price_sma{window}_ratio'] = df['close'] / df[f'sma{window}']
        df[f'price_ema{window}_ratio'] = df['close'] / df[f'ema{window}']
    
    # Moving average crossovers
    df['ema9_20_cross'] = (df['ema9'] - df['ema20']).pct_change()
    df['ema20_50_cross'] = (df['ema20'] - df['ema50']).pct_change()
    df['ema50_100_cross'] = (df['ema50'] - df['ema100']).pct_change()
    
    # Volatility indicators
    df['atr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr14'] = df['atr'].rolling(window=14).mean()
    df['atr_pct'] = df['atr14'] / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    df['volatility_50'] = df['close'].pct_change().rolling(window=50).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Multiple RSI periods
    for window in [6, 14, 21]:
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        df[f'rsi{window}'] = 100 - (100 / (1 + rs))
    
    # Advanced RSI features
    df['rsi_ma5'] = df['rsi14'].rolling(window=5).mean()
    df['rsi_ma_diff'] = df['rsi14'] - df['rsi_ma5']
    
    # Bollinger Bands
    for window in [20, 50]:
        df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
        df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + (df[f'bb_std_{window}'] * 2)
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - (df[f'bb_std_{window}'] * 2)
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        df[f'bb_pct_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 0.00001)
    
    # MACD
    # Multiple MACD configurations
    macd_configs = [
        (8, 21, 5),   # Fast
        (12, 26, 9),  # Standard
        (21, 55, 13), # Slow
    ]
    
    for fast, slow, signal in macd_configs:
        fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
        df[f'macd_{fast}_{slow}'] = fast_ema - slow_ema
        df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
        df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
        df[f'macd_hist_diff_{fast}_{slow}'] = df[f'macd_hist_{fast}_{slow}'].diff()
    
    # Stochastic Oscillator
    for k_period in [9, 14]:
        for d_period in [3, 5]:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            df[f'stoch_k_{k_period}'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 0.00001))
            df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
            df[f'stoch_diff_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'] - df[f'stoch_d_{k_period}_{d_period}']
    
    # ADX (Average Directional Index)
    high_change = df['high'] - df['high'].shift(1)
    low_change = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((high_change > low_change) & (high_change > 0), high_change, 0)
    minus_dm = np.where((low_change > high_change) & (low_change > 0), low_change, 0)
    
    df['plus_dm14'] = pd.Series(plus_dm).rolling(window=14).mean()
    df['minus_dm14'] = pd.Series(minus_dm).rolling(window=14).mean()
    
    atr14 = df['atr'].rolling(window=14).mean()
    df['plus_di14'] = 100 * df['plus_dm14'] / atr14
    df['minus_di14'] = 100 * df['minus_dm14'] / atr14
    dx = 100 * np.abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'] + 0.00001)
    df['adx14'] = dx.rolling(window=14).mean()
    
    # Ichimoku Cloud
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    df['cloud_diff'] = df['senkou_span_a'] - df['senkou_span_b']
    df['price_vs_cloud'] = df['close'] - ((df['senkou_span_a'] + df['senkou_span_b']) / 2)
    
    # Volume indicators
    if 'volume' in df.columns:
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ma_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_oscillator'] = df['volume_ma5'] / df['volume_ma20'] - 1
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma5'] = df['obv'].rolling(window=5).mean()
        df['obv_diff'] = df['obv'] - df['obv_ma5']
        
        # Chaikin Money Flow (CMF)
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 0.00001)
        mf_volume = mf_multiplier * df['volume']
        df['cmf14'] = mf_volume.rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Price action patterns
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
    df['body_vs_shadow'] = df['body_size'] / (df['upper_shadow'] + df['lower_shadow'] + 0.00001)
    
    # Price momentum
    for window in [3, 5, 14, 21]:
        df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
        df[f'price_rate_of_change_{window}'] = df['close'].pct_change(periods=window) * 100
    
    # Rate of Change (ROC) of various indicators
    for indicator in ['rsi14', 'macd_12_26', 'adx14']:
        if indicator in df.columns:
            df[f'{indicator}_roc3'] = df[indicator].pct_change(periods=3)
            df[f'{indicator}_roc5'] = df[indicator].pct_change(periods=5)
    
    # Market regime features
    df['trend_strength'] = np.abs(df['ema20'] / df['ema50'] - 1) * 100
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(window=100).mean()).astype(int)
    
    # Advanced targets for multi-classification
    # Create target variables with horizons of 1, 3, 5, and 10 periods ahead
    for horizon in [1, 3, 5, 10]:
        # Directional target (up/down)
        df[f'direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        
        # Magnitude target (pct change)
        df[f'return_{horizon}'] = df['close'].pct_change(periods=horizon).shift(-horizon)
        
    return df


def load_multi_timeframe_data(symbol="SOLUSD", timeframes=None):
    """
    Load data from multiple timeframes for integrated training.
    
    Args:
        symbol (str): Trading symbol
        timeframes (list): List of timeframes to load
        
    Returns:
        dict: Dictionary of DataFrames for each timeframe
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    timeframe_data = {}
    
    for tf in timeframes:
        file_path = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Data file for {symbol} {tf} not found: {file_path}")
            continue
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate advanced indicators
            df = calculate_advanced_indicators(df)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            timeframe_data[tf] = df
            logger.info(f"Loaded and processed {len(df)} rows from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {tf} data: {e}")
            import traceback
            traceback.print_exc()
    
    return timeframe_data


def align_multi_timeframe_data(timeframe_data, primary_timeframe="1h"):
    """
    Align data from multiple timeframes to the timestamps of the primary timeframe.
    
    Args:
        timeframe_data (dict): Dictionary of DataFrames for each timeframe
        primary_timeframe (str): The primary timeframe to align others to
        
    Returns:
        pd.DataFrame: Combined DataFrame with aligned multi-timeframe data
    """
    if primary_timeframe not in timeframe_data:
        logger.error(f"Primary timeframe {primary_timeframe} data not found")
        return None
    
    # Get primary timeframe data
    primary_df = timeframe_data[primary_timeframe].copy()
    
    # Process each non-primary timeframe
    for tf, df in timeframe_data.items():
        if tf == primary_timeframe:
            continue
        
        # Create a suffix for the timeframe
        suffix = f"_{tf}"
        
        # Select key features from this timeframe
        key_features = [
            'open', 'high', 'low', 'close', 'volume',
            'ema9', 'ema20', 'ema50', 'ema100', 
            'rsi14', 'volatility_20', 'macd_12_26', 'adx14',
            'bb_width_20', 'trend_strength'
        ]
        
        # Filter features that exist in the dataframe
        existing_features = [f for f in key_features if f in df.columns]
        
        # Create a dataframe with selected features
        tf_df = df[['timestamp'] + existing_features].copy()
        
        # Rename columns to add timeframe suffix
        for col in existing_features:
            tf_df.rename(columns={col: f"{col}{suffix}"}, inplace=True)
        
        # Align to primary timeframe using forward fill method
        primary_df = pd.merge_asof(
            primary_df, 
            tf_df, 
            on='timestamp',
            direction='backward'
        )
    
    # Ensure all aligned data is present (no NaNs)
    primary_df = primary_df.fillna(method='ffill').dropna()
    
    return primary_df


def detect_market_regimes(df, lookback=100):
    """
    Detect market regimes in the data to train regime-specific models.
    
    Args:
        df (pd.DataFrame): DataFrame with price data and indicators
        lookback (int): Lookback period for regime comparison
        
    Returns:
        pd.DataFrame: DataFrame with added regime labels
    """
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Volatility measure
    volatility = df['close'].pct_change().rolling(window=20).std()
    historical_volatility = volatility.rolling(window=lookback).mean()
    volatility_ratio = volatility / historical_volatility
    
    # Trend measure
    trend_strength = np.abs(df['ema20'] / df['ema50'] - 1)
    historical_trend = trend_strength.rolling(window=lookback).mean()
    trend_ratio = trend_strength / historical_trend
    
    # Regime identification
    # 4 regimes: normal, volatile, trending, volatile + trending
    df['is_volatile'] = (volatility_ratio > 1.25).astype(int)
    df['is_trending'] = (trend_ratio > 1.25).astype(int)
    
    df['regime'] = 'normal'
    df.loc[df['is_volatile'] == 1, 'regime'] = 'volatile'
    df.loc[df['is_trending'] == 1, 'regime'] = 'trending'
    df.loc[(df['is_volatile'] == 1) & (df['is_trending'] == 1), 'regime'] = 'volatile_trending'
    
    # Create numeric regime code
    regime_map = {'normal': 0, 'volatile': 1, 'trending': 2, 'volatile_trending': 3}
    df['regime_code'] = df['regime'].map(regime_map)
    
    regime_counts = df['regime'].value_counts()
    logger.info(f"Market regime distribution: {regime_counts.to_dict()}")
    
    return df


def prepare_model_data(aligned_data, test_size=0.15, validation_size=0.15, sequence_length=24):
    """
    Prepare data for model training, including detecting market regimes and creating sequences.
    
    Args:
        aligned_data (pd.DataFrame): Aligned multi-timeframe data
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of training data for validation
        sequence_length (int): Sequence length for time series models
        
    Returns:
        tuple: Training and validation data, feature names, regime data
    """
    # Copy dataframe to avoid modifying original
    df = aligned_data.copy()
    
    # Detect market regimes
    df = detect_market_regimes(df)
    
    # Add return and target columns
    df['return'] = df['close'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # Binary up/down for next candle
    
    # Multiple target horizons
    for h in [3, 5, 10]:
        df[f'target_{h}'] = (df['close'].shift(-h) > df['close']).astype(int)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Select features for modeling
    exclude_cols = [
        'timestamp', 'return', 'target', 'target_3', 'target_5', 'target_10',
        'regime', 'direction_1', 'direction_3', 'direction_5', 'direction_10',
        'return_1', 'return_3', 'return_5', 'return_10'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract feature data
    X = df[feature_cols].values
    feature_names = feature_cols
    
    # Extract target data (using target_column to allow different prediction horizons)
    y = df['target'].values
    
    # Keep regime for stratified sampling and specialized models
    regimes = df['regime_code'].values
    
    # Split by time (chronological order)
    train_size = int(len(X) * (1 - test_size - validation_size))
    val_size = int(len(X) * validation_size)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    regimes_train = regimes[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    regimes_val = regimes[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    regimes_test = regimes[train_size+val_size:]
    
    # Create sequences for time series models
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # Create sequences for regimes
    regimes_train_seq = create_sequences_single(regimes_train, sequence_length)
    regimes_val_seq = create_sequences_single(regimes_val, sequence_length)
    regimes_test_seq = create_sequences_single(regimes_test, sequence_length)
    
    # Calculate mean and standard deviation for normalization
    scaler = StandardScaler()
    # Reshape to 2D for scaling
    X_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    scaler.fit(X_flat)
    
    # Normalize the data
    X_train_seq_norm = normalize_sequences(X_train_seq, scaler)
    X_val_seq_norm = normalize_sequences(X_val_seq, scaler)
    X_test_seq_norm = normalize_sequences(X_test_seq, scaler)
    
    # Extract normalization parameters
    norm_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    
    # Package the data
    train_data = (X_train_seq_norm, y_train_seq, regimes_train_seq)
    val_data = (X_val_seq_norm, y_val_seq, regimes_val_seq)
    test_data = (X_test_seq_norm, y_test_seq, regimes_test_seq)
    
    logger.info(f"Prepared sequences - Train: {X_train_seq_norm.shape}, Val: {X_val_seq_norm.shape}, Test: {X_test_seq_norm.shape}")
    
    return train_data, val_data, test_data, feature_names, norm_params


def create_sequences(X, y, seq_length=24):
    """
    Create sequences for time series models.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        seq_length (int): Sequence length
        
    Returns:
        tuple: (X_seq, y_seq)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)


def create_sequences_single(X, seq_length=24):
    """
    Create sequences for a single feature (e.g., regime labels).
    
    Args:
        X (numpy.ndarray): Feature vector
        seq_length (int): Sequence length
        
    Returns:
        numpy.ndarray: Sequences
    """
    X_seq = []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
    
    return np.array(X_seq)


def normalize_sequences(X_seq, scaler):
    """
    Normalize sequences using a fitted scaler.
    
    Args:
        X_seq (numpy.ndarray): Sequence data
        scaler: Fitted scaler
        
    Returns:
        numpy.ndarray: Normalized sequences
    """
    # Get original shape
    orig_shape = X_seq.shape
    
    # Reshape to 2D for normalization
    X_flat = X_seq.reshape(-1, orig_shape[-1])
    
    # Normalize
    X_flat_norm = scaler.transform(X_flat)
    
    # Reshape back to original shape
    X_seq_norm = X_flat_norm.reshape(orig_shape)
    
    return X_seq_norm


def build_advanced_tcn_model(input_shape, n_filters=64, kernel_size=3, dilations=None, 
                           dropout_rate=0.3, l1_reg=0.0001, l2_reg=0.0001):
    """
    Build an advanced TCN model with regularization and batch normalization.
    
    Args:
        input_shape (tuple): Shape of input data
        n_filters (int): Number of filters in TCN layers
        kernel_size (int): Size of convolutional kernel
        dilations (list): List of dilation rates
        dropout_rate (float): Dropout rate
        l1_reg (float): L1 regularization coefficient
        l2_reg (float): L2 regularization coefficient
        
    Returns:
        Model: Advanced TCN model
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    
    inputs = Input(shape=input_shape)
    
    x = TCN(
        nb_filters=n_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        return_sequences=True,
        dropout_rate=dropout_rate,
        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
        use_skip_connections=True,
        use_batch_norm=True
    )(inputs)
    
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_advanced_cnn_model(input_shape, n_filters=64, kernel_size=3, dropout_rate=0.3,
                           l1_reg=0.0001, l2_reg=0.0001):
    """
    Build an advanced CNN model with regularization and batch normalization.
    
    Args:
        input_shape (tuple): Shape of input data
        n_filters (int): Number of filters in Conv1D layers
        kernel_size (int): Size of convolutional kernel
        dropout_rate (float): Dropout rate
        l1_reg (float): L1 regularization coefficient
        l2_reg (float): L2 regularization coefficient
        
    Returns:
        Model: Advanced CNN model
    """
    inputs = Input(shape=input_shape)
    
    # Multi-scale convolutional layers
    conv1 = Conv1D(filters=n_filters, kernel_size=2, activation='relu', 
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv2 = BatchNormalization()(conv2)
    
    conv3 = Conv1D(filters=n_filters, kernel_size=5, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv3 = BatchNormalization()(conv3)
    
    # Downsample to make shapes compatible for concatenation
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    
    # Concatenate multi-scale features
    concat = Concatenate()([pool1, pool2, pool3])
    x = SpatialDropout1D(dropout_rate)(concat)
    
    # Second level convolution
    x = Conv1D(filters=n_filters*2, kernel_size=3, activation='relu',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_advanced_lstm_model(input_shape, units=128, dropout_rate=0.3, recurrent_dropout=0.3,
                            l1_reg=0.0001, l2_reg=0.0001):
    """
    Build an advanced LSTM model with regularization and attention.
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        recurrent_dropout (float): Recurrent dropout rate
        l1_reg (float): L1 regularization coefficient
        l2_reg (float): L2 regularization coefficient
        
    Returns:
        Model: Advanced LSTM model
    """
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with return sequences
    x = LSTM(units=units, return_sequences=True, 
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            dropout=dropout_rate, recurrent_dropout=recurrent_dropout)(inputs)
    x = BatchNormalization()(x)
    
    # Self-attention mechanism
    attention = MultiHeadAttention(
        key_dim=units//4, num_heads=4, dropout=dropout_rate
    )(x, x)
    x = Add()([x, attention])
    x = LayerNormalization()(x)
    
    # Second LSTM layer
    x = LSTM(units=units//2, 
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            dropout=dropout_rate, recurrent_dropout=recurrent_dropout)(x)
    x = BatchNormalization()(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_advanced_attention_model(input_shape, units=128, num_heads=8, dropout_rate=0.3,
                                 l1_reg=0.0001, l2_reg=0.0001):
    """
    Build an advanced multi-head self-attention model.
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of units in LSTM layer
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
        l1_reg (float): L1 regularization coefficient
        l2_reg (float): L2 regularization coefficient
        
    Returns:
        Model: Advanced attention model
    """
    inputs = Input(shape=input_shape)
    
    # Initial feature extraction
    x = Conv1D(filters=units, kernel_size=1, activation='relu',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    x = BatchNormalization()(x)
    
    # Transformer-style blocks
    for i in range(3):  # Multiple layers of attention
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=units//num_heads, dropout=dropout_rate
        )(x, x)
        
        # Add & Normalize (residual connection)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Feed-forward network
        ffn_output = Dense(units*4, activation='relu',
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(units,
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(ffn_output)
        
        # Add & Normalize (residual connection)
        x = Add()([x, ffn_output])
        x = LayerNormalization()(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu',
             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(32, activation='relu',
             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_advanced_hybrid_model(input_shape, cnn_filters=64, lstm_units=128, 
                              num_heads=4, dropout_rate=0.3, l1_reg=0.0001, l2_reg=0.0001):
    """
    Build an advanced hybrid model combining CNN, LSTM, and Attention mechanisms.
    
    Args:
        input_shape (tuple): Shape of input data
        cnn_filters (int): Number of CNN filters
        lstm_units (int): Number of LSTM units
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
        l1_reg (float): L1 regularization coefficient
        l2_reg (float): L2 regularization coefficient
        
    Returns:
        Model: Advanced hybrid model
    """
    inputs = Input(shape=input_shape)
    
    # Multi-scale CNN feature extraction
    conv1 = Conv1D(filters=cnn_filters, kernel_size=2, activation='relu', 
                  padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv1 = BatchNormalization()(conv1)
    
    conv3 = Conv1D(filters=cnn_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv3 = BatchNormalization()(conv3)
    
    conv5 = Conv1D(filters=cnn_filters, kernel_size=5, activation='relu',
                  padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    conv5 = BatchNormalization()(conv5)
    
    # Concatenate multi-scale features
    concat = Concatenate()([conv1, conv3, conv5])
    x = SpatialDropout1D(dropout_rate)(concat)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(lstm_units, return_sequences=True,
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                          recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                          dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
    x = BatchNormalization()(x)
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=lstm_units//2, dropout=dropout_rate
    )(x, x)
    
    # Add & Normalize (residual connection)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Global pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    
    # Concatenate different pooling strategies
    x = Concatenate()([avg_pool, max_pool])
    
    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model_with_regimes(model_type, train_data, val_data, input_shape, 
                           callbacks=None, batch_size=32, epochs=100):
    """
    Train a specific model type with regime awareness.
    
    Args:
        model_type (str): Type of model to train
        train_data (tuple): Training data (X, y, regimes)
        val_data (tuple): Validation data (X, y, regimes)
        input_shape (tuple): Shape of input data
        callbacks (list): Callbacks for training
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        
    Returns:
        tuple: (model, history)
    """
    X_train, y_train, regimes_train = train_data
    X_val, y_val, regimes_val = val_data
    
    # Build model based on type
    if model_type == "tcn":
        model = build_advanced_tcn_model(input_shape)
    elif model_type == "cnn":
        model = build_advanced_cnn_model(input_shape)
    elif model_type == "lstm":
        model = build_advanced_lstm_model(input_shape)
    elif model_type == "gru":
        model = build_advanced_lstm_model(input_shape, units=96)  # GRU variant
    elif model_type == "bilstm":
        model = build_advanced_lstm_model(input_shape, units=96)  # BiLSTM variant
    elif model_type == "attention":
        model = build_advanced_attention_model(input_shape)
    elif model_type == "transformer":
        model = build_advanced_attention_model(input_shape, num_heads=8)  # Transformer variant
    elif model_type == "hybrid":
        model = build_advanced_hybrid_model(input_shape)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None
    
    # Setup default callbacks if none provided
    if callbacks is None:
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, model_type, "best_model.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 1.0},  # Can be adjusted for class imbalance
        verbose=1
    )
    
    return model, history


def build_regime_specific_models(train_data, val_data, input_shape, model_type="hybrid"):
    """
    Build regime-specific models, one for each market regime.
    
    Args:
        train_data (tuple): Training data (X, y, regimes)
        val_data (tuple): Validation data (X, y, regimes)
        input_shape (tuple): Shape of input data
        model_type (str): Type of model to use
        
    Returns:
        dict: Dictionary of trained models, one per regime
    """
    X_train, y_train, regimes_train = train_data
    X_val, y_val, regimes_val = val_data
    
    regime_models = {}
    regime_codes = [0, 1, 2, 3]  # normal, volatile, trending, volatile_trending
    
    for regime in regime_codes:
        # Filter data for this regime
        train_mask = regimes_train[:, -1] == regime  # Use last timestep's regime
        val_mask = regimes_val[:, -1] == regime
        
        # Skip if not enough samples for this regime
        min_samples = 100
        if np.sum(train_mask) < min_samples or np.sum(val_mask) < min_samples // 5:
            logger.warning(f"Not enough samples for regime {regime} (train: {np.sum(train_mask)}, val: {np.sum(val_mask)})")
            continue
        
        # Create filtered datasets
        X_train_regime = X_train[train_mask]
        y_train_regime = y_train[train_mask]
        X_val_regime = X_val[val_mask]
        y_val_regime = y_val[val_mask]
        
        # Setup regime-specific model with optimizer regime name
        model_name = f"{model_type}_regime_{regime}"
        
        # Create callbacks for this regime
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, "regime_specific", f"regime_{regime}.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        logger.info(f"Training regime-specific model for regime {regime} with {len(X_train_regime)} samples")
        
        # Build model based on type
        if model_type == "tcn":
            model = build_advanced_tcn_model(input_shape)
        elif model_type == "hybrid":
            model = build_advanced_hybrid_model(input_shape)
        else:
            model = build_advanced_hybrid_model(input_shape)  # Default to hybrid
        
        # Train model
        model.fit(
            X_train_regime, y_train_regime,
            validation_data=(X_val_regime, y_val_regime),
            epochs=TRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model
        regime_models[regime] = model
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(X_val_regime, y_val_regime, verbose=0)
        logger.info(f"Regime {regime} model validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    return regime_models


def build_stacked_ensemble(train_data, val_data, base_models):
    """
    Build a stacked ensemble model using the trained base models.
    
    Args:
        train_data (tuple): Training data (X, y, regimes)
        val_data (tuple): Validation data (X, y, regimes)
        base_models (dict): Dictionary of trained base models
        
    Returns:
        tuple: (stacked_model, meta_features)
    """
    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    
    # Generate predictions from base models for meta-learner training
    meta_X_train = []
    meta_X_val = []
    
    for model_name, model in base_models.items():
        train_preds = model.predict(X_train, verbose=0).flatten()
        val_preds = model.predict(X_val, verbose=0).flatten()
        
        meta_X_train.append(train_preds)
        meta_X_val.append(val_preds)
    
    # Stack predictions into a single array
    meta_X_train = np.column_stack(meta_X_train)
    meta_X_val = np.column_stack(meta_X_val)
    
    # Train a meta-learner (logistic regression)
    meta_learner = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    meta_learner.fit(meta_X_train, y_train)
    
    # Evaluate meta-learner
    val_acc = meta_learner.score(meta_X_val, y_val)
    logger.info(f"Stacked ensemble validation accuracy: {val_acc:.4f}")
    
    # Save meta-learner
    meta_learner_path = os.path.join(MODELS_DIR, "ensemble", "meta_learner.pkl")
    try:
        import joblib
        joblib.dump(meta_learner, meta_learner_path)
        logger.info(f"Saved meta-learner to {meta_learner_path}")
    except Exception as e:
        logger.error(f"Failed to save meta-learner: {e}")
    
    # Save meta features mapping
    meta_features = {model_name: i for i, model_name in enumerate(base_models.keys())}
    meta_features_path = os.path.join(MODELS_DIR, "ensemble", "meta_features.json")
    with open(meta_features_path, 'w') as f:
        json.dump(meta_features, f)
    
    return meta_learner, meta_features


def save_model_with_metadata(model, model_type, symbol, feature_names, norm_params):
    """
    Save model and associated metadata.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        symbol (str): Trading symbol
        feature_names (list): Feature names
        norm_params (dict): Normalization parameters
    """
    # Save model
    model_path = os.path.join(MODELS_DIR, model_type, f"{symbol}.h5")
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save feature names
    feature_path = os.path.join(MODELS_DIR, model_type, "feature_names.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save normalization parameters
    norm_path = os.path.join(MODELS_DIR, model_type, "norm_params.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f)


def evaluate_models(models, test_data, model_types=None):
    """
    Evaluate trained models on test data.
    
    Args:
        models (dict): Dictionary of trained models
        test_data (tuple): Test data (X, y, regimes)
        model_types (list): List of model types to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    X_test, y_test, regimes_test = test_data
    results = {}
    
    # If model_types not specified, evaluate all models
    if model_types is None:
        model_types = list(models.keys())
    
    # Evaluate each model
    for model_type in model_types:
        if model_type not in models:
            continue
        
        model = models[model_type]
        
        # Generate predictions
        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store results
        results[model_type] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        logger.info(f"{model_type} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Evaluate by regime
        for regime in range(4):  # 0, 1, 2, 3
            regime_mask = regimes_test[:, -1] == regime
            if np.sum(regime_mask) > 0:
                regime_acc = accuracy_score(y_test[regime_mask], y_pred[regime_mask])
                results[model_type][f'regime_{regime}_accuracy'] = regime_acc
                logger.info(f"  - Regime {regime} Accuracy: {regime_acc:.4f} (samples: {np.sum(regime_mask)})")
    
    return results


def main(args):
    """
    Main function for advanced ML training.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    logger.info(f"Starting advanced ML training for {args.symbol}")
    
    # Load and align multi-timeframe data
    # Filter out timeframes that don't have data files available
    available_timeframes = []
    for tf in args.timeframes:
        file_path = os.path.join(DATA_DIR, f"{args.symbol}_{tf}.csv")
        if os.path.exists(file_path):
            available_timeframes.append(tf)
        else:
            logger.warning(f"Data file for {args.symbol} {tf} not found. Skipping this timeframe.")
    
    if not available_timeframes:
        logger.error(f"No data files found for any requested timeframes: {args.timeframes}")
        return
        
    if args.primary_timeframe not in available_timeframes:
        logger.error(f"Primary timeframe {args.primary_timeframe} not available. Available: {available_timeframes}")
        return
    
    logger.info(f"Training with available timeframes: {available_timeframes}")
    timeframe_data = load_multi_timeframe_data(args.symbol, available_timeframes)
    
    if not timeframe_data or args.primary_timeframe not in timeframe_data:
        logger.error(f"Failed to load data for {args.symbol} {args.primary_timeframe}")
        return
    
    # Align data to primary timeframe
    aligned_data = align_multi_timeframe_data(timeframe_data, args.primary_timeframe)
    
    if aligned_data is None or len(aligned_data) < 1000:  # Require at least 1000 rows
        logger.error(f"Insufficient aligned data for {args.symbol} {args.primary_timeframe}")
        return
    
    logger.info(f"Prepared aligned data with {len(aligned_data)} rows and {aligned_data.shape[1]} features")
    
    # Prepare model data with sequences
    train_data, val_data, test_data, feature_names, norm_params = prepare_model_data(
        aligned_data, 
        test_size=args.test_size, 
        validation_size=args.validation_size,
        sequence_length=args.seq_length
    )
    
    # Extract input shape from training data
    input_shape = train_data[0].shape[1:]
    logger.info(f"Input shape: {input_shape}")
    
    # Train models
    models = {}
    histories = {}
    
    for model_type in args.models:
        logger.info(f"Training {model_type} model")
        
        # Setup model-specific callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=PATIENCE // 3,
                min_lr=0.0001
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, model_type, f"{args.symbol}.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        model, history = train_model_with_regimes(
            model_type=model_type,
            train_data=train_data,
            val_data=val_data,
            input_shape=input_shape,
            callbacks=callbacks,
            batch_size=BATCH_SIZE,
            epochs=TRAIN_EPOCHS
        )
        
        if model is not None:
            # Save model and metadata
            save_model_with_metadata(
                model=model,
                model_type=model_type,
                symbol=args.symbol,
                feature_names=feature_names,
                norm_params=norm_params
            )
            
            # Store model for evaluation
            models[model_type] = model
            histories[model_type] = history
    
    # Build regime-specific models using hybrid architecture
    if args.train_regime_models:
        logger.info("Training regime-specific models")
        regime_models = build_regime_specific_models(
            train_data=train_data,
            val_data=val_data,
            input_shape=input_shape,
            model_type="hybrid"  # Use hybrid architecture for regime models
        )
        
        # Save regime metadata
        regime_metadata = {
            f"regime_{regime}": {"model_type": "hybrid"} 
            for regime in regime_models.keys()
        }
        
        regime_metadata_path = os.path.join(MODELS_DIR, "regime_specific", "metadata.json")
        with open(regime_metadata_path, 'w') as f:
            json.dump(regime_metadata, f)
    
    # Build stacked ensemble if we have multiple models
    if len(models) > 1:
        logger.info("Building stacked ensemble")
        meta_learner, meta_features = build_stacked_ensemble(
            train_data=train_data,
            val_data=val_data,
            base_models=models
        )
    
    # Evaluate models on test data
    logger.info("Evaluating models on test data")
    eval_results = evaluate_models(models, test_data)
    
    # Save evaluation results
    eval_results_path = os.path.join(MODELS_DIR, f"{args.symbol}_evaluation.json")
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Print summary
    logger.info("\nModel Evaluation Results:")
    for model_type, metrics in eval_results.items():
        logger.info(f"{model_type.upper()}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Advanced ML training completed in {elapsed_time / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced ML Training for Kraken Trading Bot")
    parser.add_argument("--symbol", type=str, default="SOLUSD", help="Trading symbol")
    parser.add_argument("--primary-timeframe", type=str, default="1h", help="Primary timeframe")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"], help="Timeframes to use")
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES, help="Models to train")
    parser.add_argument("--seq-length", type=int, default=48, help="Sequence length")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test size ratio")
    parser.add_argument("--validation-size", type=float, default=0.15, help="Validation size ratio")
    parser.add_argument("--train-regime-models", action="store_true", help="Train regime-specific models")
    
    args = parser.parse_args()
    main(args)