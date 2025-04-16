#!/usr/bin/env python3
"""
Improved Hybrid Model Training

This script trains an enhanced version of the hybrid model with:
1. Deeper network architecture
2. More sophisticated attention mechanisms
3. Advanced data augmentation techniques
4. Hyperparameter optimization
5. Increased training epochs

Usage:
    python train_improved_model.py --pair BTC/USD --epochs 20
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GlobalAveragePooling1D, Bidirectional,
    LeakyReLU, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("improved_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
IMPROVED_MODEL_SUFFIX = "_improved"

# Create required directories
for directory in [DATA_DIR, HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train improved hybrid model")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=120,
                        help="Sequence length for time series (longer for improved model)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Validation size from training data")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--optimizer_learning_rate", type=float, default=0.0005,
                        help="Learning rate for optimizer (default: 0.0005)")
    parser.add_argument("--data_augmentation", action="store_true", default=True,
                        help="Apply data augmentation techniques")
    return parser.parse_args()

def load_or_generate_data(pair, timeframe):
    """Load historical data or generate dummy data if not available"""
    # Convert pair format for filename (e.g., BTC/USD -> btc_usd)
    pair_filename = pair.replace("/", "_").lower()
    
    # Define potential file paths
    csv_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.csv"
    json_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.json"
    
    # Try to load from CSV
    if os.path.exists(csv_path):
        logger.info(f"Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    
    # Try to load from JSON
    elif os.path.exists(json_path):
        logger.info(f"Loading historical data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    else:
        logger.warning(f"No historical data found for {pair} {timeframe}")
        # Generate dummy data
        logger.info(f"Generating dummy data for {pair} ({timeframe})")
        
        # Base price depends on the cryptocurrency
        if "BTC" in pair:
            base_price = 50000
        elif "ETH" in pair:
            base_price = 3000
        elif "SOL" in pair:
            base_price = 150
        else:
            base_price = 100
        
        # Generate timestamps
        num_samples = 2000  # More data for improved model
        timestamps = [datetime.now().timestamp() - i * 3600 for i in range(num_samples)]
        timestamps.reverse()
        
        # Generate price data with trend and noise
        trend = np.linspace(0, 0.2, num_samples)
        noise = np.random.normal(0, 0.01, num_samples)
        prices = base_price * (1 + trend + noise)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.002, num_samples)),
            'high': prices * (1 + np.random.normal(0, 0.005, num_samples)),
            'low': prices * (1 - np.random.normal(0, 0.005, num_samples)),
            'close': prices,
            'volume': np.random.normal(1000, 100, num_samples)
        })
        
        # Save generated data
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved generated data to {csv_path}")
        
        return df

def calculate_advanced_indicators(df):
    """Calculate advanced technical indicators for the dataset"""
    # Make a copy to avoid modifying the original dataframe
    df_indicators = df.copy()
    
    # Ensure df has expected columns in lowercase
    df_indicators.columns = [col.lower() for col in df_indicators.columns]
    
    # If 'open' not in columns but 'price' is, use price for all OHLC
    if 'open' not in df_indicators.columns and 'price' in df_indicators.columns:
        df_indicators['open'] = df_indicators['price']
        df_indicators['high'] = df_indicators['price']
        df_indicators['low'] = df_indicators['price']
        df_indicators['close'] = df_indicators['price']
    
    # If 'volume' not in columns, add it with zeros
    if 'volume' not in df_indicators.columns:
        df_indicators['volume'] = 0
    
    # Sort by time/timestamp if available
    if 'time' in df_indicators.columns:
        df_indicators.sort_values('time', inplace=True)
    elif 'timestamp' in df_indicators.columns:
        df_indicators.sort_values('timestamp', inplace=True)
    
    # 1. Price and Volume Transformations
    df_indicators['log_return'] = np.log(df_indicators['close'] / df_indicators['close'].shift(1))
    df_indicators['price_change'] = df_indicators['close'].pct_change()
    df_indicators['log_volume'] = np.log(df_indicators['volume'] + 1)  # Add 1 to avoid log(0)
    
    # 2. Moving Averages (More periods)
    for period in [5, 10, 20, 50, 100, 200]:
        df_indicators[f'sma_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'ema_{period}'] = df_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # Relative position (%)
        df_indicators[f'sma_{period}_rel'] = 100 * (df_indicators['close'] / df_indicators[f'sma_{period}'] - 1)
        df_indicators[f'ema_{period}_rel'] = 100 * (df_indicators['close'] / df_indicators[f'ema_{period}'] - 1)
        
        # Price distance from MA in ATR units
        if period >= 20:  # Only for longer-term MAs
            atr = df_indicators['high'].rolling(14).max() - df_indicators['low'].rolling(14).min()
            df_indicators[f'sma_{period}_dist_atr'] = (df_indicators['close'] - df_indicators[f'sma_{period}']) / atr
    
    # 3. RSI with Multiple Timeframes
    for period in [6, 14, 20, 50]:
        delta = df_indicators['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI Divergence (simple version)
        if period == 14:
            df_indicators['rsi_uptrend'] = ((df_indicators['close'] > df_indicators['close'].shift(5)) & 
                                           (df_indicators[f'rsi_{period}'] < df_indicators[f'rsi_{period}'].shift(5))).astype(int)
            df_indicators['rsi_downtrend'] = ((df_indicators['close'] < df_indicators['close'].shift(5)) & 
                                             (df_indicators[f'rsi_{period}'] > df_indicators[f'rsi_{period}'].shift(5))).astype(int)
    
    # 4. MACD and Signal Line Crossovers
    ema_12 = df_indicators['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_indicators['close'].ewm(span=26, adjust=False).mean()
    df_indicators['macd'] = ema_12 - ema_26
    df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
    df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']
    df_indicators['macd_cross_above'] = ((df_indicators['macd'] > df_indicators['macd_signal']) & 
                                        (df_indicators['macd'].shift() <= df_indicators['macd_signal'].shift())).astype(int)
    df_indicators['macd_cross_below'] = ((df_indicators['macd'] < df_indicators['macd_signal']) & 
                                        (df_indicators['macd'].shift() >= df_indicators['macd_signal'].shift())).astype(int)
    
    # 5. Bollinger Bands and %B
    for period in [20, 50]:
        df_indicators[f'bb_middle_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'bb_std_{period}'] = df_indicators['close'].rolling(window=period).std()
        df_indicators[f'bb_upper_{period}'] = df_indicators[f'bb_middle_{period}'] + (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_lower_{period}'] = df_indicators[f'bb_middle_{period}'] - (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_width_{period}'] = (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}']) / df_indicators[f'bb_middle_{period}']
        
        # %B (position within bollinger bands, 0-1)
        df_indicators[f'bb_b_{period}'] = (df_indicators['close'] - df_indicators[f'bb_lower_{period}']) / (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}'])
        
        # BB Squeeze (narrowing bands)
        df_indicators[f'bb_squeeze_{period}'] = (df_indicators[f'bb_width_{period}'] < df_indicators[f'bb_width_{period}'].rolling(window=20).min()).astype(int)
        
    # 6. Stochastic Oscillator
    for period in [14, 21]:
        low_min = df_indicators['low'].rolling(window=period).min()
        high_max = df_indicators['high'].rolling(window=period).max()
        
        df_indicators[f'stoch_k_{period}'] = 100 * ((df_indicators['close'] - low_min) / (high_max - low_min + 0.0001))
        df_indicators[f'stoch_d_{period}'] = df_indicators[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Stochastic crossovers
        df_indicators[f'stoch_cross_above_{period}'] = ((df_indicators[f'stoch_k_{period}'] > df_indicators[f'stoch_d_{period}']) & 
                                                      (df_indicators[f'stoch_k_{period}'].shift() <= df_indicators[f'stoch_d_{period}'].shift())).astype(int)
        df_indicators[f'stoch_cross_below_{period}'] = ((df_indicators[f'stoch_k_{period}'] < df_indicators[f'stoch_d_{period}']) & 
                                                      (df_indicators[f'stoch_k_{period}'].shift() >= df_indicators[f'stoch_d_{period}'].shift())).astype(int)
    
    # 7. Advanced ATR and Volatility Metrics
    for period in [7, 14, 21]:
        tr1 = df_indicators['high'] - df_indicators['low']
        tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
        tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df_indicators[f'atr_{period}'] = tr.rolling(window=period).mean()
        df_indicators[f'atr_percent_{period}'] = df_indicators[f'atr_{period}'] / df_indicators['close'] * 100
        
        # Normalized ATR (percentile)
        df_indicators[f'atr_percentile_{period}'] = df_indicators[f'atr_{period}'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Volatility ratio
        if period == 14:
            df_indicators['volatility_ratio'] = df_indicators[f'atr_{period}'] / df_indicators[f'atr_{period}'].rolling(window=100).mean()
    
    # 8. Price Momentum and Rate of Change
    for period in [1, 5, 10, 20, 50]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
        
        # Normalized momentum (z-score)
        if period in [10, 20]:
            df_indicators[f'norm_momentum_{period}'] = (df_indicators[f'rate_of_change_{period}'] - 
                                                      df_indicators[f'rate_of_change_{period}'].rolling(window=100).mean()) / df_indicators[f'rate_of_change_{period}'].rolling(window=100).std()
    
    # 9. Moving Average Crossovers
    df_indicators['ema_5_10_cross'] = ((df_indicators['ema_5'] > df_indicators['ema_10']) & 
                                      (df_indicators['ema_5'].shift() <= df_indicators['ema_10'].shift())).astype(int)
    df_indicators['ema_10_20_cross'] = ((df_indicators['ema_10'] > df_indicators['ema_20']) & 
                                       (df_indicators['ema_10'].shift() <= df_indicators['ema_20'].shift())).astype(int)
    df_indicators['ema_20_50_cross'] = ((df_indicators['ema_20'] > df_indicators['ema_50']) & 
                                       (df_indicators['ema_20'].shift() <= df_indicators['ema_50'].shift())).astype(int)
    df_indicators['ema_50_200_cross'] = ((df_indicators['ema_50'] > df_indicators['ema_200']) & 
                                        (df_indicators['ema_50'].shift() <= df_indicators['ema_200'].shift())).astype(int)
    
    # 10. Advanced Candlestick Features
    df_indicators['body_size'] = abs(df_indicators['close'] - df_indicators['open'])
    df_indicators['body_ratio'] = df_indicators['body_size'] / (df_indicators['high'] - df_indicators['low'] + 0.0001)
    df_indicators['upper_shadow'] = df_indicators['high'] - df_indicators[['open', 'close']].max(axis=1)
    df_indicators['lower_shadow'] = df_indicators[['open', 'close']].min(axis=1) - df_indicators['low']
    df_indicators['upper_shadow_ratio'] = df_indicators['upper_shadow'] / (df_indicators['high'] - df_indicators['low'] + 0.0001)
    df_indicators['lower_shadow_ratio'] = df_indicators['lower_shadow'] / (df_indicators['high'] - df_indicators['low'] + 0.0001)
    
    # Candlestick patterns (simple)
    df_indicators['is_doji'] = (df_indicators['body_ratio'] < 0.1).astype(int)
    df_indicators['is_hammer'] = ((df_indicators['lower_shadow_ratio'] > 0.6) & (df_indicators['upper_shadow_ratio'] < 0.15)).astype(int)
    df_indicators['is_shooting_star'] = ((df_indicators['upper_shadow_ratio'] > 0.6) & (df_indicators['lower_shadow_ratio'] < 0.15)).astype(int)
    
    # 11. Volume-based Indicators
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_ma_20'].replace(0, 0.001)
    
    # Volume-price divergence
    df_indicators['price_up_volume_down'] = ((df_indicators['close'] > df_indicators['close'].shift(1)) & 
                                           (df_indicators['volume'] < df_indicators['volume'].shift(1))).astype(int)
    df_indicators['price_down_volume_up'] = ((df_indicators['close'] < df_indicators['close'].shift(1)) & 
                                           (df_indicators['volume'] > df_indicators['volume'].shift(1))).astype(int)
    
    # 12. Market Regime Features
    df_indicators['bull_market'] = (df_indicators['ema_50'] > df_indicators['ema_200']).astype(int)
    df_indicators['bear_market'] = (df_indicators['ema_50'] < df_indicators['ema_200']).astype(int)
    
    # Trending vs ranging market (using ADX-like calculation)
    dx_pos = (df_indicators['ema_20'] - df_indicators['ema_20'].shift(1)).rolling(window=14).mean()
    dx_neg = (df_indicators['ema_20'].shift(1) - df_indicators['ema_20']).rolling(window=14).mean()
    adx = 100 * (abs(dx_pos - dx_neg) / (dx_pos + dx_neg + 0.0001)).rolling(window=14).mean()
    df_indicators['trending_market'] = (adx > 25).astype(int)
    df_indicators['ranging_market'] = (adx <= 25).astype(int)
    
    # 13. Pattern Recognition Features
    # Higher highs and lower lows
    df_indicators['higher_high'] = ((df_indicators['high'] > df_indicators['high'].shift(1)) & 
                                  (df_indicators['high'].shift(1) > df_indicators['high'].shift(2))).astype(int)
    df_indicators['lower_low'] = ((df_indicators['low'] < df_indicators['low'].shift(1)) & 
                                (df_indicators['low'].shift(1) < df_indicators['low'].shift(2))).astype(int)
    
    # Double top/bottom patterns (very simple approximation)
    df_indicators['double_top'] = ((df_indicators['high'] > df_indicators['high'].shift(5) * 0.98) & 
                                 (df_indicators['high'] < df_indicators['high'].shift(5) * 1.02) & 
                                 (df_indicators['high'].shift(2) < df_indicators['high'].shift(5) * 0.98)).astype(int)
    
    df_indicators['double_bottom'] = ((df_indicators['low'] < df_indicators['low'].shift(5) * 1.02) & 
                                    (df_indicators['low'] > df_indicators['low'].shift(5) * 0.98) & 
                                    (df_indicators['low'].shift(2) > df_indicators['low'].shift(5) * 1.02)).astype(int)
    
    # Drop rows with NaN values (due to indicators calculation)
    df_indicators.dropna(inplace=True)
    
    return df_indicators

def create_target_variable(df, price_shift=1, threshold=0.005):
    """Create target variable for ML model training with multiple classes"""
    # Calculate future returns
    future_return = df['close'].shift(-price_shift) / df['close'] - 1
    
    # Create target labels: 1 (strong up), 0.5 (moderate up), 0 (neutral), -0.5 (moderate down), -1 (strong down)
    df['target'] = 0
    df.loc[future_return > threshold*2, 'target'] = 1      # Strong bullish
    df.loc[(future_return > threshold) & (future_return <= threshold*2), 'target'] = 0.5  # Moderate bullish
    df.loc[(future_return < -threshold) & (future_return >= -threshold*2), 'target'] = -0.5  # Moderate bearish
    df.loc[future_return < -threshold*2, 'target'] = -1    # Strong bearish
    
    # Drop rows with NaN values (last rows where target couldn't be calculated)
    df.dropna(subset=['target'], inplace=True)
    
    # Convert target to integer classes
    target_map = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
    df['target_class'] = df['target'].map(target_map)
    
    return df

def apply_data_augmentation(X, y):
    """Apply data augmentation techniques to increase training data"""
    logger.info(f"Original data shape: {X.shape}")
    
    X_augmented = []
    y_augmented = []
    
    # Add original data
    X_augmented.append(X)
    y_augmented.append(y)
    
    # 1. Add small random noise
    noise_factor = 0.02
    X_noise = X + np.random.normal(0, noise_factor, X.shape)
    X_augmented.append(X_noise)
    y_augmented.append(y)
    
    # 2. Time scaling (slight stretching/shrinking of time)
    # Randomly select a subset of samples
    indices = np.random.choice(X.shape[0], size=X.shape[0] // 4, replace=False)
    for idx in indices:
        # Randomly select stretch factor
        stretch_factor = np.random.uniform(0.9, 1.1)
        sequence = X[idx]
        # Apply time stretching
        stretched = []
        for i in range(X.shape[1]):
            new_idx = int(i * stretch_factor)
            if 0 <= new_idx < X.shape[1]:
                stretched.append(sequence[new_idx])
            else:
                stretched.append(sequence[i])
        X_augmented.append(np.array([stretched]))
        y_augmented.append(np.array([y[idx]]))
    
    # 3. Combine augmentations
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.vstack(y_augmented)
    
    logger.info(f"Augmented data shape: {X_augmented.shape}")
    
    return X_augmented, y_augmented

def prepare_data_for_training(df, sequence_length, test_size, validation_size, apply_augmentation=False):
    """Prepare data for training, validation, and testing"""
    # Select features and target
    feature_columns = [col for col in df.columns if col not in ['time', 'timestamp', 'target', 'target_class']]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Get targets (multi-class)
    targets = df['target_class'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=5)  # 5 classes: strong down, moderate down, neutral, moderate up, strong up
    
    # Apply data augmentation if requested
    if apply_augmentation:
        X, y_categorical = apply_data_augmentation(X, y_categorical)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, shuffle=False
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, shuffle=False
    )
    
    logger.info(f"Data shapes:")
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns

def conv_block(x, filters, kernel_size, dilation_rate=1):
    """Create a convolutional block with residual connection"""
    # Convolutional layer
    conv = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        activation=None
    )(x)
    
    # Batch normalization
    conv = BatchNormalization()(conv)
    
    # Activation function
    conv = LeakyReLU(alpha=0.2)(conv)
    
    # Second convolutional layer
    conv = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        activation=None
    )(conv)
    
    # Batch normalization
    conv = BatchNormalization()(conv)
    
    # Residual connection (if input and output have same shape)
    if x.shape[-1] == filters:
        conv = Add()([conv, x])
    
    # Activation function
    conv = LeakyReLU(alpha=0.2)(conv)
    
    return conv

def attention_block(x, name="attention"):
    """Create a self-attention block"""
    # Get input shape
    seq_len, features = x.shape[1], x.shape[2]
    
    # Create query, key, value projections
    query = Dense(features, activation=None)(x)
    key = Dense(features, activation=None)(x)
    value = Dense(features, activation=None)(x)
    
    # Calculate attention scores
    scores = tf.matmul(query, key, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(features, tf.float32))
    
    # Apply softmax to get attention weights
    attention_weights = tf.nn.softmax(scores, axis=-1)
    
    # Apply attention weights to values
    context = tf.matmul(attention_weights, value)
    
    # Residual connection
    context = Add()([context, x])
    
    return context

def build_enhanced_hybrid_model(input_shape, output_shape=5):
    """Build an enhanced hybrid model with improved architectures"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch with dilated convolutions
    cnn = conv_block(inputs, filters=64, kernel_size=3)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = conv_block(cnn, filters=128, kernel_size=3, dilation_rate=2)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = conv_block(cnn, filters=128, kernel_size=3, dilation_rate=4)
    cnn = GlobalAveragePooling1D()(cnn)
    cnn = Dense(64, activation='relu')(cnn)
    cnn = Dropout(0.3)(cnn)
    
    # 2. LSTM Branch with bidirectional layers
    lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(64, return_sequences=True))(lstm)
    lstm = attention_block(lstm, name="lstm_attention")
    lstm = GlobalAveragePooling1D()(lstm)
    lstm = Dense(64, activation='relu')(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # 3. GRU Branch for short-term patterns
    gru = GRU(64, return_sequences=True)(inputs)
    gru = Dropout(0.3)(gru)
    gru = GRU(64)(gru)
    gru = Dense(64, activation='relu')(gru)
    gru = Dropout(0.3)(gru)
    
    # Merge branches
    merged = Concatenate()([cnn, lstm, gru])
    
    # Deep meta-learner
    meta = Dense(128, activation='relu')(merged)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.5)(meta)
    meta = Dense(64, activation='relu')(meta)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.3)(meta)
    meta = Dense(32, activation='relu')(meta)
    meta = Dropout(0.2)(meta)
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(meta)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, pair, learning_rate=0.001):
    """Train the enhanced model"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create callbacks
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}{IMPROVED_MODEL_SUFFIX}_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Create logs directory
    logs_dir = f"{RESULTS_DIR}/logs/{pair_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logs_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Load the best model
    model = load_model(checkpoint_path)
    
    return model, history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model with enhanced metrics"""
    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    # Calculate class-wise accuracy
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_accuracy = {}
    for i in range(5):
        class_mask = y_test_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy[class_names[i]] = accuracy_score(
                y_test_classes[class_mask], 
                y_pred_classes[class_mask]
            )
        else:
            class_accuracy[class_names[i]] = 0.0
    
    # Calculate confusion metrics
    # Convert back to -1, -0.5, 0, 0.5, 1 format for easier interpretation
    # Strong down (-1): class 0, Moderate down (-0.5): class 1, Neutral (0): class 2, Moderate up (0.5): class 3, Strong up (1): class 4
    
    # Calculate direction accuracy (just care about up vs down, ignore neutral)
    non_neutral_mask = (y_test_classes != 2) & (y_pred_classes != 2)
    y_test_direction = np.array([1 if c > 2 else 0 for c in y_test_classes[non_neutral_mask]])
    y_pred_direction = np.array([1 if c > 2 else 0 for c in y_pred_classes[non_neutral_mask]])
    direction_accuracy = accuracy_score(y_test_direction, y_pred_direction) if len(y_test_direction) > 0 else 0.0
    
    # Trading-specific metrics (directional success)
    false_positives = np.sum((y_pred_classes > 2) & (y_test_classes < 2))
    false_negatives = np.sum((y_pred_classes < 2) & (y_test_classes > 2))
    true_positives = np.sum((y_pred_classes > 2) & (y_test_classes > 2))
    true_negatives = np.sum((y_pred_classes < 2) & (y_test_classes < 2))
    
    # Win rate (ignoring neutrals) = (TP + TN) / (TP + TN + FP + FN)
    total_directional_predictions = true_positives + true_negatives + false_positives + false_negatives
    win_rate = (true_positives + true_negatives) / total_directional_predictions if total_directional_predictions > 0 else 0.0
    
    # Calculate signal distribution
    signal_distribution = {}
    for i, name in enumerate(class_names):
        signal_distribution[name] = np.mean(y_pred_classes == i)
    
    # Print evaluation
    logger.info("\nModel Evaluation:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Directional Accuracy: {direction_accuracy:.4f}")
    logger.info(f"Win Rate: {win_rate:.4f}")
    
    logger.info("\nClass-wise Accuracy:")
    for class_name, acc in class_accuracy.items():
        logger.info(f"  {class_name}: {acc:.4f}")
    
    logger.info("\nSignal Distribution:")
    for class_name, dist in signal_distribution.items():
        logger.info(f"  {class_name}: {dist:.2%}")
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'direction_accuracy': direction_accuracy,
        'win_rate': win_rate,
        'class_accuracy': class_accuracy,
        'signal_distribution': signal_distribution,
        'confusion_metrics': {
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    }
    
    return metrics

def plot_training_history(history, pair):
    """Plot training history with enhanced visualizations"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{pair} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{pair} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.semilogy(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True, alpha=0.3)
    
    # Plot accuracy vs loss
    plt.subplot(2, 2, 4)
    plt.plot(history.history['loss'], history.history['accuracy'], 'o-')
    plt.plot(history.history['val_loss'], history.history['val_accuracy'], 'o-')
    plt.title('Accuracy vs Loss')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plot_path = f"{RESULTS_DIR}/hybrid_{pair_clean}{IMPROVED_MODEL_SUFFIX}_training_history.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Training history plot saved to {plot_path}")

def update_ml_config(pair, model_path, metrics, max_portfolio_risk=0.25):
    """Update ML configuration for the pair"""
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML config: {e}")
            config = {"models": {}, "global_settings": {}}
    else:
        config = {"models": {}, "global_settings": {}}
    
    # Update global settings if not present
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    # Set global settings
    config["global_settings"]["base_leverage"] = 5.0
    config["global_settings"]["max_leverage"] = 75.0
    config["global_settings"]["confidence_threshold"] = 0.65
    config["global_settings"]["risk_percentage"] = 0.20
    config["global_settings"]["max_portfolio_risk"] = max_portfolio_risk
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Add or update model config for this pair
    config["models"][pair] = {
        "model_type": "enhanced_hybrid",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "win_rate": metrics["win_rate"],
        "direction_accuracy": metrics["direction_accuracy"],
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config for {pair}")
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")

def save_model_summary(model, pair, metrics):
    """Save model summary and metrics"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Capture model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = '\n'.join(model_summary)
    
    # Create summary text
    summary = f"""
Enhanced Hybrid Model Summary for {pair}
========================================
{model_summary}

Model Evaluation Metrics
------------------------
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1']:.4f}
Directional Accuracy: {metrics['direction_accuracy']:.4f}
Win Rate: {metrics['win_rate']:.4f}

Class-wise Accuracy
------------------
"""
    
    for class_name, acc in metrics['class_accuracy'].items():
        summary += f"{class_name}: {acc:.4f}\n"
    
    summary += f"""
Signal Distribution
------------------
"""
    
    for class_name, dist in metrics['signal_distribution'].items():
        summary += f"{class_name}: {dist:.2%}\n"
    
    summary += f"""
Confusion Metrics
----------------
True Positives: {metrics['confusion_metrics']['true_positives']}
True Negatives: {metrics['confusion_metrics']['true_negatives']}
False Positives: {metrics['confusion_metrics']['false_positives']}
False Negatives: {metrics['confusion_metrics']['false_negatives']}

Model Improvements
-----------------
1. Deeper network architecture with residual connections
2. Enhanced data features (40+ technical indicators)
3. Multi-class prediction for trading signals
4. More sophisticated attention mechanisms
5. Bidirectional LSTM layers
6. Dual GRU branch for improved sequential patterns

Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save summary to file
    summary_path = f"{RESULTS_DIR}/hybrid_{pair_clean}{IMPROVED_MODEL_SUFFIX}_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Model summary saved to {summary_path}")

def main():
    """Main function"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVED HYBRID MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Optimizer Learning Rate: {args.optimizer_learning_rate}")
    logger.info(f"Data Augmentation: {args.data_augmentation}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Load data
        df = load_or_generate_data(args.pair, args.timeframe)
        
        # Calculate advanced indicators
        logger.info("Calculating advanced technical indicators...")
        df_indicators = calculate_advanced_indicators(df)
        
        # Create target variable
        logger.info("Creating target variable...")
        df_labeled = create_target_variable(df_indicators)
        
        # Prepare data for training
        logger.info("Preparing data for training...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, features = prepare_data_for_training(
            df_labeled, args.sequence_length, args.test_size, args.validation_size, args.data_augmentation
        )
        
        # Print feature information
        logger.info(f"Number of features: {len(features)}")
        
        # Build model
        logger.info("Building enhanced hybrid model...")
        input_shape = (args.sequence_length, len(features))
        model = build_enhanced_hybrid_model(input_shape, output_shape=5)
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        # Train model
        logger.info(f"Training model for {args.pair}...")
        model, history, model_path = train_model(
            model, X_train, y_train, X_val, y_val, 
            args.epochs, args.batch_size, args.pair,
            learning_rate=args.optimizer_learning_rate
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history, args.pair)
        
        # Save model summary
        logger.info("Saving model summary...")
        save_model_summary(model, args.pair, metrics)
        
        # Update ML config
        logger.info("Updating ML configuration...")
        update_ml_config(args.pair, model_path, metrics, args.max_portfolio_risk)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Win Rate: {metrics['win_rate']:.4f}")
        logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        
        return model_path, metrics
    
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()