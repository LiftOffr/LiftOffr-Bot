#!/usr/bin/env python3
"""
Multi-Timeframe Model Trainer

This script implements the three-phase approach to multi-timeframe trading:
1. Train models on individual timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
2. Build a unified multi-timeframe model
3. Create an ensemble meta-model

Usage:
    python multi_timeframe_trainer.py --pair BTC/USD
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GlobalAveragePooling1D, Bidirectional,
    TimeDistributed, Lambda, Reshape
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mtf_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
MTF_MODEL_DIR = "mtf_models"
ENSEMBLE_DIR = "ensemble_models"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Default pairs and timeframes
ALL_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
TRAINING_TIMEFRAMES = ['15m', '1h', '4h', '1d']  # Focus on most important timeframes
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, MTF_MODEL_DIR, 
                 ENSEMBLE_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Timeframe Model Trainer")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help=f"Trading pair to train models for (options: {', '.join(ALL_PAIRS)})")
    parser.add_argument("--timeframes", type=str, default="15m,1h,4h,1d",
                        help=f"Comma-separated list of timeframes (default: 15m,1h,4h,1d)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series (default: 60)")
    parser.add_argument("--predict_horizon", type=int, default=4,
                        help="Number of intervals to predict ahead (default: 4)")
    parser.add_argument("--update_ml_config", action="store_true", default=True,
                        help="Update ML configuration with trained models")
    parser.add_argument("--phase", type=str, default="all",
                        help="Training phase (options: individual, mtf, ensemble, all)")
    parser.add_argument("--fetch_missing", action="store_true", default=False,
                        help="Fetch missing data if needed")
    return parser.parse_args()

def load_historical_data(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load historical data for a pair and timeframe"""
    pair_clean = pair.replace("/", "_")
    file_path = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}.csv"
    
    if os.path.exists(file_path):
        logger.info(f"Loading historical data for {pair} ({timeframe}) from {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                if isinstance(df['timestamp'][0], str):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded {len(df)} records for {pair} ({timeframe})")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None
    else:
        logger.warning(f"Historical data not found: {file_path}")
        return None

def fetch_missing_data(pair: str, timeframe: str, days: int = 7) -> bool:
    """Fetch missing data for a pair and timeframe"""
    try:
        import subprocess
        
        logger.info(f"Fetching {timeframe} data for {pair}...")
        
        command = [
            "python", "fetch_kraken_15m_data.py",
            "--pair", pair,
            "--timeframe", timeframe,
            "--days", str(days)
        ]
        
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Check if file was created
        pair_clean = pair.replace("/", "_")
        file_path = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}.csv"
        
        if os.path.exists(file_path):
            logger.info(f"Successfully fetched {timeframe} data for {pair}")
            return True
        else:
            logger.error(f"Failed to fetch {timeframe} data for {pair}")
            return False
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return False

def calculate_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Calculate technical indicators optimized for specific timeframes"""
    logger.info(f"Calculating technical indicators for {timeframe} timeframe...")
    
    # Make a copy to avoid modifying the original
    df_indicators = df.copy()
    
    # Common indicators for all timeframes
    # 1. Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df_indicators[f'sma_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'ema_{period}'] = df_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # Relative to price
        df_indicators[f'sma_{period}_rel'] = (df_indicators['close'] / df_indicators[f'sma_{period}'] - 1) * 100
        df_indicators[f'ema_{period}_rel'] = (df_indicators['close'] / df_indicators[f'ema_{period}'] - 1) * 100
    
    # 2. RSI
    for period in [6, 14, 20, 50]:
        delta = df_indicators['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    ema_12 = df_indicators['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_indicators['close'].ewm(span=26, adjust=False).mean()
    df_indicators['macd'] = ema_12 - ema_26
    df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
    df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']
    
    # 4. Bollinger Bands
    for period in [20, 50]:
        df_indicators[f'bb_middle_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'bb_std_{period}'] = df_indicators['close'].rolling(window=period).std()
        df_indicators[f'bb_upper_{period}'] = df_indicators[f'bb_middle_{period}'] + (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_lower_{period}'] = df_indicators[f'bb_middle_{period}'] - (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_width_{period}'] = (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}']) / df_indicators[f'bb_middle_{period}']
        df_indicators[f'bb_b_{period}'] = (df_indicators['close'] - df_indicators[f'bb_lower_{period}']) / (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}'] + 0.0001)
    
    # 5. ATR (Average True Range)
    for period in [7, 14, 21]:
        tr1 = df_indicators['high'] - df_indicators['low']
        tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
        tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_indicators[f'atr_{period}'] = tr.rolling(window=period).mean()
        df_indicators[f'atr_percent_{period}'] = df_indicators[f'atr_{period}'] / df_indicators['close'] * 100
    
    # 6. Momentum
    for period in [5, 10, 20, 50]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
    
    # 7. Stochastic Oscillator
    for period in [14, 21]:
        low_min = df_indicators['low'].rolling(window=period).min()
        high_max = df_indicators['high'].rolling(window=period).max()
        df_indicators[f'stoch_k_{period}'] = 100 * ((df_indicators['close'] - low_min) / (high_max - low_min + 0.0001))
        df_indicators[f'stoch_d_{period}'] = df_indicators[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # 8. Volume Indicators
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_ma_20'].replace(0, 0.001)
    
    # 9. Trend Indicators
    df_indicators['bull_market'] = (df_indicators['ema_50'] > df_indicators['ema_200']).astype(int)
    df_indicators['bear_market'] = (df_indicators['ema_50'] < df_indicators['ema_200']).astype(int)
    
    # 10. Volatility Metrics
    df_indicators['volatility_14'] = df_indicators['close'].rolling(window=14).std() / df_indicators['close'] * 100
    
    # Add timeframe-specific indicators
    if timeframe in ['1m', '5m', '15m']:
        # Short-term indicators for lower timeframes
        df_indicators['micro_trend'] = (df_indicators['close'] > df_indicators['ema_5']).astype(int)
        df_indicators['micro_momentum'] = df_indicators['close'].diff(3) / df_indicators['close'].shift(3) * 100
        
        # Volume spikes for scalping
        df_indicators['volume_spike'] = (df_indicators['volume'] > df_indicators['volume'].rolling(window=10).mean() * 2).astype(int)
        
        # Price acceleration
        df_indicators['price_acceleration'] = df_indicators['close'].diff().diff()
        
        # Fast stochastic for quick reversals
        df_indicators['fast_stoch_k'] = df_indicators['stoch_k_14'].rolling(window=3).mean()
        df_indicators['fast_stoch_d'] = df_indicators['fast_stoch_k'].rolling(window=3).mean()
    
    elif timeframe in ['30m', '1h']:
        # Medium-term indicators
        df_indicators['ema_crossover'] = ((df_indicators['ema_10'] > df_indicators['ema_20']) & 
                                         (df_indicators['ema_10'].shift() <= df_indicators['ema_20'].shift())).astype(int)
        
        # ADX for trend strength
        plus_dm = df_indicators['high'].diff()
        minus_dm = df_indicators['low'].diff(-1)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'tr1': df_indicators['high'] - df_indicators['low'],
            'tr2': abs(df_indicators['high'] - df_indicators['close'].shift()),
            'tr3': abs(df_indicators['low'] - df_indicators['close'].shift())
        }).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df_indicators['adx'] = dx.rolling(window=14).mean()
        
        # Ichimoku Cloud
        high_9 = df_indicators['high'].rolling(window=9).max()
        low_9 = df_indicators['low'].rolling(window=9).min()
        df_indicators['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df_indicators['high'].rolling(window=26).max()
        low_26 = df_indicators['low'].rolling(window=26).min()
        df_indicators['kijun_sen'] = (high_26 + low_26) / 2
        
        df_indicators['senkou_span_a'] = ((df_indicators['tenkan_sen'] + df_indicators['kijun_sen']) / 2).shift(26)
        
        high_52 = df_indicators['high'].rolling(window=52).max()
        low_52 = df_indicators['low'].rolling(window=52).min()
        df_indicators['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    elif timeframe in ['4h', '1d']:
        # Long-term indicators
        # VWAP (Volume-Weighted Average Price)
        df_indicators['vwap'] = (df_indicators['volume'] * df_indicators['close']).cumsum() / df_indicators['volume'].cumsum()
        
        # OBV (On-Balance Volume)
        df_indicators['obv'] = (df_indicators['volume'] * np.where(df_indicators['close'] > df_indicators['close'].shift(), 1, -1)).cumsum()
        
        # Mass Index
        high_low = df_indicators['high'] - df_indicators['low']
        high_low_ema = high_low.ewm(span=9, adjust=False).mean()
        ema_ratio = high_low_ema / high_low_ema.ewm(span=9, adjust=False).mean()
        df_indicators['mass_index'] = ema_ratio.rolling(window=25).sum()
        
        # Chaikin Money Flow
        mf_multiplier = ((df_indicators['close'] - df_indicators['low']) - (df_indicators['high'] - df_indicators['close'])) / (df_indicators['high'] - df_indicators['low'])
        mf_volume = mf_multiplier * df_indicators['volume']
        df_indicators['cmf'] = mf_volume.rolling(window=20).sum() / df_indicators['volume'].rolling(window=20).sum()
    
    # Drop rows with NaN values
    df_indicators.dropna(inplace=True)
    
    logger.info(f"Calculated {len(df_indicators.columns) - len(df.columns)} indicators for {timeframe}")
    logger.info(f"Data shape after indicator calculation: {df_indicators.shape}")
    
    return df_indicators

def create_target_variable(
    df: pd.DataFrame, 
    timeframe: str, 
    predict_horizon: int, 
    threshold: float = 0.005
) -> pd.DataFrame:
    """Create target variable for ML model training optimized for timeframe"""
    logger.info(f"Creating target variable for {timeframe} with horizon {predict_horizon}...")
    
    # Scale threshold based on timeframe volatility
    if timeframe in ['1m', '5m', '15m']:
        # Lower timeframes need smaller thresholds due to smaller price movements
        actual_threshold = threshold * 0.5
    elif timeframe in ['4h', '1d']:
        # Higher timeframes need larger thresholds
        actual_threshold = threshold * 2.0
    else:
        actual_threshold = threshold
    
    logger.info(f"Using threshold {actual_threshold:.4f} for {timeframe}")
    
    # Calculate future returns
    future_return = df['close'].shift(-predict_horizon) / df['close'] - 1
    
    # Create target labels: 1 (strong up), 0.5 (moderate up), 0 (neutral), -0.5 (moderate down), -1 (strong down)
    df['target'] = 0
    df.loc[future_return > actual_threshold*2, 'target'] = 1      # Strong bullish
    df.loc[(future_return > actual_threshold) & (future_return <= actual_threshold*2), 'target'] = 0.5  # Moderate bullish
    df.loc[(future_return < -actual_threshold) & (future_return >= -actual_threshold*2), 'target'] = -0.5  # Moderate bearish
    df.loc[future_return < -actual_threshold*2, 'target'] = -1    # Strong bearish
    
    # Drop rows with NaN values (last rows where target couldn't be calculated)
    df.dropna(subset=['target'], inplace=True)
    
    # Convert target to integer classes
    target_map = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
    df['target_class'] = df['target'].map(target_map)
    
    # Log class distribution
    class_counts = df['target_class'].value_counts().sort_index()
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    
    logger.info(f"Target class distribution for {timeframe}:")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        pct = count / len(df) * 100
        logger.info(f"  {name}: {count} samples ({pct:.1f}%)")
    
    return df

def prepare_data_for_training(
    df: pd.DataFrame, 
    sequence_length: int, 
    test_size: float = 0.2, 
    validation_size: float = 0.2
) -> Tuple:
    """Prepare data for training, validation, and testing"""
    logger.info(f"Preparing sequences with length {sequence_length}...")
    
    # Select features and target
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'target_class']]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Get targets
    targets = df['target_class'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=5)  # 5 classes
    
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

def build_individual_model(
    input_shape: Tuple, 
    timeframe: str, 
    output_shape: int = 5
) -> Model:
    """Build a model optimized for specific timeframe"""
    logger.info(f"Building model for {timeframe} timeframe...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    if timeframe in ['1m', '5m', '15m']:
        # Lower timeframes need focus on micro patterns and noise filtering
        # CNN branch for pattern recognition
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    elif timeframe in ['30m', '1h']:
        # Medium timeframes need balanced approach
        # LSTM with bidirectional layers for sequence patterns
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    else:  # ['4h', '1d']
        # Higher timeframes need focus on trend detection
        # TCN-like branch with dilated convolutions
        tcn_layers = []
        num_filters = 64
        kernel_size = 3
        
        # Use multiple dilated convolutions with different dilation rates
        for dilation_rate in [1, 2, 4, 8]:
            conv = Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu'
            )(inputs)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.2)(conv)
            tcn_layers.append(conv)
        
        # Merge TCN layers
        if len(tcn_layers) > 1:
            x = Concatenate(axis=-1)(tcn_layers)
        else:
            x = tcn_layers[0]
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    # Deep neural network final layers
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer (5 classes)
    outputs = Dense(output_shape, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model for {timeframe} created")
    
    return model

def train_individual_model(
    model: Model, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    pair: str,
    timeframe: str,
    args
) -> Tuple:
    """Train model for individual timeframe"""
    logger.info(f"Training model for {pair} ({timeframe})...")
    
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create checkpoint path
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/individual_{pair_clean}_{timeframe}_model.h5"
    
    # Create callbacks
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
    logs_dir = f"{RESULTS_DIR}/logs/{pair_clean}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logs_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Load the best model
    model = load_model(checkpoint_path)
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    logger.info(f"Training for {pair} ({timeframe}) completed in {training_time_str}")
    
    return model, history, checkpoint_path, training_time

def evaluate_individual_model(
    model: Model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    pair: str,
    timeframe: str
) -> Dict:
    """Evaluate model for individual timeframe"""
    logger.info(f"Evaluating {pair} ({timeframe}) model...")
    
    # Basic evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    
    # Class mapping
    class_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
    y_pred_signal = np.array([class_map[c] for c in y_pred_class])
    y_test_signal = np.array([class_map[c] for c in y_test_class])
    
    # Calculate class-wise accuracy
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = y_test_class == i
        if np.sum(class_mask) > 0:
            class_accuracy[name] = np.mean(y_pred_class[class_mask] == i)
        else:
            class_accuracy[name] = 0.0
    
    # Calculate direction accuracy
    non_neutral_mask = (y_test_signal != 0) & (y_pred_signal != 0)
    direction_accuracy = np.mean(np.sign(y_pred_signal[non_neutral_mask]) == np.sign(y_test_signal[non_neutral_mask]))
    
    # Calculate precision, recall, F1 for non-neutral classes
    binary_y_test = (y_test_signal > 0).astype(int)  # 1 for positive, 0 for negative
    binary_y_pred = (y_pred_signal > 0).astype(int)
    
    # Filter out neutral predictions
    non_neutral_indices = np.where((y_test_signal != 0) & (y_pred_signal != 0))[0]
    if len(non_neutral_indices) > 0:
        filtered_y_test = binary_y_test[non_neutral_indices]
        filtered_y_pred = binary_y_pred[non_neutral_indices]
        
        precision = precision_score(filtered_y_test, filtered_y_pred)
        recall = recall_score(filtered_y_test, filtered_y_pred)
        f1 = f1_score(filtered_y_test, filtered_y_pred)
    else:
        precision = 0
        recall = 0
        f1 = 0
    
    # Combine metrics
    metrics = {
        "accuracy": accuracy,
        "direction_accuracy": direction_accuracy,
        "class_accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    # Print metrics
    logger.info(f"{pair} ({timeframe}) evaluation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics

def save_individual_model_report(
    pair: str,
    timeframe: str,
    model_path: str,
    metrics: Dict,
    training_time: float,
    args
) -> str:
    """Save report for individual timeframe model"""
    # Create report filename
    pair_clean = pair.replace("/", "_").lower()
    report_file = f"{RESULTS_DIR}/individual_model_report_{pair_clean}_{timeframe}.json"
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Create report data
    report = {
        "pair": pair,
        "timeframe": timeframe,
        "model_path": model_path,
        "model_type": "individual",
        "training_time": training_time_str,
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "predict_horizon": args.predict_horizon
        },
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return ""

def update_ml_config_with_individual_model(
    pair: str,
    timeframe: str,
    model_path: str,
    metrics: Dict
) -> bool:
    """Update ML configuration with individual timeframe model"""
    logger.info(f"Updating ML configuration with {pair} ({timeframe}) model...")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
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
    config["global_settings"]["max_portfolio_risk"] = 0.25
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Create model key that includes timeframe
    model_key = f"{pair}_{timeframe}"
    
    # Add or update model config
    config["models"][model_key] = {
        "pair": pair,
        "timeframe": timeframe,
        "model_type": "individual",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "direction_accuracy": metrics["direction_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config for {pair} ({timeframe})")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def build_mtf_model(
    input_shapes: Dict[str, Tuple], 
    timeframes: List[str], 
    output_shape: int = 5
) -> Model:
    """Build multi-timeframe model that uses inputs from multiple timeframes"""
    logger.info(f"Building multi-timeframe model with timeframes: {', '.join(timeframes)}...")
    
    # Create branches for each timeframe
    branches = {}
    inputs = {}
    
    for timeframe in timeframes:
        # Input for this timeframe
        inputs[timeframe] = Input(shape=input_shapes[timeframe], name=f"input_{timeframe}")
        
        # Branch architecture
        if timeframe in ['1m', '5m', '15m']:
            # CNN branch for lower timeframes
            x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs[timeframe])
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = GlobalAveragePooling1D()(x)
        
        elif timeframe in ['30m', '1h']:
            # LSTM branch for medium timeframes
            x = Bidirectional(LSTM(64, return_sequences=True))(inputs[timeframe])
            x = Dropout(0.3)(x)
            x = Bidirectional(LSTM(64, return_sequences=False))(x)
        
        else:  # ['4h', '1d']
            # TCN-like branch for higher timeframes
            tcn_layers = []
            num_filters = 64
            kernel_size = 3
            
            # Use multiple dilated convolutions with different dilation rates
            for dilation_rate in [1, 2, 4, 8]:
                conv = Conv1D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding='causal',
                    dilation_rate=dilation_rate,
                    activation='relu'
                )(inputs[timeframe])
                conv = BatchNormalization()(conv)
                conv = Dropout(0.2)(conv)
                tcn_layers.append(conv)
            
            # Merge TCN layers
            if len(tcn_layers) > 1:
                x = Concatenate(axis=-1)(tcn_layers)
            else:
                x = tcn_layers[0]
            
            x = GlobalAveragePooling1D()(x)
        
        # Final dense layer for this branch
        x = Dense(64, activation='relu', name=f"dense_{timeframe}")(x)
        x = Dropout(0.3)(x)
        
        # Store branch output
        branches[timeframe] = x
    
    # Merge all branches
    if len(branches) > 1:
        merged = Concatenate()([branch for branch in branches.values()])
    else:
        merged = list(branches.values())[0]
    
    # Add attention layer to focus on important timeframes
    attention = Dense(64, activation='tanh')(merged)
    attention = Dense(1, activation='sigmoid')(attention)
    attention_weighted = tf.keras.layers.Multiply()([merged, attention])
    
    # Deep neural network final layers
    x = Dense(128, activation='relu')(attention_weighted)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer (5 classes)
    outputs = Dense(output_shape, activation='softmax')(x)
    
    # Create model
    model = Model(
        inputs=[inputs[tf] for tf in timeframes],
        outputs=outputs
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Multi-timeframe model created")
    
    return model

def align_mtf_data(
    data_dict: Dict[str, pd.DataFrame], 
    sequence_lengths: Dict[str, int]
) -> Tuple[Dict[str, pd.DataFrame], List[datetime]]:
    """Align data from multiple timeframes to common timestamps"""
    logger.info("Aligning data from multiple timeframes...")
    
    # Get the timeframe with the least amount of data
    min_timeframe = min(data_dict.items(), key=lambda x: len(x[1]))[0]
    min_df = data_dict[min_timeframe]
    
    # Get reference timestamps from the minimum timeframe
    reference_timestamps = min_df['timestamp'].tolist()
    
    # Align all dataframes to these reference timestamps
    aligned_data = {}
    common_timestamps = []
    
    for timestamp in reference_timestamps:
        # Check if this timestamp exists in all timeframes
        valid_for_all = True
        
        for timeframe, df in data_dict.items():
            # Find closest timestamp in this timeframe
            # For higher timeframes, find the candle that contains this timestamp
            if timeframe in ['4h', '1d']:
                # For higher timeframes, find the candle that contains this timestamp
                mask = (df['timestamp'] <= timestamp)
                if not mask.any():
                    valid_for_all = False
                    break
            else:
                # For lower timeframes, find exact or closest match
                if timestamp not in df['timestamp'].values:
                    valid_for_all = False
                    break
        
        if valid_for_all:
            common_timestamps.append(timestamp)
    
    logger.info(f"Found {len(common_timestamps)} common timestamps across all timeframes")
    
    # Create aligned dataframes
    for timeframe, df in data_dict.items():
        aligned_df = pd.DataFrame()
        
        for timestamp in common_timestamps:
            # Find index of closest timestamp in this timeframe
            if timeframe in ['4h', '1d']:
                # For higher timeframes, find the candle that contains this timestamp
                mask = (df['timestamp'] <= timestamp)
                if mask.any():
                    idx = mask.idxmax()
                    aligned_df = pd.concat([aligned_df, df.iloc[[idx]]], ignore_index=True)
            else:
                # For lower timeframes, find exact match
                idx = df[df['timestamp'] == timestamp].index
                if len(idx) > 0:
                    aligned_df = pd.concat([aligned_df, df.iloc[[idx[0]]]], ignore_index=True)
        
        aligned_data[timeframe] = aligned_df
    
    return aligned_data, common_timestamps

def prepare_mtf_data(
    aligned_data: Dict[str, pd.DataFrame],
    common_timestamps: List[datetime],
    feature_columns: Dict[str, List[str]],
    scalers: Dict[str, MinMaxScaler],
    sequence_lengths: Dict[str, int],
    test_size: float = 0.2,
    validation_size: float = 0.2
) -> Tuple:
    """Prepare data for multi-timeframe model training"""
    logger.info("Preparing data for multi-timeframe model...")
    
    # Get common target from the lowest timeframe (most granular)
    lowest_timeframe = min(aligned_data.keys())
    lowest_df = aligned_data[lowest_timeframe]
    
    # Use target from lowest timeframe
    targets = lowest_df['target_class'].values
    
    # Create sequences for each timeframe
    X_dict = {}
    for timeframe, df in aligned_data.items():
        # Get features
        features = df[feature_columns[timeframe]].values
        
        # Normalize features
        features = scalers[timeframe].transform(features)
        
        # Create sequences
        X_timeframe = []
        seq_len = sequence_lengths[timeframe]
        
        for i in range(seq_len, len(features)):
            X_timeframe.append(features[i-seq_len:i])
        
        X_dict[timeframe] = np.array(X_timeframe)
    
    # Find the minimum number of sequences across all timeframes
    min_sequences = min([len(X) for X in X_dict.values()])
    
    # Trim sequences and targets to match the minimum
    for timeframe in X_dict:
        X_dict[timeframe] = X_dict[timeframe][-min_sequences:]
    
    # Trim targets to match
    targets = targets[-min_sequences:]
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(targets, num_classes=5)  # 5 classes
    
    # Split into train and test
    train_indices, test_indices = train_test_split(
        np.arange(min_sequences), test_size=test_size, shuffle=False
    )
    
    # Further split train into train and validation
    train_indices, val_indices = train_test_split(
        train_indices, test_size=validation_size, shuffle=False
    )
    
    # Create train, validation, and test data for each timeframe
    X_train, X_val, X_test = {}, {}, {}
    for timeframe in X_dict:
        X_train[timeframe] = X_dict[timeframe][train_indices]
        X_val[timeframe] = X_dict[timeframe][val_indices]
        X_test[timeframe] = X_dict[timeframe][test_indices]
    
    # Create train, validation, and test targets
    y_train = y_categorical[train_indices]
    y_val = y_categorical[val_indices]
    y_test = y_categorical[test_indices]
    
    logger.info(f"Data shapes:")
    for timeframe in X_dict:
        logger.info(f"  {timeframe} - X_train: {X_train[timeframe].shape}, X_val: {X_val[timeframe].shape}, X_test: {X_test[timeframe].shape}")
    logger.info(f"  y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_mtf_model(
    model: Model, 
    X_train: Dict[str, np.ndarray], 
    y_train: np.ndarray, 
    X_val: Dict[str, np.ndarray], 
    y_val: np.ndarray, 
    pair: str,
    timeframes: List[str],
    args
) -> Tuple:
    """Train multi-timeframe model"""
    logger.info(f"Training multi-timeframe model for {pair} with timeframes: {', '.join(timeframes)}...")
    
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create checkpoint path
    checkpoint_path = f"{MTF_MODEL_DIR}/mtf_{pair_clean}_model.h5"
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # Create logs directory
    logs_dir = f"{RESULTS_DIR}/logs/mtf_{pair_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logs_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Prepare input data
    X_train_list = [X_train[tf] for tf in timeframes]
    X_val_list = [X_val[tf] for tf in timeframes]
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train_list, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val_list, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Load the best model
    model = load_model(checkpoint_path)
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    logger.info(f"Multi-timeframe model training completed in {training_time_str}")
    
    return model, history, checkpoint_path, training_time

def evaluate_mtf_model(
    model: Model, 
    X_test: Dict[str, np.ndarray], 
    y_test: np.ndarray, 
    pair: str,
    timeframes: List[str]
) -> Dict:
    """Evaluate multi-timeframe model"""
    logger.info(f"Evaluating multi-timeframe model for {pair}...")
    
    # Prepare input data
    X_test_list = [X_test[tf] for tf in timeframes]
    
    # Basic evaluation
    loss, accuracy = model.evaluate(X_test_list, y_test, verbose=0)
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test_list, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    
    # Class mapping
    class_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
    y_pred_signal = np.array([class_map[c] for c in y_pred_class])
    y_test_signal = np.array([class_map[c] for c in y_test_class])
    
    # Calculate class-wise accuracy
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = y_test_class == i
        if np.sum(class_mask) > 0:
            class_accuracy[name] = np.mean(y_pred_class[class_mask] == i)
        else:
            class_accuracy[name] = 0.0
    
    # Calculate direction accuracy
    non_neutral_mask = (y_test_signal != 0) & (y_pred_signal != 0)
    if np.sum(non_neutral_mask) > 0:
        direction_accuracy = np.mean(np.sign(y_pred_signal[non_neutral_mask]) == np.sign(y_test_signal[non_neutral_mask]))
    else:
        direction_accuracy = 0.0
    
    # Calculate precision, recall, F1 for non-neutral classes
    binary_y_test = (y_test_signal > 0).astype(int)  # 1 for positive, 0 for negative
    binary_y_pred = (y_pred_signal > 0).astype(int)
    
    # Filter out neutral predictions
    non_neutral_indices = np.where((y_test_signal != 0) & (y_pred_signal != 0))[0]
    if len(non_neutral_indices) > 0:
        filtered_y_test = binary_y_test[non_neutral_indices]
        filtered_y_pred = binary_y_pred[non_neutral_indices]
        
        precision = precision_score(filtered_y_test, filtered_y_pred)
        recall = recall_score(filtered_y_test, filtered_y_pred)
        f1 = f1_score(filtered_y_test, filtered_y_pred)
    else:
        precision = 0
        recall = 0
        f1 = 0
    
    # Combine metrics
    metrics = {
        "accuracy": accuracy,
        "direction_accuracy": direction_accuracy,
        "class_accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    # Print metrics
    logger.info(f"Multi-timeframe model metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics

def save_mtf_model_report(
    pair: str,
    timeframes: List[str],
    model_path: str,
    metrics: Dict,
    training_time: float,
    args
) -> str:
    """Save report for multi-timeframe model"""
    # Create report filename
    pair_clean = pair.replace("/", "_").lower()
    report_file = f"{RESULTS_DIR}/mtf_model_report_{pair_clean}.json"
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Create report data
    report = {
        "pair": pair,
        "timeframes": timeframes,
        "model_path": model_path,
        "model_type": "mtf",
        "training_time": training_time_str,
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "predict_horizon": args.predict_horizon
        },
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved multi-timeframe model report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return ""

def update_ml_config_with_mtf_model(
    pair: str,
    timeframes: List[str],
    model_path: str,
    metrics: Dict
) -> bool:
    """Update ML configuration with multi-timeframe model"""
    logger.info(f"Updating ML configuration with multi-timeframe model for {pair}...")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
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
    config["global_settings"]["max_portfolio_risk"] = 0.25
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Create model key
    model_key = f"{pair}_mtf"
    
    # Add or update model config
    config["models"][model_key] = {
        "pair": pair,
        "timeframe": "mtf",
        "timeframes": timeframes,
        "model_type": "mtf",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "direction_accuracy": metrics["direction_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True,
        "preferred": True,  # Prefer MTF model over individual models
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with multi-timeframe model for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def build_ensemble_model(
    individual_models: Dict[str, str],
    pair: str,
    timeframes: List[str]
) -> Optional[str]:
    """Build ensemble model from individual timeframe models"""
    logger.info(f"Building ensemble model for {pair} with timeframes: {', '.join(timeframes)}...")
    
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create ensemble path
    ensemble_path = f"{ENSEMBLE_DIR}/ensemble_{pair_clean}_model.json"
    
    # Create ensemble configuration
    ensemble_config = {
        "pair": pair,
        "timeframes": timeframes,
        "models": individual_models,
        "weights": {tf: 1.0 / len(timeframes) for tf in timeframes},  # Equal weights initially
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save ensemble configuration
    try:
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info(f"Saved ensemble model configuration to {ensemble_path}")
        return ensemble_path
    except Exception as e:
        logger.error(f"Error saving ensemble configuration: {e}")
        return None

def update_ml_config_with_ensemble(
    pair: str,
    timeframes: List[str],
    ensemble_path: str
) -> bool:
    """Update ML configuration with ensemble model"""
    logger.info(f"Updating ML configuration with ensemble model for {pair}...")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
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
    config["global_settings"]["max_portfolio_risk"] = 0.25
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Create model key
    model_key = f"{pair}_ensemble"
    
    # Add or update model config
    config["models"][model_key] = {
        "pair": pair,
        "timeframe": "ensemble",
        "timeframes": timeframes,
        "model_type": "ensemble",
        "model_path": ensemble_path,
        "accuracy": 0.0,  # Will be determined during trading
        "direction_accuracy": 0.0,  # Will be determined during trading
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True,
        "preferred": True,  # Prefer ensemble model
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with ensemble model for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def generate_summary_report(
    individual_reports: Dict[str, Dict],
    mtf_reports: Dict[str, Dict],
    ensemble_configs: Dict[str, Dict],
    pair: str
) -> str:
    """Generate summary report for all models"""
    # Create report filename
    pair_clean = pair.replace("/", "_").lower()
    report_file = f"{RESULTS_DIR}/summary_report_{pair_clean}.md"
    
    # Create report
    report = f"# Multi-Timeframe Training Summary Report for {pair}\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add overview table for individual models
    report += "## Individual Timeframe Models\n\n"
    report += "| Timeframe | Accuracy | Direction Accuracy | Precision | Recall | F1 Score |\n"
    report += "|-----------|----------|-------------------|-----------|--------|----------|\n"
    
    for timeframe, data in sorted(individual_reports.items()):
        metrics = data.get("metrics", {})
        accuracy = metrics.get("accuracy", 0)
        direction_accuracy = metrics.get("direction_accuracy", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1_score = metrics.get("f1_score", 0)
        
        report += f"| {timeframe} | {accuracy:.2%} | {direction_accuracy:.2%} | {precision:.2%} | {recall:.2%} | {f1_score:.2%} |\n"
    
    # Add multi-timeframe model performance
    if mtf_reports:
        report += "\n## Multi-Timeframe Model\n\n"
        for mtf_name, data in mtf_reports.items():
            metrics = data.get("metrics", {})
            accuracy = metrics.get("accuracy", 0)
            direction_accuracy = metrics.get("direction_accuracy", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)
            timeframes = data.get("timeframes", [])
            
            report += f"Timeframes: {', '.join(timeframes)}\n\n"
            report += f"- **Accuracy**: {accuracy:.2%}\n"
            report += f"- **Direction Accuracy**: {direction_accuracy:.2%}\n"
            report += f"- **Precision**: {precision:.2%}\n"
            report += f"- **Recall**: {recall:.2%}\n"
            report += f"- **F1 Score**: {f1_score:.2%}\n"
    
    # Add ensemble model configurations
    if ensemble_configs:
        report += "\n## Ensemble Model\n\n"
        for ensemble_name, data in ensemble_configs.items():
            timeframes = data.get("timeframes", [])
            weights = data.get("weights", {})
            
            report += f"Timeframes: {', '.join(timeframes)}\n\n"
            report += "Weights:\n"
            for tf, weight in weights.items():
                report += f"- {tf}: {weight:.2f}\n"
    
    # Add detailed class accuracy for individual models
    report += "\n## Detailed Class Accuracy by Timeframe\n\n"
    
    for timeframe, data in sorted(individual_reports.items()):
        report += f"### {timeframe}\n\n"
        
        class_accuracy = data.get("metrics", {}).get("class_accuracy", {})
        for class_name, accuracy in class_accuracy.items():
            report += f"- {class_name}: {accuracy:.2%}\n"
        
        report += "\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    
    # Find best individual model
    best_individual = max(
        individual_reports.items(),
        key=lambda x: x[1].get("metrics", {}).get("f1_score", 0)
    )
    best_timeframe = best_individual[0]
    best_f1 = best_individual[1].get("metrics", {}).get("f1_score", 0)
    
    report += f"1. **Best Individual Model**: {best_timeframe} timeframe with F1 score of {best_f1:.2%}\n"
    
    # Compare with MTF model
    if mtf_reports:
        mtf_f1 = list(mtf_reports.values())[0].get("metrics", {}).get("f1_score", 0)
        if mtf_f1 > best_f1:
            report += "2. The Multi-Timeframe model outperforms all individual models\n"
        else:
            report += f"2. The {best_timeframe} model outperforms the Multi-Timeframe model\n"
    
    # Recommend ensemble approach
    report += "3. Consider using the Ensemble model for production, as it can adapt to changing market conditions\n"
    report += "4. For optimal results, adjust ensemble weights based on recent performance\n"
    
    # Trading recommendations
    report += "\n## Trading Recommendations\n\n"
    report += "Based on model performance, consider the following trading strategy:\n\n"
    report += "1. **Entry signals**: Use the ensemble model for entry signals\n"
    report += "2. **Position sizing**: Scale position size based on model confidence\n"
    report += "3. **Leverage**: Start with base leverage of 5x and adjust up to 75x based on prediction confidence\n"
    report += "4. **Exit signals**: Use the 15m timeframe model for short-term exit signals\n"
    report += "5. **Risk management**: Implement dynamic stop-loss based on the ATR indicator\n"
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved summary report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
        return ""

def train_individual_timeframe_models(pair: str, timeframes: List[str], args) -> Dict[str, Dict]:
    """Train models for individual timeframes"""
    logger.info(f"Training individual timeframe models for {pair}...")
    
    # Dictionary to store reports for each timeframe
    individual_reports = {}
    
    # Train model for each timeframe
    for timeframe in timeframes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {pair} ({timeframe})")
        logger.info(f"{'='*80}\n")
        
        # Load data
        df = load_historical_data(pair, timeframe)
        
        if df is None and args.fetch_missing:
            # Try to fetch missing data
            fetch_missing_data(pair, timeframe)
            
            # Try to load again
            df = load_historical_data(pair, timeframe)
        
        if df is None:
            logger.error(f"No data available for {pair} ({timeframe}). Skipping...")
            continue
        
        # Calculate indicators
        df_indicators = calculate_indicators(df, timeframe)
        
        # Create target variable
        df_labeled = create_target_variable(df_indicators, timeframe, args.predict_horizon)
        
        # Prepare data for training
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns = prepare_data_for_training(
            df_labeled, args.sequence_length, test_size=0.2, validation_size=0.2
        )
        
        # Build model
        input_shape = (args.sequence_length, len(feature_columns))
        model = build_individual_model(input_shape, timeframe)
        
        # Train model
        model, history, model_path, training_time = train_individual_model(
            model, X_train, y_train, X_val, y_val, pair, timeframe, args
        )
        
        # Evaluate model
        metrics = evaluate_individual_model(
            model, X_test, y_test, pair, timeframe
        )
        
        # Save report
        report_path = save_individual_model_report(
            pair, timeframe, model_path, metrics, training_time, args
        )
        
        # Update ML config
        if args.update_ml_config:
            update_ml_config_with_individual_model(
                pair, timeframe, model_path, metrics
            )
        
        # Store report
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
                individual_reports[timeframe] = report
        except Exception as e:
            logger.error(f"Error loading report: {e}")
    
    return individual_reports

def train_mtf_model_phase(pair: str, timeframes: List[str], args) -> Dict[str, Dict]:
    """Train multi-timeframe model"""
    logger.info(f"Training multi-timeframe model for {pair}...")
    
    # Dictionary to store reports
    mtf_reports = {}
    
    # Dictionary to store data, scalers, and feature columns for each timeframe
    data_dict = {}
    scaler_dict = {}
    feature_columns_dict = {}
    sequence_lengths_dict = {}
    
    # Load and prepare data for each timeframe
    for timeframe in timeframes:
        # Load data
        df = load_historical_data(pair, timeframe)
        
        if df is None and args.fetch_missing:
            # Try to fetch missing data
            fetch_missing_data(pair, timeframe)
            
            # Try to load again
            df = load_historical_data(pair, timeframe)
        
        if df is None:
            logger.error(f"No data available for {pair} ({timeframe}). Skipping...")
            continue
        
        # Calculate indicators
        df_indicators = calculate_indicators(df, timeframe)
        
        # Create target variable
        df_labeled = create_target_variable(df_indicators, timeframe, args.predict_horizon)
        
        # Store data
        data_dict[timeframe] = df_labeled
        
        # Prepare sequence length
        # Higher timeframes need shorter sequences
        if timeframe in ['4h', '1d']:
            sequence_lengths_dict[timeframe] = max(20, args.sequence_length // 4)
        else:
            sequence_lengths_dict[timeframe] = args.sequence_length
        
        # Select features
        feature_columns = [col for col in df_labeled.columns if col not in ['timestamp', 'target', 'target_class']]
        feature_columns_dict[timeframe] = feature_columns
        
        # Fit scaler
        scaler = MinMaxScaler()
        scaler.fit(df_labeled[feature_columns])
        scaler_dict[timeframe] = scaler
    
    # Check if we have enough data
    if len(data_dict) < 2:
        logger.error(f"Not enough timeframes with data for {pair}. Skipping MTF model...")
        return mtf_reports
    
    # Align data from multiple timeframes
    aligned_data, common_timestamps = align_mtf_data(data_dict, sequence_lengths_dict)
    
    # Prepare data for MTF model
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_mtf_data(
        aligned_data, common_timestamps, feature_columns_dict, scaler_dict, sequence_lengths_dict,
        test_size=0.2, validation_size=0.2
    )
    
    # Build MTF model
    input_shapes = {tf: (sequence_lengths_dict[tf], len(feature_columns_dict[tf])) for tf in aligned_data}
    mtf_model = build_mtf_model(input_shapes, list(aligned_data.keys()))
    
    # Train MTF model
    mtf_model, history, model_path, training_time = train_mtf_model(
        mtf_model, X_train, y_train, X_val, y_val, pair, list(aligned_data.keys()), args
    )
    
    # Evaluate MTF model
    metrics = evaluate_mtf_model(
        mtf_model, X_test, y_test, pair, list(aligned_data.keys())
    )
    
    # Save report
    report_path = save_mtf_model_report(
        pair, list(aligned_data.keys()), model_path, metrics, training_time, args
    )
    
    # Update ML config
    if args.update_ml_config:
        update_ml_config_with_mtf_model(
            pair, list(aligned_data.keys()), model_path, metrics
        )
    
    # Store report
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
            mtf_reports["mtf"] = report
    except Exception as e:
        logger.error(f"Error loading report: {e}")
    
    return mtf_reports

def build_ensemble_phase(pair: str, timeframes: List[str], individual_reports: Dict[str, Dict]) -> Dict[str, Dict]:
    """Build ensemble model"""
    logger.info(f"Building ensemble model for {pair}...")
    
    # Dictionary to store ensemble configurations
    ensemble_configs = {}
    
    # Get model paths from individual reports
    individual_models = {}
    for timeframe, report in individual_reports.items():
        model_path = report.get("model_path")
        if model_path and os.path.exists(model_path):
            individual_models[timeframe] = model_path
    
    if len(individual_models) < 2:
        logger.error(f"Not enough individual models for {pair}. Skipping ensemble...")
        return ensemble_configs
    
    # Build ensemble model
    ensemble_path = build_ensemble_model(individual_models, pair, list(individual_models.keys()))
    
    if ensemble_path:
        # Update ML config
        update_ml_config_with_ensemble(pair, list(individual_models.keys()), ensemble_path)
        
        # Store ensemble configuration
        try:
            with open(ensemble_path, 'r') as f:
                ensemble_config = json.load(f)
                ensemble_configs["ensemble"] = ensemble_config
        except Exception as e:
            logger.error(f"Error loading ensemble configuration: {e}")
    
    return ensemble_configs

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine timeframes
    if args.timeframes == "all":
        timeframes = TRAINING_TIMEFRAMES
    else:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-TIMEFRAME MODEL TRAINER")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Prediction Horizon: {args.predict_horizon}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Update ML Config: {args.update_ml_config}")
    logger.info(f"Fetch Missing Data: {args.fetch_missing}")
    logger.info("=" * 80 + "\n")
    
    # Dictionary to store reports for each phase
    individual_reports = {}
    mtf_reports = {}
    ensemble_configs = {}
    
    # Train individual timeframe models
    if args.phase in ["individual", "all"]:
        individual_reports = train_individual_timeframe_models(args.pair, timeframes, args)
    
    # Train multi-timeframe model
    if args.phase in ["mtf", "all"] and len(timeframes) >= 2:
        mtf_reports = train_mtf_model_phase(args.pair, timeframes, args)
    
    # Build ensemble model
    if args.phase in ["ensemble", "all"] and individual_reports:
        ensemble_configs = build_ensemble_phase(args.pair, timeframes, individual_reports)
    
    # Generate summary report
    if individual_reports or mtf_reports or ensemble_configs:
        summary_report = generate_summary_report(individual_reports, mtf_reports, ensemble_configs, args.pair)
        
        if summary_report:
            logger.info(f"Generated summary report: {summary_report}")
    
    # Print success message
    logger.info("\nMulti-timeframe training completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())