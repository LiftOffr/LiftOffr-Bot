#!/usr/bin/env python3
"""
Multi-Timeframe Trainer

This script implements a sophisticated multi-timeframe training approach:
1. Trains individual models for each timeframe (15m, 1h, 4h, 1d)
2. Creates unified models that combine data from all timeframes
3. Builds ensemble meta-models that integrate predictions from all timeframes

Usage:
    python multi_timeframe_trainer.py --pair BTC/USD [--all] [--skip-individual]
                                     [--skip-unified] [--skip-ensemble]
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
TIMEFRAMES = ['15m', '1h', '4h', '1d']
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bbands_upper', 'bbands_middle', 'bbands_lower',
    'atr_14'
]
TARGET_COLUMNS = ['next_close_pct']
SUPPORTED_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Timeframe Model Trainer')
    parser.add_argument('--pair', type=str, default='BTC/USD',
                        help='Trading pair to train for (e.g., BTC/USD)')
    parser.add_argument('--all', action='store_true',
                        help='Train models for all supported trading pairs')
    parser.add_argument('--skip-individual', action='store_true',
                        help='Skip training individual timeframe models')
    parser.add_argument('--skip-unified', action='store_true',
                        help='Skip training unified models')
    parser.add_argument('--skip-ensemble', action='store_true',
                        help='Skip training ensemble models')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining of existing models')
    return parser.parse_args()

def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical data for a trading pair and timeframe
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        
    Returns:
        DataFrame with historical data
    """
    pair_symbol = pair.replace('/', '_')
    file_path = f'historical_data/{pair_symbol}_{timeframe}.csv'
    
    if not os.path.exists(file_path):
        logging.warning(f"Data file not found: {file_path}")
        return None
    
    # Load data from CSV
    df = pd.read_csv(file_path)
    
    # Make sure we have a datetime index
    if 'timestamp' in df.columns:
        # Try to parse the timestamp as a datetime string first
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        except (ValueError, TypeError):
            # If that fails, try parsing as Unix timestamp (seconds since epoch)
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            except:
                logging.error(f"Failed to parse timestamp column in the data")
                return None
                
        df.set_index('datetime', inplace=True)
    
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added features
    """
    if df is None or len(df) == 0:
        return None
        
    # Copy the dataframe to avoid modifying the original
    df = df.copy()
    
    # Calculate EMA indicators
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate Bollinger Bands
    df['bbands_middle'] = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    df['bbands_upper'] = df['bbands_middle'] + (std_dev * 2)
    df['bbands_lower'] = df['bbands_middle'] - (std_dev * 2)
    
    # Calculate ATR (14-period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Target: next period's percentage change
    df['next_close'] = df['close'].shift(-1)
    df['next_close_pct'] = (df['next_close'] / df['close'] - 1) * 100
    
    # Drop any rows with missing values
    df.dropna(inplace=True)
    
    return df

def prepare_sequence_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sequence_length: int,
    test_size: float = 0.2,
    validation_size: float = 0.2
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare sequence data for LSTM/TCN models
    
    Args:
        df: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sequence_length: Length of input sequences
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        
    Returns:
        Dictionary with train/val/test data and scalers
    """
    if df is None or len(df) <= sequence_length:
        return None, None

    # Extract features and targets
    features = df[feature_columns].values
    targets = df[target_columns].values
    
    # Scale features
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # Scale targets
    target_scaler = StandardScaler()
    targets_scaled = target_scaler.fit_transform(targets)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(targets_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, shuffle=False
    )
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }
    
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
    }
    
    return data, scalers

def create_hybrid_model(
    input_shape: Tuple[int, int],
    output_dim: int = 1,
    tcn_units: int = 64,
    tcn_kernel_size: int = 3,
    tcn_dilations: List[int] = [1, 2, 4, 8, 16],
    lstm_units: int = 64,
    cnn_filters: int = 32,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Create a hybrid model with CNN, LSTM, and TCN branches
    
    Args:
        input_shape: Input shape (sequence_length, features)
        output_dim: Number of output dimensions
        tcn_units: Number of TCN units
        tcn_kernel_size: TCN kernel size
        tcn_dilations: TCN dilations
        lstm_units: Number of LSTM units
        cnn_filters: Number of CNN filters
        dropout_rate: Dropout rate
        
    Returns:
        Hybrid model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # CNN branch
    cnn_branch = layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn_branch = layers.MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = layers.Conv1D(filters=cnn_filters*2, kernel_size=3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = layers.GlobalAveragePooling1D()(cnn_branch)
    cnn_branch = layers.Dense(32, activation='relu')(cnn_branch)
    
    # LSTM branch with custom attention mechanism
    lstm_branch = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    
    # Custom attention implementation
    # Query, key, and value are all the same in self-attention
    query = lstm_branch
    value = lstm_branch
    
    # Calculate attention scores
    attention_scores = layers.Dense(lstm_units * 2)(query)  # Bidirectional has 2x units
    attention_scores = layers.Activation('tanh')(attention_scores)
    attention_scores = layers.Dense(1)(attention_scores)  # (batch, seq_len, 1)
    
    # Apply softmax to get attention weights
    attention_weights = layers.Softmax(axis=1)(attention_scores)  # (batch, seq_len, 1)
    
    # Apply attention weights to the value
    context_vector = layers.Multiply()([value, attention_weights])  # (batch, seq_len, units*2)
    context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)  # (batch, units*2)
    
    # Continue with LSTM processing
    lstm_branch = layers.Bidirectional(layers.LSTM(lstm_units // 2))(lstm_branch)
    lstm_branch = layers.Dense(32, activation='relu')(lstm_branch)
    
    # Custom TCN-like branch using 1D convolutions with dilations
    tcn_branch = inputs
    for dilation_rate in tcn_dilations:
        tcn_branch = layers.Conv1D(
            filters=tcn_units,
            kernel_size=tcn_kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu'
        )(tcn_branch)
        tcn_branch = layers.BatchNormalization()(tcn_branch)
        tcn_branch = layers.Dropout(dropout_rate)(tcn_branch)
    
    # Global pooling to reduce sequence dimension
    tcn_branch = layers.GlobalAveragePooling1D()(tcn_branch)
    tcn_branch = layers.Dense(32, activation='relu')(tcn_branch)
    
    # Combine branches
    combined = layers.concatenate([cnn_branch, lstm_branch, tcn_branch])
    combined = layers.Dropout(dropout_rate)(combined)
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    
    # Output layer
    outputs = layers.Dense(output_dim, activation='linear')(combined)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def train_individual_timeframe_model(
    pair: str,
    timeframe: str,
    force: bool = False
) -> Optional[Tuple[tf.keras.Model, Dict[str, Any]]]:
    """
    Train a model for a specific timeframe
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        force: Whether to force retraining
        
    Returns:
        Tuple of (trained model, model info)
    """
    pair_symbol = pair.replace('/', '_')
    model_name = f"{pair_symbol}_{timeframe}"
    model_path = f"ml_models/{model_name}.h5"
    info_path = f"ml_models/{model_name}_info.json"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(info_path) and not force:
        logging.info(f"Model already exists for {pair} ({timeframe}). Use --force to retrain.")
        return None
    
    logging.info(f"Training individual model for {pair} ({timeframe})...")
    
    # Load and prepare data
    df = load_data(pair, timeframe)
    if df is None:
        logging.error(f"Failed to load data for {pair} ({timeframe})")
        return None
    
    # Add features
    df = add_features(df)
    if df is None or len(df) == 0:
        logging.error(f"Failed to add features for {pair} ({timeframe})")
        return None
    
    # Prepare sequence data
    data, scalers = prepare_sequence_data(
        df, FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
    )
    if data is None:
        logging.error(f"Failed to prepare sequence data for {pair} ({timeframe})")
        return None
    
    # Create and train model
    input_shape = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    model = create_hybrid_model(input_shape, len(TARGET_COLUMNS))
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    train_metrics = model.evaluate(data['X_train'], data['y_train'], verbose=0)
    val_metrics = model.evaluate(data['X_val'], data['y_val'], verbose=0)
    test_metrics = model.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    # Make predictions on test set
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = scalers['target_scaler'].inverse_transform(y_pred_scaled)
    y_true = scalers['target_scaler'].inverse_transform(data['y_test'])
    
    # Calculate additional metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_pred_direction = np.sign(y_pred)
    y_true_direction = np.sign(y_true)
    directional_accuracy = np.mean(y_pred_direction == y_true_direction)
    
    # Save model and info
    os.makedirs('ml_models', exist_ok=True)
    model.save(model_path)
    
    model_info = {
        'pair': pair,
        'timeframe': timeframe,
        'model_type': 'hybrid',
        'feature_columns': FEATURE_COLUMNS,
        'target_columns': TARGET_COLUMNS,
        'sequence_length': SEQUENCE_LENGTH,
        'training_time': training_time,
        'epochs': len(history.history['loss']),
        'metrics': {
            'train_loss': float(train_metrics[0]),
            'train_mae': float(train_metrics[1]),
            'val_loss': float(val_metrics[0]),
            'val_mae': float(val_metrics[1]),
            'test_loss': float(test_metrics[0]),
            'test_mae': float(test_metrics[1]),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logging.info(f"Saved model and info for {pair} ({timeframe})")
    logging.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
    
    return model, model_info

def align_timeframe_data(
    pair: str,
    required_timeframes: List[str] = TIMEFRAMES
) -> Optional[pd.DataFrame]:
    """
    Align data from multiple timeframes to create a unified dataset
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        required_timeframes: List of required timeframes
        
    Returns:
        DataFrame with aligned data from all timeframes
    """
    pair_symbol = pair.replace('/', '_')
    dataframes = {}
    
    # Load and process data for each timeframe
    for tf in required_timeframes:
        df = load_data(pair, tf)
        if df is None:
            logging.error(f"Missing data for {pair} ({tf})")
            return None
        
        # Add features and select needed columns
        df = add_features(df)
        if df is None or len(df) == 0:
            logging.error(f"Failed to add features for {pair} ({tf})")
            return None
        
        # Keep needed columns and add timeframe suffix
        needed_cols = FEATURE_COLUMNS + TARGET_COLUMNS
        df = df[needed_cols]
        df = df.add_suffix(f'_{tf}')
        
        dataframes[tf] = df
    
    # Start with the highest resolution timeframe (15m)
    aligned_df = dataframes['15m'].copy()
    
    # For each higher timeframe, assign its values to the corresponding 15m candles
    for tf in required_timeframes[1:]:  # 1h, 4h, 1d
        df_higher = dataframes[tf]
        
        # Resample the higher timeframe to 15m frequency (forward-fill)
        # Default to None in case none of the conditions are met
        df_resampled = None
        
        if tf == '1h':
            df_resampled = df_higher.resample('15T').ffill()
        elif tf == '4h':
            df_resampled = df_higher.resample('15T').ffill()
        elif tf == '1d':
            df_resampled = df_higher.resample('15T').ffill()
        
        # If resampling failed for some reason, skip this timeframe
        if df_resampled is None:
            logging.warning(f"Failed to resample {tf} data for {pair}, skipping this timeframe")
            continue
        
        # Merge with the aligned dataframe
        aligned_df = aligned_df.join(df_resampled, how='inner')
    
    # Drop rows with missing values
    aligned_df.dropna(inplace=True)
    
    # Ensure we have enough data
    if len(aligned_df) <= SEQUENCE_LENGTH:
        logging.error(f"Not enough aligned data for {pair}")
        return None
    
    return aligned_df

def train_unified_model(
    pair: str,
    force: bool = False
) -> Optional[Tuple[tf.keras.Model, Dict[str, Any]]]:
    """
    Train a unified model that combines data from all timeframes
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        force: Whether to force retraining
        
    Returns:
        Tuple of (trained model, model info)
    """
    pair_symbol = pair.replace('/', '_')
    model_name = f"{pair_symbol}_unified"
    model_path = f"ml_models/{model_name}.h5"
    info_path = f"ml_models/{model_name}_info.json"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(info_path) and not force:
        logging.info(f"Unified model already exists for {pair}. Use --force to retrain.")
        return None
    
    logging.info(f"Training unified model for {pair}...")
    
    # Align data from all timeframes
    aligned_df = align_timeframe_data(pair)
    if aligned_df is None:
        logging.error(f"Failed to align timeframe data for {pair}")
        return None
    
    # Define feature and target columns for the unified model
    unified_feature_cols = []
    for tf in TIMEFRAMES:
        unified_feature_cols.extend([f"{col}_{tf}" for col in FEATURE_COLUMNS])
    
    unified_target_cols = [f"next_close_pct_15m"]
    
    # Prepare sequence data
    data, scalers = prepare_sequence_data(
        aligned_df, unified_feature_cols, unified_target_cols,
        SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
    )
    if data is None:
        logging.error(f"Failed to prepare sequence data for unified model")
        return None
    
    # Create and train model with expanded input shape
    input_shape = (SEQUENCE_LENGTH, len(unified_feature_cols))
    model = create_hybrid_model(
        input_shape, len(unified_target_cols),
        tcn_units=96, lstm_units=96, cnn_filters=48
    )
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    train_metrics = model.evaluate(data['X_train'], data['y_train'], verbose=0)
    val_metrics = model.evaluate(data['X_val'], data['y_val'], verbose=0)
    test_metrics = model.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    # Make predictions on test set
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = scalers['target_scaler'].inverse_transform(y_pred_scaled)
    y_true = scalers['target_scaler'].inverse_transform(data['y_test'])
    
    # Calculate additional metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_pred_direction = np.sign(y_pred)
    y_true_direction = np.sign(y_true)
    directional_accuracy = np.mean(y_pred_direction == y_true_direction)
    
    # Save model and info
    os.makedirs('ml_models', exist_ok=True)
    model.save(model_path)
    
    model_info = {
        'pair': pair,
        'model_type': 'unified',
        'timeframes': TIMEFRAMES,
        'feature_columns': unified_feature_cols,
        'target_columns': unified_target_cols,
        'sequence_length': SEQUENCE_LENGTH,
        'training_time': training_time,
        'epochs': len(history.history['loss']),
        'metrics': {
            'train_loss': float(train_metrics[0]),
            'train_mae': float(train_metrics[1]),
            'val_loss': float(val_metrics[0]),
            'val_mae': float(val_metrics[1]),
            'test_loss': float(test_metrics[0]),
            'test_mae': float(test_metrics[1]),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logging.info(f"Saved unified model and info for {pair}")
    logging.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
    
    return model, model_info

def train_ensemble_model(
    pair: str,
    force: bool = False
) -> Optional[Tuple[tf.keras.Model, Dict[str, Any]]]:
    """
    Train an ensemble meta-model that combines predictions from all timeframe models
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        force: Whether to force retraining
        
    Returns:
        Tuple of (trained model, model info)
    """
    pair_symbol = pair.replace('/', '_')
    model_name = f"{pair_symbol}_ensemble"
    model_path = f"ml_models/{model_name}.h5"
    info_path = f"ml_models/{model_name}_info.json"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(info_path) and not force:
        logging.info(f"Ensemble model already exists for {pair}. Use --force to retrain.")
        return None
    
    logging.info(f"Training ensemble model for {pair}...")
    
    # Check if individual timeframe models exist
    individual_models = {}
    individual_info = {}
    for tf in TIMEFRAMES:
        model_file = f"ml_models/{pair_symbol}_{tf}.h5"
        info_file = f"ml_models/{pair_symbol}_{tf}_info.json"
        
        if not os.path.exists(model_file) or not os.path.exists(info_file):
            logging.error(f"Missing individual model for {pair} ({tf})")
            return None
        
        individual_models[tf] = keras.models.load_model(model_file)
        with open(info_file, 'r') as f:
            individual_info[tf] = json.load(f)
    
    # Also check if unified model exists
    unified_model_file = f"ml_models/{pair_symbol}_unified.h5"
    unified_info_file = f"ml_models/{pair_symbol}_unified_info.json"  # Fixed filename
    
    if os.path.exists(unified_model_file) and os.path.exists(unified_info_file):
        unified_model = keras.models.load_model(unified_model_file)
        with open(unified_info_file, 'r') as f:
            unified_info = json.load(f)
        logging.info(f"Loaded unified model for {pair}")
    else:
        logging.info(f"Unified model not found for {pair}")
        unified_model = None
        unified_info = None
    
    # Align data from all timeframes
    aligned_df = align_timeframe_data(pair)
    if aligned_df is None:
        logging.error(f"Failed to align timeframe data for {pair}")
        return None
    
    # Generate predictions from individual timeframe models
    meta_features = []
    
    # For each timeframe, generate predictions using the appropriate model
    for tf in TIMEFRAMES:
        # Extract features for this timeframe
        tf_feature_cols = [f"{col}_{tf}" for col in FEATURE_COLUMNS]
        tf_target_cols = [f"next_close_pct_{tf}"]
        
        # Prepare sequence data
        data, scalers = prepare_sequence_data(
            aligned_df, tf_feature_cols, tf_target_cols,
            SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
        )
        
        if data is None:
            logging.error(f"Failed to prepare sequence data for {pair} ({tf})")
            return None
        
        # Generate predictions
        y_pred_scaled = individual_models[tf].predict(data['X_test'])
        
        # Store predictions as meta-features
        meta_features.append(y_pred_scaled)
    
    # Add unified model predictions if available
    if unified_model is not None:
        # Extract features for unified model
        unified_feature_cols = []
        for tf in TIMEFRAMES:
            unified_feature_cols.extend([f"{col}_{tf}" for col in FEATURE_COLUMNS])
        
        unified_target_cols = [f"next_close_pct_15m"]
        
        # Prepare sequence data
        unified_data, unified_scalers = prepare_sequence_data(
            aligned_df, unified_feature_cols, unified_target_cols,
            SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
        )
        
        if unified_data is not None:
            # Generate predictions
            unified_y_pred_scaled = unified_model.predict(unified_data['X_test'])
            
            # Store predictions as meta-features
            meta_features.append(unified_y_pred_scaled)
    
    # Combine predictions into meta-features
    X_meta = np.concatenate(meta_features, axis=1)
    
    # Use 15m actual values as targets for the meta-model
    # Make sure we have data from the last timeframe in the loop
    y_meta = None
    final_scalers = None
    
    # Get the data from the 15m timeframe models specifically
    for tf in TIMEFRAMES:
        if tf == '15m':
            tf_feature_cols = [f"{col}_{tf}" for col in FEATURE_COLUMNS]
            tf_target_cols = [f"next_close_pct_{tf}"]
            
            temp_data, temp_scalers = prepare_sequence_data(
                aligned_df, tf_feature_cols, tf_target_cols,
                SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
            )
            
            if temp_data is not None:
                y_meta = temp_data['y_test']
                final_scalers = temp_scalers
                logging.info(f"Using {tf} targets for ensemble meta-model")
                break
    
    # If we couldn't get 15m data, try any timeframe
    if y_meta is None:
        for tf in TIMEFRAMES:
            tf_feature_cols = [f"{col}_{tf}" for col in FEATURE_COLUMNS]
            tf_target_cols = [f"next_close_pct_{tf}"]
            
            temp_data, temp_scalers = prepare_sequence_data(
                aligned_df, tf_feature_cols, tf_target_cols,
                SEQUENCE_LENGTH, TEST_SIZE, VALIDATION_SIZE
            )
            
            if temp_data is not None:
                y_meta = temp_data['y_test']
                final_scalers = temp_scalers
                logging.info(f"Using {tf} targets for ensemble meta-model")
                break
    
    if y_meta is None or final_scalers is None:
        logging.error(f"Could not find target data for ensemble model for {pair}")
        return None
    
    # Split meta-features into training and testing sets
    meta_split_idx = int(len(X_meta) * 0.7)
    X_meta_train = X_meta[:meta_split_idx]
    y_meta_train = y_meta[:meta_split_idx]
    X_meta_test = X_meta[meta_split_idx:]
    y_meta_test = y_meta[meta_split_idx:]
    
    # Create and train ensemble meta-model
    meta_model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_meta.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    meta_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Train meta-model
    start_time = time.time()
    meta_history = meta_model.fit(
        X_meta_train, y_meta_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate meta-model
    meta_metrics = meta_model.evaluate(X_meta_test, y_meta_test, verbose=0)
    
    # Make predictions
    y_meta_pred = meta_model.predict(X_meta_test)
    
    # Calculate additional metrics
    y_true = final_scalers['target_scaler'].inverse_transform(y_meta_test)
    y_pred = final_scalers['target_scaler'].inverse_transform(y_meta_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_pred_direction = np.sign(y_pred)
    y_true_direction = np.sign(y_true)
    directional_accuracy = np.mean(y_pred_direction == y_true_direction)
    
    # Save model and info
    os.makedirs('ml_models', exist_ok=True)
    meta_model.save(model_path)
    
    ensemble_info = {
        'pair': pair,
        'model_type': 'ensemble',
        'timeframes': TIMEFRAMES,
        'includes_unified': unified_model is not None,
        'training_time': training_time,
        'epochs': len(meta_history.history['loss']),
        'metrics': {
            'loss': float(meta_metrics[0]),
            'mae': float(meta_metrics[1]),
            'rmse': float(rmse),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        },
        'component_models': {
            tf: {
                'accuracy': individual_info[tf]['metrics']['directional_accuracy']
            } for tf in TIMEFRAMES
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open(info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    logging.info(f"Saved ensemble model and info for {pair}")
    logging.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
    
    return meta_model, ensemble_info

def train_all_for_pair(pair: str, args: argparse.Namespace) -> bool:
    """
    Train all model types for a single trading pair
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        args: Command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    pair_symbol = pair.replace('/', '_')
    logging.info(f"=== Training all models for {pair} ===")
    
    # Train individual timeframe models
    if not args.skip_individual:
        for tf in TIMEFRAMES:
            result = train_individual_timeframe_model(pair, tf, args.force)
            if result is None and not os.path.exists(f"ml_models/{pair_symbol}_{tf}.h5"):
                logging.error(f"Failed to train individual model for {pair} ({tf})")
                return False
    
    # Train unified model
    if not args.skip_unified:
        result = train_unified_model(pair, args.force)
        if result is None and not os.path.exists(f"ml_models/{pair_symbol}_unified.h5"):
            logging.warning(f"Failed to train unified model for {pair}")
            # Continue anyway, as ensemble model can work with just individual models
    
    # Train ensemble model
    if not args.skip_ensemble:
        result = train_ensemble_model(pair, args.force)
        if result is None and not os.path.exists(f"ml_models/{pair_symbol}_ensemble.h5"):
            logging.error(f"Failed to train ensemble model for {pair}")
            return False
    
    logging.info(f"=== Completed training for {pair} ===")
    return True

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create directory for models if it doesn't exist
    os.makedirs('ml_models', exist_ok=True)
    
    # Determine which pairs to process
    pairs_to_process = SUPPORTED_PAIRS if args.all else [args.pair]
    
    results = {}
    for pair in pairs_to_process:
        success = train_all_for_pair(pair, args)
        results[pair] = success
    
    # Print summary
    logging.info("=== Training Summary ===")
    for pair, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"{pair}: {status}")
    
    # Overall success
    overall_success = all(results.values())
    if overall_success:
        logging.info("All models trained successfully!")
    else:
        logging.warning("Some models failed to train. Check the logs for details.")
    
    return overall_success

if __name__ == "__main__":
    main()