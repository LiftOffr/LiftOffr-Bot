#!/usr/bin/env python3
"""
Train Models with Small Timeframes (1m and 5m)

This script implements specialized training for models using 1m and 5m data, 
with techniques to avoid overfitting on high-frequency data and ensure models
generalize properly to live trading conditions.

Usage:
    python train_with_small_timeframes.py [--pair PAIR] [--all] [--force] [--days DAYS]
"""

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_small_timeframes.log')
    ]
)

# Constants
TIMEFRAMES = ['1m', '5m']
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
SEQUENCE_LENGTH = 120  # Longer sequence for small timeframes
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
NOISE_FACTOR = 0.01  # Factor for adding noise to prevent overfitting


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Small Timeframe Model Trainer')
    parser.add_argument('--pair', type=str, default='BTC/USD',
                      help='Trading pair to train for (e.g., BTC/USD)')
    parser.add_argument('--all', action='store_true',
                      help='Train models for all supported trading pairs')
    parser.add_argument('--force', action='store_true',
                      help='Force retraining of existing models')
    parser.add_argument('--days', type=int, default=14,
                      help='Number of days of data to use for training')
    return parser.parse_args()


def load_small_timeframe_data(pair: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical data for a trading pair and small timeframe
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1m, 5m)
        
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
    
    # Add categorical labels based on next_close_pct
    # This creates 5 classes: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up
    bins = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]
    labels = [0, 1, 2, 3, 4]  # 0: Strong Down, 4: Strong Up
    df['direction_class'] = pd.cut(df['next_close_pct'], bins=bins, labels=labels)
    
    # Drop any rows with missing values
    df.dropna(inplace=True)
    
    return df


def prepare_sequence_data_with_augmentation(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sequence_length: int,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    noise_factor: float = 0.01
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare sequence data with data augmentation to prevent overfitting
    
    Args:
        df: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sequence_length: Length of input sequences
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        noise_factor: Factor for adding Gaussian noise to training data
        
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
    
    # Split into train and test sets using chronological order (no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]
    
    # Split train into train and validation
    val_split_idx = int(len(X_train_val) * (1 - validation_size))
    X_train, X_val = X_train_val[:val_split_idx], X_train_val[val_split_idx:]
    y_train, y_val = y_train_val[:val_split_idx], y_train_val[val_split_idx:]
    
    # Data augmentation for training set only
    # Add Gaussian noise to training data to prevent overfitting
    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor, X_train.shape)
        X_train_noisy = X_train + noise
        X_train = np.concatenate([X_train, X_train_noisy], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
    
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


def create_enhanced_small_timeframe_model(
    input_shape: Tuple[int, int],
    output_dim: int = 1,
    lstm_units: int = 128,
    cnn_filters: int = 64,
    regularization: float = 0.001,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Create an enhanced model optimized for small timeframes
    
    Args:
        input_shape: Input shape (sequence_length, features)
        output_dim: Number of output dimensions
        lstm_units: Number of LSTM units
        cnn_filters: Number of CNN filters
        regularization: L2 regularization strength
        dropout_rate: Dropout rate
        
    Returns:
        Enhanced model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Add dropout to input as a form of noise during training
    input_dropout = layers.SpatialDropout1D(dropout_rate/2)(inputs)
    
    # CNN path for local feature extraction
    cnn_branch = layers.Conv1D(
        filters=cnn_filters, 
        kernel_size=5, 
        padding='same',
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(regularization)
    )(input_dropout)
    cnn_branch = layers.BatchNormalization()(cnn_branch)
    cnn_branch = layers.MaxPooling1D(pool_size=2)(cnn_branch)
    
    cnn_branch = layers.Conv1D(
        filters=cnn_filters*2, 
        kernel_size=3, 
        padding='same',
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(regularization)
    )(cnn_branch)
    cnn_branch = layers.BatchNormalization()(cnn_branch)
    cnn_branch = layers.MaxPooling1D(pool_size=2)(cnn_branch)
    
    # LSTM path with attention for sequence modeling
    lstm_branch = layers.Bidirectional(layers.LSTM(
        lstm_units, 
        return_sequences=True,
        recurrent_regularizer=keras.regularizers.l2(regularization)
    ))(input_dropout)
    lstm_branch = layers.BatchNormalization()(lstm_branch)
    
    # Custom attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_branch)
    attention_weights = layers.Softmax(axis=1)(attention)
    context_vector = layers.Multiply()([lstm_branch, attention_weights])
    context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
    
    # Additional LSTM layer
    lstm_branch2 = layers.Bidirectional(layers.LSTM(
        lstm_units//2, 
        return_sequences=False,
        recurrent_regularizer=keras.regularizers.l2(regularization)
    ))(lstm_branch)
    lstm_branch2 = layers.BatchNormalization()(lstm_branch2)
    
    # Flatten CNN branch
    cnn_branch = layers.Flatten()(cnn_branch)
    
    # Combine all branches
    combined = layers.concatenate([cnn_branch, lstm_branch2, context_vector])
    combined = layers.Dropout(dropout_rate)(combined)
    
    # Dense layers with residual connections
    dense1 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization))(combined)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(dropout_rate)(dense1)
    
    dense2 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization))(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(dropout_rate)(dense2)
    
    # Output layer
    outputs = layers.Dense(output_dim, activation='linear')(dense2)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Huber loss (more robust to outliers than MSE)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=keras.losses.Huber(),
        metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model


def calculate_trading_metrics(y_true, y_pred):
    """
    Calculate trading-specific metrics beyond standard ML metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of trading metrics
    """
    # Convert predictions to trading signals (direction only)
    y_true_dir = np.sign(y_true)
    y_pred_dir = np.sign(y_pred)
    
    # Calculate directional accuracy
    directional_accuracy = np.mean(y_true_dir == y_pred_dir)
    
    # Calculate win rate (correct positive predictions)
    positive_signals = (y_pred_dir > 0)
    if np.sum(positive_signals) > 0:
        win_rate = np.mean(y_true_dir[positive_signals] > 0)
    else:
        win_rate = 0.0
    
    # Calculate profit factor-like metric
    y_true_pos = y_true * (y_pred_dir > 0)  # True returns when prediction is positive
    y_true_neg = y_true * (y_pred_dir < 0) * -1  # True returns when prediction is negative (inverted)
    
    gross_profit = np.sum(y_true_pos[y_true_pos > 0])
    gross_loss = np.sum(y_true_pos[y_true_pos < 0])
    
    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0.0
    
    # Calculate Sharpe ratio-like metric (assuming risk-free rate of 0)
    returns = y_true * y_pred_dir  # Returns based on predicted direction
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumsum(returns)
    max_return = np.maximum.accumulate(cumulative_returns)
    drawdown = max_return - cumulative_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate precision/recall for up moves (class 1)
    precision = precision_score(y_true_dir > 0, y_pred_dir > 0, zero_division=0)
    recall = recall_score(y_true_dir > 0, y_pred_dir > 0, zero_division=0)
    f1 = f1_score(y_true_dir > 0, y_pred_dir > 0, zero_division=0)
    
    return {
        'directional_accuracy': directional_accuracy,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def train_small_timeframe_model(
    pair: str,
    timeframe: str,
    force: bool = False,
    days_to_use: int = 14
) -> Optional[Dict[str, Any]]:
    """
    Train a model specifically optimized for small timeframes
    
    Args:
        pair: Trading pair (e.g., BTC/USD)
        timeframe: Timeframe (e.g., 1m, 5m)
        force: Whether to force retraining
        days_to_use: Number of days of data to use
        
    Returns:
        Dictionary with model info
    """
    pair_symbol = pair.replace('/', '_')
    model_name = f"{pair_symbol}_{timeframe}"
    model_path = f"ml_models/{model_name}.h5"
    info_path = f"ml_models/{model_name}_info.json"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(info_path) and not force:
        logging.info(f"Model already exists for {pair} ({timeframe}). Use --force to retrain.")
        return None
    
    logging.info(f"Training small timeframe model for {pair} ({timeframe})...")
    
    # Load and prepare data
    df = load_small_timeframe_data(pair, timeframe)
    if df is None:
        logging.error(f"Failed to load data for {pair} ({timeframe})")
        return None
    
    # Limit to the specified number of days if requested
    if days_to_use > 0:
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_use)
        df = df[df.index >= cutoff_date]
        logging.info(f"Using {len(df)} rows of data from the last {days_to_use} days")
    
    # Add features
    df = add_features(df)
    if df is None or len(df) == 0:
        logging.error(f"Failed to add features for {pair} ({timeframe})")
        return None
    
    # Prepare sequence data with augmentation
    data, scalers = prepare_sequence_data_with_augmentation(
        df, FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, 
        TEST_SIZE, VALIDATION_SIZE, NOISE_FACTOR
    )
    if data is None:
        logging.error(f"Failed to prepare sequence data for {pair} ({timeframe})")
        return None
    
    # Report dataset size
    logging.info(f"Training data shape: {data['X_train'].shape}")
    logging.info(f"Validation data shape: {data['X_val'].shape}")
    logging.info(f"Test data shape: {data['X_test'].shape}")
    
    # Create and train model
    input_shape = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    model = create_enhanced_small_timeframe_model(input_shape, len(TARGET_COLUMNS))
    
    # Early stopping and reduce learning rate on plateau
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
    ]
    
    # Train model
    start_time = time.time()
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
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
    
    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(y_true, y_pred)
    
    # Standard ML metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Save model and info
    os.makedirs('ml_models', exist_ok=True)
    model.save(model_path)
    
    model_info = {
        'pair': pair,
        'timeframe': timeframe,
        'model_type': 'enhanced_small_timeframe',
        'feature_columns': FEATURE_COLUMNS,
        'target_columns': TARGET_COLUMNS,
        'sequence_length': SEQUENCE_LENGTH,
        'training_time': training_time,
        'epochs': len(history.history['loss']),
        'early_stopping_epoch': np.argmin(history.history['val_loss']) + 1,
        'metrics': {
            # Standard ML metrics
            'train_loss': float(train_metrics[0]),
            'train_mae': float(train_metrics[1]),
            'val_loss': float(val_metrics[0]),
            'val_mae': float(val_metrics[1]),
            'test_loss': float(test_metrics[0]),
            'test_mae': float(test_metrics[1]),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            
            # Trading-specific metrics
            'directional_accuracy': float(trading_metrics['directional_accuracy']),
            'win_rate': float(trading_metrics['win_rate']),
            'profit_factor': float(trading_metrics['profit_factor']),
            'sharpe_ratio': float(trading_metrics['sharpe_ratio']),
            'max_drawdown': float(trading_metrics['max_drawdown']),
            'precision': float(trading_metrics['precision']),
            'recall': float(trading_metrics['recall']),
            'f1_score': float(trading_metrics['f1_score'])
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logging.info(f"Saved model and info for {pair} ({timeframe})")
    logging.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    logging.info(f"Trading Metrics - Directional Accuracy: {trading_metrics['directional_accuracy']:.4f}, "
                f"Win Rate: {trading_metrics['win_rate']:.4f}, "
                f"Profit Factor: {trading_metrics['profit_factor']:.4f}, "
                f"F1 Score: {trading_metrics['f1_score']:.4f}")
    
    # Log early stopping point
    logging.info(f"Training stopped at epoch {model_info['early_stopping_epoch']} out of {len(history.history['loss'])}")
    
    return model_info


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create model directory if it doesn't exist
    os.makedirs('ml_models', exist_ok=True)
    
    # First, check if we have the 1m and 5m data
    for timeframe in TIMEFRAMES:
        if args.all:
            # Check if we have data for all pairs
            missing_data = False
            for pair in SUPPORTED_PAIRS:
                pair_symbol = pair.replace('/', '_')
                file_path = f'historical_data/{pair_symbol}_{timeframe}.csv'
                if not os.path.exists(file_path):
                    logging.warning(f"Missing data for {pair} ({timeframe})")
                    missing_data = True
            
            if missing_data:
                logging.info("Fetching missing data...")
                result = os.system(f"python fetch_kraken_small_timeframes.py --days {args.days}")
                if result != 0:
                    logging.error("Failed to fetch historical data")
                    return False
        else:
            # Check if we have data for the selected pair
            pair_symbol = args.pair.replace('/', '_')
            file_path = f'historical_data/{pair_symbol}_{timeframe}.csv'
            if not os.path.exists(file_path):
                logging.warning(f"Missing data for {args.pair} ({timeframe})")
                logging.info("Fetching missing data...")
                result = os.system(f"python fetch_kraken_small_timeframes.py --days {args.days} --pairs {args.pair}")
                if result != 0:
                    logging.error("Failed to fetch historical data")
                    return False
    
    # Determine which pairs to process
    if args.all:
        pairs_to_process = SUPPORTED_PAIRS.copy()
    else:
        if args.pair in SUPPORTED_PAIRS:
            pairs_to_process = [args.pair]
        else:
            logging.error(f"Unsupported pair: {args.pair}")
            return False
    
    # Train models for each pair and timeframe
    results = {}
    for pair in pairs_to_process:
        pair_results = {}
        for timeframe in TIMEFRAMES:
            model_info = train_small_timeframe_model(pair, timeframe, args.force, args.days)
            pair_results[timeframe] = model_info is not None
        
        results[pair] = pair_results
        
        # Log a blank line for better readability
        logging.info("")
    
    # Print summary
    logging.info("=== Training Summary ===")
    all_success = True
    for pair, pair_results in results.items():
        for timeframe, success in pair_results.items():
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"{pair} ({timeframe}): {status}")
            if not success:
                all_success = False
    
    return all_success


if __name__ == "__main__":
    main()