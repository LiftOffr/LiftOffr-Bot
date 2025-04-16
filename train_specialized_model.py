#!/usr/bin/env python3
"""
Train Specialized Models for a Specific Trading Pair

This script trains specialized machine learning models optimized for different
aspects of trading (entry timing, exit timing, position sizing, and trade cancellation)
for a specific cryptocurrency pair and timeframe.

Usage:
    python train_specialized_model.py --pair SOL/USD --timeframe 1h [--model-type entry|exit|cancel|sizing|all] [--force]

Example:
    python train_specialized_model.py --pair BTC/USD --timeframe 1h
    python train_specialized_model.py --pair ETH/USD --timeframe 15m --model-type entry
    python train_specialized_model.py --pair SOL/USD --all --force
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, Bidirectional, Conv1D, Dense, Dropout,
                           Flatten, Input, LayerNormalization, MaxPooling1D,
                           concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

# Constants
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MODEL_TYPES = ['entry', 'exit', 'cancel', 'sizing', 'all']
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train specialized models for a trading pair')
    parser.add_argument('--pair', type=str, required=True, choices=SUPPORTED_PAIRS,
                       help='Trading pair (e.g., SOL/USD)')
    parser.add_argument('--timeframe', type=str, choices=TIMEFRAMES,
                       help='Timeframe (e.g., 15m, 1h, 4h, 1d)')
    parser.add_argument('--model-type', type=str, choices=MODEL_TYPES, default='all',
                       help='Type of model to train')
    parser.add_argument('--force', action='store_true',
                       help='Force retraining of existing models')
    parser.add_argument('--all', action='store_true',
                       help='Train models for all timeframes')
    return parser.parse_args()


def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical data for a trading pair and timeframe
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        
    Returns:
        DataFrame with historical data
    """
    pair_file = pair.replace('/', '_')
    file_path = f"historical_data/{pair_file}_{timeframe}.csv"
    
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added features
    """
    if df is None or len(df) < 50:
        logging.error("Insufficient data for feature calculation")
        return None
    
    try:
        # Ensure column names are standardized
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logging.error(f"Missing required column: {col}")
                return None
        
        # Convert timestamp if needed
        if df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # Bollinger Bands
            if window in [20]:
                df[f'bb_{window}_upper'] = df[f'ma_{window}'] + 2 * df['close'].rolling(window=window).std()
                df[f'bb_{window}_lower'] = df[f'ma_{window}'] - 2 * df['close'].rolling(window=window).std()
                df[f'bb_{window}_width'] = (df[f'bb_{window}_upper'] - df[f'bb_{window}_lower']) / df[f'ma_{window}']
                df[f'bb_{window}_position'] = (df['close'] - df[f'bb_{window}_lower']) / (df[f'bb_{window}_upper'] - df[f'bb_{window}_lower'])
        
        # Exponential moving averages
        for window in [5, 10, 20, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
            
        # Moving average convergence/divergence (MACD)
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Average True Range (ATR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        df['atr_14_ratio'] = df['atr_14'] / df['close']
        
        # Target variables for different model types
        # Entry signal (1 if price increases by 2%+ in next 24 candles)
        future_returns = df['close'].pct_change(periods=24).shift(-24)
        df['target_entry'] = (future_returns > 0.02).astype(int)
        
        # Exit signal (1 if price decreases by 1%+ in next 12 candles)
        future_drawdown = df['close'].rolling(12).min().shift(-12) / df['close'] - 1
        df['target_exit'] = (future_drawdown < -0.01).astype(int)
        
        # Cancel signal (1 if price moves against entry by 1%+ without first moving 2%+ in favor)
        # Simplified version
        df['target_cancel'] = ((future_returns < 0) & (future_returns > -0.01)).astype(int)
        
        # Position sizing (normalized future returns, clipped)
        df['target_sizing'] = np.clip(future_returns, -0.1, 0.1) / 0.1
        
        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        logging.info(f"Added features, final dataframe shape: {df.shape}")
        return df
    
    except Exception as e:
        logging.error(f"Error adding features: {e}")
        return None


def prepare_specialized_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    model_type: str,
    sequence_length: int,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    noise_factor: float = 0.01
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Prepare specialized data for different model types
    
    Args:
        df: DataFrame with features and targets
        feature_columns: List of feature column names
        model_type: Type of model to prepare data for ('entry', 'exit', 'cancel', 'sizing')
        sequence_length: Length of input sequences
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        noise_factor: Factor for adding noise to training data
        
    Returns:
        Tuple of (data dict, metadata dict)
    """
    try:
        # Select target column based on model type
        target_column = f'target_{model_type}'
        if target_column not in df.columns:
            logging.error(f"Target column {target_column} not found in dataframe")
            return None
        
        # Extract features and target
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(df) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length-1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, shuffle=False
        )
        
        # Normalize features
        feature_scaler = MinMaxScaler()
        n_samples_train, n_steps, n_features = X_train.shape
        
        # Reshape to 2D for scaling
        X_train_reshaped = X_train.reshape(n_samples_train * n_steps, n_features)
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
        
        # Reshape back to 3D
        X_train = X_train_scaled.reshape(n_samples_train, n_steps, n_features)
        
        # Apply the same scaling to validation and test sets
        X_val = feature_scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test = feature_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        # Add noise to training data for regularization
        if noise_factor > 0:
            noise = np.random.normal(0, noise_factor, X_train.shape)
            X_train = X_train + noise
            X_train = np.clip(X_train, 0, 1)  # Keep values in [0,1]
        
        # For regression models, scale the target
        if model_type == 'sizing':
            target_scaler = MinMaxScaler(feature_range=(-1, 1))
            y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
            scaler_metadata = {
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler
            }
        else:
            scaler_metadata = {
                'feature_scaler': feature_scaler
            }
        
        # Calculate class weights for classification problems
        if model_type != 'sizing':
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            total = neg_count + pos_count
            
            # Handle edge case where all examples are of one class
            if neg_count == 0 or pos_count == 0:
                class_weight = {0: 1.0, 1: 1.0}
            else:
                weight_for_0 = (1 / neg_count) * (total / 2.0)
                weight_for_1 = (1 / pos_count) * (total / 2.0)
                class_weight = {0: weight_for_0, 1: weight_for_1}
            
            # Adjust weights based on model type
            if model_type == 'entry':
                # For entry models, prioritize precision (reduce false positives)
                class_weight[1] *= 1.2
            elif model_type == 'exit':
                # For exit models, prioritize recall (reduce false negatives)
                class_weight[1] *= 1.5
            
            class_weights = class_weight
            positive_ratio = pos_count / total
            
            metadata = {
                'feature_columns': feature_columns,
                'sequence_length': sequence_length,
                'class_weights': class_weights,
                'positive_ratio': positive_ratio,
                'scalers': scaler_metadata,
                'is_classification': True
            }
        else:
            metadata = {
                'feature_columns': feature_columns,
                'sequence_length': sequence_length,
                'scalers': scaler_metadata,
                'is_classification': False
            }
        
        # Package data
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        logging.info(f"Prepared data for {model_type} model")
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Validation set shape: {X_val.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        if model_type != 'sizing':
            logging.info(f"Class distribution in training: {positive_ratio:.2%} positive")
        
        return data, metadata
    
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None, None


def create_specialized_model(
    model_type: str,
    input_shape: Tuple[int, int],
    is_classification: bool = True,
    dropout_rate: float = 0.3,
    regularization: float = 0.001,
    learning_rate: float = 0.001,
    loss_weight_up: float = 1.0,
    loss_weight_down: float = 1.0
) -> tf.keras.Model:
    """
    Create a specialized model for a specific purpose
    
    Args:
        model_type: Type of model ('entry', 'exit', 'cancel', 'sizing')
        input_shape: Input shape (sequence_length, features)
        is_classification: Whether this is a classification model
        dropout_rate: Dropout rate for regularization
        regularization: L2 regularization strength
        learning_rate: Learning rate for optimizer
        loss_weight_up: Weight for upward move class (or positive errors in regression)
        loss_weight_down: Weight for downward move class (or negative errors in regression)
        
    Returns:
        Specialized TensorFlow model
    """
    try:
        # Define inputs
        input_layer = Input(shape=input_shape)
        
        # CNN branch
        conv1 = Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(regularization)
        )(input_layer)
        conv1 = LayerNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        
        conv2 = Conv1D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(regularization)
        )(conv1)
        conv2 = LayerNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        
        conv2 = Flatten()(conv2)
        
        # LSTM branch
        lstm1 = Bidirectional(LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=l2(regularization),
            recurrent_regularizer=l2(regularization/10)
        ))(input_layer)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(dropout_rate)(lstm1)
        
        lstm2 = Bidirectional(LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=l2(regularization),
            recurrent_regularizer=l2(regularization/10)
        ))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(dropout_rate)(lstm2)
        
        # Combine branches
        combined = concatenate([conv2, lstm2])
        
        # Dense layers
        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(regularization))(combined)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(64, activation='relu', kernel_regularizer=l2(regularization))(dense1)
        dense2 = LayerNormalization()(dense2)
        dense2 = Dropout(dropout_rate/2)(dense2)
        
        # Output layer
        if is_classification:
            output_layer = Dense(1, activation='sigmoid', name='output')(dense2)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output_layer = Dense(1, activation='tanh', name='output')(dense2)  # tanh for normalized regression
            
            # Custom loss function that penalizes underestimation more for position sizing
            def asymmetric_mse(y_true, y_pred):
                error = y_true - y_pred
                positive_error = tf.maximum(0., error)
                negative_error = tf.maximum(0., -error)
                return tf.reduce_mean(loss_weight_up * tf.square(positive_error) + 
                                     loss_weight_down * tf.square(negative_error))
            
            loss = asymmetric_mse
            metrics = ['mae', 'mse']
        
        # Compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Adjust learning rate based on model type
        if model_type == 'entry':
            # Entry models need to be more precise, so slower learning
            adjusted_lr = learning_rate * 0.8
        elif model_type == 'exit':
            # Exit models need to be more responsive, so faster learning
            adjusted_lr = learning_rate * 1.2
        else:
            adjusted_lr = learning_rate
        
        optimizer = Adam(learning_rate=adjusted_lr)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logging.info(f"Created specialized {model_type} model")
        model.summary(print_fn=logging.info)
        
        return model
    
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return None


def train_specialized_model(
    pair: str,
    timeframe: str,
    model_type: str,
    force: bool = False
) -> Optional[Tuple[tf.keras.Model, Dict]]:
    """
    Train a specialized model for a specific purpose
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        model_type: Type of model to train ('entry', 'exit', 'cancel', 'sizing')
        force: Whether to force retraining
        
    Returns:
        Tuple of (trained model, model info)
    """
    try:
        # Create directories if they don't exist
        os.makedirs('ml_models', exist_ok=True)
        
        # Format filenames
        pair_formatted = pair.replace('/', '_')
        model_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_model.h5"
        info_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_info.json"
        
        # Check if model already exists
        if os.path.exists(model_path) and os.path.exists(info_path) and not force:
            logging.info(f"Model for {pair} ({timeframe}, {model_type}) already exists. Use --force to retrain.")
            
            # Load existing model info
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            # Load model
            try:
                model = load_model(model_path, compile=False)
                logging.info(f"Loaded existing model from {model_path}")
                return model, model_info
            except Exception as e:
                logging.error(f"Error loading existing model: {e}")
                logging.info("Will train a new model")
        
        # Load and prepare data
        df = load_data(pair, timeframe)
        if df is None:
            return None, None
        
        df = add_features(df)
        if df is None or len(df) < 100:
            logging.error(f"Insufficient data after feature engineering")
            return None, None
        
        # Select features
        price_features = ['open', 'high', 'low', 'close', 'volume',
                          'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                          'volume_change', 'volume_ma_ratio']
        
        ma_features = [f'ma_{window}' for window in [5, 10, 20, 50]] + \
                      [f'ma_{window}_ratio' for window in [5, 10, 20, 50]]
        
        bb_features = [f'bb_20_upper', f'bb_20_lower', f'bb_20_width', f'bb_20_position']
        
        ema_features = [f'ema_{window}' for window in [5, 10, 20, 50]] + \
                       [f'ema_{window}_ratio' for window in [5, 10, 20, 50]]
        
        momentum_features = ['macd', 'macd_signal', 'macd_hist', 'rsi_14', 
                            'atr_14', 'atr_14_ratio']
        
        # Use different feature sets based on model type
        if model_type in ['entry', 'exit']:
            # Full feature set for entry/exit decisions
            feature_columns = price_features + ma_features + bb_features + ema_features + momentum_features
        elif model_type == 'cancel':
            # Focus on volatility and momentum for cancel decisions
            feature_columns = price_features + bb_features + ['rsi_14', 'atr_14', 'atr_14_ratio']
        elif model_type == 'sizing':
            # Focus on trend strength for position sizing
            feature_columns = price_features + ma_features + ema_features + momentum_features
        else:
            logging.error(f"Unknown model type: {model_type}")
            return None, None
        
        # Determine sequence length based on timeframe
        if timeframe == '1m':
            sequence_length = 60  # 1 hour
        elif timeframe == '5m':
            sequence_length = 48  # 4 hours
        elif timeframe == '15m':
            sequence_length = 32  # 8 hours
        elif timeframe == '1h':
            sequence_length = 24  # 1 day
        elif timeframe == '4h':
            sequence_length = 18  # 3 days
        elif timeframe == '1d':
            sequence_length = 14  # 2 weeks
        else:
            sequence_length = 20  # default
        
        # Prepare data
        is_classification = model_type != 'sizing'
        data, metadata = prepare_specialized_data(
            df, 
            feature_columns, 
            model_type, 
            sequence_length,
            test_size=0.2,
            validation_size=0.2,
            noise_factor=0.01
        )
        
        if data is None or metadata is None:
            return None, None
        
        # Adjust hyperparameters based on model type
        if model_type == 'entry':
            dropout_rate = 0.35  # More regularization to prevent false positives
            learning_rate = 0.0008
            loss_weight_up = 1.0
            loss_weight_down = 1.0
        elif model_type == 'exit':
            dropout_rate = 0.25  # Less regularization to catch more exit signals
            learning_rate = 0.0012
            loss_weight_up = 1.0
            loss_weight_down = 1.0
        elif model_type == 'cancel':
            dropout_rate = 0.3
            learning_rate = 0.001
            loss_weight_up = 1.0
            loss_weight_down = 1.0
        elif model_type == 'sizing':
            dropout_rate = 0.3
            learning_rate = 0.001
            # For sizing, penalize underestimation more than overestimation
            loss_weight_up = 1.5  # Penalty for underestimating position size
            loss_weight_down = 1.0  # Penalty for overestimating position size
        
        # Create model
        model = create_specialized_model(
            model_type,
            input_shape=(data['X_train'].shape[1], data['X_train'].shape[2]),
            is_classification=is_classification,
            dropout_rate=dropout_rate,
            regularization=0.001,
            learning_rate=learning_rate,
            loss_weight_up=loss_weight_up,
            loss_weight_down=loss_weight_down
        )
        
        if model is None:
            return None, None
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        # Set class weights for classification models
        if is_classification:
            class_weight = metadata['class_weights']
        else:
            class_weight = None
        
        # Train model
        batch_size = 32
        if timeframe in ['1m', '5m']:
            # Use smaller batch size for higher frequency data
            batch_size = 64
        
        epochs = 100  # Will use early stopping
        
        history = model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate model
        evaluation = model.evaluate(data['X_test'], data['y_test'], verbose=1)
        metrics = {}
        
        if is_classification:
            # For classification models
            metrics['test_loss'] = float(evaluation[0])
            metrics['test_accuracy'] = float(evaluation[1])
            
            # Calculate precision, recall, F1
            y_pred_prob = model.predict(data['X_test'])
            y_pred = (y_pred_prob > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                data['y_test'], y_pred, average='binary'
            )
            
            # Calculate AUC
            try:
                auc = roc_auc_score(data['y_test'], y_pred_prob)
                metrics['auc'] = float(auc)
            except:
                metrics['auc'] = 0.5  # Default if calculation fails
            
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1_score'] = float(f1)
            
            # Calculate profit factor (simplified)
            true_positives = ((y_pred == 1) & (data['y_test'] == 1)).sum()
            false_positives = ((y_pred == 1) & (data['y_test'] == 0)).sum()
            
            # Assume fixed profit and loss per trade
            avg_profit = 0.02  # 2% profit target
            avg_loss = 0.01   # 1% stop loss
            
            total_profit = true_positives * avg_profit
            total_loss = false_positives * avg_loss
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            metrics['profit_factor'] = float(min(profit_factor, 10.0))  # Cap at 10
            
            # Calculate win rate
            predicted_trades = (y_pred == 1).sum()
            winning_trades = true_positives
            win_rate = winning_trades / predicted_trades if predicted_trades > 0 else 0
            metrics['win_rate'] = float(win_rate)
            
            # Calculate Sharpe ratio (simplified)
            if predicted_trades > 0:
                returns = [avg_profit if hit else -avg_loss for hit in (y_pred == 1) & (data['y_test'] == 1)]
                mean_return = sum(returns) / len(returns)
                std_return = np.std(returns) if len(returns) > 1 else 1.0
                sharpe = mean_return / std_return if std_return > 0 else 0
                metrics['sharpe_ratio'] = float(sharpe)
            else:
                metrics['sharpe_ratio'] = 0.0
            
        else:
            # For regression models
            metrics['test_loss'] = float(evaluation[0])
            metrics['test_mae'] = float(evaluation[1])
            metrics['test_mse'] = float(evaluation[2])
            
            # Calculate R-squared
            y_pred = model.predict(data['X_test']).flatten()
            ss_res = np.sum((data['y_test'] - y_pred) ** 2)
            ss_tot = np.sum((data['y_test'] - np.mean(data['y_test'])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metrics['r_squared'] = float(r2)
        
        # Save model info
        train_history = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        model_info = {
            'pair': pair,
            'timeframe': timeframe,
            'model_type': model_type,
            'feature_columns': feature_columns,
            'sequence_length': sequence_length,
            'is_classification': is_classification,
            'metrics': metrics,
            'training_history': train_history,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save model info
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logging.info(f"Trained and saved {model_type} model for {pair} ({timeframe})")
        logging.info(f"Metrics: {metrics}")
        
        # Load best model from checkpoint
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
        
        return model, model_info
    
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None, None


def create_ensemble_model(
    pair: str,
    timeframe: str,
    force: bool = False
) -> bool:
    """
    Create ensemble model from specialized models
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        force: Whether to force recreation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Format filenames
        pair_formatted = pair.replace('/', '_')
        ensemble_info_path = f"ml_models/{pair_formatted}_{timeframe}_ensemble_info.json"
        
        # Check if ensemble info already exists
        if os.path.exists(ensemble_info_path) and not force:
            logging.info(f"Ensemble info for {pair} ({timeframe}) already exists. Use --force to recreate.")
            return True
        
        # Check if all required models exist
        model_types = ['entry', 'exit', 'cancel', 'sizing']
        models_info = {}
        
        for model_type in model_types:
            info_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_info.json"
            if not os.path.exists(info_path):
                logging.error(f"Missing model info for {model_type}. Please train all models first.")
                return False
            
            with open(info_path, 'r') as f:
                models_info[model_type] = json.load(f)
        
        # Create ensemble info
        ensemble_info = {
            'pair': pair,
            'timeframe': timeframe,
            'component_models': list(models_info.keys()),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {}
        }
        
        # Combine metrics from component models
        for model_type, info in models_info.items():
            if 'metrics' in info:
                for metric, value in info['metrics'].items():
                    ensemble_key = f"{model_type}_{metric}"
                    ensemble_info['metrics'][ensemble_key] = value
        
        # Calculate ensemble metrics
        # For simplicity, use entry model's main metrics
        if 'entry' in models_info and 'metrics' in models_info['entry']:
            entry_metrics = models_info['entry']['metrics']
            for key in ['win_rate', 'profit_factor', 'sharpe_ratio']:
                if key in entry_metrics:
                    ensemble_info['metrics'][key] = entry_metrics[key]
        
        # Save ensemble info
        with open(ensemble_info_path, 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        logging.info(f"Created ensemble model info for {pair} ({timeframe})")
        return True
    
    except Exception as e:
        logging.error(f"Error creating ensemble model: {e}")
        return False


def train_for_pair_and_timeframe(
    pair: str,
    timeframe: str,
    model_type: str = 'all',
    force: bool = False
) -> Dict[str, bool]:
    """
    Train all model types for a specific pair and timeframe
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 15m, 1h, 4h, 1d)
        model_type: Specific model type to train (or 'all')
        force: Whether to force retraining
        
    Returns:
        Dictionary of training results for each model type
    """
    logging.info(f"Training models for {pair} ({timeframe})")
    
    results = {}
    
    # Determine which models to train
    model_types_to_train = ['entry', 'exit', 'cancel', 'sizing'] if model_type == 'all' else [model_type]
    
    # Train each model type
    for model_type in model_types_to_train:
        logging.info(f"Training {model_type} model for {pair} ({timeframe})")
        model, info = train_specialized_model(pair, timeframe, model_type, force)
        results[model_type] = model is not None and info is not None
    
    # Create ensemble if all models were trained successfully
    if model_type == 'all' and all(results.values()):
        results['ensemble'] = create_ensemble_model(pair, timeframe, force)
    
    return results


def train_for_pair(
    pair: str,
    timeframe: str = None,
    model_type: str = 'all',
    force: bool = False,
    all_timeframes: bool = False
) -> Dict[str, Dict[str, bool]]:
    """
    Train models for all timeframes for a specific pair
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Specific timeframe to train (or None for all)
        model_type: Specific model type to train (or 'all')
        force: Whether to force retraining
        all_timeframes: Whether to train all timeframes
        
    Returns:
        Dictionary of training results for each timeframe
    """
    logging.info(f"Training models for {pair}")
    
    # Ensure historical data exists
    pair_formatted = pair.replace('/', '_')
    
    # Determine which timeframes to train
    timeframes_to_train = TIMEFRAMES if (timeframe is None or all_timeframes) else [timeframe]
    
    # Check if data files exist
    for tf in timeframes_to_train:
        file_path = f"historical_data/{pair_formatted}_{tf}.csv"
        if not os.path.exists(file_path):
            logging.error(f"Missing data file for {pair} ({tf}): {file_path}")
            timeframes_to_train.remove(tf)
    
    if not timeframes_to_train:
        logging.error(f"No data files found for {pair}. Please fetch data first.")
        return {}
    
    # Train models for each timeframe
    results = {}
    for tf in timeframes_to_train:
        results[tf] = train_for_pair_and_timeframe(pair, tf, model_type, force)
    
    return results


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create required directories
    os.makedirs('ml_models', exist_ok=True)
    os.makedirs('historical_data', exist_ok=True)
    
    # Train models
    if args.all:
        if args.timeframe:
            logging.warning("Both --all and --timeframe specified. Using --all (all timeframes).")
        results = train_for_pair(args.pair, model_type=args.model_type, force=args.force, all_timeframes=True)
    else:
        if args.timeframe:
            results = train_for_pair(args.pair, args.timeframe, args.model_type, args.force)
        else:
            logging.error("Either --timeframe or --all must be specified")
            return False
    
    # Print summary
    logging.info("=== Training Summary ===")
    for timeframe, timeframe_results in results.items():
        success_count = sum(1 for result in timeframe_results.values() if result)
        total_count = len(timeframe_results)
        logging.info(f"{args.pair} ({timeframe}): {success_count}/{total_count} models trained successfully")
    
    return True


if __name__ == "__main__":
    main()