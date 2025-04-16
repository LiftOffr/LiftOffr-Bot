#!/usr/bin/env python3
"""
Create Ensemble Model with Attention Mechanisms

This script creates an advanced ensemble model that integrates:
1. LSTM with Self-Attention
2. GRU with Temporal Attention
3. TCN (Temporal Convolutional Network)
4. Transformer-style Multi-Head Attention
5. Ensemble meta-learner to combine predictions

Based on the training roadmap, this creates a powerful ensemble model
that can capture various market patterns and dynamics for improved
trading predictions.

Usage:
    python create_ensemble_with_attention.py --pair BTC/USD --timeframe 1h
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Concatenate,
    GlobalAveragePooling1D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Import custom attention mechanisms
from attention_mechanism import (
    SelfAttention, ScaledDotProductAttention, MultiHeadAttention,
    TemporalAttention, FeatureAttention, build_hybrid_attention_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Create required directories
for directory in [DATA_DIR, HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create ensemble model with attention mechanisms")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Validation size from training data")
    parser.add_argument("--base_leverage", type=float, default=5.0,
                        help="Base leverage for trading")
    parser.add_argument("--max_leverage", type=float, default=75.0,
                        help="Maximum leverage for high-confidence trades")
    return parser.parse_args()

def load_historical_data(pair, timeframe):
    """Load historical data for a trading pair"""
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
        # Create dummy data for testing purposes
        logger.info("Creating dummy data for testing purposes")
        
        # Create a date range for the last 1000 hours
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=1000, freq=timeframe)
        
        # Generate random price data with a trend
        base_price = 10000 if "BTC" in pair else 1000
        trend = np.linspace(0, 0.2, 1000)  # Small upward trend
        noise = np.random.normal(0, 0.01, 1000)  # Small random noise
        prices = base_price * (1 + trend + noise)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.normal(0, 0.005, 1000)),
            'low': prices * (1 - np.random.normal(0, 0.005, 1000)),
            'close': prices * (1 + np.random.normal(0, 0.001, 1000)),
            'volume': np.random.normal(1000, 100, 1000)
        })
        
        return df

def calculate_indicators(df):
    """Calculate technical indicators for the dataset"""
    # Make a copy to avoid modifying the original dataframe
    df_indicators = df.copy()
    
    # Ensure df has expected columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col.lower() in map(str.lower, df_indicators.columns) for col in required_columns):
        # Try lowercase column names
        df_indicators.columns = [col.lower() for col in df_indicators.columns]
    
    # If still missing required columns, try to adapt
    if 'open' not in df_indicators.columns and 'price' in df_indicators.columns:
        df_indicators['open'] = df_indicators['price']
        df_indicators['high'] = df_indicators['price']
        df_indicators['low'] = df_indicators['price']
        df_indicators['close'] = df_indicators['price']
    
    if 'volume' not in df_indicators.columns:
        df_indicators['volume'] = 0  # Default to zero if no volume data
    
    # Sort by time/timestamp if available
    if 'time' in df_indicators.columns:
        df_indicators.sort_values('time', inplace=True)
    elif 'timestamp' in df_indicators.columns:
        df_indicators.sort_values('timestamp', inplace=True)
    
    # Calculate technical indicators
    # 1. Moving Averages
    df_indicators['sma_5'] = df_indicators['close'].rolling(window=5).mean()
    df_indicators['sma_10'] = df_indicators['close'].rolling(window=10).mean()
    df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
    df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
    df_indicators['sma_100'] = df_indicators['close'].rolling(window=100).mean()
    df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()
    
    df_indicators['ema_5'] = df_indicators['close'].ewm(span=5, adjust=False).mean()
    df_indicators['ema_10'] = df_indicators['close'].ewm(span=10, adjust=False).mean()
    df_indicators['ema_20'] = df_indicators['close'].ewm(span=20, adjust=False).mean()
    df_indicators['ema_50'] = df_indicators['close'].ewm(span=50, adjust=False).mean()
    df_indicators['ema_100'] = df_indicators['close'].ewm(span=100, adjust=False).mean()
    df_indicators['ema_200'] = df_indicators['close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df_indicators['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate RSI for different periods
    for period in [6, 14, 20, 50]:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        df_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
    df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
    df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']
    
    # 4. Bollinger Bands
    for period in [10, 20, 50]:
        df_indicators[f'bb_middle_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'bb_std_{period}'] = df_indicators['close'].rolling(window=period).std()
        df_indicators[f'bb_upper_{period}'] = df_indicators[f'bb_middle_{period}'] + (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_lower_{period}'] = df_indicators[f'bb_middle_{period}'] - (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_width_{period}'] = (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}']) / df_indicators[f'bb_middle_{period}']
    
    # 5. ATR (Average True Range)
    tr1 = df_indicators['high'] - df_indicators['low']
    tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
    tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    for period in [7, 14, 21]:
        df_indicators[f'atr_{period}'] = tr.rolling(window=period).mean()
    
    # 6. Volume-based indicators
    df_indicators['volume_ma_10'] = df_indicators['volume'].rolling(window=10).mean()
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ma_50'] = df_indicators['volume'].rolling(window=50).mean()
    df_indicators['volume_ratio_10'] = df_indicators['volume'] / df_indicators['volume_ma_10']
    df_indicators['volume_ratio_20'] = df_indicators['volume'] / df_indicators['volume_ma_20']
    
    # 7. Price Momentum
    for period in [5, 10, 20, 50]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
    
    # 8. Volatility
    for period in [10, 20, 50]:
        df_indicators[f'volatility_{period}'] = df_indicators['close'].rolling(window=period).std() / df_indicators['close'].rolling(window=period).mean() * 100
    
    # 9. Moving Average Slopes
    for ma in ['sma', 'ema']:
        for period in [20, 50, 100]:
            df_indicators[f'{ma}_{period}_slope'] = df_indicators[f'{ma}_{period}'].diff(3) / 3
    
    # 10. Candlestick Features
    df_indicators['body_size'] = abs(df_indicators['close'] - df_indicators['open'])
    df_indicators['body_ratio'] = df_indicators['body_size'] / (df_indicators['high'] - df_indicators['low'])
    df_indicators['upper_shadow'] = df_indicators['high'] - df_indicators[['open', 'close']].max(axis=1)
    df_indicators['lower_shadow'] = df_indicators[['open', 'close']].min(axis=1) - df_indicators['low']
    df_indicators['is_bullish'] = (df_indicators['close'] > df_indicators['open']).astype(int)
    
    # 11. Price Distance from Moving Averages
    for ma in ['sma', 'ema']:
        for period in [20, 50, 100, 200]:
            df_indicators[f'{ma}_{period}_distance'] = (df_indicators['close'] - df_indicators[f'{ma}_{period}']) / df_indicators[f'{ma}_{period}'] * 100
    
    # 12. Moving Average Crossovers
    df_indicators['sma_5_10_cross'] = (df_indicators['sma_5'] > df_indicators['sma_10']).astype(int)
    df_indicators['sma_10_20_cross'] = (df_indicators['sma_10'] > df_indicators['sma_20']).astype(int)
    df_indicators['sma_20_50_cross'] = (df_indicators['sma_20'] > df_indicators['sma_50']).astype(int)
    df_indicators['sma_50_100_cross'] = (df_indicators['sma_50'] > df_indicators['sma_100']).astype(int)
    df_indicators['sma_100_200_cross'] = (df_indicators['sma_100'] > df_indicators['sma_200']).astype(int)
    
    df_indicators['ema_5_10_cross'] = (df_indicators['ema_5'] > df_indicators['ema_10']).astype(int)
    df_indicators['ema_10_20_cross'] = (df_indicators['ema_10'] > df_indicators['ema_20']).astype(int)
    df_indicators['ema_20_50_cross'] = (df_indicators['ema_20'] > df_indicators['ema_50']).astype(int)
    df_indicators['ema_50_100_cross'] = (df_indicators['ema_50'] > df_indicators['ema_100']).astype(int)
    df_indicators['ema_100_200_cross'] = (df_indicators['ema_100'] > df_indicators['ema_200']).astype(int)
    
    # Drop rows with NaN values
    df_indicators.dropna(inplace=True)
    
    return df_indicators

def create_target_variable(df, price_shift=1, threshold=0.005):
    """Create target variable for ML model training"""
    # Calculate future returns
    future_return = df['close'].shift(-price_shift) / df['close'] - 1
    
    # Create target labels: 1 (up), 0 (neutral), -1 (down)
    df['target'] = 0
    df.loc[future_return > threshold, 'target'] = 1
    df.loc[future_return < -threshold, 'target'] = -1
    
    # Drop NaN values (last rows where target couldn't be calculated)
    df.dropna(subset=['target'], inplace=True)
    
    return df

def prepare_data_for_training(df, sequence_length, test_size, validation_size):
    """Prepare data for training, validation, and testing"""
    # Select features and target
    price_features = ['open', 'high', 'low', 'close', 'volume']
    technical_features = [col for col in df.columns if col not in price_features + ['time', 'timestamp', 'target']]
    
    # Combine all features
    all_features = price_features + technical_features
    
    # Normalize features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[all_features])
    
    # Get targets
    targets = df['target'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # Add 1 to shift from [-1,0,1] to [0,1,2]
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, all_features

def create_base_models(input_shape, output_shape=3):
    """Create base models for ensemble"""
    models = {}
    
    # 1. LSTM with Self-Attention
    lstm_input = Input(shape=input_shape)
    lstm = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_input)
    lstm_attention, _ = SelfAttention()(lstm)
    lstm_dense = Dense(64, activation='relu')(lstm_attention)
    lstm_dropout = Dropout(0.3)(lstm_dense)
    lstm_output = Dense(output_shape, activation='softmax')(lstm_dropout)
    models['lstm_self_attention'] = Model(inputs=lstm_input, outputs=lstm_output)
    
    # 2. GRU with Temporal Attention
    gru_input = Input(shape=input_shape)
    gru = tf.keras.layers.GRU(128, return_sequences=True)(gru_input)
    gru_attention, _ = TemporalAttention()(gru)
    gru_pooling = GlobalAveragePooling1D()(gru_attention)
    gru_dense = Dense(64, activation='relu')(gru_pooling)
    gru_dropout = Dropout(0.3)(gru_dense)
    gru_output = Dense(output_shape, activation='softmax')(gru_dropout)
    models['gru_temporal_attention'] = Model(inputs=gru_input, outputs=gru_output)
    
    # 3. CNN
    cnn_input = Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    flatten = Flatten()(pool2)
    cnn_dense = Dense(64, activation='relu')(flatten)
    cnn_dropout = Dropout(0.3)(cnn_dense)
    cnn_output = Dense(output_shape, activation='softmax')(cnn_dropout)
    models['cnn'] = Model(inputs=cnn_input, outputs=cnn_output)
    
    # 4. Transformer with Multi-Head Attention
    transformer_input = Input(shape=input_shape)
    transformer = MultiHeadAttention()(transformer_input)
    transformer_pooling = GlobalAveragePooling1D()(transformer)
    transformer_dense = Dense(64, activation='relu')(transformer_pooling)
    transformer_dropout = Dropout(0.3)(transformer_dense)
    transformer_output = Dense(output_shape, activation='softmax')(transformer_dropout)
    models['transformer'] = Model(inputs=transformer_input, outputs=transformer_output)
    
    # Compile all models
    for name, model in models.items():
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return models

def train_base_models(models, X_train, y_train, X_val, y_val, epochs, batch_size, pair):
    """Train base models for ensemble"""
    trained_models = {}
    histories = {}
    
    # Clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        
        # Create callbacks
        checkpoint_path = f"{MODEL_WEIGHTS_DIR}/{name}_{pair_clean}_model.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        # Load best model
        trained_models[name] = load_model(
            checkpoint_path,
            custom_objects={
                'SelfAttention': SelfAttention,
                'TemporalAttention': TemporalAttention,
                'MultiHeadAttention': MultiHeadAttention
            }
        )
        
        histories[name] = history
    
    return trained_models, histories

def create_stacked_ensemble(base_models, input_shape, output_shape=3):
    """Create stacked ensemble model from base models"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Get predictions from base models
    predictions = []
    for name, model in base_models.items():
        # Create a clone of the model
        prediction_model = Model(inputs=model.input, outputs=model.output)
        # Freeze the model weights
        prediction_model.trainable = False
        # Get predictions
        predictions.append(prediction_model(inputs))
    
    # Concatenate predictions
    concatenated = Concatenate()(predictions)
    
    # Meta-learner (Dense layers)
    meta = Dense(128, activation='relu')(concatenated)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.5)(meta)
    meta = Dense(64, activation='relu')(meta)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.3)(meta)
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(meta)
    
    # Create model
    ensemble_model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble_model

def train_ensemble_model(ensemble_model, X_train, y_train, X_val, y_val, epochs, batch_size, pair):
    """Train ensemble model"""
    # Clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create callbacks
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/ensemble_{pair_clean}_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = ensemble_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Load best model
    trained_model = load_model(
        checkpoint_path,
        custom_objects={
            'SelfAttention': SelfAttention,
            'TemporalAttention': TemporalAttention,
            'MultiHeadAttention': MultiHeadAttention
        }
    )
    
    return trained_model, history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Convert back to original labels (-1, 0, 1)
    y_pred_labels = y_pred_classes - 1
    y_test_labels = y_test_classes - 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    # Calculate additional trading-specific metrics
    signal_counts = np.bincount(y_pred_classes, minlength=3)
    
    # Calculate win rate (excluding neutral predictions)
    non_neutral_mask = (y_pred_labels != 0) & (y_test_labels != 0)
    win_rate = np.mean(y_pred_labels[non_neutral_mask] == y_test_labels[non_neutral_mask]) if np.sum(non_neutral_mask) > 0 else 0
    
    # Print evaluation
    logger.info("\nModel Evaluation:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Win Rate: {win_rate:.4f}")
    logger.info("\nSignal Distribution:")
    logger.info(f"Bearish (-1): {signal_counts[0]} ({signal_counts[0]/len(y_pred_classes):.2%})")
    logger.info(f"Neutral (0): {signal_counts[1]} ({signal_counts[1]/len(y_pred_classes):.2%})")
    logger.info(f"Bullish (1): {signal_counts[2]} ({signal_counts[2]/len(y_pred_classes):.2%})")
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'win_rate': win_rate,
        'signal_distribution': {
            'bearish': signal_counts[0] / len(y_pred_classes),
            'neutral': signal_counts[1] / len(y_pred_classes),
            'bullish': signal_counts[2] / len(y_pred_classes)
        }
    }
    
    return metrics

def plot_training_history(history, pair, model_name='ensemble'):
    """Plot training history"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{pair} {model_name.capitalize()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{pair} {model_name.capitalize()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Save plot
    plt.tight_layout()
    plot_path = f"{RESULTS_DIR}/{model_name}_{pair_clean}_training_history.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Training history plot saved to {plot_path}")

def update_ml_config(pair, model_path, metrics, args):
    """Update ML configuration for the model"""
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
    config["global_settings"]["base_leverage"] = args.base_leverage
    config["global_settings"]["max_leverage"] = args.max_leverage
    config["global_settings"]["confidence_threshold"] = 0.65
    config["global_settings"]["risk_percentage"] = 0.20
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Add or update model config for this pair
    config["models"][pair] = {
        "model_type": "ensemble",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "win_rate": metrics["win_rate"],
        "base_leverage": args.base_leverage,
        "max_leverage": args.max_leverage,
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

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load historical data
    df = load_historical_data(args.pair, args.timeframe)
    
    # Calculate technical indicators
    df_indicators = calculate_indicators(df)
    
    # Create target variable
    df_labeled = create_target_variable(df_indicators)
    
    # Prepare data for training
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, features = prepare_data_for_training(
        df_labeled, args.sequence_length, args.test_size, args.validation_size
    )
    
    # Get input shape
    input_shape = (args.sequence_length, len(features))
    
    # Create base models
    base_models = create_base_models(input_shape)
    
    # Train base models
    trained_base_models, base_histories = train_base_models(
        base_models,
        X_train, y_train,
        X_val, y_val,
        args.epochs // 2,  # Use fewer epochs for base models
        args.batch_size,
        args.pair
    )
    
    # Create ensemble model
    ensemble_model = create_stacked_ensemble(trained_base_models, input_shape)
    
    # Train ensemble model
    trained_ensemble, ensemble_history, model_path = train_ensemble_model(
        ensemble_model,
        X_train, y_train,
        X_val, y_val,
        args.epochs // 2,  # Use fewer epochs for ensemble model
        args.batch_size * 2,  # Use larger batch size for ensemble model
        args.pair
    )
    
    # Evaluate ensemble model
    metrics = evaluate_model(trained_ensemble, X_test, y_test)
    
    # Plot training history
    plot_training_history(ensemble_history, args.pair)
    
    # Update ML config
    update_ml_config(args.pair, model_path, metrics, args)
    
    logger.info(f"Ensemble model with attention mechanisms created and trained for {args.pair}")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Update ML config with new model path and settings")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error creating ensemble model: {e}")
        import traceback
        traceback.print_exc()