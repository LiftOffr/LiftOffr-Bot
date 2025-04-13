#!/usr/bin/env python3
"""
Train Ensemble Models for Kraken Trading Bot

This script trains multiple model architectures for the ensemble:
- TCN (Temporal Convolutional Network)
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- BiLSTM (Bidirectional LSTM)
- Attention (Self-Attention Mechanism)
- Transformer (Transformer Architecture)
- Hybrid (Combined CNN-LSTM with Attention)

Each model is saved in its own directory with the necessary metadata
for the ensemble to use them effectively.
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, SpatialDropout1D
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure model directories exist
for model_type in MODEL_TYPES:
    os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)

def load_and_prepare_data(symbol="SOLUSD", timeframe="1h", test_size=0.2, validation_size=0.2):
    """
    Load and prepare data for model training
    
    Args:
        symbol (str): Trading symbol (e.g., "SOLUSD")
        timeframe (str): Timeframe for data (e.g., "1h")
        test_size (float): Fraction of data for testing
        validation_size (float): Fraction of training data for validation
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, norm_params)
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create basic features
        df['return'] = df['close'].pct_change()
        df['target'] = df['return'].shift(-1)  # Target is next period's return
        df['direction'] = (df['target'] > 0).astype(int)  # 1 if price goes up, 0 if down
        
        # Calculate indicators
        # Price based indicators
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Price relative to moving averages
        df['price_sma5_ratio'] = df['close'] / df['sma5']
        df['price_sma10_ratio'] = df['close'] / df['sma10']
        df['price_sma20_ratio'] = df['close'] / df['sma20']
        df['price_ema9_ratio'] = df['close'] / df['ema9']
        df['price_ema21_ratio'] = df['close'] / df['ema21']
        
        # Volatility indicators
        df['volatility'] = df['close'].rolling(window=20).std() / df['close']
        df['atr'] = df['high'] - df['low']  # Simple ATR
        
        # Volume indicators
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.00001)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features for model training
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma5', 'sma10', 'sma20', 'ema9', 'ema21',
            'price_sma5_ratio', 'price_sma10_ratio', 'price_sma20_ratio',
            'price_ema9_ratio', 'price_ema21_ratio',
            'volatility', 'atr',
            'volume_ratio', 'price_change', 'high_low_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position'
        ]
        
        # Normalize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df[feature_columns])
        y = df['direction'].values
        
        # Store normalization parameters
        norm_params = {
            'mean': scaler.data_min_.tolist(),
            'std': (scaler.data_max_ - scaler.data_min_).tolist()
        }
        
        # Split data chronologically
        train_size = int(len(X) * (1 - test_size))
        X_train_val, X_test = X[:train_size], X[train_size:]
        y_train_val, y_test = y[:train_size], y[train_size:]
        
        # Split training data into train and validation
        val_size = int(len(X_train_val) * validation_size)
        X_train, X_val = X_train_val[:-val_size], X_train_val[-val_size:]
        y_train, y_val = y_train_val[:-val_size], y_train_val[-val_size:]
        
        logger.info(f"Data prepared: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        logger.info(f"Validation data: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
        logger.info(f"Test data: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns, norm_params
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sequences(X, y, seq_length=24):
    """
    Create sequences for time series models
    
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

def build_tcn_model(input_shape, n_filters=64, kernel_size=3, dilations=None, 
                   dropout_rate=0.2, return_sequences=False):
    """
    Build a Temporal Convolutional Network (TCN) model
    
    Args:
        input_shape (tuple): Shape of input data
        n_filters (int): Number of filters in TCN layers
        kernel_size (int): Size of convolutional kernel
        dilations (list): List of dilation rates
        dropout_rate (float): Dropout rate
        return_sequences (bool): Whether to return sequences
        
    Returns:
        Model: TCN model
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    
    inputs = Input(shape=input_shape)
    
    x = TCN(
        nb_filters=n_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        return_sequences=return_sequences,
        dropout_rate=dropout_rate,
        use_skip_connections=True,
        use_batch_norm=True
    )(inputs)
    
    if return_sequences:
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_model(input_shape, n_filters=64, kernel_size=3, dropout_rate=0.2):
    """
    Build a Convolutional Neural Network (CNN) model
    
    Args:
        input_shape (tuple): Shape of input data
        n_filters (int): Number of filters in Conv1D layers
        kernel_size (int): Size of convolutional kernel
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: CNN model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, 
                    activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    
    # Second convolutional block
    model.add(Conv1D(filters=n_filters*2, kernel_size=kernel_size, 
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    
    # Third convolutional block
    model.add(Conv1D(filters=n_filters*2, kernel_size=kernel_size, 
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(dropout_rate))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lstm_model(input_shape, units=64, dropout_rate=0.2):
    """
    Build a Long Short-Term Memory (LSTM) model
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with return sequences
    model.add(LSTM(units=units, return_sequences=True, 
                  input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units=units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_gru_model(input_shape, units=64, dropout_rate=0.2):
    """
    Build a Gated Recurrent Unit (GRU) model
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of GRU units
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: GRU model
    """
    model = Sequential()
    
    # First GRU layer with return sequences
    model.add(GRU(units=units, return_sequences=True, 
                 input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second GRU layer
    model.add(GRU(units=units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_bilstm_model(input_shape, units=64, dropout_rate=0.2):
    """
    Build a Bidirectional LSTM model
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: BiLSTM model
    """
    model = Sequential()
    
    # First BiLSTM layer with return sequences
    model.add(Bidirectional(LSTM(units=units, return_sequences=True), 
                           input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second BiLSTM layer
    model.add(Bidirectional(LSTM(units=units)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_attention_model(input_shape, units=64, num_heads=2, dropout_rate=0.2):
    """
    Build a Self-Attention based model
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: Attention model
    """
    inputs = Input(shape=input_shape)
    
    # LSTM layer to extract temporal features
    x = LSTM(units=units, return_sequences=True)(inputs)
    
    # Self-attention layer
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=units // num_heads
    )(x, x)
    
    # Add & Normalize (residual connection)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_transformer_model(input_shape, num_heads=4, ff_dim=128, 
                          num_transformer_blocks=2, dropout_rate=0.2):
    """
    Build a Transformer model
    
    Args:
        input_shape (tuple): Shape of input data
        num_heads (int): Number of attention heads
        ff_dim (int): Feed forward dimension
        num_transformer_blocks (int): Number of transformer blocks
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: Transformer model
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[-1] // num_heads
        )(x, x)
        
        # Add & Normalize (residual connection)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward
        ff = Dense(ff_dim, activation='relu')(x)
        ff = Dense(input_shape[-1])(ff)
        
        # Add & Normalize (residual connection)
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_hybrid_model(input_shape, cnn_filters=64, lstm_units=64, 
                     num_heads=2, dropout_rate=0.2):
    """
    Build a hybrid CNN-LSTM-Attention model
    
    Args:
        input_shape (tuple): Shape of input data
        cnn_filters (int): Number of CNN filters
        lstm_units (int): Number of LSTM units
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
        
    Returns:
        Model: Hybrid model
    """
    inputs = Input(shape=input_shape)
    
    # CNN layers to extract local patterns
    x = Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # LSTM layer to capture temporal dependencies
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    
    # Self-attention layer to focus on important features
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=lstm_units // num_heads
    )(x, x)
    
    # Add & Normalize (residual connection)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Global max pooling to reduce dimensionality
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model_type, X_train, y_train, X_val, y_val, seq_length=24, 
               batch_size=32, epochs=50, patience=10):
    """
    Train a specific model type
    
    Args:
        model_type (str): Type of model to train
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation targets
        seq_length (int): Sequence length for time series
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Patience for early stopping
        
    Returns:
        tuple: (model, history)
    """
    # Create sequences for time series models
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    input_shape = X_train_seq.shape[1:]
    logger.info(f"Training {model_type} model with input_shape={input_shape}")
    
    # Build model based on type
    model = None
    if model_type == "tcn":
        model = build_tcn_model(input_shape)
    elif model_type == "cnn":
        model = build_cnn_model(input_shape)
    elif model_type == "lstm":
        model = build_lstm_model(input_shape)
    elif model_type == "gru":
        model = build_gru_model(input_shape)
    elif model_type == "bilstm":
        model = build_bilstm_model(input_shape)
    elif model_type == "attention":
        model = build_attention_model(input_shape)
    elif model_type == "transformer":
        model = build_transformer_model(input_shape)
    elif model_type == "hybrid":
        model = build_hybrid_model(input_shape)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, model_type, f"best_model.h5"),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def save_model(model, model_type, symbol, feature_names, norm_params):
    """
    Save model and associated metadata
    
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

def evaluate_model(model, X_test, y_test, seq_length=24):
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        seq_length (int): Sequence length
        
    Returns:
        dict: Evaluation metrics
    """
    # Create sequences for test data
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    # Evaluate model
    results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Predict and calculate metrics
    y_pred = model.predict(X_test_seq, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_binary.flatten() == y_test_seq)
    
    # Calculate directional accuracy (percentage of correct price direction predictions)
    directional_accuracy = np.mean(y_pred_binary.flatten() == y_test_seq)
    
    # Return metrics
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'directional_accuracy': directional_accuracy
    }
    
    return metrics

def main(args):
    """
    Main function to train and evaluate ensemble models
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    logger.info(f"Starting ensemble model training for {args.symbol}_{args.timeframe}")
    
    # Load and prepare data
    data = load_and_prepare_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        test_size=args.test_size,
        validation_size=args.validation_size
    )
    
    if data is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, norm_params = data
    
    # Train models
    results = {}
    for model_type in args.models:
        logger.info(f"Training {model_type} model...")
        model, history = train_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience
        )
        
        if model is not None:
            # Save model and metadata
            save_model(
                model=model,
                model_type=model_type,
                symbol=args.symbol,
                feature_names=feature_names,
                norm_params=norm_params
            )
            
            # Evaluate model
            metrics = evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                seq_length=args.seq_length
            )
            
            results[model_type] = metrics
            logger.info(f"{model_type} model evaluation: {metrics}")
    
    # Print summary of results
    logger.info("\nModel Training Results Summary:")
    for model_type, metrics in results.items():
        logger.info(f"{model_type.upper()}: Accuracy={metrics['accuracy']:.4f}, Directional Accuracy={metrics['directional_accuracy']:.4f}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Ensemble training completed in {elapsed_time / 60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble models for Kraken Trading Bot")
    parser.add_argument("--symbol", type=str, default="SOLUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES, help="Models to train")
    parser.add_argument("--seq_length", type=int, default=24, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    parser.add_argument("--validation_size", type=float, default=0.2, help="Validation size ratio")
    
    args = parser.parse_args()
    main(args)