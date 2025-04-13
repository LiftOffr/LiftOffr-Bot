#!/usr/bin/env python3
"""
Train ML Models for Kraken Trading Bot

This script trains the ML models used in the trading bot, including:
1. TCN (Temporal Convolutional Network)
2. CNN (Convolutional Neural Network)
3. LSTM (Long Short-Term Memory)
4. GRU (Gated Recurrent Unit)
5. BiLSTM (Bidirectional LSTM)
6. Attention-based models
7. Transformer models
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add

# Import tcn only if available
try:
    from tcn import TCN
    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False
    print("TCN package not available. Install with: pip install keras-tcn")

# Import ML data integrator
from ml_data_integrator import MLDataIntegrator

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
TRAIN_TEST_SPLIT = 0.8
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 0.001

def ensure_directories():
    """Ensure all necessary directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_type in ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid"]:
        os.makedirs(f"{MODELS_DIR}/{model_type}", exist_ok=True)

def get_callbacks(model_type, trading_pair="SOLUSD"):
    """Get callbacks for model training"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=DEFAULT_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f"{MODELS_DIR}/{model_type}/{trading_pair}.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

def evaluate_model(model, X_test, y_test_return, y_test_direction):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        X_test: Test data
        y_test_return: Test targets (returns)
        y_test_direction: Test targets (direction)
        
    Returns:
        dict: Evaluation metrics
    """
    # Evaluate on test data
    predictions = model.predict(X_test)
    predictions = predictions.flatten()
    
    # Convert to direction predictions
    direction_predictions = (predictions > 0).astype(int)
    
    # Calculate accuracy for direction
    direction_accuracy = np.mean(direction_predictions == y_test_direction)
    
    # Calculate MSE for returns
    mse = np.mean((predictions - y_test_return) ** 2)
    
    # Calculate MAE for returns
    mae = np.mean(np.abs(predictions - y_test_return))
    
    # Calculate hit rate (percentage of correct direction predictions)
    hit_rate = direction_accuracy
    
    # Calculate Sharpe ratio (simplified)
    pred_returns = predictions
    sharpe = np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0
    
    # Return evaluation metrics
    return {
        "mse": float(mse),
        "mae": float(mae),
        "direction_accuracy": float(direction_accuracy),
        "hit_rate": float(hit_rate),
        "sharpe": float(sharpe)
    }

def train_tcn_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a TCN (Temporal Convolutional Network) model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    if not TCN_AVAILABLE:
        logger.error("TCN package not available. Skipping TCN model training.")
        return None, None
    
    try:
        logger.info("Training TCN model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/tcn/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build TCN model
        input_layer = Input(shape=input_shape)
        
        # Add TCN layer
        tcn_layer = TCN(
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16),
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=False
        )(input_layer)
        
        # Add Dense layers
        x = Dense(64, activation='relu')(tcn_layer)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output_layer = Dense(1, activation='tanh')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("tcn"),
            verbose=1
        )
        
        logger.info("TCN model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training TCN model: {e}")
        traceback.print_exc()
        return None, None

def train_cnn_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a CNN (Convolutional Neural Network) model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training CNN model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/cnn/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build CNN model
        model = Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='tanh')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("cnn"),
            verbose=1
        )
        
        logger.info("CNN model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
        traceback.print_exc()
        return None, None

def train_lstm_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train an LSTM (Long Short-Term Memory) model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training LSTM model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/lstm/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='tanh')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("lstm"),
            verbose=1
        )
        
        logger.info("LSTM model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        traceback.print_exc()
        return None, None

def train_gru_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a GRU (Gated Recurrent Unit) model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training GRU model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/gru/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build GRU model
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            GRU(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='tanh')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("gru"),
            verbose=1
        )
        
        logger.info("GRU model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training GRU model: {e}")
        traceback.print_exc()
        return None, None

def train_bilstm_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a BiLSTM (Bidirectional LSTM) model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training BiLSTM model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/bilstm/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build BiLSTM model
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            Bidirectional(LSTM(64, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='tanh')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("bilstm"),
            verbose=1
        )
        
        logger.info("BiLSTM model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training BiLSTM model: {e}")
        traceback.print_exc()
        return None, None

def train_attention_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train an Attention-based LSTM model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training Attention model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/attention/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build Attention model
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Self-attention layer
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(lstm_out, lstm_out)
        attention = BatchNormalization()(attention)
        
        # Add & Normalize
        x = Add()([lstm_out, attention])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='tanh')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("attention"),
            verbose=1
        )
        
        logger.info("Attention model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training Attention model: {e}")
        traceback.print_exc()
        return None, None

def train_transformer_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a Transformer model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training Transformer model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/transformer/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build Transformer model
        inputs = Input(shape=input_shape)
        
        # Transformer Encoder
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
            # Normalization and Attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            attention_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            attention_output = Dropout(dropout)(attention_output)
            out1 = Add()([inputs, attention_output])
            
            # Feed Forward
            x = LayerNormalization(epsilon=1e-6)(out1)
            x = Dense(ff_dim, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return Add()([out1, x])
        
        # Apply transformer layers
        x = inputs
        for i in range(3):  # Multiple transformer blocks
            x = transformer_encoder(x, head_size=32, num_heads=8, ff_dim=128, dropout=0.1)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='tanh')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("transformer"),
            verbose=1
        )
        
        logger.info("Transformer model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training Transformer model: {e}")
        traceback.print_exc()
        return None, None

def train_hybrid_model(X_train, y_train, X_val, y_val, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train a Hybrid model combining CNN, LSTM, and Attention mechanisms
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training Hybrid model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Normalize data
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1))
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Save normalization parameters
        norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(f"{MODELS_DIR}/hybrid/norm_params.json", 'w') as f:
            json.dump(norm_params, f)
        
        # Build Hybrid model
        inputs = Input(shape=input_shape)
        
        # CNN branch
        cnn = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(128, 3, padding='same', activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(64, activation='relu')(cnn)
        cnn = Dropout(0.2)(cnn)
        
        # LSTM branch
        lstm = LSTM(64, return_sequences=True)(inputs)
        lstm = BatchNormalization()(lstm)
        
        # GRU branch
        gru = GRU(64, return_sequences=True)(inputs)
        gru = BatchNormalization()(gru)
        
        # Combine LSTM and GRU
        recurrent = tf.keras.layers.Concatenate()([lstm, gru])
        
        # Add attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(recurrent, recurrent)
        attention = BatchNormalization()(attention)
        recurrent = Add()([recurrent, attention])
        
        # Global pooling
        recurrent_features = GlobalAveragePooling1D()(recurrent)
        
        # Combine CNN and Recurrent features
        combined = tf.keras.layers.Concatenate()([cnn, recurrent_features])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        # Output layer
        outputs = Dense(1, activation='tanh')(combined)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks("hybrid"),
            verbose=1
        )
        
        logger.info("Hybrid model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training Hybrid model: {e}")
        traceback.print_exc()
        return None, None

def export_and_compare_models(trading_pair="SOLUSD", timeframe="1h"):
    """
    Export and compare all trained models
    
    Args:
        trading_pair (str): Trading pair
        timeframe (str): Timeframe
    
    Returns:
        dict: Comparison of model performance
    """
    # Load test data for all models
    model_types = ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid"]
    model_results = {}
    
    for model_type in model_types:
        try:
            model_path = f"{MODELS_DIR}/{model_type}/{trading_pair}.h5"
            eval_path = f"{MODELS_DIR}/{model_type}/{trading_pair}_evaluation.json"
            
            if os.path.exists(model_path) and os.path.exists(f"{MODELS_DIR}/{model_type}/X_test.npy"):
                # Load test data
                X_test = np.load(f"{MODELS_DIR}/{model_type}/X_test.npy")
                y_test_return = np.load(f"{MODELS_DIR}/{model_type}/y_test_return.npy")
                y_test_direction = np.load(f"{MODELS_DIR}/{model_type}/y_test_direction.npy")
                
                # Load normalization parameters
                with open(f"{MODELS_DIR}/{model_type}/norm_params.json", 'r') as f:
                    norm_params = json.load(f)
                
                mean = np.array(norm_params['mean'])
                std = np.array(norm_params['std'])
                
                # Normalize test data
                X_test_norm = (X_test - mean) / std
                
                # Load model
                model = tf.keras.models.load_model(model_path)
                
                # Evaluate model
                results = evaluate_model(model, X_test_norm, y_test_return, y_test_direction)
                
                # Save evaluation results
                with open(eval_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Store results
                model_results[model_type] = results
                
                logger.info(f"Evaluated {model_type} model with MSE: {results['mse']:.6f}, Direction Accuracy: {results['direction_accuracy']:.4f}")
            else:
                logger.warning(f"Model or test data not found for {model_type}")
        
        except Exception as e:
            logger.error(f"Error evaluating {model_type} model: {e}")
            traceback.print_exc()
    
    # Combine all results and save comparison
    if model_results:
        with open(f"{MODELS_DIR}/model_comparison.json", 'w') as f:
            json.dump(model_results, f, indent=2)
        
        # Print comparison
        logger.info("\nModel Comparison:")
        logger.info("-" * 80)
        logger.info(f"{'Model Type':<15} {'MSE':<10} {'MAE':<10} {'Direction Acc.':<15} {'Hit Rate':<10} {'Sharpe':<10}")
        logger.info("-" * 80)
        
        for model_type, results in model_results.items():
            logger.info(f"{model_type:<15} {results['mse']:<10.6f} {results['mae']:<10.6f} {results['direction_accuracy']:<15.4f} {results['hit_rate']:<10.4f} {results['sharpe']:<10.4f}")
        
        logger.info("-" * 80)
    
    return model_results

def train_all_models(trading_pair="SOL/USD", timeframe="1h", batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Train all ML models for the trading bot
    
    Args:
        trading_pair (str): Trading pair to train models for
        timeframe (str): Timeframe to use for training
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        dict: Dictionary of trained models and their performance
    """
    # Ensure directories exist
    ensure_directories()
    
    # Create ticker symbol without /
    ticker = trading_pair.replace("/", "")
    
    # Initialize data integrator
    data_integrator = MLDataIntegrator(trading_pair)
    
    # Load historical data
    data_integrator.load_data()
    
    # Process datasets with technical indicators
    data_integrator.process_datasets()
    
    # Integrate data from all timeframes
    integrated_df = data_integrator.integrate_timeframes(primary_timeframe=timeframe)
    
    if integrated_df is None or len(integrated_df) < 200:
        logger.error(f"Insufficient data for training (found {len(integrated_df) if integrated_df is not None else 0} rows). Minimum 200 rows required.")
        return None
    
    # Prepare training data for return prediction
    data_tuple = data_integrator.prepare_training_data(integrated_df, target_column=f"close_{timeframe}")
    
    if data_tuple[0] is None:
        logger.error("Failed to prepare training data.")
        return None
    
    (X_train, y_train_return, y_train_direction, 
     X_val, y_val_return, y_val_direction,
     X_test, y_test_return, y_test_direction,
     feature_names) = data_tuple
    
    logger.info(f"Prepared {len(X_train)} training samples, {len(X_val)} validation samples, and {len(X_test)} test samples.")
    
    # Train models
    models_and_performance = {}
    
    # TCN model
    tcn_model, tcn_history = train_tcn_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if tcn_model is not None:
        data_integrator.save_train_stats(tcn_model, tcn_history, "tcn")
        data_integrator.save_processed_data(data_tuple, "tcn")
    
    # CNN model
    cnn_model, cnn_history = train_cnn_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if cnn_model is not None:
        data_integrator.save_train_stats(cnn_model, cnn_history, "cnn")
        data_integrator.save_processed_data(data_tuple, "cnn")
    
    # LSTM model
    lstm_model, lstm_history = train_lstm_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if lstm_model is not None:
        data_integrator.save_train_stats(lstm_model, lstm_history, "lstm")
        data_integrator.save_processed_data(data_tuple, "lstm")
    
    # GRU model
    gru_model, gru_history = train_gru_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if gru_model is not None:
        data_integrator.save_train_stats(gru_model, gru_history, "gru")
        data_integrator.save_processed_data(data_tuple, "gru")
    
    # BiLSTM model
    bilstm_model, bilstm_history = train_bilstm_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if bilstm_model is not None:
        data_integrator.save_train_stats(bilstm_model, bilstm_history, "bilstm")
        data_integrator.save_processed_data(data_tuple, "bilstm")
    
    # Attention model
    attention_model, attention_history = train_attention_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if attention_model is not None:
        data_integrator.save_train_stats(attention_model, attention_history, "attention")
        data_integrator.save_processed_data(data_tuple, "attention")
    
    # Transformer model
    transformer_model, transformer_history = train_transformer_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if transformer_model is not None:
        data_integrator.save_train_stats(transformer_model, transformer_history, "transformer")
        data_integrator.save_processed_data(data_tuple, "transformer")
    
    # Hybrid model
    hybrid_model, hybrid_history = train_hybrid_model(X_train, y_train_return, X_val, y_val_return, batch_size, epochs)
    if hybrid_model is not None:
        data_integrator.save_train_stats(hybrid_model, hybrid_history, "hybrid")
        data_integrator.save_processed_data(data_tuple, "hybrid")
    
    # Export and compare all models
    model_comparison = export_and_compare_models(ticker, timeframe)
    
    return model_comparison

def main():
    """Main function to train all ML models"""
    logger.info("Starting ML model training for Kraken Trading Bot")
    
    trading_pair = "SOL/USD"
    timeframe = "1h"
    
    model_comparison = train_all_models(trading_pair, timeframe)
    
    if model_comparison:
        logger.info("All models trained successfully.")
        
        # Find best model
        best_model = min(model_comparison.items(), key=lambda x: x[1]['mse'])
        logger.info(f"Best model based on MSE: {best_model[0]} with MSE: {best_model[1]['mse']:.6f}")
        
        best_dir_model = max(model_comparison.items(), key=lambda x: x[1]['direction_accuracy'])
        logger.info(f"Best model based on Direction Accuracy: {best_dir_model[0]} with Accuracy: {best_dir_model[1]['direction_accuracy']:.4f}")
    else:
        logger.error("Failed to train models.")

if __name__ == "__main__":
    main()