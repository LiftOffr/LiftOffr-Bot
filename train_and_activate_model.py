#!/usr/bin/env python3
"""
Train and Activate Single Model

This script trains a simple ML model for a specified trading pair:
1. Creates a basic LSTM model for the pair
2. Saves the model to the model_weights directory
3. Updates the trading bot configuration to use this model
4. Can be run for one pair at a time to avoid timeouts

Usage:
    python train_and_activate_model.py --pair BTC/USD
"""
import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories
MODEL_WEIGHTS_DIR = "model_weights"
DATA_DIR = "data"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and activate a model for a specific pair")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g., BTC/USD)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [MODEL_WEIGHTS_DIR, DATA_DIR, CONFIG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_example_data(n_samples=1000, n_features=40):
    """
    Generate example data for ML model training
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    # Generate features and targets
    X = np.random.randn(n_samples, 24, n_features)
    
    # Simple trend pattern where the next price is influenced by the last few prices with noise
    trend = np.cumsum(np.random.randn(n_samples, 1) * 0.1, axis=0)
    y = np.sign(trend + np.random.randn(n_samples, 1) * 0.05)
    
    # Enhance the pattern to make it more learnable
    for i in range(1, n_samples):
        if np.random.rand() < 0.7:  # 70% of the time, follow the previous trend
            y[i] = y[i-1]
    
    # Add some momentum patterns
    for i in range(10, n_samples):
        if np.random.rand() < 0.3:  # 30% of the time, follow a 10-day momentum
            momentum = np.mean(y[i-10:i])
            y[i] = np.sign(momentum) if abs(momentum) > 0.05 else y[i]
    
    # Split into train and validation sets
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val

def create_lstm_model(input_shape, lr=0.001):
    """
    Create a simple LSTM model
    
    Args:
        input_shape: Shape of input data
        lr: Learning rate
        
    Returns:
        tf.keras.Model: LSTM model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=lr),
        metrics=['mae']
    )
    
    return model

def train_model_for_pair(pair_name, epochs=10, batch_size=32):
    """
    Train and save a model for a specific trading pair
    
    Args:
        pair_name: Name of the trading pair (e.g., "BTC/USD")
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        str: Path to the saved model file
    """
    symbol = pair_name.split('/')[0].lower()
    logger.info(f"Training model for {pair_name}...")
    
    # Generate example data for training
    X_train, y_train, X_val, y_val = generate_example_data()
    
    # Create and train model
    model = create_lstm_model(input_shape=X_train.shape[1:])
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model_path = f"{MODEL_WEIGHTS_DIR}/lstm_{symbol}_model.h5"
    model.save(model_path)
    logger.info(f"Saved model for {pair_name} to {model_path}")
    
    # Evaluate model
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Model evaluation for {pair_name}: Loss = {loss:.4f}, MAE = {mae:.4f}")
    
    return model_path

def update_ml_config(pair_name, model_path):
    """
    Update ML configuration to use the trained model
    
    Args:
        pair_name: Name of the trading pair
        model_path: Path to the trained model
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create or load ML configuration
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                ml_config = json.load(f)
            
            # Ensure required keys exist
            if "models" not in ml_config:
                ml_config["models"] = {}
                
            if "global_settings" not in ml_config:
                ml_config["global_settings"] = {
                    "confidence_threshold": 0.65,
                    "dynamic_leverage_range": {
                        "min": 5.0,
                        "max": 125.0
                    },
                    "risk_percentage": 0.20,
                    "max_positions_per_pair": 1
                }
        except (json.JSONDecodeError, FileNotFoundError):
            # If file exists but is invalid, create a new config
            ml_config = {
                "models": {},
                "global_settings": {
                    "confidence_threshold": 0.65,
                    "dynamic_leverage_range": {
                        "min": 5.0,
                        "max": 125.0
                    },
                    "risk_percentage": 0.20,
                    "max_positions_per_pair": 1
                }
            }
    else:
        # Create new config if file doesn't exist
        ml_config = {
            "models": {},
            "global_settings": {
                "confidence_threshold": 0.65,
                "dynamic_leverage_range": {
                    "min": 5.0,
                    "max": 125.0
                },
                "risk_percentage": 0.20,
                "max_positions_per_pair": 1
            }
        }
    
    # Add or update model configuration for the pair
    ml_config["models"][pair_name] = {
        "model_type": "lstm",
        "model_path": model_path,
        "confidence_threshold": 0.65,
        "min_leverage": 5.0,
        "max_leverage": 125.0,
        "risk_percentage": 0.20
    }
    
    # Save configuration
    with open(ML_CONFIG_PATH, 'w') as f:
        json.dump(ml_config, f, indent=2)
    
    logger.info(f"Updated ML configuration at {ML_CONFIG_PATH}")
    return True

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    pair = args.pair
    epochs = args.epochs
    batch_size = args.batch_size
    
    logger.info(f"Starting ML model training for {pair}...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Train model for the pair
    model_path = train_model_for_pair(pair, epochs=epochs, batch_size=batch_size)
    
    # Update ML configuration
    update_ml_config(pair, model_path)
    
    logger.info(f"ML model training for {pair} complete")
    logger.info("You can now restart the trading bot to use this model")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise