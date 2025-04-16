#!/usr/bin/env python3
"""
Train One Pair Model

This script trains an advanced model for a single trading pair to avoid timeouts:
1. Creates a TCN-LSTM hybrid model
2. Saves the model to the model_weights directory
3. Updates the trading bot configuration for this pair
"""
import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D
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

# Maximum leverage setting (reduced from 125x to 75x)
MAX_LEVERAGE = 75.0
MIN_LEVERAGE = 5.0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train model for a single pair")
    parser.add_argument("--pair", type=str, required=True,
                        help="Trading pair to train the model for (e.g., BTC/USD)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--confidence-threshold", type=float, default=0.6,
                        help="Confidence threshold for trading signals")
    parser.add_argument("--reset-portfolio", action="store_true",
                        help="Reset portfolio to $20,000 if this is specified")
    return parser.parse_args()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [MODEL_WEIGHTS_DIR, DATA_DIR, CONFIG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_advanced_data(n_samples=1000, n_features=60):
    """
    Generate advanced example data for ML model training
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    # Generate features with more time steps
    X = np.random.randn(n_samples, 24, n_features)
    
    # Generate target with trend and noise
    trend = np.cumsum(np.random.randn(n_samples, 1) * 0.1, axis=0)
    y = np.sign(trend + np.random.randn(n_samples, 1) * 0.05)
    
    # Add momentum patterns
    for i in range(10, n_samples):
        if np.random.rand() < 0.7:  # 70% of the time, follow recent trend
            recent_trend = np.mean(y[i-10:i])
            if abs(recent_trend) > 0.3:  # Strong trend
                y[i] = np.sign(recent_trend)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val

def create_hybrid_model(input_shape, lr=0.001):
    """
    Create a hybrid TCN-LSTM model
    
    Args:
        input_shape: Shape of input data
        lr: Learning rate
        
    Returns:
        tf.keras.Model: Hybrid model
    """
    model = Sequential([
        # Convolutional layer (TCN-like)
        Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # LSTM layers
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense output layers
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='tanh')  # Output between -1 and 1
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=lr),
        metrics=['mae']
    )
    
    return model

def train_model_for_pair(pair_name, epochs=5, batch_size=32):
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
    logger.info(f"Training advanced model for {pair_name}...")
    
    # Generate example data for training
    X_train, y_train, X_val, y_val = generate_advanced_data()
    
    # Create and train model
    model = create_hybrid_model(input_shape=X_train.shape[1:])
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{symbol}_model.h5"
    model.save(model_path)
    logger.info(f"Saved model for {pair_name} to {model_path}")
    
    # Evaluate model
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Model evaluation for {pair_name}: Loss = {loss:.4f}, MAE = {mae:.4f}")
    
    # Calculate accuracy (direction prediction)
    pred = model.predict(X_val)
    pred_direction = np.sign(pred)
    true_direction = np.sign(y_val)
    accuracy = np.mean(pred_direction == true_direction)
    logger.info(f"Direction prediction accuracy for {pair_name}: {accuracy:.2%}")
    
    return model_path

def update_ml_config(pair, model_path, confidence_threshold=0.6):
    """
    Update ML configuration for a specific pair
    
    Args:
        pair: Trading pair name
        model_path: Path to the trained model
        confidence_threshold: Confidence threshold for trading signals
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
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
                        "confidence_threshold": confidence_threshold,
                        "dynamic_leverage_range": {
                            "min": MIN_LEVERAGE,
                            "max": MAX_LEVERAGE
                        },
                        "risk_percentage": 0.20,
                        "max_positions_per_pair": 1
                    }
                else:
                    # Update global settings
                    ml_config["global_settings"]["dynamic_leverage_range"] = {
                        "min": MIN_LEVERAGE,
                        "max": MAX_LEVERAGE
                    }
            except Exception:
                # If file exists but is invalid, create a new config
                ml_config = {
                    "models": {},
                    "global_settings": {
                        "confidence_threshold": confidence_threshold,
                        "dynamic_leverage_range": {
                            "min": MIN_LEVERAGE,
                            "max": MAX_LEVERAGE
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
                    "confidence_threshold": confidence_threshold,
                    "dynamic_leverage_range": {
                        "min": MIN_LEVERAGE,
                        "max": MAX_LEVERAGE
                    },
                    "risk_percentage": 0.20,
                    "max_positions_per_pair": 1
                }
            }
        
        # Update model configuration for the pair
        ml_config["models"][pair] = {
            "model_type": "hybrid",
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "min_leverage": MIN_LEVERAGE,
            "max_leverage": MAX_LEVERAGE,
            "risk_percentage": 0.20
        }
        
        # Save configuration
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML configuration for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False

def reset_portfolio():
    """Reset portfolio to initial state with $20,000"""
    try:
        portfolio_path = f"{DATA_DIR}/sandbox_portfolio.json"
        portfolio = {
            "balance": 20000.0,
            "initial_balance": 20000.0,
            "last_updated": "2025-04-16T00:00:00.000000"
        }
        
        # Save portfolio
        with open(portfolio_path, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        # Also clear positions
        positions_path = f"{DATA_DIR}/sandbox_positions.json"
        with open(positions_path, 'w') as f:
            json.dump({}, f, indent=2)
        
        logger.info(f"Reset portfolio to $20,000 and cleared all positions")
        return True
    except Exception as e:
        logger.error(f"Error resetting portfolio: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    pair = args.pair
    epochs = args.epochs
    batch_size = args.batch_size
    confidence_threshold = args.confidence_threshold
    do_reset_portfolio = args.reset_portfolio
    
    logger.info(f"Starting advanced ML model training for {pair}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Train model for the pair
    model_path = train_model_for_pair(pair, epochs=epochs, batch_size=batch_size)
    
    # Update ML configuration
    update_ml_config(pair, model_path, confidence_threshold)
    
    # Reset portfolio if requested
    if do_reset_portfolio:
        reset_portfolio()
        logger.info("Portfolio reset to $20,000")
    
    logger.info(f"Advanced ML model training for {pair} complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise