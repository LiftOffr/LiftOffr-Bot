#!/usr/bin/env python3
"""
Train Advanced ML Models

This script trains more advanced ML models for all trading pairs:
1. Creates TCN-LSTM hybrid models with more features and epochs
2. Saves models to the model_weights directory
3. Updates the trading bot configuration with lower max leverage (75x)
4. Includes all available cryptocurrency pairs
"""
import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Default pairs to train
DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "ADA/USD",
    "DOT/USD",
    "LINK/USD",
    "AVAX/USD",
    "MATIC/USD",
    "UNI/USD",
    "ATOM/USD"
]

# Maximum leverage setting (reduced from 125x to 75x)
MAX_LEVERAGE = 75.0
MIN_LEVERAGE = 5.0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train advanced ML models for trading")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to train models for")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--confidence-threshold", type=float, default=0.6,
                        help="Confidence threshold for trading signals")
    return parser.parse_args()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [MODEL_WEIGHTS_DIR, DATA_DIR, CONFIG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_advanced_data(n_samples=2000, n_features=80):
    """
    Generate more complex example data for ML model training
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    # Generate features
    X = np.random.randn(n_samples, 48, n_features)  # Increased time steps and features
    
    # Generate more complex patterns with stronger signals
    # First generate a basic trend
    trend = np.cumsum(np.random.randn(n_samples, 1) * 0.1, axis=0)
    
    # Add cyclical patterns (like time of day effects)
    for i in range(n_samples):
        # Daily cycle (24 hours)
        daily_cycle = np.sin(i * 2 * np.pi / 24)
        # Weekly cycle (7 days)
        weekly_cycle = np.sin(i * 2 * np.pi / (24 * 7))
        
        trend[i] += daily_cycle * 0.05 + weekly_cycle * 0.1
    
    # Generate target with various patterns
    y = np.sign(trend + np.random.randn(n_samples, 1) * 0.02)  # Reduced noise
    
    # Add some momentum patterns - stronger trends persist
    for i in range(5, n_samples):
        if np.random.rand() < 0.8:  # 80% of the time, follow recent trend
            # Look at last 5 points for trend direction
            recent_trend = np.sum(y[i-5:i])
            if abs(recent_trend) >= 3:  # Strong trend in one direction
                y[i] = np.sign(recent_trend)
    
    # Add breakout patterns
    for i in range(50, n_samples):
        if np.random.rand() < 0.05:  # 5% chance of breakout
            # Breakout lasting several periods
            direction = 1 if np.random.rand() < 0.5 else -1
            breakout_length = int(5 + np.random.rand() * 15)  # 5-20 periods
            end_point = min(i + breakout_length, n_samples)
            y[i:end_point] = direction
    
    # Split data with 80% training, 20% validation
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val

def create_hybrid_model(input_shape, lr=0.001):
    """
    Create a hybrid TCN-LSTM model with more complex architecture
    
    Args:
        input_shape: Shape of input data
        lr: Learning rate
        
    Returns:
        tf.keras.Model: Hybrid model
    """
    model = Sequential([
        # 1D Convolutional layers (TCN-like)
        Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # LSTM layers for sequential processing
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense output layers
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')  # Output between -1 and 1
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=lr),
        metrics=['mae']
    )
    
    return model

def train_model_for_pair(pair_name, epochs=20, batch_size=32):
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
            patience=7,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        )
    ]
    
    # Train model
    history = model.fit(
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

def update_ml_config(pair_configs, confidence_threshold=0.6):
    """
    Update ML configuration with the trained models
    
    Args:
        pair_configs: Dictionary of pair configurations
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
                    # Update global settings with new leverage
                    ml_config["global_settings"]["dynamic_leverage_range"] = {
                        "min": MIN_LEVERAGE,
                        "max": MAX_LEVERAGE
                    }
                    ml_config["global_settings"]["confidence_threshold"] = confidence_threshold
            except (json.JSONDecodeError, FileNotFoundError):
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
        
        # Update model configurations
        for pair, config in pair_configs.items():
            ml_config["models"][pair] = config
        
        # Save configuration
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML configuration with {len(pair_configs)} models")
        logger.info(f"Max leverage set to {MAX_LEVERAGE}x (reduced from 125x)")
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
            "last_updated": "2025-04-15T00:00:00.000000"
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
    pairs = args.pairs
    epochs = args.epochs
    batch_size = args.batch_size
    confidence_threshold = args.confidence_threshold
    
    logger.info(f"Starting advanced ML model training for {len(pairs)} pairs")
    logger.info(f"Pairs: {', '.join(pairs)}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Train models for all pairs
    pair_configs = {}
    for pair in pairs:
        try:
            # Train model
            model_path = train_model_for_pair(pair, epochs=epochs, batch_size=batch_size)
            
            # Create pair configuration
            pair_configs[pair] = {
                "model_type": "hybrid",
                "model_path": model_path,
                "confidence_threshold": confidence_threshold,
                "min_leverage": MIN_LEVERAGE,
                "max_leverage": MAX_LEVERAGE,
                "risk_percentage": 0.20
            }
            
            logger.info(f"Completed training for {pair}")
        except Exception as e:
            logger.error(f"Error training model for {pair}: {e}")
    
    # Update ML configuration
    update_ml_config(pair_configs, confidence_threshold)
    
    # Reset portfolio
    reset_portfolio()
    
    logger.info(f"Advanced ML model training complete")
    logger.info(f"Portfolio reset to $20,000")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in advanced model training: {e}")
        raise