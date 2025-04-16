#!/usr/bin/env python3
"""
Create Dummy Models

This script creates basic model files for all trading pairs:
1. Creates dummy model files in the model_weights directory
2. Updates the ML configuration with model file paths
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories
MODEL_WEIGHTS_DIR = "model_weights"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Default pairs
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

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [MODEL_WEIGHTS_DIR, CONFIG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_simple_model():
    """
    Create a simple LSTM model
    
    Returns:
        tf.keras.Model: Simple LSTM model
    """
    model = Sequential([
        LSTM(32, input_shape=(24, 30), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='tanh')
    ])
    
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae']
    )
    
    return model

def create_model_for_pair(pair):
    """
    Create a model for a specific trading pair
    
    Args:
        pair: Trading pair name (e.g., "BTC/USD")
        
    Returns:
        str: Path to the saved model file
    """
    symbol = pair.split('/')[0].lower()
    logger.info(f"Creating model for {pair}...")
    
    # Create model
    model = create_simple_model()
    
    # Save model
    model_path = f"{MODEL_WEIGHTS_DIR}/{symbol}_model.h5"
    model.save(model_path)
    logger.info(f"Saved model for {pair} to {model_path}")
    
    return model_path

def update_ml_config():
    """
    Update ML configuration with model file paths
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing ML configuration
        with open(ML_CONFIG_PATH, 'r') as f:
            ml_config = json.load(f)
        
        # Update model paths
        for pair in DEFAULT_PAIRS:
            symbol = pair.split('/')[0].lower()
            model_path = f"{MODEL_WEIGHTS_DIR}/{symbol}_model.h5"
            
            if pair in ml_config["models"]:
                ml_config["models"][pair]["model_path"] = model_path
                logger.info(f"Updated model path for {pair}: {model_path}")
        
        # Save updated configuration
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML configuration with model paths")
        return True
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False

def main():
    """Main function"""
    logger.info("Creating model files for all trading pairs")
    
    # Ensure directories exist
    ensure_directories()
    
    # Create model for each pair
    for pair in DEFAULT_PAIRS:
        try:
            create_model_for_pair(pair)
        except Exception as e:
            logger.error(f"Error creating model for {pair}: {e}")
    
    # Update ML configuration
    update_ml_config()
    
    logger.info("Model creation complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error creating models: {e}")
        raise