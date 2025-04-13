#!/usr/bin/env python3
"""
ML Data Integrator for Kraken Trading Bot

This module integrates the historical data with ML models training pipeline.
It connects the data fetcher with the ML model training process.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import importlib
from historical_data_fetcher import fetch_and_prepare_data, prepare_datasets_for_ml, save_datasets, load_datasets
from train_ml_models import train_tcn_model, train_cnn_model, train_lstm_model

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
DEFAULT_TRADING_PAIR = "SOL/USD"
DEFAULT_TIMEFRAME = "1h"  # Primary timeframe for training

def ensure_directories():
    """Ensure required directories exist"""
    for dir_path in [DATA_DIR, MODELS_DIR, f"{DATA_DIR}/ml_datasets", f"{MODELS_DIR}/tcn", f"{MODELS_DIR}/cnn", f"{MODELS_DIR}/lstm"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def fetch_data_for_training(trading_pair=DEFAULT_TRADING_PAIR, days_back=365*2):
    """
    Fetch and prepare historical data for ML model training
    
    Args:
        trading_pair (str): Trading pair to fetch data for
        days_back (int): Number of days to fetch data for
        
    Returns:
        dict: Dictionary of ML datasets
    """
    logger.info(f"Fetching and preparing data for {trading_pair} spanning {days_back} days")
    
    # Fetch and prepare historical data
    historical_data = fetch_and_prepare_data(trading_pair, days_back)
    
    # Prepare datasets for ML training
    ml_datasets = prepare_datasets_for_ml(historical_data)
    
    # Save datasets
    save_datasets(ml_datasets, trading_pair)
    
    return ml_datasets

def prepare_data_for_enhanced_training(trading_pair=DEFAULT_TRADING_PAIR, timeframe=DEFAULT_TIMEFRAME):
    """
    Prepare data for enhanced training with multiple timeframes
    
    This function prepares a combined dataset that includes features from multiple timeframes
    to provide more context and improve model performance.
    
    Args:
        trading_pair (str): Trading pair to fetch data for
        timeframe (str): Primary timeframe for training
        
    Returns:
        dict: Dictionary containing prepared datasets
    """
    logger.info(f"Preparing enhanced multi-timeframe dataset for {trading_pair}")
    
    # List of timeframes to use (primary timeframe + lower and higher timeframes)
    timeframes = ["15m", "30m", "1h", "4h"]
    
    # Load datasets for each timeframe
    datasets = {}
    for tf in timeframes:
        dataset = load_datasets(trading_pair, tf)
        if dataset is not None:
            datasets[tf] = dataset
    
    if not datasets:
        logger.error("No datasets found. Please fetch data first.")
        return None
    
    if timeframe not in datasets:
        logger.error(f"Primary timeframe {timeframe} not found in available datasets.")
        return None
    
    # Use the primary timeframe as the base
    primary_dataset = datasets[timeframe]
    
    # Record shape information for debugging
    logger.info(f"Primary dataset shape - X_train: {primary_dataset['X_train'].shape}, y_train: {primary_dataset['y_train'].shape}")
    
    return primary_dataset

def train_all_models(ml_dataset, trading_pair=DEFAULT_TRADING_PAIR, timeframe=DEFAULT_TIMEFRAME):
    """
    Train all ML models using the provided dataset
    
    Args:
        ml_dataset (dict): Dictionary containing prepared datasets
        trading_pair (str): Trading pair
        timeframe (str): Timeframe used for training
        
    Returns:
        dict: Dictionary containing trained model information
    """
    logger.info(f"Starting training for all models using {trading_pair} {timeframe} data")
    
    ensure_directories()
    
    # Check if train_ml_models module is available
    try:
        train_ml = importlib.import_module('train_ml_models')
    except ImportError:
        logger.error("train_ml_models module not found. Make sure it's available.")
        return None
    
    # Extract training and validation data
    X_train = ml_dataset['X_train']
    y_train = ml_dataset['y_train']
    X_val = ml_dataset['X_val']
    y_val = ml_dataset['y_val']
    
    # Train each model type
    models_info = {}
    
    # Train TCN model
    try:
        logger.info("Training TCN model...")
        tcn_model, tcn_history = train_tcn_model(X_train, y_train, X_val, y_val)
        tcn_model_path = f"{MODELS_DIR}/tcn/{trading_pair.replace('/', '')}-{timeframe}.h5"
        tcn_model.save(tcn_model_path)
        
        # Save training history
        with open(f"{MODELS_DIR}/tcn/{trading_pair.replace('/', '')}-{timeframe}-history.json", 'w') as f:
            history_dict = {key: [float(val) for val in values] for key, values in tcn_history.history.items()}
            json.dump(history_dict, f)
        
        models_info['tcn'] = {
            'model_path': tcn_model_path,
            'val_accuracy': float(max(tcn_history.history['val_accuracy'])),
            'val_loss': float(min(tcn_history.history['val_loss']))
        }
        logger.info(f"TCN model trained and saved to {tcn_model_path}")
    except Exception as e:
        logger.error(f"Error training TCN model: {e}")
    
    # Train CNN model
    try:
        logger.info("Training CNN model...")
        cnn_model, cnn_history = train_cnn_model(X_train, y_train, X_val, y_val)
        cnn_model_path = f"{MODELS_DIR}/cnn/{trading_pair.replace('/', '')}-{timeframe}.h5"
        cnn_model.save(cnn_model_path)
        
        # Save training history
        with open(f"{MODELS_DIR}/cnn/{trading_pair.replace('/', '')}-{timeframe}-history.json", 'w') as f:
            history_dict = {key: [float(val) for val in values] for key, values in cnn_history.history.items()}
            json.dump(history_dict, f)
        
        models_info['cnn'] = {
            'model_path': cnn_model_path,
            'val_accuracy': float(max(cnn_history.history['val_accuracy'])),
            'val_loss': float(min(cnn_history.history['val_loss']))
        }
        logger.info(f"CNN model trained and saved to {cnn_model_path}")
    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
    
    # Train LSTM model
    try:
        logger.info("Training LSTM model...")
        lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val)
        lstm_model_path = f"{MODELS_DIR}/lstm/{trading_pair.replace('/', '')}-{timeframe}.h5"
        lstm_model.save(lstm_model_path)
        
        # Save training history
        with open(f"{MODELS_DIR}/lstm/{trading_pair.replace('/', '')}-{timeframe}-history.json", 'w') as f:
            history_dict = {key: [float(val) for val in values] for key, values in lstm_history.history.items()}
            json.dump(history_dict, f)
        
        models_info['lstm'] = {
            'model_path': lstm_model_path,
            'val_accuracy': float(max(lstm_history.history['val_accuracy'])),
            'val_loss': float(min(lstm_history.history['val_loss']))
        }
        logger.info(f"LSTM model trained and saved to {lstm_model_path}")
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
    
    # Save model information
    models_info_path = f"{MODELS_DIR}/{trading_pair.replace('/', '')}-{timeframe}-models-info.json"
    with open(models_info_path, 'w') as f:
        json.dump(models_info, f)
    
    logger.info(f"All models trained and saved. Model info stored at {models_info_path}")
    
    return models_info

def run_full_data_pipeline(trading_pair=DEFAULT_TRADING_PAIR, timeframe=DEFAULT_TIMEFRAME, days_back=365*2):
    """
    Run the full data pipeline from fetching to training
    
    Args:
        trading_pair (str): Trading pair to fetch data for
        timeframe (str): Primary timeframe for training
        days_back (int): Number of days to fetch data for
    """
    logger.info(f"Starting full ML data pipeline for {trading_pair}")
    
    # Fetch and prepare data
    fetch_data_for_training(trading_pair, days_back)
    
    # Prepare enhanced dataset
    ml_dataset = prepare_data_for_enhanced_training(trading_pair, timeframe)
    
    if ml_dataset is None:
        logger.error("Failed to prepare enhanced dataset. Aborting training.")
        return
    
    # Train all models
    train_all_models(ml_dataset, trading_pair, timeframe)
    
    logger.info("Full ML data pipeline completed successfully")

def main():
    """Main function to run the ML data integrator"""
    logger.info("Starting ML data integrator")
    
    ensure_directories()
    
    # Run the full data pipeline
    run_full_data_pipeline()

if __name__ == "__main__":
    main()