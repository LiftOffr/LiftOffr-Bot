#!/usr/bin/env python3
"""
Script to train ML models on historical trading data.
Uses the MarketMLModel class to train TCN, CNN, LSTM, and combined models.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_models import MarketMLModel, prepare_market_data_from_trades
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)

def plot_training_history(history, save_path='plots'):
    """
    Plot training and validation loss for all models
    
    Args:
        history (dict): Dictionary with training histories for each model
        save_path (str): Directory to save plot images
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot TCN model loss
    plt.subplot(2, 2, 1)
    plt.plot(history['tcn']['loss'], label='Training Loss')
    plt.plot(history['tcn']['val_loss'], label='Validation Loss')
    plt.title('TCN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot CNN model loss
    plt.subplot(2, 2, 2)
    plt.plot(history['cnn']['loss'], label='Training Loss')
    plt.plot(history['cnn']['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot LSTM model loss
    plt.subplot(2, 2, 3)
    plt.plot(history['lstm']['loss'], label='Training Loss')
    plt.plot(history['lstm']['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Combined model loss
    plt.subplot(2, 2, 4)
    plt.plot(history['combined']['loss'], label='Training Loss')
    plt.plot(history['combined']['val_loss'], label='Validation Loss')
    plt.title('Combined Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png')
    plt.close()
    
    # Also plot comparison of validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['tcn']['val_loss'], label='TCN')
    plt.plot(history['cnn']['val_loss'], label='CNN')
    plt.plot(history['lstm']['val_loss'], label='LSTM')
    plt.plot(history['combined']['val_loss'], label='Combined')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/validation_loss_comparison.png')
    plt.close()


def evaluate_models(model, X_val, y_val):
    """
    Evaluate all models and compare their performance
    
    Args:
        model (MarketMLModel): Trained model instance
        X_val (np.array): Validation features
        y_val (np.array): Validation targets
        
    Returns:
        dict: Evaluation metrics for each model
    """
    # Evaluate TCN model
    tcn_loss = model.tcn_model.evaluate(X_val, y_val, verbose=0)
    
    # Evaluate CNN model
    cnn_loss = model.cnn_model.evaluate(X_val, y_val, verbose=0)
    
    # Evaluate LSTM model
    lstm_loss = model.lstm_model.evaluate(X_val, y_val, verbose=0)
    
    # Evaluate combined model
    combined_loss = model.combined_model.evaluate(X_val, y_val, verbose=0)
    
    results = {
        'tcn': {'loss': tcn_loss},
        'cnn': {'loss': cnn_loss},
        'lstm': {'loss': lstm_loss},
        'combined': {'loss': combined_loss}
    }
    
    # Print results
    logging.info("Model Evaluation:")
    logging.info(f"TCN Model Loss: {tcn_loss:.4f}")
    logging.info(f"CNN Model Loss: {cnn_loss:.4f}")
    logging.info(f"LSTM Model Loss: {lstm_loss:.4f}")
    logging.info(f"Combined Model Loss: {combined_loss:.4f}")
    
    return results


def plot_predictions(model, X_val, y_val, num_samples=5, save_path='plots'):
    """
    Plot predictions vs actual values for all models
    
    Args:
        model (MarketMLModel): Trained model instance
        X_val (np.array): Validation features
        y_val (np.array): Validation targets
        num_samples (int): Number of samples to plot
        save_path (str): Directory to save plot images
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Sample indices
    indices = np.random.choice(len(X_val), size=num_samples, replace=False)
    
    for idx in indices:
        # Get sample
        X_sample = X_val[idx:idx+1]
        y_true = y_val[idx]
        
        # Get predictions from all models
        y_pred_tcn = model.tcn_model.predict(X_sample)[0]
        y_pred_cnn = model.cnn_model.predict(X_sample)[0]
        y_pred_lstm = model.lstm_model.predict(X_sample)[0]
        y_pred_combined = model.combined_model.predict(X_sample)[0]
        
        # Inverse transform predictions and true values
        y_true = model.target_scaler.inverse_transform(y_true.reshape(1, -1))[0]
        y_pred_tcn = model.target_scaler.inverse_transform(y_pred_tcn.reshape(1, -1))[0]
        y_pred_cnn = model.target_scaler.inverse_transform(y_pred_cnn.reshape(1, -1))[0]
        y_pred_lstm = model.target_scaler.inverse_transform(y_pred_lstm.reshape(1, -1))[0]
        y_pred_combined = model.target_scaler.inverse_transform(y_pred_combined.reshape(1, -1))[0]
        
        # Plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(y_true))
        
        plt.plot(x, y_true, 'o-', label='Actual', linewidth=2)
        plt.plot(x, y_pred_tcn, 'o--', label='TCN Prediction')
        plt.plot(x, y_pred_cnn, 'o--', label='CNN Prediction')
        plt.plot(x, y_pred_lstm, 'o--', label='LSTM Prediction')
        plt.plot(x, y_pred_combined, 'o--', label='Combined Prediction', linewidth=2)
        
        plt.title(f'Model Predictions vs Actual (Sample {idx})')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_path}/prediction_sample_{idx}.png')
        plt.close()


def main(args):
    # Load and prepare data
    logging.info("Loading and preparing market data...")
    
    if args.csv_file and os.path.exists(args.csv_file):
        market_data = prepare_market_data_from_trades(args.csv_file)
    else:
        # Try to use trades.csv if no file is specified
        default_path = 'trades.csv'
        if os.path.exists(default_path):
            market_data = prepare_market_data_from_trades(default_path)
        else:
            logging.error(f"Could not find trades data file: {args.csv_file or default_path}")
            return 1
    
    if market_data is None or len(market_data) < 100:
        logging.error("Not enough market data for training (need at least 100 rows)")
        return 1
    
    logging.info(f"Loaded {len(market_data)} market data samples")
    
    # Initialize model
    model = MarketMLModel(sequence_length=args.sequence_length, n_features=market_data.shape[1] - 1)
    
    # Prepare data for training
    X_train, X_val, y_train, y_val = model.prepare_data(market_data, target_column='close')
    
    logging.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    
    # Build models
    model.build_tcn_model()
    model.build_cnn_model()
    model.build_lstm_model()
    model.build_combined_model()
    
    # Train models
    history = model.train_models(
        X_train, y_train, X_val, y_val, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate models
    evaluate_models(model, X_val, y_val)
    
    # Plot predictions
    plot_predictions(model, X_val, y_val, num_samples=5)
    
    # Save scalers for future use
    model.save_scalers()
    
    logging.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models for trading bot')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with market data')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length for time series')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    sys.exit(main(args))