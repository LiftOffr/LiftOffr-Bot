#!/usr/bin/env python3
"""
Script to test predictions from the trained ML model
"""

import os
import logging
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
MODELS_DIR = "models/lstm"
MODEL_FILE = os.path.join(MODELS_DIR, "model.h5")

def load_and_preprocess_data(symbol="SOLUSD", timeframe="1h", seq_length=24):
    """
    Load and preprocess data for prediction
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe for data
        seq_length (int): Sequence length for LSTM input
    
    Returns:
        tuple: (X, df)
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None, None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create features
        df['return'] = df['close'].pct_change()
        df['direction'] = (df['return'] > 0).astype(int)
        
        # Calculate technical indicators
        # Simple Moving Averages
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        
        # Price relative to SMA
        df['price_sma5_ratio'] = df['close'] / df['sma5']
        df['price_sma10_ratio'] = df['close'] / df['sma10']
        df['price_sma20_ratio'] = df['close'] / df['sma20']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close']
        
        # Volume features
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features for ML
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma5', 'sma10', 'sma20', 
            'price_sma5_ratio', 'price_sma10_ratio', 'price_sma20_ratio',
            'volatility', 'volume_ratio'
        ]
        
        # Normalize features
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[feature_columns]),
            columns=feature_columns
        )
        
        # Create sequences for LSTM
        X = []
        for i in range(len(df_normalized) - seq_length + 1):
            X.append(df_normalized[feature_columns].iloc[i:i+seq_length].values)
        
        X = np.array(X)
        
        return X, df
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None

def make_predictions(model, X):
    """
    Make predictions using the trained model
    
    Args:
        model: Trained model
        X (numpy.ndarray): Input data
        
    Returns:
        numpy.ndarray: Predictions
    """
    predictions = model.predict(X)
    return predictions

def main():
    """Test the trained model by making predictions"""
    logger.info("Testing ML model predictions")
    
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        logger.error(f"Model file not found: {MODEL_FILE}")
        return
    
    # Load model
    try:
        model = load_model(MODEL_FILE)
        logger.info(f"Loaded model from {MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load and preprocess data
    X, df = load_and_preprocess_data(symbol="SOLUSD", timeframe="1h")
    
    if X is None:
        logger.error("Failed to prepare data for prediction. Exiting.")
        return
    
    # Make predictions
    logger.info(f"Making predictions on {len(X)} samples...")
    predictions = make_predictions(model, X)
    
    # Analyze predictions
    df_subset = df.iloc[24:].copy()  # Start from the first complete sequence
    df_subset = df_subset.reset_index(drop=True)
    
    # Check lengths to make sure they match
    if len(df_subset) != len(predictions):
        logger.warning(f"Length mismatch: df_subset length {len(df_subset)}, predictions length {len(predictions)}")
        # Adjust the dataframe to match predictions length
        if len(df_subset) > len(predictions):
            df_subset = df_subset.iloc[:len(predictions)]
        else:
            predictions = predictions[:len(df_subset)]
            
    df_subset['prediction'] = predictions
    df_subset['actual_return'] = df_subset['return'].shift(-1)  # Actual next return
    df_subset['prediction_direction'] = (df_subset['prediction'] > 0).astype(int)
    df_subset['actual_direction'] = (df_subset['actual_return'] > 0).astype(int)
    df_subset = df_subset.dropna()
    
    # Calculate accuracy
    direction_accuracy = (df_subset['prediction_direction'] == df_subset['actual_direction']).mean()
    
    # Display sample predictions
    logger.info(f"Direction prediction accuracy: {direction_accuracy:.4f}")
    logger.info("\nSample predictions (recent data):")
    
    # Display most recent predictions
    sample_size = min(10, len(df_subset))
    recent_samples = df_subset.tail(sample_size)
    
    for _, row in recent_samples.iterrows():
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        price = row['close']
        pred = row['prediction']
        pred_dir = "UP" if row['prediction_direction'] else "DOWN"
        actual_dir = "UP" if row['actual_direction'] else "DOWN"
        match = "✓" if row['prediction_direction'] == row['actual_direction'] else "✗"
        
        logger.info(f"{timestamp} | Price: ${price:.2f} | Pred: {pred:.6f} | Direction: {pred_dir} vs {actual_dir} {match}")
    
    logger.info("Prediction testing completed.")

if __name__ == "__main__":
    main()