#!/usr/bin/env python3
"""
Simplified ML model training for Kraken Trading Bot
focusing on LSTM model only
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

def ensure_directories():
    """Ensure all necessary directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_prepare_data(symbol="SOLUSD", timeframe="1h", seq_length=24):
    """
    Load and prepare data for ML training
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe for data
        seq_length (int): Sequence length for LSTM input
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
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
        
        # Target variable (next period return)
        df_normalized['target_return'] = df['return'].shift(-1)
        df_normalized['target_direction'] = df['direction'].shift(-1)
        
        # Drop NaN values again
        df_normalized = df_normalized.dropna()
        
        # Create sequences for LSTM
        X, y = [], []
        
        for i in range(len(df_normalized) - seq_length):
            X.append(df_normalized[feature_columns].iloc[i:i+seq_length].values)
            y.append(df_normalized['target_return'].iloc[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info(f"Prepared data: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        logger.info(f"Validation data: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None

def build_lstm_model(input_shape):
    """
    Build LSTM model for time series prediction
    
    Args:
        input_shape (tuple): Shape of input data
    
    Returns:
        Model: Keras LSTM model
    """
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
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=5):
    """
    Train LSTM model
    
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
    # Build model
    model = build_lstm_model(X_train.shape[1:])
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "model.h5"),
            monitor='val_loss',
            save_best_only=False,  # Save all checkpoints
            save_freq='epoch',     # Save after each epoch
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model explicitly
    model.save(os.path.join(MODELS_DIR, "final_model.h5"))
    
    return model, history

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
    
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    predictions = model.predict(X_val)
    
    # Calculate MSE
    mse = np.mean((predictions.flatten() - y_val) ** 2)
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions.flatten() - y_val))
    
    # Calculate direction accuracy
    direction_predictions = (predictions > 0).astype(int)
    direction_actual = (y_val > 0).astype(int)
    direction_accuracy = np.mean(direction_predictions.flatten() == direction_actual)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'direction_accuracy': float(direction_accuracy)
    }

def main():
    """Main function to train LSTM model"""
    logger.info("Starting simplified ML model training for Kraken Trading Bot")
    
    # Ensure directories exist
    ensure_directories()
    
    # Load and prepare data
    X_train, y_train, X_val, y_val = load_and_prepare_data(symbol="SOLUSD", timeframe="1h")
    
    if X_train is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=5)  # Using fewer epochs for quicker training
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val)
    
    logger.info(f"Model evaluation: MSE = {metrics['mse']:.6f}, MAE = {metrics['mae']:.6f}")
    logger.info(f"Direction accuracy: {metrics['direction_accuracy']:.4f}")
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()