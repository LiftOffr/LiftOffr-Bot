#!/usr/bin/env python3
"""
Train SOL/USD ML Model

This script trains an optimized ML model specifically for SOL/USD:
1. Loads historical data for SOL/USD
2. Creates enhanced features with technical indicators
3. Builds and trains an LSTM model with optimized parameters
4. Saves the trained model and performance metrics

Usage:
    python train_solusd_model.py
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. Model training will not work.")

# Constants
PAIR = "SOL/USD"
LOOKBACK = 24
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_SIZE = 0.8
VALIDATION_SPLIT = 0.1
OUTPUT_DIR = "models/lstm"

def load_dataset():
    """
    Load historical data for SOL/USD.
    
    Returns:
        DataFrame with historical data
    """
    # Normalize pair format
    pair_path = PAIR.replace('/', '')
    
    # Try to locate the dataset
    dataset_paths = [
        f"training_data/{pair_path}_1h_enhanced.csv",
        f"historical_data/{pair_path}_1h.csv"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            logger.info(f"Loading dataset from {path}")
            df = pd.read_csv(path)
            logger.info(f"Dataset shape: {df.shape}")
            return df
    
    logger.error(f"No dataset found for {PAIR}")
    return None

def prepare_features(df):
    """
    Prepare features for model training.
    
    Args:
        df: DataFrame with historical data
        
    Returns:
        DataFrame with prepared features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    
    # Identify price column
    price_columns = ['close', 'Close', 'price', 'Price']
    price_column = None
    
    for col in price_columns:
        if col in df_processed.columns:
            price_column = col
            break
    
    if not price_column:
        logger.error("Could not identify price column")
        return None
    
    # Use existing features if the dataset is already enhanced
    existing_features = ['rsi', 'ema9', 'ema21', 'ema50', 'atr', 'bb_width', 'adx']
    has_existing_features = all(feature in df_processed.columns for feature in existing_features)
    
    if not has_existing_features:
        logger.info("Adding technical indicators as features")
        
        # Price changes
        df_processed['price_change'] = df_processed[price_column].pct_change()
        df_processed['price_change_1'] = df_processed[price_column].diff()
        
        # Moving averages
        for window in [9, 21, 50, 100]:
            df_processed[f'ema{window}'] = df_processed[price_column].ewm(span=window, adjust=False).mean()
        
        # RSI (14-period)
        delta = df_processed[price_column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df_processed['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        window = 20
        std_dev = 2
        
        df_processed['bb_middle'] = df_processed[price_column].rolling(window=window).mean()
        rolling_std = df_processed[price_column].rolling(window=window).std()
        
        df_processed['bb_upper'] = df_processed['bb_middle'] + (rolling_std * std_dev)
        df_processed['bb_lower'] = df_processed['bb_middle'] - (rolling_std * std_dev)
        df_processed['bb_width'] = (df_processed['bb_upper'] - df_processed['bb_lower']) / df_processed['bb_middle']
    
    # Add target: price direction for next period (1 = up, 0 = down)
    df_processed['target'] = np.where(df_processed[price_column].shift(-1) > df_processed[price_column], 1, 0)
    
    # Drop NaN values
    df_processed = df_processed.dropna()
    
    logger.info(f"Processed dataset shape: {df_processed.shape}")
    logger.info(f"Target distribution: {df_processed['target'].value_counts().to_dict()}")
    
    return df_processed, price_column

def prepare_sequences(df, price_column):
    """
    Prepare sequences for LSTM model.
    
    Args:
        df: DataFrame with prepared features
        price_column: Column name of price data
        
    Returns:
        X_train, y_train, X_test, y_test, feature_scaler
    """
    # Drop timestamp column if it exists
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    
    # Drop columns that shouldn't be features
    columns_to_drop = ['target']
    if price_column in df.columns:
        # Keep price column
        pass
    
    # Select features
    feature_columns = [col for col in df.columns if col not in columns_to_drop]
    
    # Separate features and target
    X = df[feature_columns].values
    y = df['target'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(LOOKBACK, len(X_scaled)):
        X_sequences.append(X_scaled[i-LOOKBACK:i])
        y_sequences.append(y[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split into train and test sets
    split_idx = int(len(X_sequences) * TRAIN_SIZE)
    
    X_train = X_sequences[:split_idx]
    y_train = y_sequences[:split_idx]
    
    X_test = X_sequences[split_idx:]
    y_test = y_sequences[split_idx:]
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_columns, scaler

def build_model(input_shape):
    """
    Build LSTM model with optimized hyperparameters.
    
    Args:
        input_shape: Shape of input sequences
        
    Returns:
        Compiled model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Trained model, training history
    """
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    
    return model, history, {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }

def plot_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save plot
    pair_path = PAIR.replace('/', '')
    plt.savefig(f"{OUTPUT_DIR}/{pair_path}_history.png")
    plt.close()

def save_model(model, feature_columns, metrics):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        feature_columns: Feature column names
        metrics: Model metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Normalize pair format
    pair_path = PAIR.replace('/', '')
    
    # Save model
    model_path = f"{OUTPUT_DIR}/{pair_path}.h5"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'pair': PAIR,
        'lookback': LOOKBACK,
        'feature_columns': feature_columns,
        'metrics': metrics,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = f"{OUTPUT_DIR}/{pair_path}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info(f"TRAINING ML MODEL FOR {PAIR}")
    logger.info("=" * 80)
    
    # Check if TensorFlow is available
    if not HAS_TENSORFLOW:
        logger.error("TensorFlow is not available. Cannot train model.")
        return 1
    
    # Disable GPU if not needed
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    logger.info("Using CPU for training")
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return 1
    
    # Prepare features
    df_processed, price_column = prepare_features(df)
    if df_processed is None:
        return 1
    
    # Prepare sequences
    X_train, y_train, X_test, y_test, feature_columns, scaler = prepare_sequences(df_processed, price_column)
    
    # Build model
    input_shape = (LOOKBACK, X_train.shape[2])
    model = build_model(input_shape)
    
    # Train model
    model, history, metrics = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_history(history)
    
    # Save model
    save_model(model, feature_columns, metrics)
    
    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())