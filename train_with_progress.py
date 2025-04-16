#!/usr/bin/env python3
"""
Train Hybrid Model with Progress Updates

This script trains a simplified version of the hybrid model and provides
regular progress updates to show training advancement. It focuses on:
1. Quick iterations for demonstration purposes
2. Simplified architecture while preserving key components
3. Fast progress tracking and intermediate results

Usage:
    python train_with_progress.py --pair BTC/USD --epochs 5
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Create required directories
for directory in [DATA_DIR, HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train hybrid model with progress updates")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    return parser.parse_args()

def generate_sample_data(num_samples=1000, features=25, sequence_length=60):
    """Generate sample data for quick demonstration"""
    # Generate timestamps
    timestamps = np.arange(num_samples)
    
    # Generate price data
    base_price = 50000
    trend = np.linspace(0, 0.2, num_samples)
    noise = np.random.normal(0, 0.01, num_samples)
    prices = base_price * (1 + trend + noise)
    
    # Create price DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.002, num_samples)),
        'high': prices * (1 + np.random.normal(0, 0.005, num_samples)),
        'low': prices * (1 - np.random.normal(0, 0.005, num_samples)),
        'close': prices,
        'volume': np.random.normal(1000, 100, num_samples)
    })
    
    # Add some basic technical indicators
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi_14'] = np.random.uniform(0, 100, num_samples)  # Simplified RSI
    df['atr_14'] = np.random.uniform(100, 500, num_samples)  # Simplified ATR
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    
    # Create target variable (next period price direction)
    future_return = df['close'].shift(-1) / df['close'] - 1
    df['target'] = 0
    df.loc[future_return > 0.005, 'target'] = 1
    df.loc[future_return < -0.005, 'target'] = -1
    
    # Drop last row where target is NaN
    df.dropna(inplace=True)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(df)):
        # Get sequence of price data and indicators
        sequence = df.iloc[i-sequence_length:i].drop(['timestamp', 'target'], axis=1).values
        X.append(sequence)
        y.append(df.iloc[i]['target'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # Add 1 to shift from [-1,0,1] to [0,1,2]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_simplified_hybrid_model(input_shape, output_shape=3):
    """Build a simplified hybrid model for faster demonstration"""
    # Print progress
    print("\nBuilding hybrid model architecture...")
    print("CNN branch: Capturing local price patterns")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    print("  - Added Conv1D layer with 32 filters")
    cnn = MaxPooling1D(pool_size=2)(cnn)
    print("  - Added MaxPooling1D layer")
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn)
    print("  - Added Conv1D layer with 64 filters")
    cnn = Flatten()(cnn)
    cnn = Dense(32, activation='relu')(cnn)
    print("  - Added Dense layer with 32 units")
    cnn = Dropout(0.3)(cnn)
    print("  - Added Dropout layer (0.3)")
    
    # 2. LSTM Branch
    print("\nLSTM branch: Capturing sequence memory")
    lstm = LSTM(32, return_sequences=False)(inputs)
    print("  - Added LSTM layer with 32 units")
    lstm = Dense(32, activation='relu')(lstm)
    print("  - Added Dense layer with 32 units")
    lstm = Dropout(0.3)(lstm)
    print("  - Added Dropout layer (0.3)")
    
    # Merge branches
    print("\nFusion: Combining CNN and LSTM branches")
    merged = Concatenate()([cnn, lstm])
    print("  - Concatenated CNN and LSTM outputs")
    
    # Dense layers
    dense = Dense(64, activation='relu')(merged)
    print("  - Added Dense layer with 64 units")
    dense = BatchNormalization()(dense)
    print("  - Added BatchNormalization layer")
    dense = Dropout(0.5)(dense)
    print("  - Added Dropout layer (0.5)")
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(dense)
    print("  - Added Output layer with softmax activation")
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\nModel compiled with Adam optimizer and categorical_crossentropy loss")
    
    # Print model summary
    print("\nModel Architecture Summary:")
    model.summary(print_fn=print)
    
    return model

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training progress"""
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1} started at {datetime.now().strftime('%H:%M:%S')}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"  Training accuracy: {logs['accuracy']:.4f}")
        print(f"  Validation accuracy: {logs['val_accuracy']:.4f}")
        print(f"  Training loss: {logs['loss']:.4f}")
        print(f"  Validation loss: {logs['val_loss']:.4f}")
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            print(f"  Batch {batch}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, pair):
    """Train the hybrid model with progress updates"""
    # Print training start
    print(f"\nStarting model training for {pair}...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create callbacks
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_quick_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    progress_callback = TrainingProgressCallback()
    
    # Track training start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, progress_callback],
        verbose=0  # Turn off default verbosity
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    # Load best model
    print(f"Loading best model from {checkpoint_path}")
    model = load_model(checkpoint_path)
    
    # Return trained model and history
    return model, history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with detailed metrics"""
    print(f"\nEvaluating model on test set...")
    print(f"Test samples: {len(X_test)}")
    
    # Get model predictions
    print("Getting model predictions...")
    y_pred = model.predict(X_test)
    print("Predictions complete")
    
    # Convert predictions and test labels to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test_classes)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Calculate class distribution
    class_counts = np.bincount(y_pred_classes, minlength=3)
    class_names = ["Bearish (-1)", "Neutral (0)", "Bullish (1)"]
    print("\nPrediction distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_counts[i]} samples ({class_counts[i]/len(y_pred_classes):.2%})")
    
    # Convert to trading signals (-1, 0, 1)
    y_pred_signals = y_pred_classes - 1
    y_test_signals = y_test_classes - 1
    
    # Calculate trading metrics (excluding neutral predictions)
    non_neutral_idx = (y_pred_signals != 0) & (y_test_signals != 0)
    if np.sum(non_neutral_idx) > 0:
        trading_accuracy = np.mean(y_pred_signals[non_neutral_idx] == y_test_signals[non_neutral_idx])
        print(f"\nTrading accuracy (excluding neutral): {trading_accuracy:.4f}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'class_distribution': {
            'bearish': class_counts[0] / len(y_pred_classes),
            'neutral': class_counts[1] / len(y_pred_classes),
            'bullish': class_counts[2] / len(y_pred_classes)
        }
    }

def update_ml_config(pair, model_path, metrics, max_portfolio_risk=0.25):
    """Update ML configuration for the trained model"""
    print(f"\nUpdating ML configuration for {pair}...")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            print(f"Loaded existing ML configuration with {len(config.get('models', {}))} models")
        except Exception as e:
            print(f"Error loading ML configuration: {e}")
            config = {"models": {}, "global_settings": {}}
    else:
        print("Creating new ML configuration")
        config = {"models": {}, "global_settings": {}}
    
    # Update global settings
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    config["global_settings"]["base_leverage"] = 5.0
    config["global_settings"]["max_leverage"] = 75.0
    config["global_settings"]["confidence_threshold"] = 0.65
    config["global_settings"]["risk_percentage"] = 0.20
    config["global_settings"]["max_portfolio_risk"] = max_portfolio_risk
    
    # Update model config
    if "models" not in config:
        config["models"] = {}
    
    # Add or update model config for this pair
    config["models"][pair] = {
        "model_type": "hybrid_quick",
        "model_path": model_path,
        "accuracy": metrics.get("accuracy", 0),
        "win_rate": 0.5,  # Placeholder
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Updated ML configuration saved to {ML_CONFIG_PATH}")
    except Exception as e:
        print(f"Error saving ML configuration: {e}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "=" * 80)
    print("HYBRID MODEL TRAINING WITH PROGRESS UPDATES")
    print("=" * 80)
    print(f"Pair: {args.pair}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    print("=" * 80 + "\n")
    
    # Track start time
    start_time = time.time()
    
    # Generate sample data for quick demonstration
    print("Generating sample data for demonstration...")
    X_train, y_train, X_val, y_val, X_test, y_test = generate_sample_data(
        num_samples=1000,
        features=10,
        sequence_length=args.sequence_length
    )
    
    # Get input shape
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # Build model
    model = build_simplified_hybrid_model(input_shape)
    
    # Train model
    model, history, model_path = train_model(
        model, X_train, y_train, X_val, y_val,
        args.epochs, args.batch_size, args.pair
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Update ML config
    update_ml_config(args.pair, model_path, metrics, args.max_portfolio_risk)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    print("\nTraining demonstration completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()