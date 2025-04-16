#!/usr/bin/env python3
"""
Train Hybrid Model with Faster Progress

This script trains a simplified version of the hybrid model designed to run
quickly even on CPU-only environments. It provides:
1. Fewer parameters for lighter computation load
2. Smaller training dataset for quick demonstration
3. Regular progress updates with detailed statistics
4. Automatic integration with the trading system

Usage:
    python train_with_faster_progress.py --pair BTC/USD --epochs 3
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
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")
else:
    print("Using CPU for training - this will be slower")

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
    parser = argparse.ArgumentParser(description="Train hybrid model with faster progress")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=40,
                        help="Sequence length for time series")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--smaller_model", action="store_true", default=True,
                        help="Use smaller model architecture for faster training")
    parser.add_argument("--add_to_config", action="store_true", default=True,
                        help="Add trained model to ML configuration")
    return parser.parse_args()

def generate_sample_data(num_samples=800, features=15, sequence_length=40):
    """Generate sample data for quick demonstration"""
    print(f"Generating sample data with {num_samples} samples, {features} features, and sequence length {sequence_length}")
    
    # Generate timestamps
    timestamps = np.arange(num_samples)
    
    # Generate price data (with a clear uptrend and some reversal patterns)
    base_price = 50000  # Starting price
    
    # Create trend with some cycles
    t = np.linspace(0, 8*np.pi, num_samples)
    trend = 0.3 * np.exp(0.001 * t) + 0.1 * np.sin(t/4)
    
    # Add some noise
    noise = np.random.normal(0, 0.005, num_samples)
    
    # Create price series
    prices = base_price * (1 + trend + noise)
    
    # Create price DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.002, num_samples)),
        'high': prices * (1 + np.random.normal(0, 0.004, num_samples)),
        'low': prices * (1 - np.random.normal(0, 0.004, num_samples)),
        'close': prices,
        'volume': np.random.normal(1000, 100, num_samples)
    })
    
    # Add some basic technical indicators
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi_14'] = 50 + 25*np.sin(t/2) + np.random.normal(0, 5, num_samples)  # Simplified RSI
    df['atr_14'] = 0.02 * prices + np.random.normal(0, 50, num_samples)  # Simplified ATR
    df['macd'] = df['sma_5'] - df['sma_20']
    df['macd_signal'] = df['macd'].rolling(window=9).mean()
    
    # Add momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility
    df['volatility'] = df['high'] / df['low'] - 1
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    
    # Create target variable (next period price direction with 5 classes)
    future_return = df['close'].shift(-1) / df['close'] - 1
    
    # 5-class target: strong down, moderate down, neutral, moderate up, strong up
    df['target'] = 2  # neutral by default
    df.loc[future_return > 0.005, 'target'] = 3  # moderate up
    df.loc[future_return > 0.01, 'target'] = 4  # strong up
    df.loc[future_return < -0.005, 'target'] = 1  # moderate down
    df.loc[future_return < -0.01, 'target'] = 0  # strong down
    
    # Drop last row where target is NaN
    df.dropna(inplace=True)
    
    # Print class distribution
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_counts = df['target'].value_counts().sort_index()
    print("\nClass distribution in generated data:")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        pct = count / len(df) * 100
        print(f"  {name}: {count} samples ({pct:.1f}%)")
    
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
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=5)  # 5 classes for improved model
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_fast_hybrid_model(input_shape, output_shape=5, smaller=True):
    """Build a fast hybrid model for quicker demonstration"""
    print("\nBuilding hybrid model architecture...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    if smaller:
        # Smaller model for faster training
        print("Using smaller model architecture for faster training")
        
        # 1. CNN Branch
        cnn = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
        print("  - Added Conv1D layer with 16 filters")
        cnn = MaxPooling1D(pool_size=2)(cnn)
        print("  - Added MaxPooling1D layer")
        cnn = Flatten()(cnn)
        cnn = Dense(16, activation='relu')(cnn)
        print("  - Added Dense layer with 16 units")
        cnn = Dropout(0.2)(cnn)
        
        # 2. LSTM Branch
        lstm = LSTM(16, return_sequences=False)(inputs)
        print("  - Added LSTM layer with 16 units")
        lstm = Dense(16, activation='relu')(lstm)
        print("  - Added Dense layer with 16 units")
        lstm = Dropout(0.2)(lstm)
        
        # Merge branches
        merged = Concatenate()([cnn, lstm])
        print("  - Concatenated CNN and LSTM branches")
        
        # Dense layer
        dense = Dense(32, activation='relu')(merged)
        print("  - Added Dense layer with 32 units")
        dense = Dropout(0.3)(dense)
        
    else:
        # Standard model (still optimized for faster training)
        print("Using standard model architecture")
        
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
        
        # 2. LSTM Branch
        lstm = LSTM(32, return_sequences=True)(inputs)
        print("  - Added LSTM layer with 32 units (return sequences)")
        lstm = LSTM(32)(lstm)
        print("  - Added LSTM layer with 32 units")
        lstm = Dense(32, activation='relu')(lstm)
        print("  - Added Dense layer with 32 units")
        lstm = Dropout(0.3)(lstm)
        
        # Merge branches
        merged = Concatenate()([cnn, lstm])
        print("  - Concatenated CNN and LSTM branches")
        
        # Dense layers
        dense = Dense(64, activation='relu')(merged)
        print("  - Added Dense layer with 64 units")
        dense = BatchNormalization()(dense)
        print("  - Added BatchNormalization layer")
        dense = Dropout(0.5)(dense)
        
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

class FastTrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback for faster training with progress updates"""
    def __init__(self, epochs):
        super(FastTrainingCallback, self).__init__()
        self.epochs = epochs
        self.start_time = None
        self.epoch_start_time = None
        self.batch_times = []
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\nTraining started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total epochs: {self.epochs}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []
        print(f"\nEpoch {epoch+1}/{self.epochs} started at {datetime.now().strftime('%H:%M:%S')}")
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % 5 == 0:  # Print every 5 batches
            batch_time = time.time() - (self.epoch_start_time + sum(self.batch_times))
            self.batch_times.append(batch_time)
            
            # Get logs
            loss = logs.get('loss', 0)
            accuracy = logs.get('accuracy', 0)
            
            print(f"  Batch {batch}: loss={loss:.4f}, accuracy={accuracy:.4f}, time={batch_time:.2f}s")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        elapsed_time = time.time() - self.start_time
        remaining_time = (epoch_time * (self.epochs - epoch - 1))
        
        print(f"\nEpoch {epoch+1}/{self.epochs} completed in {epoch_time:.2f}s")
        print(f"  Training accuracy: {logs['accuracy']:.4f}")
        print(f"  Validation accuracy: {logs['val_accuracy']:.4f}")
        print(f"  Training loss: {logs['loss']:.4f}")
        print(f"  Validation loss: {logs['val_loss']:.4f}")
        
        minutes, seconds = divmod(int(elapsed_time), 60)
        hours, minutes = divmod(minutes, 60)
        print(f"  Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        minutes, seconds = divmod(int(remaining_time), 60)
        hours, minutes = divmod(minutes, 60)
        print(f"  Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Progress bar
        progress = (epoch + 1) / self.epochs
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"  Progress: [{bar}] {progress:.1%}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        minutes, seconds = divmod(int(total_time), 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\nTraining completed in {hours:02d}:{minutes:02d}:{seconds:02d}")

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, pair):
    """Train the model with fast progress updates"""
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
    
    fast_callback = FastTrainingCallback(epochs)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, fast_callback],
        verbose=0  # Turn off default verbosity
    )
    
    # Load best model
    print(f"\nLoading best model from {checkpoint_path}")
    try:
        model = load_model(checkpoint_path)
        print("Best model loaded successfully")
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using last model state")
    
    return model, history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with detailed metrics"""
    print(f"\nEvaluating model on test set...")
    print(f"Test samples: {len(X_test)}")
    
    # Get model predictions
    print("Getting model predictions...")
    y_pred = model.predict(X_test, verbose=0)
    print("Predictions complete")
    
    # Convert predictions and test labels to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test_classes)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Calculate class distribution
    class_counts = np.bincount(y_pred_classes, minlength=5)
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    print("\nPrediction distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_counts[i]} samples ({class_counts[i]/len(y_pred_classes):.2%})")
    
    # Calculate confusion matrix
    confusion_matrix = np.zeros((5, 5), dtype=int)
    for i in range(len(y_test_classes)):
        confusion_matrix[y_test_classes[i], y_pred_classes[i]] += 1
    
    print("\nConfusion Matrix:")
    print("  Predicted →")
    print("  ↓ Actual | Strong Down | Moderate Down | Neutral | Moderate Up | Strong Up")
    print("  ---------|-------------|---------------|---------|-------------|----------")
    for i, name in enumerate(class_names):
        row = [confusion_matrix[i, j] for j in range(5)]
        print(f"  {name:11}| {row[0]:11d} | {row[1]:13d} | {row[2]:7d} | {row[3]:11d} | {row[4]:9d}")
    
    # Calculate class-wise accuracy
    class_accuracy = {}
    for i, name in enumerate(class_names):
        if sum(confusion_matrix[i, :]) > 0:
            class_accuracy[name] = confusion_matrix[i, i] / sum(confusion_matrix[i, :])
        else:
            class_accuracy[name] = 0
    
    print("\nClass-wise Accuracy:")
    for name, acc in class_accuracy.items():
        print(f"  {name}: {acc:.4f}")
    
    # Calculate trading direction accuracy
    directional_classes = {
        'bearish': [0, 1],  # Strong down, moderate down
        'neutral': [2],     # Neutral
        'bullish': [3, 4]   # Moderate up, strong up
    }
    
    y_test_direction = np.zeros(len(y_test_classes), dtype=int)
    y_pred_direction = np.zeros(len(y_pred_classes), dtype=int)
    
    for direction, classes in directional_classes.items():
        for cls in classes:
            if direction == 'bearish':
                y_test_direction[y_test_classes == cls] = -1
                y_pred_direction[y_pred_classes == cls] = -1
            elif direction == 'bullish':
                y_test_direction[y_test_classes == cls] = 1
                y_pred_direction[y_pred_classes == cls] = 1
            # Neutral remains 0
    
    # Calculate direction accuracy (excluding neutrals)
    non_neutral_mask = (y_test_direction != 0) & (y_pred_direction != 0)
    if np.sum(non_neutral_mask) > 0:
        direction_accuracy = np.mean(y_test_direction[non_neutral_mask] == y_pred_direction[non_neutral_mask])
        print(f"\nTrade Direction Accuracy (excluding neutral): {direction_accuracy:.4f}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'direction_accuracy': direction_accuracy if 'direction_accuracy' in locals() else 0,
        'class_distribution': {
            class_names[i]: class_counts[i] / len(y_pred_classes) for i in range(5)
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
        "direction_accuracy": metrics.get("direction_accuracy", 0),
        "win_rate": 0.6,  # Placeholder
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
        return True
    except Exception as e:
        print(f"Error saving ML configuration: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "=" * 80)
    print("HYBRID MODEL TRAINING WITH FASTER PROGRESS")
    print("=" * 80)
    print(f"Pair: {args.pair}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    print(f"Smaller Model: {args.smaller_model}")
    print("=" * 80 + "\n")
    
    # Track start time
    start_time = time.time()
    
    try:
        # Generate sample data for quick demonstration
        print("Preparing data for training...")
        X_train, y_train, X_val, y_val, X_test, y_test = generate_sample_data(
            num_samples=800,
            features=15,
            sequence_length=args.sequence_length
        )
        
        # Get input shape
        input_shape = X_train.shape[1:]
        print(f"Input shape: {input_shape}")
        
        # Build model
        model = build_fast_hybrid_model(input_shape, smaller=args.smaller_model)
        
        # Train model
        model, history, model_path = train_model(
            model, X_train, y_train, X_val, y_val,
            args.epochs, args.batch_size, args.pair
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Update ML config if requested
        if args.add_to_config:
            update_success = update_ml_config(args.pair, model_path, metrics, args.max_portfolio_risk)
            if update_success:
                print(f"Successfully added {args.pair} model to ML configuration")
            else:
                print(f"Failed to add {args.pair} model to ML configuration")
        
        # Calculate total time
        total_time = time.time() - start_time
        minutes, seconds = divmod(int(total_time), 60)
        hours, minutes = divmod(minutes, 60)
        
        print(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print("\nTraining completed successfully!")
        
        # Print model path for reference
        print(f"Model saved to: {model_path}")
        
        return 0
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())