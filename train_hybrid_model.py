#!/usr/bin/env python3
"""
Hybrid Model Training for Crypto Trading

This script implements a hybrid model architecture combining:
1. CNN branch for local price patterns
2. LSTM branch for sequence memory
3. TCN branch for temporal dynamics
4. Meta-learner to combine outputs

Based on the training roadmap, this creates an advanced model for capturing
various market dynamics and improving prediction accuracy.

Usage:
    python train_hybrid_model.py --pair BTC/USD --epochs 50 --batch_size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GRU, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Try to import TCN, install if not available
try:
    from tcn import TCN
    tcn_available = True
except ImportError:
    print("TCN package not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
    from tcn import TCN
    tcn_available = True

# Configure paths
DATA_DIR = "data"
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"

# Create required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train hybrid ML model for crypto trading")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Validation size from training data")
    return parser.parse_args()

def load_historical_data(pair, timeframe):
    """Load historical data for a trading pair"""
    # Convert pair format for filename (e.g., BTC/USD -> btc_usd)
    pair_filename = pair.replace("/", "_").lower()
    
    # Define potential file paths
    csv_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.csv"
    json_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.json"
    
    # Try to load from CSV
    if os.path.exists(csv_path):
        print(f"Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    
    # Try to load from JSON
    elif os.path.exists(json_path):
        print(f"Loading historical data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    else:
        print(f"No historical data found for {pair} {timeframe}")
        return None

def prepare_data(df, sequence_length, test_size, validation_size):
    """Prepare training data with technical indicators and features"""
    # Ensure df has expected columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        # Try lowercase column names
        df.columns = [col.lower() for col in df.columns]
    
    # If still missing required columns, try to adapt
    if 'open' not in df.columns and 'price' in df.columns:
        df['open'] = df['price']
        df['high'] = df['price']
        df['low'] = df['price']
        df['close'] = df['price']
    
    if 'volume' not in df.columns:
        df['volume'] = 0  # Default to zero if no volume data
    
    # Sort by time/timestamp if available
    if 'time' in df.columns:
        df.sort_values('time', inplace=True)
    elif 'timestamp' in df.columns:
        df.sort_values('timestamp', inplace=True)
    
    # Calculate technical indicators
    # 1. Moving Averages
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 5. ATR (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # 6. Volume-based indicators
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 7. Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(10)
    
    # 8. Volatility
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # 9. Slope of EMA
    df['ema_20_slope'] = df['ema_20'].diff(3) / 3
    df['ema_50_slope'] = df['ema_50'].diff(5) / 5
    
    # 10. Candlestick features
    df['body_size'] = abs(df['close'] - df['open'])
    df['wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
    df['wick_lower'] = np.minimum(df['open'], df['close']) - df['low']
    
    # Drop rows with NaN values (due to indicators calculation)
    df.dropna(inplace=True)
    
    # Create target variable (next period price direction)
    # 1 if price increases by at least 0.5%, -1 if it decreases by at least 0.5%, 0 otherwise
    price_shift = 1  # Predict next period
    threshold = 0.005  # 0.5% threshold
    future_return = df['close'].shift(-price_shift) / df['close'] - 1
    df['target'] = 0
    df.loc[future_return > threshold, 'target'] = 1
    df.loc[future_return < -threshold, 'target'] = -1
    
    # Drop last rows where target is NaN
    df.dropna(inplace=True)
    
    # Select features
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'ema_9', 'ema_20', 'ema_50', 'ema_200',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
        'atr_14', 'volume_ratio', 'price_momentum', 'volatility',
        'ema_20_slope', 'ema_50_slope', 
        'body_size', 'wick_upper', 'wick_lower'
    ]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Get targets
    targets = df['target'].values
    
    # Create sequences (e.g., 60 bars of data for each prediction)
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # Add 1 to shift from [-1,0,1] to [0,1,2]
    
    # Split into train and test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_categorical[:split_idx], y_categorical[split_idx:]
    
    # Further split train into train and validation
    train_split_idx = int(len(X_train) * (1 - validation_size))
    X_train, X_val = X_train[:train_split_idx], X_train[train_split_idx:]
    y_train, y_val = y_train[:train_split_idx], y_train[train_split_idx:]
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def build_hybrid_model(input_shape, output_shape):
    """Build a hybrid model with CNN, LSTM, and TCN branches"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(64, activation='relu')(cnn)
    cnn = Dropout(0.3)(cnn)
    
    # 2. LSTM Branch
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64)(lstm)
    lstm = Dense(64, activation='relu')(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # 3. TCN Branch (if available)
    if tcn_available:
        tcn = TCN(nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8], 
                  return_sequences=False, activation='relu')(inputs)
        tcn = Dense(64, activation='relu')(tcn)
        tcn = Dropout(0.3)(tcn)
        
        # Merge branches
        merged = Concatenate()([cnn, lstm, tcn])
    else:
        # Merge without TCN if not available
        merged = Concatenate()([cnn, lstm])
    
    # Meta-learner (Dense layers)
    meta = Dense(128, activation='relu')(merged)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.5)(meta)
    meta = Dense(64, activation='relu')(meta)
    meta = BatchNormalization()(meta)
    meta = Dropout(0.3)(meta)
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(meta)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, pair):
    """Train the hybrid model"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create callbacks
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    return history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Convert back to original labels (-1, 0, 1)
    y_pred_labels = y_pred_classes - 1
    y_test_labels = y_test_classes - 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    # Calculate additional trading-specific metrics
    signal_counts = np.bincount(y_pred_classes, minlength=3)
    win_rate = np.sum((y_pred_labels == y_test_labels) & (y_pred_labels != 0)) / np.sum(y_pred_labels != 0) if np.sum(y_pred_labels != 0) > 0 else 0
    
    # Print evaluation
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Win Rate: {win_rate:.4f}")
    print("\nSignal Distribution:")
    print(f"Bearish (-1): {signal_counts[0]} ({signal_counts[0]/len(y_pred_classes):.2%})")
    print(f"Neutral (0): {signal_counts[1]} ({signal_counts[1]/len(y_pred_classes):.2%})")
    print(f"Bullish (1): {signal_counts[2]} ({signal_counts[2]/len(y_pred_classes):.2%})")
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'win_rate': win_rate,
        'signal_distribution': {
            'bearish': signal_counts[0] / len(y_pred_classes),
            'neutral': signal_counts[1] / len(y_pred_classes),
            'bullish': signal_counts[2] / len(y_pred_classes)
        }
    }
    
    return metrics

def plot_training_history(history, pair):
    """Plot training history"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{pair} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{pair} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Save plot
    plt.tight_layout()
    plot_path = f"{RESULTS_DIR}/hybrid_{pair_clean}_training_history.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")

def save_model_summary(model, pair, metrics):
    """Save model summary and metrics"""
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Capture model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = '\n'.join(model_summary)
    
    # Create summary text
    summary = f"""
Hybrid Model Summary for {pair}
==============================
{model_summary}

Model Evaluation Metrics
------------------------
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1']:.4f}
Win Rate: {metrics['win_rate']:.4f}

Signal Distribution
------------------
Bearish (-1): {metrics['signal_distribution']['bearish']:.2%}
Neutral (0): {metrics['signal_distribution']['neutral']:.2%}
Bullish (1): {metrics['signal_distribution']['bullish']:.2%}

Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    # Save summary to file
    summary_path = f"{RESULTS_DIR}/hybrid_{pair_clean}_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Model summary saved to {summary_path}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load and prepare data
    df = load_historical_data(args.pair, args.timeframe)
    if df is None or len(df) < 1000:
        print(f"Insufficient historical data for {args.pair}")
        return
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(
        df,
        args.sequence_length,
        args.test_size,
        args.validation_size
    )
    
    # Build model
    input_shape = (args.sequence_length, X_train.shape[2])
    output_shape = y_train.shape[1]
    model = build_hybrid_model(input_shape, output_shape)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print(f"\nTraining model for {args.pair}...")
    history, model_path = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        args.epochs,
        args.batch_size,
        args.pair
    )
    
    # Load best model for evaluation
    best_model = load_model(model_path, custom_objects={'TCN': TCN} if tcn_available else None)
    
    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history, args.pair)
    
    # Save model summary
    save_model_summary(best_model, args.pair, metrics)
    
    print(f"\nModel training complete for {args.pair}")
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in hybrid model training: {e}")
        raise