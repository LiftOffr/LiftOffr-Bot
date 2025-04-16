#!/usr/bin/env python3
"""
Train Hybrid Model for One Cryptocurrency Pair

This script trains a hybrid model for a single cryptocurrency pair with:
1. Reduced epochs (10)
2. Simplified architecture for faster training
3. Saving model and results

Usage:
    python train_one_crypto_pair.py --pair BTC/USD --epochs 10
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
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
TRADING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [DATA_DIR, HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train hybrid model for one cryptocurrency pair")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help="Trading pair to train model for (e.g., BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train/test split")
    parser.add_argument("--validation_size", type=float, default=0.2,
                        help="Validation size from training data")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    return parser.parse_args()

def generate_dummy_data(pair, timeframe, num_samples=1000):
    """Generate dummy data for training if historical data is not available"""
    logger.info(f"Generating dummy data for {pair} ({timeframe})")
    
    # Base price depends on the cryptocurrency
    if "BTC" in pair:
        base_price = 50000
    elif "ETH" in pair:
        base_price = 3000
    elif "SOL" in pair:
        base_price = 150
    else:
        base_price = 100
    
    # Generate timestamps
    timestamps = [datetime.now().timestamp() - i * 3600 for i in range(num_samples)]
    timestamps.reverse()
    
    # Generate price data with trend and noise
    trend = np.linspace(0, 0.2, num_samples)
    noise = np.random.normal(0, 0.01, num_samples)
    prices = base_price * (1 + trend + noise)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.002, num_samples)),
        'high': prices * (1 + np.random.normal(0, 0.005, num_samples)),
        'low': prices * (1 - np.random.normal(0, 0.005, num_samples)),
        'close': prices,
        'volume': np.random.normal(1000, 100, num_samples)
    })
    
    return df

def load_or_generate_data(pair, timeframe):
    """Load historical data or generate dummy data if not available"""
    # Convert pair format for filename (e.g., BTC/USD -> btc_usd)
    pair_filename = pair.replace("/", "_").lower()
    
    # Define potential file paths
    csv_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.csv"
    json_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.json"
    
    # Try to load from CSV
    if os.path.exists(csv_path):
        logger.info(f"Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    
    # Try to load from JSON
    elif os.path.exists(json_path):
        logger.info(f"Loading historical data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    else:
        logger.warning(f"No historical data found for {pair} {timeframe}")
        # Generate dummy data
        df = generate_dummy_data(pair, timeframe)
        # Save generated data
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved generated data to {csv_path}")
        return df

def calculate_indicators(df):
    """Calculate technical indicators for the dataset"""
    # Make a copy to avoid modifying the original dataframe
    df_indicators = df.copy()
    
    # Ensure df has expected columns in lowercase
    df_indicators.columns = [col.lower() for col in df_indicators.columns]
    
    # If 'open' not in columns but 'price' is, use price for all OHLC
    if 'open' not in df_indicators.columns and 'price' in df_indicators.columns:
        df_indicators['open'] = df_indicators['price']
        df_indicators['high'] = df_indicators['price']
        df_indicators['low'] = df_indicators['price']
        df_indicators['close'] = df_indicators['price']
    
    # If 'volume' not in columns, add it with zeros
    if 'volume' not in df_indicators.columns:
        df_indicators['volume'] = 0
    
    # Sort by time/timestamp if available
    if 'time' in df_indicators.columns:
        df_indicators.sort_values('time', inplace=True)
    elif 'timestamp' in df_indicators.columns:
        df_indicators.sort_values('timestamp', inplace=True)
    
    # Calculate technical indicators
    # 1. Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df_indicators[f'sma_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'ema_{period}'] = df_indicators['close'].ewm(span=period, adjust=False).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df_indicators['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for period in [6, 14, 20]:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    ema_12 = df_indicators['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_indicators['close'].ewm(span=26, adjust=False).mean()
    df_indicators['macd'] = ema_12 - ema_26
    df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
    df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']
    
    # 4. Bollinger Bands
    for period in [20]:
        df_indicators[f'bb_middle_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'bb_std_{period}'] = df_indicators['close'].rolling(window=period).std()
        df_indicators[f'bb_upper_{period}'] = df_indicators[f'bb_middle_{period}'] + (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_lower_{period}'] = df_indicators[f'bb_middle_{period}'] - (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_width_{period}'] = (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}']) / df_indicators[f'bb_middle_{period}']
    
    # 5. ATR (Average True Range)
    tr1 = df_indicators['high'] - df_indicators['low']
    tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
    tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_indicators['atr_14'] = tr.rolling(window=14).mean()
    
    # 6. Price Momentum
    for period in [10, 20]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
    
    # 7. Volatility
    df_indicators['volatility_20'] = df_indicators['close'].rolling(window=20).std() / df_indicators['close'].rolling(window=20).mean() * 100
    
    # 8. Distance from Moving Averages
    for period in [20, 50]:
        df_indicators[f'dist_from_sma_{period}'] = (df_indicators['close'] - df_indicators[f'sma_{period}']) / df_indicators[f'sma_{period}'] * 100
    
    # 9. Volume-based indicators
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_ma_20'].replace(0, 0.001)
    
    # 10. Candlestick features
    df_indicators['body_size'] = abs(df_indicators['close'] - df_indicators['open'])
    df_indicators['upper_shadow'] = df_indicators['high'] - np.maximum(df_indicators['open'], df_indicators['close'])
    df_indicators['lower_shadow'] = np.minimum(df_indicators['open'], df_indicators['close']) - df_indicators['low']
    
    # Drop rows with NaN values (due to indicators calculation)
    df_indicators.dropna(inplace=True)
    
    return df_indicators

def create_target_variable(df, price_shift=1, threshold=0.005):
    """Create target variable for ML model training"""
    # Calculate future returns
    future_return = df['close'].shift(-price_shift) / df['close'] - 1
    
    # Create target labels: 1 (up), 0 (neutral), -1 (down)
    df['target'] = 0
    df.loc[future_return > threshold, 'target'] = 1
    df.loc[future_return < -threshold, 'target'] = -1
    
    # Drop rows with NaN values (last rows where target couldn't be calculated)
    df.dropna(subset=['target'], inplace=True)
    
    return df

def prepare_data_for_training(df, sequence_length, test_size, validation_size):
    """Prepare data for training, validation, and testing"""
    # Select features and target
    feature_columns = [col for col in df.columns if col not in ['time', 'timestamp', 'target']]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Get targets
    targets = df['target'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # Add 1 to shift from [-1,0,1] to [0,1,2]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, shuffle=False
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, shuffle=False
    )
    
    logger.info(f"Data shapes:")
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns

def build_simplified_hybrid_model(input_shape, output_shape=3):
    """Build a simplified hybrid model for faster training"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(32, activation='relu')(cnn)
    
    # 2. LSTM Branch
    lstm = LSTM(32, return_sequences=False)(inputs)
    lstm = Dense(32, activation='relu')(lstm)
    
    # Merge branches
    merged = Concatenate()([cnn, lstm])
    
    # Dense layers
    dense = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.3)(dense)
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(dropout)
    
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
    """Train the model"""
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
        patience=5,
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
    
    # Calculate trading-specific metrics
    signal_counts = np.bincount(y_pred_classes, minlength=3)
    
    # Calculate win rate (excluding neutral predictions)
    non_neutral_mask = (y_pred_labels != 0) & (y_test_labels != 0)
    win_rate = np.mean(y_pred_labels[non_neutral_mask] == y_test_labels[non_neutral_mask]) if np.sum(non_neutral_mask) > 0 else 0
    
    # Print evaluation
    logger.info("\nModel Evaluation:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Win Rate: {win_rate:.4f}")
    logger.info("\nSignal Distribution:")
    logger.info(f"Bearish (-1): {signal_counts[0]} ({signal_counts[0]/len(y_pred_classes):.2%})")
    logger.info(f"Neutral (0): {signal_counts[1]} ({signal_counts[1]/len(y_pred_classes):.2%})")
    logger.info(f"Bullish (1): {signal_counts[2]} ({signal_counts[2]/len(y_pred_classes):.2%})")
    
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
    
    logger.info(f"Training history plot saved to {plot_path}")

def update_ml_config(pair, model_path, metrics, max_portfolio_risk=0.25):
    """Update ML configuration for the pair"""
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML config: {e}")
            config = {"models": {}, "global_settings": {}}
    else:
        config = {"models": {}, "global_settings": {}}
    
    # Update global settings if not present
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    # Set global settings
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
        "model_type": "hybrid",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "win_rate": metrics["win_rate"],
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
        logger.info(f"Updated ML config for {pair}")
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")

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
    
    logger.info(f"Model summary saved to {summary_path}")

def train_pair(pair, args):
    """Train a hybrid model for a single pair"""
    logger.info(f"\n{'='*80}\nTraining hybrid model for {pair}\n{'='*80}")
    
    # Load data
    df = load_or_generate_data(pair, args.timeframe)
    
    # Calculate indicators
    df_indicators = calculate_indicators(df)
    
    # Create target variable
    df_labeled = create_target_variable(df_indicators)
    
    # Prepare data for training
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, features = prepare_data_for_training(
        df_labeled, args.sequence_length, args.test_size, args.validation_size
    )
    
    # Build model
    input_shape = (args.sequence_length, len(features))
    model = build_simplified_hybrid_model(input_shape)
    
    # Train model
    history, model_path = train_model(
        model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size, pair
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history, pair)
    
    # Save model summary
    save_model_summary(model, pair, metrics)
    
    # Update ML config
    update_ml_config(pair, model_path, metrics, args.max_portfolio_risk)
    
    logger.info(f"Training complete for {pair}\n")
    
    return model_path, metrics

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Train model for specified pair
    model_path, metrics = train_pair(args.pair, args)
    
    logger.info(f"Model trained and saved to {model_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Win Rate: {metrics['win_rate']:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()