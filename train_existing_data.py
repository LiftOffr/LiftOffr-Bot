#!/usr/bin/env python3
"""
Train Models with Existing Data

This script trains the improved hybrid model using existing OHLCV data
and reports comprehensive performance metrics including PnL, win rate,
and Sharpe ratio for each cryptocurrency pair.

Usage:
    python train_existing_data.py --pair BTC/USD
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten,
    Concatenate, BatchNormalization, GlobalAveragePooling1D, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_metrics.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train models with existing data")
    parser.add_argument("--pair", type=str, default="all",
                        help=f"Trading pair to train model for (options: {', '.join(ALL_PAIRS)} or 'all')")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (lower for demonstration)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--generate_sample", action="store_true", default=True,
                        help="Generate sample data if no data exists")
    return parser.parse_args()

def load_historical_data(pair: str) -> Optional[pd.DataFrame]:
    """Load existing historical data for a pair"""
    pair_clean = pair.replace("/", "_").lower()
    
    # Try various potential file paths
    potential_paths = [
        f"{HISTORICAL_DATA_DIR}/{pair_clean}_1h.csv",
        f"{HISTORICAL_DATA_DIR}/{pair_clean}.csv",
        f"data/{pair_clean}_historical.csv",
        f"data/historical_{pair_clean}.csv"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading historical data from {path}")
                df = pd.read_csv(path)
                
                # Check if required columns exist
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.warning(f"Missing required columns in {path}: {missing_columns}")
                    
                    # Try to adapt data format
                    if 'time' in df.columns and 'timestamp' not in df.columns:
                        df['timestamp'] = df['time']
                    
                    if 'price' in df.columns:
                        if 'open' not in df.columns:
                            df['open'] = df['price']
                        if 'high' not in df.columns:
                            df['high'] = df['price']
                        if 'low' not in df.columns:
                            df['low'] = df['price']
                        if 'close' not in df.columns:
                            df['close'] = df['price']
                
                # Check again after adaptation
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"Still missing required columns after adaptation: {missing_columns}")
                    continue
                
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    if pd.api.types.is_numeric_dtype(df['timestamp']):
                        # Convert Unix timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    else:
                        # Try to parse as string
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                logger.info(f"Loaded {len(df)} records for {pair}")
                return df
            
            except Exception as e:
                logger.error(f"Error loading data from {path}: {e}")
    
    logger.warning(f"No existing historical data found for {pair}")
    return None

def generate_sample_data(pair: str, num_samples: int = 1000) -> pd.DataFrame:
    """Generate sample price data for training demonstration"""
    logger.info(f"Generating sample data for {pair} with {num_samples} samples")
    
    # Set base price depending on the cryptocurrency
    if "BTC" in pair:
        base_price = 50000
    elif "ETH" in pair:
        base_price = 3000
    elif "SOL" in pair:
        base_price = 150
    elif "ADA" in pair:
        base_price = 2
    elif "DOT" in pair:
        base_price = 25
    elif "LINK" in pair:
        base_price = 20
    elif "AVAX" in pair:
        base_price = 80
    elif "MATIC" in pair:
        base_price = 1.5
    elif "UNI" in pair:
        base_price = 15
    elif "ATOM" in pair:
        base_price = 30
    else:
        base_price = 100
    
    # Generate timestamps
    now = datetime.now()
    timestamps = [(now - timedelta(hours=i)).timestamp() for i in range(num_samples)]
    timestamps.reverse()
    
    # Generate price data with trends and cycles
    t = np.linspace(0, 8*np.pi, num_samples)
    
    # Create trend with some cycles
    trend = 0.2 * np.exp(0.001 * t) + 0.1 * np.sin(t/4)
    
    # Add some noise
    noise = np.random.normal(0, 0.01, num_samples)
    
    # Create price series
    prices = base_price * (1 + trend + noise)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s'),
        'open': prices * (1 + np.random.normal(0, 0.002, num_samples)),
        'high': prices * (1 + np.random.normal(0, 0.005, num_samples)),
        'low': prices * (1 - np.random.normal(0, 0.005, num_samples)),
        'close': prices,
        'volume': np.random.normal(1000, 100, num_samples) * base_price / 100
    })
    
    # Save to file
    pair_clean = pair.replace("/", "_").lower()
    filename = f"{HISTORICAL_DATA_DIR}/{pair_clean}_1h.csv"
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Saved generated data to {filename}")
    except Exception as e:
        logger.error(f"Error saving generated data: {e}")
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for OHLCV data"""
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying the original
    df_indicators = df.copy()
    
    # Ensure we have the expected columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df_indicators.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
    
    # 1. Moving Averages
    for period in [5, 10, 20, 50, 100]:
        df_indicators[f'sma_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'ema_{period}'] = df_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # Relative to price
        df_indicators[f'sma_{period}_rel'] = (df_indicators['close'] / df_indicators[f'sma_{period}'] - 1) * 100
        df_indicators[f'ema_{period}_rel'] = (df_indicators['close'] / df_indicators[f'ema_{period}'] - 1) * 100
    
    # 2. RSI
    for period in [6, 14, 20]:
        delta = df_indicators['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
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
        df_indicators[f'bb_b_{period}'] = (df_indicators['close'] - df_indicators[f'bb_lower_{period}']) / (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}'] + 0.0001)
    
    # 5. ATR (Average True Range)
    for period in [14]:
        tr1 = df_indicators['high'] - df_indicators['low']
        tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
        tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_indicators[f'atr_{period}'] = tr.rolling(window=period).mean()
        df_indicators[f'atr_percent_{period}'] = df_indicators[f'atr_{period}'] / df_indicators['close'] * 100
    
    # 6. Momentum
    for period in [10, 20]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
    
    # 7. Stochastic Oscillator
    for period in [14]:
        low_min = df_indicators['low'].rolling(window=period).min()
        high_max = df_indicators['high'].rolling(window=period).max()
        df_indicators[f'stoch_k_{period}'] = 100 * ((df_indicators['close'] - low_min) / (high_max - low_min + 0.0001))
        df_indicators[f'stoch_d_{period}'] = df_indicators[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # 8. Volume Indicators
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_ma_20'].replace(0, 0.001)
    
    # 9. Trend Indicators
    df_indicators['bull_market'] = (df_indicators['ema_50'] > df_indicators['ema_100']).astype(int)
    df_indicators['bear_market'] = (df_indicators['ema_50'] < df_indicators['ema_100']).astype(int)
    
    # 10. Volatility Metrics
    df_indicators['volatility_14'] = df_indicators['close'].rolling(window=14).std() / df_indicators['close'] * 100
    
    # Drop rows with NaN values (due to indicators calculation)
    df_indicators.dropna(inplace=True)
    
    logger.info(f"Calculated {len(df_indicators.columns) - len(required_columns)} technical indicators")
    logger.info(f"Data shape after indicator calculation: {df_indicators.shape}")
    
    return df_indicators

def create_target_variable(df: pd.DataFrame, price_shift: int = 1, threshold: float = 0.005) -> pd.DataFrame:
    """Create target variable for ML model training with multiple classes"""
    logger.info("Creating target variable...")
    
    # Calculate future returns
    future_return = df['close'].shift(-price_shift) / df['close'] - 1
    
    # Create target labels: 1 (strong up), 0.5 (moderate up), 0 (neutral), -0.5 (moderate down), -1 (strong down)
    df['target'] = 0
    df.loc[future_return > threshold*2, 'target'] = 1      # Strong bullish
    df.loc[(future_return > threshold) & (future_return <= threshold*2), 'target'] = 0.5  # Moderate bullish
    df.loc[(future_return < -threshold) & (future_return >= -threshold*2), 'target'] = -0.5  # Moderate bearish
    df.loc[future_return < -threshold*2, 'target'] = -1    # Strong bearish
    
    # Drop rows with NaN values (last rows where target couldn't be calculated)
    df.dropna(subset=['target'], inplace=True)
    
    # Convert target to integer classes
    target_map = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
    df['target_class'] = df['target'].map(target_map)
    
    # Log class distribution
    class_counts = df['target_class'].value_counts().sort_index()
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    
    logger.info("Target class distribution:")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        pct = count / len(df) * 100
        logger.info(f"  {name}: {count} samples ({pct:.1f}%)")
    
    return df

def prepare_data_for_training(
    df: pd.DataFrame, 
    sequence_length: int, 
    test_size: float = 0.2, 
    validation_size: float = 0.2
) -> Tuple:
    """Prepare data for training, validation, and testing"""
    logger.info(f"Preparing sequences with length {sequence_length}...")
    
    # Select features and target
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'target_class']]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Get targets (multi-class)
    targets = df['target_class'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert target to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=5)  # 5 classes
    
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

def build_hybrid_model(input_shape: Tuple, output_shape: int = 5) -> Model:
    """Build a hybrid model for CPU-friendly training"""
    logger.info("Building hybrid model...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch
    cnn = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = GlobalAveragePooling1D()(cnn)
    cnn = Dense(32, activation='relu')(cnn)
    cnn = Dropout(0.3)(cnn)
    
    # 2. LSTM Branch
    lstm = LSTM(32, return_sequences=False)(inputs)
    lstm = Dense(32, activation='relu')(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Merge branches
    merged = Concatenate()([cnn, lstm])
    
    # Deep neural network
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer (5 classes)
    outputs = Dense(output_shape, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    return model

def train_model(
    model: Model, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    pair: str,
    epochs: int,
    batch_size: int
) -> Tuple:
    """Train the model"""
    logger.info("Training model...")
    
    # Create clean pair name for file paths
    pair_clean = pair.replace("/", "_").lower()
    
    # Create checkpoint path
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_model.h5"
    
    # Create callbacks
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
    
    # Create logs directory
    logs_dir = f"{RESULTS_DIR}/logs/{pair_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logs_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, tensorboard],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Load the best model
    try:
        model = load_model(checkpoint_path)
        logger.info(f"Loaded best model from {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Could not load best model: {e}. Using final model.")
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    logger.info(f"Training completed in {training_time_str}")
    
    return model, history, checkpoint_path, training_time

def evaluate_model(
    model: Model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    feature_columns: List[str], 
    sequence_length: int,
    pair: str
) -> Dict:
    """Evaluate the trained model with detailed trading metrics"""
    logger.info("Evaluating model...")
    
    # Basic evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    
    # Class mapping
    class_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
    y_pred_signal = np.array([class_map.get(c, 0) for c in y_pred_class])
    y_test_signal = np.array([class_map.get(c, 0) for c in y_test_class])
    
    # Calculate class-wise accuracy
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = y_test_class == i
        if np.sum(class_mask) > 0:
            class_accuracy[name] = np.mean(y_pred_class[class_mask] == i)
        else:
            class_accuracy[name] = 0.0
    
    # Calculate direction accuracy (excluding neutral predictions)
    non_neutral_mask = (y_test_signal != 0) & (y_pred_signal != 0)
    if np.sum(non_neutral_mask) > 0:
        direction_accuracy = np.mean(np.sign(y_pred_signal[non_neutral_mask]) == np.sign(y_test_signal[non_neutral_mask]))
    else:
        direction_accuracy = 0.0
    
    # Simulate trading performance
    trading_metrics = simulate_trading(
        model, 
        df, 
        scaler, 
        feature_columns, 
        sequence_length,
        pair
    )
    
    # Combine metrics
    metrics = {
        "accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "direction_accuracy": direction_accuracy,
        "trading_metrics": trading_metrics
    }
    
    return metrics

def simulate_trading(
    model: Model, 
    df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    feature_columns: List[str], 
    sequence_length: int,
    pair: str
) -> Dict:
    """Simulate trading with the trained model"""
    logger.info("Simulating trading performance...")
    
    # Prepare trading simulation data
    # Use the last part of the data that wasn't used in training
    trading_data = df.iloc[-(len(df)//5):]
    
    # Normalize features
    trading_features = scaler.transform(trading_data[feature_columns])
    
    # Create sequences for prediction
    X_trading = []
    for i in range(sequence_length, len(trading_features)):
        X_trading.append(trading_features[i-sequence_length:i])
    
    X_trading = np.array(X_trading)
    
    # Get predictions
    y_pred_proba = model.predict(X_trading, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # Class mapping
    class_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
    y_pred_signal = np.array([class_map.get(c, 0) for c in y_pred_class])
    
    # Get confidence
    confidence = np.array([y_pred_proba[i, c] for i, c in enumerate(y_pred_class)])
    
    # Create trading signals DataFrame
    trading_signals = pd.DataFrame({
        'timestamp': trading_data.iloc[sequence_length:]['timestamp'].values,
        'close': trading_data.iloc[sequence_length:]['close'].values,
        'signal': y_pred_signal,
        'confidence': confidence
    })
    
    # Simulate trading
    initial_capital = 10000
    risk_per_trade = 0.02  # 2% risk per trade
    max_leverage = 75
    base_leverage = 5
    
    # Add columns for trading simulation
    trading_signals['leverage'] = base_leverage + (max_leverage - base_leverage) * (trading_signals['confidence'] - 0.2) / 0.8
    trading_signals['leverage'] = trading_signals['leverage'].clip(base_leverage, max_leverage)
    trading_signals['leverage'] = trading_signals['leverage'] * np.abs(trading_signals['signal'])
    
    # Set leverage to 0 for neutral signals
    trading_signals.loc[trading_signals['signal'] == 0, 'leverage'] = 0
    
    # Calculate position size
    trading_signals['position'] = trading_signals['signal'] * trading_signals['leverage']
    
    # Calculate returns (excluding the first row)
    trading_signals['return'] = trading_signals['close'].pct_change()
    trading_signals['strategy_return'] = trading_signals['position'].shift(1) * trading_signals['return']
    
    # Apply risk management (2% risk per trade)
    trading_signals['position_size'] = initial_capital * risk_per_trade / trading_signals['leverage'].replace(0, 1)
    
    # Calculate PnL
    trading_signals['pnl'] = trading_signals['strategy_return'] * trading_signals['position_size']
    trading_signals['cumulative_pnl'] = trading_signals['pnl'].cumsum()
    
    # Calculate equity curve
    trading_signals['equity'] = initial_capital + trading_signals['cumulative_pnl']
    
    # Calculate drawdown
    trading_signals['peak'] = trading_signals['equity'].cummax()
    trading_signals['drawdown'] = (trading_signals['equity'] - trading_signals['peak']) / trading_signals['peak']
    
    # Clean up data
    trading_signals = trading_signals.dropna()
    
    # Calculate trading metrics
    if len(trading_signals) > 0:
        # Total return
        total_return = trading_signals['equity'].iloc[-1] / initial_capital - 1
        
        # Annualized return (252 trading days per year)
        try:
            days = (pd.to_datetime(trading_signals['timestamp'].iloc[-1]) - 
                    pd.to_datetime(trading_signals['timestamp'].iloc[0])).total_seconds() / (24 * 60 * 60)
            days = max(1, days)
        except Exception:
            days = len(trading_signals) / 24  # Assuming hourly data
        
        annualized_return = (1 + total_return) ** (252 / max(1, days)) - 1
        
        # Volatility
        daily_returns = trading_signals.groupby(pd.Grouper(key='timestamp', freq='D'))['strategy_return'].sum()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / max(0.0001, volatility)
        
        # Maximum drawdown
        max_drawdown = trading_signals['drawdown'].min()
        
        # Win rate
        trades = []
        current_position = 0
        trade_pnl = 0
        
        for i, row in trading_signals.iterrows():
            if row['position'] != current_position:
                # Position changed, close previous trade
                if current_position != 0:
                    trades.append(trade_pnl)
                    trade_pnl = 0
                
                # Open new position
                current_position = row['position']
            else:
                # Update current trade PnL
                trade_pnl += row['pnl']
        
        # Close last trade
        if current_position != 0:
            trades.append(trade_pnl)
        
        # Calculate win rate
        if len(trades) > 0:
            win_rate = sum(1 for t in trades if t > 0) / len(trades)
            avg_win = sum(t for t in trades if t > 0) / max(1, sum(1 for t in trades if t > 0))
            avg_loss = sum(t for t in trades if t < 0) / max(1, sum(1 for t in trades if t < 0))
            
            # Profit factor
            gross_profit = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t < 0))
            profit_factor = gross_profit / max(0.0001, gross_loss)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Return metrics
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_equity": trading_signals['equity'].iloc[-1],
            "total_pnl": trading_signals['equity'].iloc[-1] - initial_capital
        }
    else:
        metrics = {
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "final_equity": initial_capital,
            "total_pnl": 0
        }
    
    # Save trading signals for analysis
    pair_clean = pair.replace('/', '_').lower()
    signals_file = f"{RESULTS_DIR}/trading_signals_{pair_clean}.csv"
    try:
        trading_signals.to_csv(signals_file, index=False)
        logger.info(f"Saved trading signals to {signals_file}")
    except Exception as e:
        logger.error(f"Error saving trading signals: {e}")
    
    # Log metrics
    logger.info(f"Trading metrics for {pair}:")
    logger.info(f"  Total Return: {metrics['total_return']:.2%}")
    logger.info(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    logger.info(f"  Trades: {metrics['trades']}")
    
    return metrics

def train_pair(pair: str, args) -> Dict:
    """Train model for a specific trading pair"""
    logger.info(f"\nTraining model for {pair}...")
    
    # Load or generate data
    df = load_historical_data(pair)
    
    if df is None or len(df) < 500:
        if args.generate_sample:
            logger.info(f"Generating sample data for {pair}")
            df = generate_sample_data(pair)
        else:
            logger.error(f"No data available for {pair} and sample generation disabled")
            return {"pair": pair, "success": False, "error": "No data available"}
    
    # Calculate indicators
    df_indicators = calculate_indicators(df)
    
    # Create target variable
    df_labeled = create_target_variable(df_indicators)
    
    # Prepare data for training
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns = prepare_data_for_training(
        df_labeled, args.sequence_length, test_size=0.2, validation_size=0.2
    )
    
    # Build model
    input_shape = (args.sequence_length, len(feature_columns))
    model = build_hybrid_model(input_shape)
    
    # Train model
    model, history, model_path, training_time = train_model(
        model, X_train, y_train, X_val, y_val, pair, args.epochs, args.batch_size
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, df_labeled, scaler, feature_columns, args.sequence_length, pair)
    
    # Create result
    result = {
        "pair": pair,
        "success": True,
        "model_path": model_path,
        "training_time": training_time,
        "metrics": metrics
    }
    
    # Save metrics to file
    pair_clean = pair.replace("/", "_").lower()
    metrics_file = f"{RESULTS_DIR}/metrics_{pair_clean}.json"
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
    
    return result

def generate_summary(results: Dict[str, Dict]) -> str:
    """Generate summary of training results"""
    if not results:
        return "No training results"
    
    successful_pairs = [p for p, r in results.items() if r.get("success", False)]
    if not successful_pairs:
        return "No successful training results"
    
    # Create summary
    summary = "\n" + "=" * 80 + "\n"
    summary += "TRAINING SUMMARY\n"
    summary += "=" * 80 + "\n\n"
    
    # Create performance table
    summary += "| Pair | Accuracy | Direction Acc | Win Rate | PnL | Sharpe Ratio |\n"
    summary += "|------|----------|--------------|----------|-----|-------------|\n"
    
    for pair, result in results.items():
        if result.get("success", False):
            metrics = result.get("metrics", {})
            model_acc = metrics.get("accuracy", 0)
            dir_acc = metrics.get("direction_accuracy", 0)
            trading = metrics.get("trading_metrics", {})
            win_rate = trading.get("win_rate", 0)
            pnl = trading.get("total_pnl", 0)
            sharpe = trading.get("sharpe_ratio", 0)
            
            summary += f"| {pair} | {model_acc:.2%} | {dir_acc:.2%} | {win_rate:.2%} | ${pnl:.2f} | {sharpe:.2f} |\n"
    
    summary += "\n"
    
    # Add detailed metrics
    summary += "Detailed Metrics:\n\n"
    
    for pair, result in results.items():
        if result.get("success", False):
            summary += f"{pair}:\n"
            metrics = result.get("metrics", {})
            trading = metrics.get("trading_metrics", {})
            
            summary += f"  - Model Accuracy: {metrics.get('accuracy', 0):.2%}\n"
            summary += f"  - Direction Accuracy: {metrics.get('direction_accuracy', 0):.2%}\n"
            summary += f"  - Win Rate: {trading.get('win_rate', 0):.2%}\n"
            summary += f"  - PnL: ${trading.get('total_pnl', 0):.2f}\n"
            summary += f"  - Sharpe Ratio: {trading.get('sharpe_ratio', 0):.2f}\n"
            summary += f"  - Max Drawdown: {trading.get('max_drawdown', 0):.2%}\n"
            summary += f"  - Trades: {trading.get('trades', 0)}\n"
            summary += f"  - Profit Factor: {trading.get('profit_factor', 0):.2f}\n\n"
    
    return summary

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine pairs to train
    if args.pair.lower() == "all":
        pairs = ALL_PAIRS
    else:
        pairs = [args.pair]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN MODELS WITH EXISTING DATA")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Generate Sample Data: {args.generate_sample}")
    logger.info("=" * 80 + "\n")
    
    # Train models for each pair
    results = {}
    
    for pair in pairs:
        result = train_pair(pair, args)
        results[pair] = result
    
    # Generate summary
    summary = generate_summary(results)
    logger.info(summary)
    
    # Save summary to file
    summary_file = f"{RESULTS_DIR}/training_summary.txt"
    try:
        with open(summary_file, 'w') as f:
            f.write(summary)
        logger.info(f"Saved summary to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    return 0 if any(r.get("success", False) for r in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())