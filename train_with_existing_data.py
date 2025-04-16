#!/usr/bin/env python3
"""
Train Improved Model with Existing Data

This script trains the enhanced hybrid model using existing historical data
and reports comprehensive performance metrics including PnL, win rate, and
Sharpe ratio.

Usage:
    python train_with_existing_data.py --pair BTC/USD --timeframe 1h
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
        logging.FileHandler("existing_data_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
ALL_TIMEFRAMES = ['1h', '4h', '1d']
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train improved model with existing data")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help=f"Trading pair to train model for (options: {', '.join(ALL_PAIRS)})")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help=f"Timeframe to use for training (options: {', '.join(ALL_TIMEFRAMES)})")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--update_ml_config", action="store_true", default=True,
                        help="Update ML configuration with training results")
    return parser.parse_args()

def load_existing_data(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load existing historical data"""
    pair_clean = pair.replace("/", "_")
    
    # Try different filename formats
    potential_files = [
        f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe}.csv",
        f"{HISTORICAL_DATA_DIR}/{pair_clean.upper()}_{timeframe}.csv",
        f"{HISTORICAL_DATA_DIR}/{pair_clean.lower()}_{timeframe}.csv",
        f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe.lower()}.csv",
        f"{HISTORICAL_DATA_DIR}/{pair_clean}_{timeframe.upper()}.csv"
    ]
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            logger.info(f"Loading historical data from {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    if isinstance(df['timestamp'][0], str):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                logger.info(f"Loaded {len(df)} records from {file_path}")
                return df
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
    
    logger.error(f"Could not find historical data for {pair} ({timeframe})")
    return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for OHLCV data"""
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying the original
    df_indicators = df.copy()
    
    # Ensure we have the expected columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df_indicators.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns in data: {missing_columns}")
        
        # Try to accommodate different column formats
        if 'price' in df_indicators.columns and 'close' not in df_indicators.columns:
            df_indicators['close'] = df_indicators['price']
            
        if 'close' in df_indicators.columns and 'open' not in df_indicators.columns:
            df_indicators['open'] = df_indicators['close'].shift(1)
            df_indicators.loc[0, 'open'] = df_indicators.loc[0, 'close']
            
        if 'close' in df_indicators.columns and 'high' not in df_indicators.columns:
            df_indicators['high'] = df_indicators['close']
            
        if 'close' in df_indicators.columns and 'low' not in df_indicators.columns:
            df_indicators['low'] = df_indicators['close']
            
        if 'volume' not in df_indicators.columns:
            df_indicators['volume'] = 0
    
    # 1. Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df_indicators[f'sma_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'ema_{period}'] = df_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # Relative to price
        df_indicators[f'sma_{period}_rel'] = (df_indicators['close'] / df_indicators[f'sma_{period}'] - 1) * 100
        df_indicators[f'ema_{period}_rel'] = (df_indicators['close'] / df_indicators[f'ema_{period}'] - 1) * 100
    
    # 2. RSI
    for period in [6, 14, 20, 50]:
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
    for period in [20, 50]:
        df_indicators[f'bb_middle_{period}'] = df_indicators['close'].rolling(window=period).mean()
        df_indicators[f'bb_std_{period}'] = df_indicators['close'].rolling(window=period).std()
        df_indicators[f'bb_upper_{period}'] = df_indicators[f'bb_middle_{period}'] + (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_lower_{period}'] = df_indicators[f'bb_middle_{period}'] - (df_indicators[f'bb_std_{period}'] * 2)
        df_indicators[f'bb_width_{period}'] = (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}']) / df_indicators[f'bb_middle_{period}']
        df_indicators[f'bb_b_{period}'] = (df_indicators['close'] - df_indicators[f'bb_lower_{period}']) / (df_indicators[f'bb_upper_{period}'] - df_indicators[f'bb_lower_{period}'] + 0.0001)
    
    # 5. ATR (Average True Range)
    for period in [7, 14, 21]:
        tr1 = df_indicators['high'] - df_indicators['low']
        tr2 = abs(df_indicators['high'] - df_indicators['close'].shift())
        tr3 = abs(df_indicators['low'] - df_indicators['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_indicators[f'atr_{period}'] = tr.rolling(window=period).mean()
        df_indicators[f'atr_percent_{period}'] = df_indicators[f'atr_{period}'] / df_indicators['close'] * 100
    
    # 6. Momentum
    for period in [5, 10, 20, 50]:
        df_indicators[f'momentum_{period}'] = df_indicators['close'] - df_indicators['close'].shift(period)
        df_indicators[f'rate_of_change_{period}'] = (df_indicators['close'] / df_indicators['close'].shift(period) - 1) * 100
    
    # 7. Stochastic Oscillator
    for period in [14, 21]:
        low_min = df_indicators['low'].rolling(window=period).min()
        high_max = df_indicators['high'].rolling(window=period).max()
        df_indicators[f'stoch_k_{period}'] = 100 * ((df_indicators['close'] - low_min) / (high_max - low_min + 0.0001))
        df_indicators[f'stoch_d_{period}'] = df_indicators[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # 8. Volume Indicators
    df_indicators['volume_ma_20'] = df_indicators['volume'].rolling(window=20).mean()
    df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_ma_20'].replace(0, 0.001)
    
    # 9. Trend Indicators
    df_indicators['bull_market'] = (df_indicators['ema_50'] > df_indicators['ema_200']).astype(int)
    df_indicators['bear_market'] = (df_indicators['ema_50'] < df_indicators['ema_200']).astype(int)
    
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

def build_enhanced_model(input_shape: Tuple, output_shape: int = 5) -> Model:
    """Build an enhanced hybrid model"""
    logger.info("Building enhanced hybrid model...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1. CNN Branch
    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = GlobalAveragePooling1D()(cnn)
    cnn = Dense(64, activation='relu')(cnn)
    cnn = Dropout(0.3)(cnn)
    
    # 2. LSTM Branch with bidirectional layers
    lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(64, return_sequences=False))(lstm)
    lstm = Dense(64, activation='relu')(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Merge branches
    merged = Concatenate()([cnn, lstm])
    
    # Deep neural network
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    
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
    args
) -> Tuple:
    """Train the enhanced model"""
    logger.info("Training model...")
    
    # Create clean pair name for file paths
    pair_clean = args.pair.replace("/", "_").lower()
    
    # Create checkpoint path
    checkpoint_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_{args.timeframe}_model.h5"
    
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
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Create logs directory
    logs_dir = f"{RESULTS_DIR}/logs/{pair_clean}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logs_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Load the best model
    model = load_model(checkpoint_path)
    
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
    sequence_length: int
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
    y_pred_signal = np.array([class_map[c] for c in y_pred_class])
    y_test_signal = np.array([class_map[c] for c in y_test_class])
    
    # Calculate class-wise accuracy
    class_names = ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"]
    class_accuracy = {}
    for i, name in enumerate(class_names):
        class_mask = y_test_class == i
        if np.sum(class_mask) > 0:
            class_accuracy[name] = np.mean(y_pred_class[class_mask] == i)
        else:
            class_accuracy[name] = 0.0
    
    # Calculate direction accuracy
    non_neutral_mask = (y_test_signal != 0) & (y_pred_signal != 0)
    direction_accuracy = np.mean(np.sign(y_pred_signal[non_neutral_mask]) == np.sign(y_test_signal[non_neutral_mask]))
    
    # Simulate trading performance
    trading_metrics = simulate_trading(
        model, 
        df, 
        scaler, 
        feature_columns, 
        sequence_length
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
    sequence_length: int
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
    y_pred_signal = np.array([class_map[c] for c in y_pred_class])
    
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
    trading_signals['position_size'] = initial_capital * risk_per_trade / trading_signals['leverage']
    
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
        days = (trading_signals['timestamp'].iloc[-1] - trading_signals['timestamp'].iloc[0]) / timedelta(days=1)
        annualized_return = (1 + total_return) ** (252 / max(1, days)) - 1
        
        # Volatility
        daily_returns = trading_signals['strategy_return'].resample('D', on='timestamp').sum()
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
            "final_equity": trading_signals['equity'].iloc[-1]
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
            "final_equity": initial_capital
        }
    
    # Save trading signals for analysis
    pair_clean = args.pair.replace('/', '_').lower()
    signals_file = f"{RESULTS_DIR}/trading_signals_{pair_clean}_{args.timeframe}.csv"
    trading_signals.to_csv(signals_file, index=False)
    logger.info(f"Saved trading signals to {signals_file}")
    
    # Log metrics
    logger.info(f"Trading metrics:")
    logger.info(f"  Total Return: {metrics['total_return']:.2%}")
    logger.info(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"  Trades: {metrics['trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    return metrics

def update_ml_config(
    pair: str, 
    timeframe: str, 
    model_path: str, 
    metrics: Dict, 
    max_portfolio_risk: float = 0.25
) -> bool:
    """Update ML configuration with training results"""
    logger.info("Updating ML configuration...")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
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
    
    # Create model key that includes timeframe
    model_key = f"{pair}_{timeframe}"
    
    # Add or update model config
    config["models"][model_key] = {
        "pair": pair,
        "timeframe": timeframe,
        "model_type": "enhanced_hybrid",
        "model_path": model_path,
        "accuracy": metrics["accuracy"],
        "direction_accuracy": metrics["direction_accuracy"],
        "win_rate": metrics["trading_metrics"]["win_rate"],
        "sharpe_ratio": metrics["trading_metrics"]["sharpe_ratio"],
        "total_return": metrics["trading_metrics"]["total_return"],
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config for {pair} ({timeframe})")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def save_training_report(
    pair: str, 
    timeframe: str, 
    metrics: Dict, 
    training_time: float, 
    model_path: str,
    args
) -> str:
    """Save training report to file"""
    # Create report filename
    pair_clean = pair.replace("/", "_").lower()
    report_file = f"{RESULTS_DIR}/training_report_{pair_clean}_{timeframe}.json"
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Create report data
    report = {
        "pair": pair,
        "timeframe": timeframe,
        "model_path": model_path,
        "training_time": training_time_str,
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "max_portfolio_risk": args.max_portfolio_risk
        },
        "metrics": {
            "model_accuracy": metrics["accuracy"],
            "direction_accuracy": metrics["direction_accuracy"],
            "class_accuracy": metrics["class_accuracy"],
            "trading_performance": metrics["trading_metrics"]
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved training report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving training report: {e}")
        return ""

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN IMPROVED MODEL WITH EXISTING DATA")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Update ML Config: {args.update_ml_config}")
    logger.info("=" * 80 + "\n")
    
    # Load existing data
    df = load_existing_data(args.pair, args.timeframe)
    if df is None or df.empty:
        logger.error("Failed to load existing data. Exiting.")
        return 1
    
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
    model = build_enhanced_model(input_shape)
    
    # Train model
    model, history, model_path, training_time = train_model(
        model, X_train, y_train, X_val, y_val, args
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, X_test, y_test, df_labeled, scaler, feature_columns, args.sequence_length
    )
    
    # Save training report
    save_training_report(
        args.pair, args.timeframe, metrics, training_time, model_path, args
    )
    
    # Update ML config
    if args.update_ml_config:
        update_ml_config(
            args.pair, args.timeframe, model_path, metrics, args.max_portfolio_risk
        )
    
    # Print success message
    logger.info("\nTraining completed successfully!")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    logger.info(f"Trading Win Rate: {metrics['trading_metrics']['win_rate']:.2%}")
    logger.info(f"Trading Return: {metrics['trading_metrics']['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['trading_metrics']['sharpe_ratio']:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())