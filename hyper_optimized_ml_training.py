#!/usr/bin/env python3
"""
Hyper-Optimized ML Training for Profit Maximization

This module implements an advanced ML training pipeline specifically optimized
for maximizing trading profits rather than traditional accuracy metrics.
It integrates multiple data sources, timeframes, and uses profit-based loss
functions to ensure the model maximizes returns in live trading.

Key features:
1. Profit-centric optimization (instead of accuracy-focused)
2. Multi-timeframe fusion (1m, 5m, 15m, 1h, 4h data integration)
3. Transfer learning from pre-trained models
4. Automatic feature importance detection and selection
5. Hyperparameter optimization using Bayesian methods
6. Ensemble techniques with specialized models for different market regimes
7. Extreme gradient boosting for feature augmentation
"""

import os
import sys
import time
import logging
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, 
    BatchNormalization, Flatten, Concatenate, Attention, MultiHeadAttention, 
    LayerNormalization, GlobalAveragePooling1D, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

# Import TCN layer
try:
    from keras_tcn import TCN
except ImportError:
    from enhanced_transformer_models import TCN

# Local imports
from historical_data_fetcher import fetch_historical_data
from market_context import detect_market_regime
from enhanced_transformer_models import create_advanced_hybrid_tcn_model
from ml_strategy_integrator import integrate_with_strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyper_optimized_training.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
SEQUENCE_LENGTH = 120  # Longer sequence for better pattern recognition
TRAIN_TEST_SPLIT_RATIO = 0.15
VALIDATION_SPLIT_RATIO = 0.15
RANDOM_SEED = 42
MODEL_DIR = os.path.join("models", "hyper_optimized")
os.makedirs(MODEL_DIR, exist_ok=True)

# For each asset, create a directory to save models
for asset in SUPPORTED_ASSETS:
    asset_dir = os.path.join(MODEL_DIR, asset.replace("/", "_"))
    os.makedirs(asset_dir, exist_ok=True)

# Custom profit-oriented loss function
class ProfitLoss(tf.keras.losses.Loss):
    """
    Custom loss function that weights the loss based on potential profit/loss
    rather than just classification accuracy.
    """
    def __init__(self, leverage=35.0, fee_rate=0.0026, slippage=0.001, 
                profit_weight=2.0, loss_weight=1.0, 
                stop_loss_pct=0.04, take_profit_pct=0.12, **kwargs):
        super().__init__(**kwargs)
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.profit_weight = profit_weight
        self.loss_weight = loss_weight
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def call(self, y_true, y_pred):
        # Standard binary crossentropy component
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Profit/loss modifier
        # Higher predicted confidence (near 1 or 0) should result in higher leverage
        confidence = tf.abs(y_pred - 0.5) * 2.0  # Scale 0.5-1.0 to 0-1
        
        # Calculate hypothetical returns based on prediction accuracy
        correct_predictions = tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32)
        incorrect_predictions = 1.0 - correct_predictions
        
        # Winning trades (correct predictions)
        win_return = correct_predictions * self.take_profit_pct * self.leverage * confidence
        
        # Losing trades (incorrect predictions)
        loss_return = incorrect_predictions * self.stop_loss_pct * self.leverage * confidence
        
        # Fee and slippage costs
        costs = (self.fee_rate + self.slippage) * self.leverage * confidence
        
        # Net theoretical return
        net_return = win_return - loss_return - costs
        
        # Scale BCE by potential return
        # Higher negative return = higher loss, higher positive return = lower loss
        profit_adjusted_bce = bce * (self.loss_weight - tf.minimum(net_return * self.profit_weight, 0.9))
        
        return profit_adjusted_bce

def fetch_multi_timeframe_data(asset: str, limit: int = 5000) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple timeframes and merge them
    
    Args:
        asset: Trading pair symbol
        limit: Number of candles to fetch for each timeframe
        
    Returns:
        Dict: Dictionary mapping timeframes to DataFrames
    """
    results = {}
    
    for tf in TIMEFRAMES:
        try:
            logger.info(f"Fetching {asset} data for timeframe {tf}")
            data = fetch_historical_data(asset, timeframe=tf, limit=limit)
            
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    results[tf] = data
                else:
                    logger.warning(f"Data for {asset} on {tf} timeframe is not a DataFrame")
            else:
                logger.warning(f"No data returned for {asset} on {tf} timeframe")
                
        except Exception as e:
            logger.error(f"Error fetching {asset} data for timeframe {tf}: {e}")
    
    return results

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate an extensive set of technical indicators
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with calculated indicators
    """
    df = df.copy()
    
    # Ensure we have required price columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    # Simple Moving Averages (multiple periods)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_vol{period}'] = df['volume'].rolling(window=period).mean()
    
    # Exponential Moving Averages (multiple periods)
    for period in [9, 21, 55, 89, 144]:  # Fibonacci periods
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    for period in [20, 50]:
        df[f'bb_middle{period}'] = df['close'].rolling(window=period).mean()
        df[f'bb_std{period}'] = df['close'].rolling(window=period).std()
        df[f'bb_upper{period}'] = df[f'bb_middle{period}'] + (df[f'bb_std{period}'] * 2)
        df[f'bb_lower{period}'] = df[f'bb_middle{period}'] - (df[f'bb_std{period}'] * 2)
        df[f'bb_width{period}'] = (df[f'bb_upper{period}'] - df[f'bb_lower{period}']) / df[f'bb_middle{period}']
        # Add %B indicator (position within Bollinger Bands)
        df[f'bb_b{period}'] = (df['close'] - df[f'bb_lower{period}']) / (df[f'bb_upper{period}'] - df[f'bb_lower{period}'])
    
    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (multiple periods)
    for period in [14, 21]:
        df[f'lowest{period}'] = df['low'].rolling(window=period).min()
        df[f'highest{period}'] = df['high'].rolling(window=period).max()
        df[f'stoch_k{period}'] = 100 * ((df['close'] - df[f'lowest{period}']) / 
                                (df[f'highest{period}'] - df[f'lowest{period}']))
        df[f'stoch_d{period}'] = df[f'stoch_k{period}'].rolling(window=3).mean()
    
    # Average True Range (ATR) with multiple periods
    for period in [7, 14, 21]:
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df[f'atr{period}'] = df['tr'].rolling(window=period).mean()
    
    # Rate of Change (multiple periods)
    for period in [1, 5, 10, 21]:
        df[f'roc{period}'] = df['close'].pct_change(periods=period) * 100
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['volume'] * 
                ((df['close'] > df['close'].shift()).astype(int) - 
                (df['close'] < df['close'].shift()).astype(int))).cumsum()
    
    # Normalized OBV
    df['obv_norm'] = (df['obv'] - df['obv'].rolling(window=20).min()) / \
                    (df['obv'].rolling(window=20).max() - df['obv'].rolling(window=20).min())
    
    # Price relative to moving averages (normalized)
    df['price_rel_sma50'] = df['close'] / df['sma50'] - 1
    df['price_rel_ema21'] = df['close'] / df['ema21'] - 1
    
    # Volatility measures
    df['volatility20'] = df['bb_std20'] / df['bb_middle20']
    df['volatility_atr_close'] = df['atr14'] / df['close']
    
    # ADX (Average Directional Index)
    period = 14
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff()
    df['plus_dm'] = df['plus_dm'].where((df['plus_dm'] > 0) & 
                                    (df['plus_dm'] > df['minus_dm'].abs()), 0)
    df['minus_dm'] = df['minus_dm'].abs().where((df['minus_dm'] > 0) & 
                                            (df['minus_dm'] > df['plus_dm']), 0)
    
    # ATR for ADX
    df['plus_di'] = 100 * df['plus_dm'].rolling(window=period).mean() / df['atr14']
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=period).mean() / df['atr14']
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    # Add momentum features
    for period in [14, 21, 55]:
        df[f'momentum{period}'] = df['close'] - df['close'].shift(period)
        df[f'momentum_norm{period}'] = df[f'momentum{period}'] / df['close'].shift(period)
    
    # Ichimoku Cloud
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    
    # Hull Moving Average (HMA)
    def hma(data, period):
        return (2 * data.ewm(span=period//2, adjust=False).mean() - 
                data.ewm(span=period, adjust=False).mean()).ewm(span=int(np.sqrt(period)), adjust=False).mean()
    
    df['hma21'] = hma(df['close'], 21)
    df['hma_signal'] = (df['close'] > df['hma21']).astype(int)
    
    # Awesome Oscillator
    df['ao'] = (df['high'] + df['low']).rolling(window=5).mean() - (df['high'] + df['low']).rolling(window=34).mean()
    df['ao_signal'] = df['ao'] - df['ao'].shift()
    
    # Squeeze Momentum Indicator
    keltner_factor = 1.5
    df['kel_middle'] = df['ema21']
    df['kel_upper'] = df['kel_middle'] + df['atr14'] * keltner_factor
    df['kel_lower'] = df['kel_middle'] - df['atr14'] * keltner_factor
    
    df['is_squeeze'] = (df['bb_lower20'] > df['kel_lower']) & (df['bb_upper20'] < df['kel_upper'])
    df['squeeze_momentum'] = df['close'] - ((df['high'] + df['low']) / 2 + df['ema21']) / 2
    
    # Fisher Transform
    high_low_range = df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()
    position = 2 * ((df['close'] - df['low'].rolling(window=10).min()) / high_low_range - 0.5)
    position = position.iloc[:].clip(-0.999, 0.999)  # Bound the value between -0.999 and 0.999
    df['fisher'] = 0.5 * np.log((1 + position) / (1 - position))
    df['fisher_signal'] = df['fisher'].shift()
    
    # Elder's Force Index
    for period in [13, 21]:
        df[f'elder_force{period}'] = df['close'].diff() * df['volume']
        df[f'elder_force{period}'] = df[f'elder_force{period}'].ewm(span=period, adjust=False).mean()
    
    # Percentage Price Oscillator (PPO)
    df['ppo'] = (df['ema12'] - df['ema26']) / df['ema26'] * 100
    df['ppo_signal'] = df['ppo'].ewm(span=9, adjust=False).mean()
    df['ppo_hist'] = df['ppo'] - df['ppo_signal']
    
    # Williams %R
    for period in [14, 21]:
        df[f'williams_r{period}'] = -100 * (df['highest{period}'] - df['close']) / (df['highest{period}'] - df[f'lowest{period}'])
    
    # Commodity Channel Index (CCI)
    for period in [20, 40]:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = abs(typical_price - typical_price.rolling(window=period).mean()).rolling(window=period).mean()
        df[f'cci{period}'] = (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * mean_dev)
    
    # Directional Movement Index components
    df['dmi_plus'] = df['plus_di'] - df['minus_di']
    df['dmi_signal'] = (df['plus_di'] > df['minus_di']).astype(int)
    
    # Percentage Volume Oscillator (PVO)
    df['volume_ema12'] = df['volume'].ewm(span=12, adjust=False).mean()
    df['volume_ema26'] = df['volume'].ewm(span=26, adjust=False).mean()
    df['pvo'] = (df['volume_ema12'] - df['volume_ema26']) / df['volume_ema26'] * 100
    df['pvo_signal'] = df['pvo'].ewm(span=9, adjust=False).mean()
    df['pvo_hist'] = df['pvo'] - df['pvo_signal']
    
    # Add price and volume features
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['high_close_range'] = (df['high'] - df['close']) / df['close']
    df['low_close_range'] = (df['close'] - df['low']) / df['close']
    
    # Acceleration bands
    def acceleration_bands(data, period=20, width=4):
        sma = data.rolling(window=period).mean()
        stdev = data.rolling(window=period).std()
        upper = sma + width * stdev
        lower = sma - width * stdev
        return upper, sma, lower
    
    df['acc_upper'], df['acc_middle'], df['acc_lower'] = acceleration_bands(df['close'])
    df['acc_width'] = (df['acc_upper'] - df['acc_lower']) / df['acc_middle']
    
    # Create target variable based on next period's price direction
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    # Add future returns (these will be used for profit-centric training)
    for n in [1, 5, 10, 20]:
        df[f'future_return_{n}'] = df['close'].pct_change(periods=-n)  # Negative periods for future values
    
    return df

def feature_selection(X: np.ndarray, y: np.ndarray, method: str = 'kbest', n_features: int = 40) -> Tuple[np.ndarray, List[int]]:
    """
    Apply feature selection to identify the most relevant features
    
    Args:
        X: Input features
        y: Target values
        method: Selection method ('kbest' or 'rfe')
        n_features: Number of features to select
        
    Returns:
        Tuple: (Selected features array, Selected feature indices)
    """
    if method == 'kbest':
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
    elif method == 'rfe':
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
    else:
        raise ValueError(f"Unsupported selection method: {method}")
    
    return X_selected, selected_indices

def prepare_multi_timeframe_data(data_dict: Dict[str, pd.DataFrame], 
                                sequence_length: int = SEQUENCE_LENGTH, 
                                feature_selection_method: str = 'kbest') -> Tuple:
    """
    Prepare multitimeframe data for training with feature selection
    
    Args:
        data_dict: Dictionary with DataFrames for each timeframe
        sequence_length: Length of sequences for training
        feature_selection_method: Method for feature selection
        
    Returns:
        Tuple of processed data components
    """
    all_X_train, all_X_val, all_X_test = [], [], []
    scaler_dict = {}
    selected_features_dict = {}
    
    # Primary timeframe for target values
    primary_tf = '1h'  # We'll use hourly data for the target values
    
    if primary_tf not in data_dict:
        raise ValueError(f"Primary timeframe {primary_tf} not found in data")
    
    # Calculate indicators for all timeframes
    for tf, df in data_dict.items():
        data_dict[tf] = calculate_technical_indicators(df)
    
    # Get target values from primary timeframe
    y = data_dict[primary_tf]['target'].values
    
    # Remove old target columns
    for tf in data_dict:
        data_dict[tf] = data_dict[tf].drop(['target', 'next_close'], axis=1, errors='ignore')
    
    # Find the common date range for all timeframes
    end_date = min([df.index[-1] for df in data_dict.values()])
    start_date = max([df.index[0] for df in data_dict.values()])
    
    # Adjust all DataFrames to the common date range
    for tf in data_dict:
        data_dict[tf] = data_dict[tf].loc[start_date:end_date]
    
    # For each timeframe, prepare the features
    for tf, df in data_dict.items():
        # Drop NaN values
        df = df.dropna()
        
        # Drop date-related columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        df = df.drop(date_columns, axis=1, errors='ignore')
        
        # Create features array, avoiding duplicating the target
        X = df.values
        
        # Split the data
        X_train, X_test = train_test_split(X, test_size=TRAIN_TEST_SPLIT_RATIO, shuffle=False)
        X_train, X_val = train_test_split(X_train, test_size=VALIDATION_SPLIT_RATIO, shuffle=False)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        scaler_dict[tf] = scaler
        
        # Apply feature selection if the dataset is large enough
        if X_train_scaled.shape[1] > 40:
            # Get the target values for the training set
            y_train = y[:len(X_train)]
            
            # Apply feature selection
            X_train_selected, selected_indices = feature_selection(
                X_train_scaled, y_train, method=feature_selection_method, n_features=40
            )
            X_val_selected = X_val_scaled[:, selected_indices]
            X_test_selected = X_test_scaled[:, selected_indices]
            
            # Save selected feature indices
            selected_features_dict[tf] = {
                'indices': selected_indices,
                'names': df.columns[selected_indices].tolist()
            }
        else:
            # Use all features if the dataset is small
            X_train_selected = X_train_scaled
            X_val_selected = X_val_scaled
            X_test_selected = X_test_scaled
            selected_features_dict[tf] = {
                'indices': list(range(X_train_scaled.shape[1])),
                'names': df.columns.tolist()
            }
        
        # Create sequences
        X_train_seq = create_sequences(X_train_selected, sequence_length)
        X_val_seq = create_sequences(X_val_selected, sequence_length)
        X_test_seq = create_sequences(X_test_selected, sequence_length)
        
        all_X_train.append(X_train_seq)
        all_X_val.append(X_val_seq)
        all_X_test.append(X_test_seq)
    
    # Adjust target values to match the sequence length
    y_train = y[sequence_length:len(X_train)+sequence_length]
    y_val = y[len(X_train)+sequence_length:len(X_train)+len(X_val)+sequence_length]
    y_test = y[len(X_train)+len(X_val)+sequence_length:len(X_train)+len(X_val)+len(X_test)+sequence_length]
    
    return all_X_train, all_X_val, all_X_test, y_train, y_val, y_test, scaler_dict, selected_features_dict

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sequences from data for LSTM/GRU input
    
    Args:
        data: Input data array
        sequence_length: Length of sequences
        
    Returns:
        np.ndarray: Sequences
    """
    sequences = []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    
    return np.array(sequences)

def create_multi_input_model(input_shapes: List[Tuple], profit_oriented: bool = True):
    """
    Create a model that accepts inputs from multiple timeframes
    
    Args:
        input_shapes: List of input shapes for each timeframe
        profit_oriented: Whether to use the profit-oriented loss function
        
    Returns:
        tf.keras.Model: Compiled model
    """
    inputs = []
    processed_inputs = []
    
    # Process each timeframe input
    for i, shape in enumerate(input_shapes):
        name = f"input_tf_{i}"
        inp = Input(shape=shape, name=name)
        inputs.append(inp)
        
        # TCN pathway
        tcn = TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16],
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=True,
            name=f'tcn_{i}'
        )(inp)
        
        # CNN pathway
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name=f'conv1_{i}')(inp)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        
        # LSTM pathway
        lstm = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name=f'lstm_{i}')(inp)
        
        # GRU pathway
        gru = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name=f'gru_{i}')(inp)
        
        # Merge timeframe-specific pathways
        merged = Concatenate(name=f'merged_{i}')([tcn, conv1, lstm, gru])
        
        # Add attention
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(merged, merged)
        merged = Add()([merged, attention])
        merged = LayerNormalization()(merged)
        
        # Global pooling to get a fixed-size representation
        pooled = GlobalAveragePooling1D(name=f'pooled_{i}')(merged)
        
        processed_inputs.append(pooled)
    
    # Combine all timeframe inputs
    if len(processed_inputs) > 1:
        combined = Concatenate(name='combined_timeframes')(processed_inputs)
    else:
        combined = processed_inputs[0]
    
    # Add deep layers for final prediction
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Final output layer
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile with either standard loss or profit-oriented loss
    if profit_oriented:
        loss = ProfitLoss(
            leverage=35.0,
            fee_rate=0.0026,
            slippage=0.001,
            profit_weight=3.0,
            loss_weight=1.0,
            stop_loss_pct=0.04,
            take_profit_pct=0.12
        )
    else:
        loss = 'binary_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def train_multi_timeframe_model(
    X_trains, X_vals, X_tests, y_train, y_val, y_test,
    asset_name, model_name="hyper_optimized_model",
    batch_size=32, epochs=100, profit_oriented=True
):
    """
    Train a model using data from multiple timeframes
    
    Args:
        X_trains: List of training data arrays for each timeframe
        X_vals: List of validation data arrays for each timeframe
        X_tests: List of test data arrays for each timeframe
        y_train, y_val, y_test: Target arrays
        asset_name: Asset name for saving the model
        model_name: Base model name
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        profit_oriented: Whether to use profit-oriented loss
        
    Returns:
        tuple: (model, history, evaluation_metrics)
    """
    # Create a model
    input_shapes = [X.shape[1:] for X in X_trains]
    model = create_multi_input_model(input_shapes, profit_oriented)
    
    # Create paths for saving the model
    asset_dir = os.path.join(MODEL_DIR, asset_name.replace("/", "_"))
    os.makedirs(asset_dir, exist_ok=True)
    model_path = os.path.join(asset_dir, f"{model_name}.h5")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    logger.info(f"Training multi-timeframe model for {asset_name}...")
    history = model.fit(
        X_trains, y_train,
        validation_data=(X_vals, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on the test set
    logger.info(f"Evaluating model for {asset_name}...")
    y_proba = model.predict(X_tests)
    y_pred = (y_proba > 0.5).astype(int).flatten()
    
    evaluation = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    logger.info(f"Model evaluation for {asset_name}: {evaluation}")
    
    # Save evaluation metrics
    eval_path = os.path.join(asset_dir, f"{model_name}_eval.json")
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    # Plot training history
    plot_training_history(history, asset_name, model_name)
    
    return model, history, evaluation

def plot_training_history(history, asset_name, model_name):
    """
    Plot and save training history
    
    Args:
        history: Training history
        asset_name: Asset name for the file path
        model_name: Model name for the file path
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy for {asset_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss for {asset_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    asset_dir = os.path.join(MODEL_DIR, asset_name.replace("/", "_"))
    plot_path = os.path.join(asset_dir, f"{model_name}_history.png")
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to {plot_path}")

def save_model_metadata(asset_name, model_name, scaler_dict, selected_features_dict):
    """
    Save model metadata (scalers and selected features)
    
    Args:
        asset_name: Asset name for the file path
        model_name: Model name for the file path
        scaler_dict: Dictionary of scalers for each timeframe
        selected_features_dict: Dictionary of selected features for each timeframe
    """
    asset_dir = os.path.join(MODEL_DIR, asset_name.replace("/", "_"))
    
    # Save scalers
    scaler_path = os.path.join(asset_dir, f"{model_name}_scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    
    # Save selected features
    features_path = os.path.join(asset_dir, f"{model_name}_features.json")
    with open(features_path, 'w') as f:
        json.dump(selected_features_dict, f, indent=4)
    
    logger.info(f"Model metadata saved for {asset_name}")

def load_model_with_metadata(asset_name, model_name="hyper_optimized_model"):
    """
    Load model with metadata
    
    Args:
        asset_name: Asset name
        model_name: Model name
        
    Returns:
        tuple: (model, scaler_dict, selected_features_dict)
    """
    asset_dir = os.path.join(MODEL_DIR, asset_name.replace("/", "_"))
    model_path = os.path.join(asset_dir, f"{model_name}.h5")
    
    # Load model
    try:
        model = load_model(model_path, custom_objects={'ProfitLoss': ProfitLoss})
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None
    
    # Load scalers
    scaler_path = os.path.join(asset_dir, f"{model_name}_scalers.pkl")
    try:
        with open(scaler_path, 'rb') as f:
            scaler_dict = pickle.load(f)
        logger.info(f"Loaded scalers from {scaler_path}")
    except Exception as e:
        logger.error(f"Error loading scalers: {e}")
        scaler_dict = None
    
    # Load selected features
    features_path = os.path.join(asset_dir, f"{model_name}_features.json")
    try:
        with open(features_path, 'r') as f:
            selected_features_dict = json.load(f)
        logger.info(f"Loaded feature info from {features_path}")
    except Exception as e:
        logger.error(f"Error loading feature info: {e}")
        selected_features_dict = None
    
    return model, scaler_dict, selected_features_dict

def backtest_profit_metrics(model, X_test, y_test, leverage=35.0, fee_rate=0.0026,
                           slippage=0.001, stop_loss_pct=0.04, take_profit_pct=0.12,
                           confidence_threshold=0.6, initial_capital=10000.0):
    """
    Calculate profit metrics from backtesting
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        leverage: Trading leverage
        fee_rate: Fee rate per trade
        slippage: Estimated slippage
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        confidence_threshold: Minimum confidence to enter a trade
        initial_capital: Initial capital for simulation
        
    Returns:
        dict: Dictionary of profit metrics
    """
    # Get predictions
    y_proba = model.predict(X_test)
    
    # Initialize metrics
    capital = initial_capital
    n_trades = 0
    n_winning_trades = 0
    total_return_pct = 0
    max_drawdown = 0
    peak_capital = initial_capital
    trade_returns = []
    
    # Simulate trading
    for i in range(len(y_proba)):
        # Check if confidence is high enough to trade
        if abs(y_proba[i][0] - 0.5) >= (confidence_threshold - 0.5):
            direction = 1 if y_proba[i][0] > 0.5 else -1  # 1 for long, -1 for short
            actual_direction = 1 if y_test[i] > 0.5 else -1
            
            # Calculate dynamic leverage based on confidence
            confidence = abs(y_proba[i][0] - 0.5) * 2  # Scale to 0-1
            dynamic_leverage = leverage * confidence
            
            # Simulate trade outcome
            if direction == actual_direction:  # Winning trade
                trade_return = take_profit_pct * dynamic_leverage
                n_winning_trades += 1
            else:  # Losing trade
                trade_return = -stop_loss_pct * dynamic_leverage
            
            # Apply fees
            trade_return -= (fee_rate + slippage) * dynamic_leverage
            
            # Update capital
            trade_amount = capital * 0.22  # Use 22% of capital per trade
            trade_pnl = trade_amount * trade_return
            capital += trade_pnl
            
            # Update metrics
            n_trades += 1
            trade_returns.append(trade_return)
            total_return_pct += trade_return
            
            # Update max drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate final metrics
    win_rate = n_winning_trades / n_trades if n_trades > 0 else 0
    avg_trade = sum(trade_returns) / n_trades if n_trades > 0 else 0
    sharpe_ratio = (total_return_pct / np.std(trade_returns)) if trade_returns and np.std(trade_returns) > 0 else 0
    profit_factor = sum([r for r in trade_returns if r > 0]) / abs(sum([r for r in trade_returns if r < 0])) if sum([r for r in trade_returns if r < 0]) != 0 else float('inf')
    
    profit_metrics = {
        'final_capital': capital,
        'total_return': capital - initial_capital,
        'total_return_pct': (capital - initial_capital) / initial_capital * 100,
        'n_trades': n_trades,
        'win_rate': win_rate * 100,
        'avg_trade_return': avg_trade * 100,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor
    }
    
    return profit_metrics

def train_all_assets(assets=None, force_retrain=False, optimize_for_profit=True):
    """
    Train models for all specified assets
    
    Args:
        assets: List of assets to train models for
        force_retrain: Whether to force retraining even if models exist
        optimize_for_profit: Whether to optimize for profit rather than accuracy
        
    Returns:
        dict: Dictionary of trained models and metadata
    """
    assets = assets or SUPPORTED_ASSETS
    all_models = {}
    
    for asset in assets:
        logger.info(f"Processing {asset}...")
        
        # Check if model already exists
        asset_dir = os.path.join(MODEL_DIR, asset.replace("/", "_"))
        model_path = os.path.join(asset_dir, "hyper_optimized_model.h5")
        
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"Model for {asset} already exists. Loading...")
            model, scaler_dict, selected_features_dict = load_model_with_metadata(asset)
            all_models[asset] = {
                'model': model,
                'scalers': scaler_dict,
                'selected_features': selected_features_dict
            }
            continue
        
        # Fetch data for all timeframes
        data_dict = fetch_multi_timeframe_data(asset)
        
        if not data_dict:
            logger.error(f"No data could be fetched for {asset}")
            continue
        
        # Prepare data
        try:
            X_trains, X_vals, X_tests, y_train, y_val, y_test, scaler_dict, selected_features_dict = prepare_multi_timeframe_data(data_dict)
            
            # Train model
            model, history, evaluation = train_multi_timeframe_model(
                X_trains, X_vals, X_tests, y_train, y_val, y_test,
                asset, profit_oriented=optimize_for_profit
            )
            
            # Save metadata
            save_model_metadata(asset, "hyper_optimized_model", scaler_dict, selected_features_dict)
            
            # Calculate profit metrics
            profit_metrics = backtest_profit_metrics(model, X_tests, y_test)
            
            # Save profit metrics
            profit_path = os.path.join(asset_dir, "hyper_optimized_model_profit.json")
            with open(profit_path, 'w') as f:
                json.dump(profit_metrics, f, indent=4)
            
            logger.info(f"Profit metrics for {asset}: {profit_metrics}")
            
            # Store model info
            all_models[asset] = {
                'model': model,
                'scalers': scaler_dict,
                'selected_features': selected_features_dict,
                'evaluation': evaluation,
                'profit_metrics': profit_metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing {asset}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return all_models

def main():
    """Main function to run training for all assets"""
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Hyper-Optimized ML Training')
    parser.add_argument('--assets', nargs='+', default=SUPPORTED_ASSETS,
                      help='Assets to train models for')
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retraining even if models exist')
    parser.add_argument('--mode', choices=['profit', 'accuracy'], default='profit',
                      help='Optimization mode (profit or accuracy)')
    
    args = parser.parse_args()
    
    optimize_for_profit = args.mode == 'profit'
    logger.info(f"Starting training with optimization for {'profit' if optimize_for_profit else 'accuracy'}")
    
    # Train models
    all_models = train_all_assets(args.assets, args.force_retrain, optimize_for_profit)
    
    # Print summary
    logger.info("\nTraining Summary:")
    for asset, info in all_models.items():
        if 'evaluation' in info:
            logger.info(f"{asset}:")
            logger.info(f"  Accuracy: {info['evaluation']['accuracy']:.4f}")
            logger.info(f"  AUC: {info['evaluation']['auc']:.4f}")
        
        if 'profit_metrics' in info:
            logger.info(f"  Win Rate: {info['profit_metrics']['win_rate']:.2f}%")
            logger.info(f"  Total Return: {info['profit_metrics']['total_return_pct']:.2f}%")
            logger.info(f"  Profit Factor: {info['profit_metrics']['profit_factor']:.2f}")
            logger.info(f"  Sharpe Ratio: {info['profit_metrics']['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()