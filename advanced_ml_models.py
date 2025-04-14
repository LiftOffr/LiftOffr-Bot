#!/usr/bin/env python3
"""
Advanced ML Models for Trading

This script implements cutting-edge ML architectures for price prediction:
1. Temporal Fusion Transformer (TFT) for multi-horizon forecasting
2. Neural ODE (Ordinary Differential Equation) for continuous time series
3. N-BEATS (Neural Basis Expansion Analysis for Time Series) for interpretable forecasting
4. DeepAR (Deep AutoRegressive) for probabilistic forecasting
5. Informer - Efficient Transformer for long sequence time-series forecasting
6. TCN (Temporal Convolutional Network) with self-attention mechanism

Usage:
    python advanced_ml_models.py --model tft/node/nbeats/deepar/informer/tcn_attention
                                --pair SOL/USD
                                [--epochs 200]
                                [--train-size 0.8]
                                [--lookback 24]
                                [--forecast-horizon 1,3,6,12,24]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_MODEL = 'tft'
DEFAULT_PAIR = 'SOL/USD'
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_LOOKBACK = 24
DEFAULT_FORECAST_HORIZONS = [1, 3, 6, 12, 24]
DEFAULT_PATIENCE = 20
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OUTPUT_DIR = 'models/advanced'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced ML Models for Trading')
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['tft', 'node', 'nbeats', 'deepar', 'informer', 'tcn_attention'],
        default=DEFAULT_MODEL,
        help=f'ML model architecture (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--pair',
        type=str,
        default=DEFAULT_PAIR,
        help=f'Trading pair (default: {DEFAULT_PAIR})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--train-size',
        type=float,
        default=DEFAULT_TRAIN_SIZE,
        help=f'Train/test split ratio (default: {DEFAULT_TRAIN_SIZE})'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f'Lookback window size (default: {DEFAULT_LOOKBACK})'
    )
    
    parser.add_argument(
        '--forecast-horizon',
        type=str,
        default=','.join(map(str, DEFAULT_FORECAST_HORIZONS)),
        help=f'Forecast horizons (comma-separated) (default: {",".join(map(str, DEFAULT_FORECAST_HORIZONS))})'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=DEFAULT_PATIENCE,
        help=f'Early stopping patience (default: {DEFAULT_PATIENCE})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    return parser.parse_args()

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_modules = [
        'numpy', 'pandas', 'tensorflow', 'sklearn', 'matplotlib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required dependencies: {', '.join(missing_modules)}")
        logger.error("Please install missing dependencies before running this script.")
        return False
    
    # Check TensorFlow version for TFT model
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        logger.info(f"TensorFlow version: {tf_version}")
        
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU is available for training")
        else:
            logger.warning("No GPU found, training will be slower on CPU only")
    
    except Exception as e:
        logger.warning(f"Error checking TensorFlow version: {str(e)}")
    
    return True

def load_dataset(pair: str, verbose: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[List[str]]]:
    """
    Load and preprocess dataset for the specified trading pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (DataFrame, feature_names) or (None, None) on failure
    """
    try:
        # Dynamically import required modules
        import numpy as np
        import pandas as pd
        
        # Normalize pair format
        pair_path = pair.replace('/', '')
        pair_symbol = pair.split('/')[0]
        pair_base = pair.split('/')[1]
        
        # Try to locate the dataset
        dataset_paths = [
            f"training_data/{pair_path}_1h_enhanced.csv",
            f"training_data/{pair_symbol}/{pair_base}_1h_enhanced.csv",
            f"training_data/{pair_path.lower()}_1h_enhanced.csv",
            f"historical_data/{pair_path}_1h.csv",
            f"historical_data/{pair_symbol}/{pair_base}_1h.csv"
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            logger.error(f"No dataset found for {pair}. Checked paths: {dataset_paths}")
            return None, None, None
        
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        if verbose:
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        # Basic preprocessing
        # Drop rows with missing values
        df = df.dropna()
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
        
        # Identify price and volume columns
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        potential_volume_columns = ['volume', 'Volume']
        potential_open_columns = ['open', 'Open']
        potential_high_columns = ['high', 'High']
        potential_low_columns = ['low', 'Low']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
        
        volume_column = None
        for col in potential_volume_columns:
            if col in df.columns:
                volume_column = col
                break
        
        open_column = None
        for col in potential_open_columns:
            if col in df.columns:
                open_column = col
                break
        
        high_column = None
        for col in potential_high_columns:
            if col in df.columns:
                high_column = col
                break
        
        low_column = None
        for col in potential_low_columns:
            if col in df.columns:
                low_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify a price column in the dataset")
            return None, None, None
        
        # Feature engineering - add technical indicators and other features
        feature_columns = []
        
        # Add price as a feature
        feature_columns.append(price_column)
        
        # Add other OHLC columns if available
        if open_column:
            feature_columns.append(open_column)
        
        if high_column:
            feature_columns.append(high_column)
        
        if low_column:
            feature_columns.append(low_column)
        
        # Add volume if available
        if volume_column:
            feature_columns.append(volume_column)
        
        # Add price differences
        df['price_diff_1'] = df[price_column].diff()
        feature_columns.append('price_diff_1')
        
        df['price_diff_2'] = df['price_diff_1'].diff()
        feature_columns.append('price_diff_2')
        
        # Add price returns
        df['price_return_1'] = df[price_column].pct_change()
        feature_columns.append('price_return_1')
        
        # Add simple moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            col_name = f'sma_{window}'
            df[col_name] = df[price_column].rolling(window=window).mean()
            feature_columns.append(col_name)
        
        # Add exponential moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            col_name = f'ema_{window}'
            df[col_name] = df[price_column].ewm(span=window, adjust=False).mean()
            feature_columns.append(col_name)
        
        # Add price momentum
        for window in [5, 10, 20, 50]:
            col_name = f'momentum_{window}'
            df[col_name] = df[price_column].pct_change(periods=window)
            feature_columns.append(col_name)
        
        # Add price volatility
        for window in [5, 10, 20, 50]:
            col_name = f'volatility_{window}'
            df[col_name] = df[price_column].rolling(window=window).std()
            feature_columns.append(col_name)
        
        # Add Bollinger Bands
        for window in [20]:
            std_multiplier = 2.0
            sma = df[price_column].rolling(window=window).mean()
            std = df[price_column].rolling(window=window).std()
            
            df[f'upper_bb_{window}'] = sma + (std * std_multiplier)
            df[f'lower_bb_{window}'] = sma - (std * std_multiplier)
            df[f'bb_width_{window}'] = (df[f'upper_bb_{window}'] - df[f'lower_bb_{window}']) / sma
            
            feature_columns.extend([f'upper_bb_{window}', f'lower_bb_{window}', f'bb_width_{window}'])
        
        # Add RSI
        for window in [14]:
            delta = df[price_column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            feature_columns.append(f'rsi_{window}')
        
        # Add MACD
        fast_ema = df[price_column].ewm(span=12, adjust=False).mean()
        slow_ema = df[price_column].ewm(span=26, adjust=False).mean()
        df['macd'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        feature_columns.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Add volume-based features if volume exists
        if volume_column:
            # Add volume moving average
            for window in [5, 10, 20, 50]:
                col_name = f'volume_sma_{window}'
                df[col_name] = df[volume_column].rolling(window=window).mean()
                feature_columns.append(col_name)
            
            # Add volume momentum
            for window in [5, 10, 20]:
                col_name = f'volume_momentum_{window}'
                df[col_name] = df[volume_column].pct_change(periods=window)
                feature_columns.append(col_name)
            
            # Add volume volatility
            for window in [5, 10, 20]:
                col_name = f'volume_volatility_{window}'
                df[col_name] = df[volume_column].rolling(window=window).std()
                feature_columns.append(col_name)
            
            # Add price-volume ratio
            df['price_volume_ratio'] = df[price_column] / df[volume_column].replace(0, 1)
            feature_columns.append('price_volume_ratio')
            
            # Add price-volume correlation
            for window in [10, 20]:
                df[f'price_volume_corr_{window}'] = df[price_column].rolling(window).corr(df[volume_column])
                feature_columns.append(f'price_volume_corr_{window}')
        
        # Add time-based features if timestamp exists
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
            
            feature_columns.extend([
                'hour', 'day_of_week', 'day_of_month', 'week_of_year',
                'month', 'quarter', 'is_month_end', 'is_month_start'
            ])
        
        # Drop rows with NaN values resulting from rolling windows
        df = df.dropna()
        
        logger.info(f"Dataset prepared with {len(df)} rows and {len(feature_columns)} features")
        
        return df, feature_columns, price_column
    
    except Exception as e:
        logger.error(f"Error loading dataset for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def prepare_sequences(
    df: Any,
    feature_columns: List[str],
    price_column: str,
    lookback: int = 24,
    forecast_horizons: List[int] = [1],
    train_size: float = 0.8,
    verbose: bool = False
) -> Tuple:
    """
    Prepare sequences for training.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        price_column: Name of the price column
        lookback: Lookback window size
        forecast_horizons: List of forecast horizons
        train_size: Train/test split ratio
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler)
    """
    try:
        # Dynamically import required modules
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # Extract features and target
        X = df[feature_columns].values
        y = df[price_column].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create target price changes for each forecast horizon
        y_targets = {}
        for horizon in forecast_horizons:
            # Future price
            future_price = df[price_column].shift(-horizon).values
            
            # Binary target (1 if price goes up, 0 if down or unchanged)
            y_binary = np.where(future_price > df[price_column].values, 1, 0)
            
            # Price change percentage
            y_pct_change = (future_price - df[price_column].values) / df[price_column].values
            
            # Price direction and magnitude
            y_targets[horizon] = {
                'binary': y_binary,
                'pct_change': y_pct_change,
                'future_price': future_price
            }
        
        # Create sequences
        X_sequences = []
        y_sequences = {h: {'binary': [], 'pct_change': [], 'future_price': []} for h in forecast_horizons}
        
        for i in range(lookback, len(X_scaled) - max(forecast_horizons)):
            # Input sequence
            X_sequences.append(X_scaled[i-lookback:i])
            
            # Output targets for each horizon
            for horizon in forecast_horizons:
                y_sequences[horizon]['binary'].append(y_targets[horizon]['binary'][i])
                y_sequences[horizon]['pct_change'].append(y_targets[horizon]['pct_change'][i])
                y_sequences[horizon]['future_price'].append(y_targets[horizon]['future_price'][i])
        
        X_sequences = np.array(X_sequences)
        for horizon in forecast_horizons:
            y_sequences[horizon]['binary'] = np.array(y_sequences[horizon]['binary'])
            y_sequences[horizon]['pct_change'] = np.array(y_sequences[horizon]['pct_change'])
            y_sequences[horizon]['future_price'] = np.array(y_sequences[horizon]['future_price'])
        
        # Split into train and test sets
        train_idx = int(len(X_sequences) * train_size)
        
        X_train = X_sequences[:train_idx]
        X_test = X_sequences[train_idx:]
        
        y_train = {h: {k: v[:train_idx] for k, v in y_sequences[h].items()} for h in forecast_horizons}
        y_test = {h: {k: v[train_idx:] for k, v in y_sequences[h].items()} for h in forecast_horizons}
        
        if verbose:
            logger.info(f"Input shape: {X_train.shape}")
            for horizon in forecast_horizons:
                logger.info(f"Horizon {horizon}: {len(y_train[horizon]['binary'])} training samples, {len(y_test[horizon]['binary'])} testing samples")
        
        return X_train, y_train, X_test, y_test, scaler
    
    except Exception as e:
        logger.error(f"Error preparing sequences: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None

def build_tft_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    hidden_units: int = 64,
    attention_heads: int = 4,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001
) -> Any:
    """
    Build Temporal Fusion Transformer (TFT) model.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        hidden_units: Number of hidden units
        attention_heads: Number of attention heads
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        
    Returns:
        Compiled TFT model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, MultiHeadAttention
        from tensorflow.keras.layers import LayerNormalization, Dropout, Concatenate
        from tensorflow.keras.optimizers import Adam
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Variable selection network
        vs_network = Dense(hidden_units, activation='elu')(inputs)
        vs_network = Dropout(dropout_rate)(vs_network)
        
        # LSTM encoder
        lstm_encoder = LSTM(hidden_units, return_sequences=True)(vs_network)
        lstm_encoder = LayerNormalization()(lstm_encoder)
        
        # Static covariates enrichment (if available)
        static_enriched = lstm_encoder
        
        # Attention block
        attn_layer = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=hidden_units // attention_heads
        )(static_enriched, static_enriched)
        
        attn_layer = LayerNormalization()(attn_layer + static_enriched)
        
        # Feed-forward network
        ffn = Dense(hidden_units, activation='elu')(attn_layer)
        ffn = Dense(hidden_units)(ffn)
        ffn = LayerNormalization()(ffn + attn_layer)
        
        # Projection for each horizon
        outputs = []
        for _ in range(num_horizons):
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{_}')(ffn[:, -1, :])
            
            # Percentage change regression
            pct_change_output = Dense(1, name=f'pct_change_output_{_}')(ffn[:, -1, :])
            
            # Future price regression
            future_price_output = Dense(1, name=f'future_price_output_{_}')(ffn[:, -1, :])
            
            outputs.extend([binary_output, pct_change_output, future_price_output])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_output_{i}': 'mse' for i in range(num_horizons)},
                **{f'future_price_output_{i}': 'mse' for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_output_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_output_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building TFT model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_neural_ode_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    hidden_units: int = 64,
    learning_rate: float = 0.001
) -> Any:
    """
    Build Neural ODE model for time series.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        hidden_units: Number of hidden units
        learning_rate: Learning rate
        
    Returns:
        Compiled Neural ODE model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization
        from tensorflow.keras.layers import Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import backend as K
        
        # Neural ODE-inspired layer
        class NeuralODELayer(tf.keras.layers.Layer):
            def __init__(self, units, integration_time=1.0, **kwargs):
                super(NeuralODELayer, self).__init__(**kwargs)
                self.units = units
                self.integration_time = integration_time
                self.dynamics_fn = tf.keras.Sequential([
                    Dense(units, activation='tanh'),
                    Dense(units)
                ])
            
            def build(self, input_shape):
                super(NeuralODELayer, self).build(input_shape)
            
            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                h0 = inputs
                
                # Euler integration (simplified ODE solver)
                dt = self.integration_time / 10  # 10 integration steps
                h = h0
                
                for _ in range(10):
                    dhdt = self.dynamics_fn(h)
                    h = h + dhdt * dt
                
                return h
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM for feature extraction
        x = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Neural ODE layer
        x = NeuralODELayer(hidden_units)(x[:, -1, :])
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layers for each horizon
        outputs = []
        for _ in range(num_horizons):
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{_}')(x)
            
            # Percentage change regression
            pct_change_output = Dense(1, name=f'pct_change_output_{_}')(x)
            
            # Future price regression
            future_price_output = Dense(1, name=f'future_price_output_{_}')(x)
            
            outputs.extend([binary_output, pct_change_output, future_price_output])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_output_{i}': 'mse' for i in range(num_horizons)},
                **{f'future_price_output_{i}': 'mse' for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_output_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_output_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building Neural ODE model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_nbeats_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    stack_types: List[str] = ['trend', 'seasonality'],
    num_blocks: int = 3,
    learning_rate: float = 0.001
) -> Any:
    """
    Build N-BEATS (Neural Basis Expansion Analysis for Time Series) model.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        stack_types: List of stack types ('trend', 'seasonality', 'generic')
        num_blocks: Number of blocks per stack
        learning_rate: Learning rate
        
    Returns:
        Compiled N-BEATS model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape, Add
        from tensorflow.keras.optimizers import Adam
        
        # N-BEATS block
        def create_nbeats_block(x, units, stack_type, block_id):
            # Forward expansion
            theta = Dense(units, activation='relu')(x)
            theta = Dense(units, activation='relu')(theta)
            theta = Dense(units, activation='relu')(theta)
            
            # Basis expansion
            if stack_type == 'trend':
                # Trend block with polynomial basis
                basis_width = input_shape[0]  # lookback window size
                basis = Dense(basis_width, activation='linear')(theta)
                backcast = Dense(input_shape[0] * input_shape[1], activation='linear')(basis)
                backcast = Reshape(input_shape)(backcast)
                
                # Forecast using the same basis
                forecast = Dense(num_horizons, activation='linear')(basis)
            
            elif stack_type == 'seasonality':
                # Seasonality block with Fourier basis
                basis_width = min(input_shape[0], 10)  # Limit basis size
                basis = Dense(basis_width, activation='linear')(theta)
                backcast = Dense(input_shape[0] * input_shape[1], activation='linear')(basis)
                backcast = Reshape(input_shape)(backcast)
                
                # Forecast using the same basis
                forecast = Dense(num_horizons, activation='linear')(basis)
            
            else:  # 'generic'
                # Generic block with flexible basis
                backcast = Dense(input_shape[0] * input_shape[1], activation='linear')(theta)
                backcast = Reshape(input_shape)(backcast)
                
                # Separate forecast path
                forecast = Dense(num_horizons, activation='linear')(theta)
            
            return backcast, forecast
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Process inputs
        x = inputs
        total_forecast = None
        
        # Create stacks and blocks
        for stack_idx, stack_type in enumerate(stack_types):
            stack_forecast = None
            
            for block_idx in range(num_blocks):
                # Create block
                backcast, block_forecast = create_nbeats_block(
                    x,
                    units=64,
                    stack_type=stack_type,
                    block_id=f"{stack_type}_{block_idx}"
                )
                
                # Residual connection for backcast
                x = Concatenate()([
                    x,
                    -backcast
                ])
                
                # Add forecast from this block
                if stack_forecast is None:
                    stack_forecast = block_forecast
                else:
                    stack_forecast = Add()([stack_forecast, block_forecast])
            
            # Add forecast from this stack
            if total_forecast is None:
                total_forecast = stack_forecast
            else:
                total_forecast = Add()([total_forecast, stack_forecast])
        
        # Final forecast processing
        forecast = Dense(num_horizons, activation='linear')(total_forecast)
        
        # Output layers for each horizon
        outputs = []
        for i in range(num_horizons):
            # Get forecast for this horizon
            horizon_forecast = tf.expand_dims(forecast[:, i], -1)
            
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{i}')(horizon_forecast)
            
            # Percentage change regression
            pct_change_output = Dense(1, name=f'pct_change_output_{i}')(horizon_forecast)
            
            # Future price regression
            future_price_output = Dense(1, name=f'future_price_output_{i}')(horizon_forecast)
            
            outputs.extend([binary_output, pct_change_output, future_price_output])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_output_{i}': 'mse' for i in range(num_horizons)},
                **{f'future_price_output_{i}': 'mse' for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_output_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_output_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building N-BEATS model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_deepar_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    hidden_units: int = 64,
    learning_rate: float = 0.001
) -> Any:
    """
    Build DeepAR (Deep AutoRegressive) model for probabilistic forecasting.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        hidden_units: Number of hidden units
        learning_rate: Learning rate
        
    Returns:
        Compiled DeepAR model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
        from tensorflow.keras.layers import Dropout, TimeDistributed, Lambda
        from tensorflow.keras.optimizers import Adam
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM encoder
        lstm = LSTM(hidden_units, return_sequences=True)(inputs)
        lstm = Dropout(0.2)(lstm)
        lstm = LSTM(hidden_units)(lstm)
        lstm = Dropout(0.2)(lstm)
        
        # DeepAR probabilistic output - predict both mean and variance
        outputs = []
        for i in range(num_horizons):
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{i}')(lstm)
            
            # Distribution parameters for percentage change
            pct_change_mean = Dense(1, name=f'pct_change_mean_{i}')(lstm)
            pct_change_std = Dense(1, activation='softplus', name=f'pct_change_std_{i}')(lstm)
            
            # Distribution parameters for future price
            future_price_mean = Dense(1, name=f'future_price_mean_{i}')(lstm)
            future_price_std = Dense(1, activation='softplus', name=f'future_price_std_{i}')(lstm)
            
            outputs.extend([
                binary_output, 
                pct_change_mean, pct_change_std,
                future_price_mean, future_price_std
            ])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_mean_{i}': lambda y_true, y_pred: tf.reduce_mean(
                    tf.square(y_true - y_pred) / model.get_layer(f'pct_change_std_{i}').output
                    + tf.math.log(model.get_layer(f'pct_change_std_{i}').output)
                ) for i in range(num_horizons)},
                **{f'pct_change_std_{i}': lambda y_true, y_pred: 0 for i in range(num_horizons)},
                **{f'future_price_mean_{i}': lambda y_true, y_pred: tf.reduce_mean(
                    tf.square(y_true - y_pred) / model.get_layer(f'future_price_std_{i}').output
                    + tf.math.log(model.get_layer(f'future_price_std_{i}').output)
                ) for i in range(num_horizons)},
                **{f'future_price_std_{i}': lambda y_true, y_pred: 0 for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_mean_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_mean_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building DeepAR model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_informer_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    hidden_units: int = 64,
    num_heads: int = 4,
    learning_rate: float = 0.001
) -> Any:
    """
    Build Informer model for long sequence time-series forecasting.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        hidden_units: Number of hidden units
        num_heads: Number of attention heads
        learning_rate: Learning rate
        
    Returns:
        Compiled Informer model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
        from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Concatenate
        from tensorflow.keras.optimizers import Adam
        
        # ProbSparse Self-Attention (simplified)
        def prob_sparse_attention(q, k, v, mask=None):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            
            # Scale attention scores
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            
            # Apply mask if provided
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
            
            # Probsparse sampling (simplified as regular softmax)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            
            output = tf.matmul(attention_weights, v)
            return output
        
        # ProbSparse Multi-head Attention
        class ProbSparseMultiHeadAttention(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads):
                super(ProbSparseMultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model
                
                assert d_model % self.num_heads == 0
                
                self.depth = d_model // self.num_heads
                
                self.wq = Dense(d_model)
                self.wk = Dense(d_model)
                self.wv = Dense(d_model)
                
                self.dense = Dense(d_model)
            
            def split_heads(self, x, batch_size):
                x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
                return tf.transpose(x, perm=[0, 2, 1, 3])
            
            def call(self, q, k, v, mask=None):
                batch_size = tf.shape(q)[0]
                
                q = self.wq(q)
                k = self.wk(k)
                v = self.wv(v)
                
                q = self.split_heads(q, batch_size)
                k = self.split_heads(k, batch_size)
                v = self.split_heads(v, batch_size)
                
                # Probsparse attention
                scaled_attention = prob_sparse_attention(q, k, v, mask)
                
                scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
                concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
                
                output = self.dense(concat_attention)
                return output
        
        # Distilling layer (downsample by convolutional layer)
        def distilling_layer(x, factor=2):
            return Conv1D(filters=x.shape[-1], kernel_size=factor, strides=factor, padding='same')(x)
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Embedding layer
        embedding_dim = hidden_units
        x = Dense(embedding_dim)(inputs)
        
        # Informer encoder with progressive downsampling
        for i in range(2):  # 2 encoder layers
            # Self-attention
            attn_output = ProbSparseMultiHeadAttention(embedding_dim, num_heads)(x, x, x)
            attn_output = Dropout(0.1)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed-forward network
            ffn_output = Dense(embedding_dim * 4, activation='relu')(x)
            ffn_output = Dense(embedding_dim)(ffn_output)
            ffn_output = Dropout(0.1)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
            
            # Distilling (downsample)
            if i < 1:  # Don't distill in the last encoder layer
                x = distilling_layer(x)
        
        # Global representation
        global_repr = GlobalAveragePooling1D()(x)
        
        # Output layers for each horizon
        outputs = []
        for i in range(num_horizons):
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{i}')(global_repr)
            
            # Percentage change regression
            pct_change_output = Dense(1, name=f'pct_change_output_{i}')(global_repr)
            
            # Future price regression
            future_price_output = Dense(1, name=f'future_price_output_{i}')(global_repr)
            
            outputs.extend([binary_output, pct_change_output, future_price_output])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_output_{i}': 'mse' for i in range(num_horizons)},
                **{f'future_price_output_{i}': 'mse' for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_output_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_output_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building Informer model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_tcn_attention_model(
    input_shape: Tuple[int, int],
    num_horizons: int = 1,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: List[int] = [1, 2, 4, 8, 16],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> Any:
    """
    Build TCN (Temporal Convolutional Network) with self-attention model.
    
    Args:
        input_shape: Input shape (lookback, features)
        num_horizons: Number of forecast horizons
        filters: Number of filters in convolutional layers
        kernel_size: Size of the convolutional kernels
        dilations: List of dilation rates for TCN layers
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        
    Returns:
        Compiled TCN with self-attention model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization
        from tensorflow.keras.layers import Activation, Add, GlobalAveragePooling1D, MultiHeadAttention
        from tensorflow.keras.optimizers import Adam
        
        # Residual block for TCN
        def residual_block(x, dilation_rate, filters, kernel_size, dropout_rate):
            # Weight normalization (simplified as batch normalization)
            bn_1 = BatchNormalization()(x)
            
            # First dilated convolution
            conv_1 = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='linear'
            )(bn_1)
            act_1 = Activation('relu')(conv_1)
            drop_1 = Dropout(dropout_rate)(act_1)
            
            # Second dilated convolution
            bn_2 = BatchNormalization()(drop_1)
            conv_2 = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='linear'
            )(bn_2)
            act_2 = Activation('relu')(conv_2)
            drop_2 = Dropout(dropout_rate)(act_2)
            
            # Skip connection
            if x.shape[-1] != filters:
                x = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
            
            # Add
            added = Add()([x, drop_2])
            
            return added
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv1D(filters=filters, kernel_size=1, padding='same')(inputs)
        
        # TCN blocks with dilated convolutions
        skip_connections = []
        for dilation_rate in dilations:
            x = residual_block(x, dilation_rate, filters, kernel_size, dropout_rate)
            skip_connections.append(x)
        
        # Add skip connections
        if len(skip_connections) > 1:
            x = Add()(skip_connections)
        
        # Self-attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=filters // 4
        )(x, x)
        
        # Combine with TCN output
        x = Add()([x, attention_output])
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(filters, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layers for each horizon
        outputs = []
        for i in range(num_horizons):
            # Binary prediction (up/down)
            binary_output = Dense(1, activation='sigmoid', name=f'binary_output_{i}')(x)
            
            # Percentage change regression
            pct_change_output = Dense(1, name=f'pct_change_output_{i}')(x)
            
            # Future price regression
            future_price_output = Dense(1, name=f'future_price_output_{i}')(x)
            
            outputs.extend([binary_output, pct_change_output, future_price_output])
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                **{f'binary_output_{i}': 'binary_crossentropy' for i in range(num_horizons)},
                **{f'pct_change_output_{i}': 'mse' for i in range(num_horizons)},
                **{f'future_price_output_{i}': 'mse' for i in range(num_horizons)}
            },
            metrics={
                **{f'binary_output_{i}': ['accuracy'] for i in range(num_horizons)},
                **{f'pct_change_output_{i}': ['mae'] for i in range(num_horizons)},
                **{f'future_price_output_{i}': ['mae'] for i in range(num_horizons)}
            }
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building TCN model with self-attention: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def train_model(
    model_type: str,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 20,
    learning_rate: float = 0.001,
    verbose: bool = False
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Train the specified model.
    
    Args:
        model_type: Type of model to build
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        learning_rate: Learning rate
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (trained model, training history)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        import tempfile
        
        # Prepare model input shape
        input_shape = X_train.shape[1:]
        
        # Determine number of horizons
        num_horizons = len(y_train.keys())
        
        # Build model based on type
        if model_type == 'tft':
            model = build_tft_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        elif model_type == 'node':
            model = build_neural_ode_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        elif model_type == 'nbeats':
            model = build_nbeats_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        elif model_type == 'deepar':
            model = build_deepar_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        elif model_type == 'informer':
            model = build_informer_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        elif model_type == 'tcn_attention':
            model = build_tcn_attention_model(
                input_shape=input_shape,
                num_horizons=num_horizons,
                learning_rate=learning_rate
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None, None
        
        if model is None:
            logger.error(f"Failed to build {model_type} model")
            return None, None
        
        # Create callbacks
        callbacks = []
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint callback with temporary file
        fd, checkpoint_path = tempfile.mkstemp()
        os.close(fd)
        
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        callbacks.append(model_checkpoint)
        
        # Prepare training and validation data
        train_data = {}
        val_data = {}
        
        for horizon in y_train.keys():
            for target_type, target_values in y_train[horizon].items():
                if model_type == 'deepar' and target_type != 'binary':
                    # Special handling for DeepAR model
                    train_data[f'{target_type}_mean_{horizon}'] = target_values
                    train_data[f'{target_type}_std_{horizon}'] = target_values  # Same values used for both mean and std
                    
                    val_data[f'{target_type}_mean_{horizon}'] = y_test[horizon][target_type]
                    val_data[f'{target_type}_std_{horizon}'] = y_test[horizon][target_type]
                else:
                    train_data[f'{target_type}_output_{horizon}'] = target_values
                    val_data[f'{target_type}_output_{horizon}'] = y_test[horizon][target_type]
        
        # Train model
        history = model.fit(
            X_train,
            train_data,
            validation_data=(X_test, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2 if verbose else 1
        )
        
        # Load best weights
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            os.remove(checkpoint_path)
        
        # Evaluate model
        eval_result = model.evaluate(X_test, val_data, verbose=0)
        
        # Convert history to dict
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(x) for x in v]
        
        # Create metrics summary
        metrics = {}
        
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = float(eval_result[i])
        
        # Add binary accuracy per horizon
        predictions = model.predict(X_test)
        
        # Store prediction metrics per horizon
        horizon_metrics = {}
        
        # Process predictions
        pred_idx = 0
        for horizon in y_test.keys():
            horizon_metrics[horizon] = {}
            
            # Binary prediction
            binary_pred = predictions[pred_idx]
            binary_true = y_test[horizon]['binary']
            
            binary_accuracy = tf.keras.metrics.binary_accuracy(
                binary_true,
                binary_pred
            ).numpy().mean()
            
            horizon_metrics[horizon]['binary_accuracy'] = float(binary_accuracy)
            pred_idx += 1
            
            # Percentage change prediction
            if model_type == 'deepar':
                pct_change_pred_mean = predictions[pred_idx]
                pct_change_pred_std = predictions[pred_idx + 1]
                pred_idx += 2
            else:
                pct_change_pred = predictions[pred_idx]
                pred_idx += 1
            
            # Future price prediction
            if model_type == 'deepar':
                future_price_pred_mean = predictions[pred_idx]
                future_price_pred_std = predictions[pred_idx + 1]
                pred_idx += 2
            else:
                future_price_pred = predictions[pred_idx]
                pred_idx += 1
        
        # Add horizon metrics to overall metrics
        metrics['horizons'] = horizon_metrics
        
        # Create final training results
        training_results = {
            'history': history_dict,
            'metrics': metrics
        }
        
        return model, training_results
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def save_model(
    model: Any,
    model_type: str,
    pair: str,
    training_results: Dict[str, Any],
    output_dir: str
) -> bool:
    """
    Save trained model and its metadata.
    
    Args:
        model: Trained model
        model_type: Type of model
        pair: Trading pair
        training_results: Training results and metrics
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize pair format for file path
        pair_path = pair.replace('/', '')
        
        # Create subdirectory for model type
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{pair_path}.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save model metadata
        metadata = {
            'pair': pair,
            'model_type': model_type,
            'training_results': training_results,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(model_dir, f"{pair_path}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("ADVANCED ML MODELS FOR TRADING")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model}")
    logger.info(f"Trading pair: {args.pair}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Lookback window: {args.lookback}")
    logger.info(f"Forecast horizons: {args.forecast_horizon}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info(f"Disable GPU: {args.no_gpu}")
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages and try again.")
        return 1
    
    # Disable GPU if requested
    if args.no_gpu:
        logger.info("Disabling GPU as requested")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Parse forecast horizons
    forecast_horizons = [int(h) for h in args.forecast_horizon.split(',')]
    
    # Load dataset
    df, feature_columns, price_column = load_dataset(args.pair, args.verbose)
    if df is None:
        logger.error(f"Failed to load dataset for {args.pair}")
        return 1
    
    # Prepare sequences
    sequences = prepare_sequences(
        df,
        feature_columns,
        price_column,
        args.lookback,
        forecast_horizons,
        args.train_size,
        args.verbose
    )
    
    if sequences[0] is None:
        logger.error("Failed to prepare sequences")
        return 1
    
    X_train, y_train, X_test, y_test, scaler = sequences
    
    # Train model
    model, training_results = train_model(
        args.model,
        X_train,
        y_train,
        X_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.patience,
        args.learning_rate,
        args.verbose
    )
    
    if model is None:
        logger.error("Failed to train model")
        return 1
    
    # Save model
    if not save_model(
        model,
        args.model,
        args.pair,
        training_results,
        args.output_dir
    ):
        logger.error("Failed to save model")
        return 1
    
    # Print results
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    # Print metrics summary
    try:
        metrics = training_results['metrics']
        logger.info("Model performance:")
        
        for name, value in metrics.items():
            if name != 'horizons':
                logger.info(f"  {name}: {value:.4f}")
        
        logger.info("Performance by horizon:")
        for horizon, horizon_metrics in metrics.get('horizons', {}).items():
            logger.info(f"  Horizon {horizon}:")
            for metric_name, metric_value in horizon_metrics.items():
                logger.info(f"    {metric_name}: {metric_value:.4f}")
    
    except Exception as e:
        logger.error(f"Error printing metrics: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())