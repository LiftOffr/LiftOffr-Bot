#!/usr/bin/env python3
"""
Advanced Ensemble Model for Kraken Trading Bot

This module implements a sophisticated ensemble architecture that combines
multiple model types with adaptive weighting mechanisms to handle volatile market conditions.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import custom model architectures
from attention_gru_model import AttentionLayer, load_attention_gru_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
ENSEMBLE_DIR = os.path.join(MODELS_DIR, "ensemble")
MODEL_TYPES = ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid", "attention_gru"]

# Ensure ensemble directory exists
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

class DynamicWeightedEnsemble:
    """
    A sophisticated ensemble model that dynamically weights the predictions
    from multiple model architectures based on their recent performance.
    """
    def __init__(self, trading_pair="SOL/USD", timeframe="1h"):
        """
        Initialize the ensemble model
        
        Args:
            trading_pair (str): Trading pair to model
            timeframe (str): Timeframe for analysis
        """
        self.trading_pair = trading_pair
        self.ticker_symbol = trading_pair.replace('/', '')
        self.timeframe = timeframe
        self.models = {}
        self.norm_params = {}
        self.feature_names = {}
        self.weights = self._initialize_weights()
        self.performance_history = {model_type: [] for model_type in MODEL_TYPES}
        self.current_regime = "normal"  # normal, volatile, trending, ranging
        self._load_models()
        
    def _initialize_weights(self):
        """
        Initialize model weights based on either saved values or defaults
        
        Returns:
            dict: Dictionary of model weights
        """
        weights_file = os.path.join(ENSEMBLE_DIR, f"{self.ticker_symbol}_weights.json")
        
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
                logger.info(f"Loaded saved weights: {weights}")
                return weights
            except Exception as e:
                logger.error(f"Error loading saved weights: {e}")
        
        # Default weights - equal weighting to start
        default_weights = {model_type: 1.0 / len(MODEL_TYPES) for model_type in MODEL_TYPES}
        logger.info(f"Using default weights: {default_weights}")
        return default_weights
        
    def _normalize_weights(self):
        """
        Normalize weights to ensure they sum to 1.0
        
        This is called after weights have been adjusted or models pruned
        to ensure a valid probability distribution of weights.
        """
        if not self.weights or not self.models:
            return
            
        # Get weights only for existing models
        valid_weights = {model_type: weight for model_type, weight in self.weights.items() 
                       if model_type in self.models}
        
        # Calculate sum of valid weights
        weight_sum = sum(valid_weights.values())
        
        # Normalize if sum is not zero
        if weight_sum > 0:
            for model_type in valid_weights:
                self.weights[model_type] = valid_weights[model_type] / weight_sum
        else:
            # If all weights are zero, assign equal weights
            for model_type in self.models:
                self.weights[model_type] = 1.0 / len(self.models)
                
        logger.debug(f"Normalized weights: {self.weights}")
    
    def _load_models(self):
        """Load all available trained models"""
        self.models = {}
        self.norm_params = {}
        self.feature_names = {}
        
        # If models directory doesn't exist, create it
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Ensure each model type directory exists
        for model_type in MODEL_TYPES:
            model_dir = os.path.join(MODELS_DIR, model_type)
            os.makedirs(model_dir, exist_ok=True)
        
        missing_models = []
        for model_type in MODEL_TYPES:
            model_path = os.path.join(MODELS_DIR, model_type, f"{self.ticker_symbol}.h5")
            norm_path = os.path.join(MODELS_DIR, model_type, "norm_params.json")
            features_path = os.path.join(MODELS_DIR, model_type, "feature_names.json")
            
            if os.path.exists(model_path) and os.path.exists(norm_path) and os.path.exists(features_path):
                try:
                    # For attention_gru model type, use the special loader
                    if model_type == "attention_gru":
                        # Use the custom loader with AttentionLayer
                        from attention_gru_model import load_attention_gru_model
                        self.models[model_type] = load_attention_gru_model(model_path)
                    else:
                        # Regular model loading for other model types
                        self.models[model_type] = load_model(model_path)
                    
                    # Load normalization parameters
                    with open(norm_path, 'r') as f:
                        self.norm_params[model_type] = json.load(f)
                    
                    # Load feature names
                    with open(features_path, 'r') as f:
                        self.feature_names[model_type] = json.load(f)
                    
                    logger.info(f"Loaded {model_type} model for {self.ticker_symbol}")
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {e}")
                    logger.error(traceback.format_exc())
                    missing_models.append(model_type)
            else:
                logger.warning(f"{model_type} model not found at {model_path}")
                missing_models.append(model_type)
        
        # Initialize placeholder models during testing/development
        if not self.models and len(missing_models) > 0:
            logger.warning("No trained models found. Using balanced weights with fallback prediction.")
            self.using_fallback_models = True
        else:
            self.using_fallback_models = False
        
        logger.info(f"Loaded {len(self.models)} models for {self.ticker_symbol}")
    
    def get_loaded_models(self):
        """
        Get information about loaded models
        
        Returns:
            dict: Dictionary with information about loaded models
        """
        loaded_models = []
        for model_type in MODEL_TYPES:
            if model_type in self.models:
                loaded_models.append({
                    'model_type': model_type,
                    'weight': self.weights.get(model_type, 0.0),
                    'features': len(self.feature_names.get(model_type, []))
                })
            else:
                loaded_models.append({
                    'model_type': model_type,
                    'weight': self.weights.get(model_type, 0.0),
                    'features': 0,
                    'status': 'not_loaded'
                })
        
        return {
            'loaded_count': len(self.models),
            'total_models': len(MODEL_TYPES),
            'using_fallback': getattr(self, 'using_fallback_models', False),
            'models': loaded_models
        }
    
    def detect_market_regime(self, market_data):
        """
        Detect the current market regime based on volatility and trend
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            str: Detected market regime
        """
        # Extract key metrics
        vol_col = f"volatility_{self.timeframe}"
        price_col = f"close_{self.timeframe}"
        
        if vol_col not in market_data.columns and "volatility" in market_data.columns:
            vol_col = "volatility"
        
        if price_col not in market_data.columns and "close" in market_data.columns:
            price_col = "close"
        
        # Calculate metrics if they don't exist
        if vol_col not in market_data.columns:
            market_data[vol_col] = market_data[price_col].pct_change().rolling(20).std()
        
        # Get latest volatility
        current_volatility = market_data[vol_col].iloc[-1]
        
        # Compare to historical volatility
        historical_volatility = market_data[vol_col].mean()
        volatility_threshold = 1.5  # 50% higher than average
        
        # Check for trend
        ema9_col = f"ema9_{self.timeframe}"
        ema21_col = f"ema21_{self.timeframe}"
        
        if ema9_col not in market_data.columns and "ema9" in market_data.columns:
            ema9_col = "ema9"
        
        if ema21_col not in market_data.columns and "ema21" in market_data.columns:
            ema21_col = "ema21"
        
        if ema9_col in market_data.columns and ema21_col in market_data.columns:
            ema9 = market_data[ema9_col].iloc[-1]
            ema21 = market_data[ema21_col].iloc[-1]
            
            # Calculate trend strength
            trend_strength = abs(ema9 / ema21 - 1)
            trend_threshold = 0.005  # 0.5%
            
            # Determine regime
            if current_volatility > historical_volatility * volatility_threshold:
                if trend_strength > trend_threshold:
                    regime = "volatile_trending"
                else:
                    regime = "volatile_ranging"
            else:
                if trend_strength > trend_threshold:
                    regime = "normal_trending"
                else:
                    regime = "normal_ranging"
        else:
            # Fallback if EMAs not available
            if current_volatility > historical_volatility * volatility_threshold:
                regime = "volatile"
            else:
                regime = "normal"
        
        # Update current regime
        self.current_regime = regime
        logger.info(f"Detected market regime: {regime}")
        
        return regime
    
    def _adjust_weights_for_regime(self):
        """
        Adjust model weights based on the current market regime
        
        Different models perform better in different market conditions. This method
        adjusts the weights to favor models that typically perform better in the
        current market regime.
        """
        regime_preferences = {
            "volatile": {
                "tcn": 1.5,      # TCN handles volatile data well
                "cnn": 1.2,
                "lstm": 1.0,
                "gru": 1.0,
                "bilstm": 1.2,   # BiLSTM captures both directions
                "attention": 1.5, # Attention focuses on relevant patterns
                "transformer": 1.8, # Transformers excel at complex patterns
                "hybrid": 2.0,   # Hybrid captures multiple perspectives
                "attention_gru": 1.7  # Attention-GRU excels at capturing time dependencies
            },
            "volatile_trending": {
                "tcn": 1.5,
                "cnn": 1.3,
                "lstm": 1.2,
                "gru": 1.2,
                "bilstm": 1.5,
                "attention": 1.6,
                "transformer": 1.7,
                "hybrid": 2.0,
                "attention_gru": 1.8  # Attention-GRU performs well in trending markets
            },
            "volatile_ranging": {
                "tcn": 1.7,
                "cnn": 1.5,
                "lstm": 1.0,
                "gru": 1.0,
                "bilstm": 1.3,
                "attention": 1.5,
                "transformer": 1.5,
                "hybrid": 2.0,
                "attention_gru": 1.6  # Attention-GRU helps identify patterns in ranging markets
            },
            "normal_trending": {
                "tcn": 1.2,
                "cnn": 1.5,
                "lstm": 1.5,
                "gru": 1.5,
                "bilstm": 1.3,
                "attention": 1.2,
                "transformer": 1.2,
                "hybrid": 1.5,
                "attention_gru": 1.4  # Attention-GRU can identify trend pattern details
            },
            "normal_ranging": {
                "tcn": 1.0,
                "cnn": 1.2,
                "lstm": 1.2,
                "gru": 1.2,
                "bilstm": 1.0,
                "attention": 1.0,
                "transformer": 1.0,
                "hybrid": 1.2,
                "attention_gru": 1.1  # Attention-GRU provides some advantage in normal ranging markets
            },
            "normal": {
                "tcn": 1.0,
                "cnn": 1.0,
                "lstm": 1.0,
                "gru": 1.0,
                "bilstm": 1.0,
                "attention": 1.0,
                "transformer": 1.0,
                "hybrid": 1.0,
                "attention_gru": 1.0  # Equal baseline weight for normal market conditions
            }
        }
        
        # Get regime preferences
        if self.current_regime in regime_preferences:
            prefs = regime_preferences[self.current_regime]
        else:
            prefs = regime_preferences["normal"]
        
        # Adjust weights
        adjusted_weights = {}
        for model_type in self.weights:
            if model_type in prefs:
                adjusted_weights[model_type] = self.weights[model_type] * prefs[model_type]
            else:
                adjusted_weights[model_type] = self.weights[model_type]
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        normalized_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return normalized_weights
    
    def _adjust_weights_by_performance(self, weights, lookback=10):
        """
        Further adjust weights based on recent model performance
        
        Args:
            weights (dict): Initial weights
            lookback (int): Number of recent predictions to consider
            
        Returns:
            dict: Performance-adjusted weights
        """
        # If we don't have enough performance history, return original weights
        if not all(len(hist) >= lookback for hist in self.performance_history.values()):
            return weights
        
        # Calculate performance scores
        performance_scores = {}
        for model_type, history in self.performance_history.items():
            if model_type in weights:
                recent_history = history[-lookback:]
                if recent_history:
                    # Calculate weighted score (recent predictions matter more)
                    weighted_sum = sum(score * (i + 1) for i, score in enumerate(recent_history))
                    weighted_count = sum(i + 1 for i in range(len(recent_history)))
                    performance_scores[model_type] = weighted_sum / weighted_count
                else:
                    performance_scores[model_type] = 0
        
        # Adjust weights based on performance
        adjusted_weights = {}
        for model_type, weight in weights.items():
            if model_type in performance_scores:
                # Scale weight by performance score (normalize to 0.5-1.5 range)
                performance_factor = 0.5 + max(0, min(1, performance_scores[model_type])) * 1.0
                adjusted_weights[model_type] = weight * performance_factor
            else:
                adjusted_weights[model_type] = weight
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in adjusted_weights.items()}
            return normalized_weights
        else:
            return weights
    
    def _preprocess_data(self, market_data):
        """
        Preprocess market data for model input
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            dict: Preprocessed data for each model type
        """
        preprocessed_data = {}
        
        # Check if we have enough data
        if market_data is None or len(market_data) < 60:  # Minimum sequence length is typically 60
            logger.warning(f"Not enough data for preprocessing. Data shape: {market_data.shape if market_data is not None else 'None'}")
            return preprocessed_data
            
        # Log available columns for debugging
        logger.debug(f"Available columns in market data: {list(market_data.columns)}")
        
        # Generate calculated features if needed
        market_data = self._ensure_required_features(market_data)
        
        for model_type, model in self.models.items():
            try:
                # Get the right columns based on feature names
                if model_type in self.feature_names:
                    feature_names = self.feature_names[model_type]
                    
                    # Extract relevant columns if they exist
                    X = []
                    missing_features = []
                    for feature in feature_names:
                        # Try different naming conventions to find the right feature
                        variants = [
                            feature,  # Original name
                            f"{feature}_{self.timeframe}",  # With timeframe suffix
                            feature.split('_')[0],  # Base name without suffix
                            f"{feature.split('_')[0]}_{self.timeframe}"  # Base name with timeframe
                        ]
                        
                        feature_found = False
                        for variant in variants:
                            if variant in market_data.columns:
                                X.append(market_data[variant].values)
                                if variant != feature:
                                    logger.debug(f"Found feature {feature} as {variant}")
                                feature_found = True
                                break
                        
                        if not feature_found:
                            missing_features.append(feature)
                            # Use zeros as placeholder (necessary for model structure)
                            X.append(np.zeros_like(market_data.index.values, dtype=float))
                    
                    # Log missing features
                    if missing_features:
                        logger.warning(f"Missing {len(missing_features)}/{len(feature_names)} features for {model_type} model: {missing_features}")
                        
                        # If too many features are missing, skip this model
                        if len(missing_features) > len(feature_names) / 3:  # Skip if more than 1/3 of features are missing
                            logger.error(f"Too many features missing for {model_type}, skipping model")
                            continue
                    
                    # Stack arrays to create input tensor
                    X = np.column_stack(X)
                    
                    # Create sequences for recurrent models
                    input_shape = model.input_shape[1:]
                    sequence_length = input_shape[0]
                    
                    # Check if we have enough data for sequences
                    if len(X) < sequence_length:
                        logger.warning(f"Not enough data points ({len(X)}) for sequence length {sequence_length}")
                        continue
                    
                    # Create sequences
                    sequences = []
                    for i in range(len(X) - sequence_length + 1):
                        sequences.append(X[i:i+sequence_length])
                    
                    # Convert to numpy array
                    X_sequences = np.array(sequences)
                    
                    # Normalize
                    if model_type in self.norm_params:
                        # Get normalization parameters
                        norm_params = self.norm_params[model_type]
                        
                        # Ensure norm_params are in the correct format
                        if isinstance(norm_params, dict) and 'mean' in norm_params and 'std' in norm_params:
                            # Convert parameters to numpy arrays if they're not already
                            if isinstance(norm_params['mean'], dict):
                                # Handle case where mean/std are stored as feature dictionaries
                                logger.debug(f"Converting dictionary normalization parameters to arrays for {model_type}")
                                # Use feature names to sort values consistently
                                feature_names = list(sorted(norm_params['mean'].keys()))
                                mean_values = [norm_params['mean'].get(f, 0) for f in feature_names]
                                std_values = [norm_params['std'].get(f, 1) for f in feature_names]
                                mean = np.array(mean_values)
                                std = np.array(std_values)
                            else:
                                # Already in array/list format
                                mean = np.array(norm_params['mean'])
                                std = np.array(norm_params['std'])
                            
                            # Avoid division by zero
                            std = np.where(std == 0, 1, std)
                            
                            # Ensure dimensions match
                            if mean.size == X_sequences.shape[2]:
                                # Apply normalization (shape adjustment for broadcasting)
                                X_sequences = (X_sequences - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
                            else:
                                logger.warning(f"Normalization parameter dimensions ({mean.size}) don't match data ({X_sequences.shape[2]}) for {model_type}")
                                # Apply a simpler normalization as fallback
                                X_sequences = (X_sequences - np.mean(X_sequences, axis=(0, 1), keepdims=True)) / (np.std(X_sequences, axis=(0, 1), keepdims=True) + 1e-8)
                        else:
                            logger.warning(f"Invalid normalization parameters format for {model_type}")
                            # Apply a simpler normalization as fallback
                            X_sequences = (X_sequences - np.mean(X_sequences, axis=(0, 1), keepdims=True)) / (np.std(X_sequences, axis=(0, 1), keepdims=True) + 1e-8)
                    
                    preprocessed_data[model_type] = X_sequences
            except Exception as e:
                logger.error(f"Error preprocessing data for {model_type}: {e}")
                logger.error(traceback.format_exc())
        
        # Log summary of processed models
        logger.info(f"Successfully preprocessed data for {len(preprocessed_data)}/{len(self.models)} models")
        
        return preprocessed_data
        
    def _ensure_required_features(self, market_data):
        """
        Ensure all commonly required features are present in the market data
        by calculating them if needed
        
        Args:
            market_data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Market data with additional calculated features
        """
        # Make a copy to avoid modifying the original
        data = market_data.copy()
        
        # Track added features for logging
        added_features = []
        
        # Ensure OHLCV columns exist with standard names
        ohlc_columns = {
            'open': ['open', f'open_{self.timeframe}', 'Open'],
            'high': ['high', f'high_{self.timeframe}', 'High'],
            'low': ['low', f'low_{self.timeframe}', 'Low'],
            'close': ['close', f'close_{self.timeframe}', 'Close'],
            'volume': ['volume', f'volume_{self.timeframe}', 'Volume']
        }
        
        # Standardize OHLCV column names
        for std_name, variants in ohlc_columns.items():
            if std_name not in data.columns:
                for variant in variants:
                    if variant in data.columns:
                        data[std_name] = data[variant]
                        logger.debug(f"Standardized column name: {variant} -> {std_name}")
                        added_features.append(f"{std_name} (from {variant})")
                        break
        
        # Ensure we have basic OHLCV data
        required_ohlcv = ['open', 'high', 'low', 'close']
        missing_ohlcv = [col for col in required_ohlcv if col not in data.columns]
        
        if missing_ohlcv:
            logger.warning(f"Missing essential OHLCV columns: {missing_ohlcv}")
            # If we're missing close but have other price data, use it as a fallback
            if 'close' not in data.columns:
                if 'open' in data.columns:
                    data['close'] = data['open']
                    logger.info("Using 'open' as fallback for missing 'close'")
                    added_features.append('close (from open)')
                elif 'last' in data.columns:
                    data['close'] = data['last']
                    logger.info("Using 'last' as fallback for missing 'close'")
                    added_features.append('close (from last)')
                    
            # If we're still missing essential data, provide basic fallbacks
            for col in missing_ohlcv:
                if col not in data.columns and 'close' in data.columns:
                    if col in ['high', 'low']:
                        # Create synthetic high/low as ±1% of close
                        factor = 1.01 if col == 'high' else 0.99
                        data[col] = data['close'] * factor
                        logger.info(f"Created synthetic '{col}' based on close price")
                        added_features.append(f"{col} (synthetic)")
                    elif col == 'open':
                        # Use close from previous period as open
                        data[col] = data['close'].shift(1)
                        data[col] = data[col].fillna(data['close'])
                        logger.info(f"Created synthetic '{col}' from shifted close")
                        added_features.append(f"{col} (from shifted close)")
        
        # Ensure we have a timestamp column if index is datetime
        if 'timestamp' not in data.columns and hasattr(data.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data.index):
            data['timestamp'] = data.index
            logger.debug("Added timestamp from index")
            added_features.append('timestamp (from index)')
        
        # Calculate RSI if not present
        if 'rsi' not in data.columns and 'RSI' not in data.columns and 'close' in data.columns:
            try:
                delta = data['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
                data['rsi'] = 100 - (100 / (1 + rs))
                data['rsi'] = data['rsi'].fillna(50)  # Fill NaN values with neutral RSI
                logger.debug("Added calculated RSI")
                added_features.append('rsi')
            except Exception as e:
                logger.warning(f"Failed to calculate RSI: {e}")
        elif 'RSI' in data.columns and 'rsi' not in data.columns:
            # Standardize RSI column name
            data['rsi'] = data['RSI']
            added_features.append('rsi (from RSI)')
        
        # Calculate Moving Averages (both SMA and EMA) if not present
        for period in [5, 9, 20, 21, 50, 100, 200]:
            sma_col = f'sma{period}'
            ema_col = f'ema{period}'
            
            # Calculate SMA if not present
            if sma_col not in data.columns and 'close' in data.columns:
                try:
                    data[sma_col] = data['close'].rolling(window=period).mean()
                    data[sma_col] = data[sma_col].fillna(data['close'])
                    logger.debug(f"Added calculated {sma_col}")
                    added_features.append(sma_col)
                except Exception as e:
                    logger.warning(f"Failed to calculate {sma_col}: {e}")
            
            # Calculate EMA if not present
            if ema_col not in data.columns and 'close' in data.columns:
                try:
                    data[ema_col] = data['close'].ewm(span=period, adjust=False).mean()
                    data[ema_col] = data[ema_col].fillna(data['close'])
                    logger.debug(f"Added calculated {ema_col}")
                    added_features.append(ema_col)
                except Exception as e:
                    logger.warning(f"Failed to calculate {ema_col}: {e}")
        
        # Calculate MACD if not present
        if 'macd' not in data.columns and 'close' in data.columns:
            try:
                ema12 = data['close'].ewm(span=12, adjust=False).mean()
                ema26 = data['close'].ewm(span=26, adjust=False).mean()
                data['macd'] = ema12 - ema26
                data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
                data['macd_hist'] = data['macd'] - data['macd_signal']
                
                # Fill NaN values with zeros
                data['macd'] = data['macd'].fillna(0)
                data['macd_signal'] = data['macd_signal'].fillna(0)
                data['macd_hist'] = data['macd_hist'].fillna(0)
                
                logger.debug("Added calculated MACD indicators")
                added_features.extend(['macd', 'macd_signal', 'macd_hist'])
            except Exception as e:
                logger.warning(f"Failed to calculate MACD: {e}")
        
        # Calculate Bollinger Bands if not present
        if ('bollinger_upper' not in data.columns and 'bollinger_lower' not in data.columns) and 'close' in data.columns:
            try:
                window = 20
                std_dev = 2
                sma = data['close'].rolling(window=window).mean()
                rolling_std = data['close'].rolling(window=window).std()
                
                data['bollinger_upper'] = sma + (rolling_std * std_dev)
                data['bollinger_lower'] = sma - (rolling_std * std_dev)
                data['bollinger_mid'] = sma
                
                # Fill NaN values
                data['bollinger_upper'] = data['bollinger_upper'].fillna(data['close'] * 1.05)
                data['bollinger_lower'] = data['bollinger_lower'].fillna(data['close'] * 0.95)
                data['bollinger_mid'] = data['bollinger_mid'].fillna(data['close'])
                
                logger.debug("Added calculated Bollinger Bands")
                added_features.extend(['bollinger_upper', 'bollinger_lower', 'bollinger_mid'])
            except Exception as e:
                logger.warning(f"Failed to calculate Bollinger Bands: {e}")
        
        # Handle both naming conventions for Bollinger Bands
        if 'bb_upper' not in data.columns and 'bollinger_upper' in data.columns:
            data['bb_upper'] = data['bollinger_upper']
            data['bb_lower'] = data['bollinger_lower']
            data['bb_middle'] = data['bollinger_mid']
            added_features.extend(['bb_upper', 'bb_lower', 'bb_middle'])
        elif 'bollinger_upper' not in data.columns and 'bb_upper' in data.columns:
            data['bollinger_upper'] = data['bb_upper']
            data['bollinger_lower'] = data['bb_lower']
            data['bollinger_mid'] = data['bb_middle']
            added_features.extend(['bollinger_upper', 'bollinger_lower', 'bollinger_mid'])
        
        # Calculate ADX if not present
        if 'adx' not in data.columns and all(col in data.columns for col in ['high', 'low', 'close']):
            try:
                period = 14
                # Calculate True Range
                data['tr1'] = data['high'] - data['low']
                data['tr2'] = abs(data['high'] - data['close'].shift(1))
                data['tr3'] = abs(data['low'] - data['close'].shift(1))
                data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
                
                # Calculate directional movement
                data['up_move'] = data['high'] - data['high'].shift(1)
                data['down_move'] = data['low'].shift(1) - data['low']
                
                # Calculate positive and negative directional movement
                data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
                data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
                
                # Calculate smoothed values
                data['tr14'] = data['tr'].rolling(window=period).sum()
                data['plus_di14'] = 100 * (data['plus_dm'].rolling(window=period).sum() / data['tr14'])
                data['minus_di14'] = 100 * (data['minus_dm'].rolling(window=period).sum() / data['tr14'])
                
                # Calculate DX and ADX
                data['dx'] = 100 * abs(data['plus_di14'] - data['minus_di14']) / (data['plus_di14'] + data['minus_di14'])
                data['adx'] = data['dx'].rolling(window=period).mean()
                
                # Fill NaN values with a default value (25 is a neutral value for ADX)
                data['adx'] = data['adx'].fillna(25)
                
                # Clean up intermediate calculations
                for col in ['tr1', 'tr2', 'tr3', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'tr14', 'plus_di14', 'minus_di14', 'dx']:
                    if col in data.columns:
                        data = data.drop(col, axis=1)
                
                logger.debug("Added calculated ADX")
                added_features.append('adx')
            except Exception as e:
                logger.warning(f"Failed to calculate ADX: {e}")
                # Provide a fallback for ADX if calculation fails
                data['adx'] = 25  # Neutral ADX value
                added_features.append('adx (fallback)')
        
        # Calculate ATR if not present
        if 'atr' not in data.columns and all(col in data.columns for col in ['high', 'low', 'close']):
            try:
                period = 14
                high_low = data['high'] - data['low']
                high_close = (data['high'] - data['close'].shift(1)).abs()
                low_close = (data['low'] - data['close'].shift(1)).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                data['atr'] = true_range.rolling(period).mean()
                
                # Handle NaN values
                data['atr'] = data['atr'].fillna(high_low)
                
                logger.debug("Added calculated ATR")
                added_features.append('atr')
            except Exception as e:
                logger.warning(f"Failed to calculate ATR: {e}")
                # Provide fallback for ATR if calculation fails
                if 'high' in data.columns and 'low' in data.columns:
                    data['atr'] = (data['high'] - data['low']).mean() * 0.2
                    added_features.append('atr (fallback)')
        
        # Calculate CCI if not present
        if 'cci' not in data.columns and all(col in data.columns for col in ['high', 'low', 'close']):
            try:
                period = 20
                tp = (data['high'] + data['low'] + data['close']) / 3
                tp_sma = tp.rolling(window=period).mean()
                tp_md = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
                data['cci'] = (tp - tp_sma) / (0.015 * tp_md)
                
                # Fill NaNs with 0 (neutral CCI value)
                data['cci'] = data['cci'].fillna(0)
                
                logger.debug("Added calculated CCI")
                added_features.append('cci')
            except Exception as e:
                logger.warning(f"Failed to calculate CCI: {e}")
                # Provide fallback for CCI if calculation fails
                data['cci'] = 0  # Neutral CCI value
                added_features.append('cci (fallback)')
        
        # Calculate MFI if not present
        if 'mfi' not in data.columns and all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            try:
                period = 14
                # Calculate typical price
                tp = (data['high'] + data['low'] + data['close']) / 3
                
                # Calculate money flow
                raw_money_flow = tp * data['volume']
                
                # Get change in typical price
                tp_diff = tp.diff()
                
                # Calculate positive and negative money flow
                positive_flow = np.where(tp_diff > 0, raw_money_flow, 0)
                negative_flow = np.where(tp_diff < 0, raw_money_flow, 0)
                
                # Calculate money flow ratio
                positive_sum = pd.Series(positive_flow).rolling(window=period).sum()
                negative_sum = pd.Series(negative_flow).rolling(window=period).sum()
                
                # Calculate MFI
                money_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 9999)
                data['mfi'] = 100 - (100 / (1 + money_ratio))
                
                # Fill NaNs with neutral value
                data['mfi'] = data['mfi'].fillna(50)
                
                logger.debug("Added calculated MFI")
                added_features.append('mfi')
            except Exception as e:
                logger.warning(f"Failed to calculate MFI: {e}")
                # Provide fallback for MFI if calculation fails
                data['mfi'] = 50  # Neutral MFI value
                added_features.append('mfi (fallback)')
        
        # Calculate OBV if not present
        if 'obv' not in data.columns and 'close' in data.columns and 'volume' in data.columns:
            try:
                # Calculate price change
                price_change = data['close'].diff()
                
                # OBV calculation
                obv = np.zeros(len(data))
                for i in range(1, len(data)):
                    if price_change.iloc[i] > 0:
                        obv[i] = obv[i-1] + data['volume'].iloc[i]
                    elif price_change.iloc[i] < 0:
                        obv[i] = obv[i-1] - data['volume'].iloc[i]
                    else:
                        obv[i] = obv[i-1]
                        
                data['obv'] = obv
                
                logger.debug("Added calculated OBV")
                added_features.append('obv')
            except Exception as e:
                logger.warning(f"Failed to calculate OBV: {e}")
                # Provide fallback for OBV if calculation fails
                data['obv'] = 0
                added_features.append('obv (fallback)')
        
        # Calculate rate of change indicators if not present
        for period in [1, 5, 10, 20]:
            roc_col = f'price_rate_of_change_{period}'
            if roc_col not in data.columns and 'close' in data.columns:
                try:
                    data[roc_col] = data['close'].pct_change(periods=period) * 100
                    data[roc_col] = data[roc_col].fillna(0)
                    logger.debug(f"Added calculated {roc_col}")
                    added_features.append(roc_col)
                except Exception as e:
                    logger.warning(f"Failed to calculate {roc_col}: {e}")
                    # Provide fallback
                    data[roc_col] = 0
                    added_features.append(f"{roc_col} (fallback)")
        
        # Calculate returns if not present
        if 'return' not in data.columns and 'close' in data.columns:
            try:
                data['return'] = data['close'].pct_change()
                data['return'] = data['return'].fillna(0)
                logger.debug("Added calculated return")
                added_features.append('return')
            except Exception as e:
                logger.warning(f"Failed to calculate return: {e}")
        
        # Calculate volatility if not present
        if 'volatility' not in data.columns and 'return' in data.columns:
            try:
                data['volatility'] = data['return'].rolling(window=20).std()
                data['volatility'] = data['volatility'].fillna(0.01)  # Default to 1% volatility
                logger.debug("Added calculated volatility")
                added_features.append('volatility')
            except Exception as e:
                logger.warning(f"Failed to calculate volatility: {e}")
                # Provide fallback
                data['volatility'] = 0.01
                added_features.append('volatility (fallback)')
        elif 'volatility' not in data.columns and 'close' in data.columns:
            try:
                returns = data['close'].pct_change()
                data['volatility'] = returns.rolling(window=20).std()
                data['volatility'] = data['volatility'].fillna(0.01)  # Default to 1% volatility
                logger.debug("Added calculated volatility from close")
                added_features.append('volatility (from close)')
            except Exception as e:
                logger.warning(f"Failed to calculate volatility from close: {e}")
                # Provide fallback
                data['volatility'] = 0.01
                added_features.append('volatility (fallback)')
        
        # Calculate trend indicators for decision-making algorithms
        if 'ema9' in data.columns and 'ema21' in data.columns:
            try:
                data['ema_trend'] = np.where(data['ema9'] > data['ema21'], 1, -1)
                logger.debug("Added calculated EMA trend")
                added_features.append('ema_trend')
            except Exception as e:
                logger.warning(f"Failed to calculate ema_trend: {e}")
                
        if 'close' in data.columns and 'sma20' in data.columns:
            try:
                data['sma_trend'] = np.where(data['close'] > data['sma20'], 1, -1)
                logger.debug("Added calculated SMA trend")
                added_features.append('sma_trend')
            except Exception as e:
                logger.warning(f"Failed to calculate sma_trend: {e}")
            
            try:
                data['price_vs_sma20'] = ((data['close'] / data['sma20']) - 1) * 100
                logger.debug("Added calculated price vs SMA20 percentage")
                added_features.append('price_vs_sma20')
            except Exception as e:
                logger.warning(f"Failed to calculate price_vs_sma20: {e}")
        
        if added_features:
            logger.info(f"Added {len(added_features)} missing features to the dataset: {', '.join(added_features[:10])}" + 
                      (f" and {len(added_features) - 10} more" if len(added_features) > 10 else ""))
            
        # Final check for NaN values and fill them
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data
    
    def predict(self, market_data):
        """
        Generate ensemble prediction from all available models
        
        Args:
            market_data (pd.DataFrame): Recent market data with indicators
            
        Returns:
            tuple: (prediction, confidence, details)
        """
        # First detect market regime
        self.detect_market_regime(market_data)
        
        # Adjust weights for current regime
        regime_weights = self._adjust_weights_for_regime()
        
        # Further adjust weights by recent performance
        adjusted_weights = self._adjust_weights_by_performance(regime_weights)
        
        # Preprocess data for each model
        preprocessed_data = self._preprocess_data(market_data)
        
        # Generate predictions
        predictions = {}
        confidences = {}
        errors = {}
        
        # Log diagnostic information
        model_count = len(self.models)
        preprocessed_count = len(preprocessed_data)
        
        logger.info(f"Generating predictions for {preprocessed_count}/{model_count} models")
        
        # If no models were preprocessed successfully, provide detailed error information
        if preprocessed_count == 0 and model_count > 0:
            logger.error("No models were successfully preprocessed. Input data may be incompatible.")
            logger.debug(f"Market data columns: {list(market_data.columns) if market_data is not None else 'None'}")
            logger.debug(f"Market data shape: {market_data.shape if market_data is not None else 'None'}")
            
            # Attempt to provide more specific diagnostic information
            if market_data is None or len(market_data) == 0:
                logger.error("Market data is empty or None")
            elif len(market_data) < 60:
                logger.error(f"Market data length {len(market_data)} is less than minimum sequence length (60)")
            
            # Check if any required features are completely missing
            for model_type, features in self.feature_names.items():
                missing = []
                for feature in features:
                    if feature not in market_data.columns:
                        missing.append(feature)
                
                if missing:
                    logger.error(f"Model {model_type} missing critical features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        
        for model_type, model in self.models.items():
            if model_type in preprocessed_data:
                try:
                    # Get the last sequence for prediction
                    X = preprocessed_data[model_type][-1:]
                    
                    # Log tensor shape for debugging
                    logger.debug(f"Input tensor shape for {model_type}: {X.shape}")
                    
                    # Generate prediction
                    if model_type == "attention_gru":
                        # For Attention-GRU models
                        raw_pred = model.predict(X, verbose=0)
                        # Handle different output shapes
                        if isinstance(raw_pred, list):
                            pred = raw_pred[0][0]
                        elif len(raw_pred.shape) > 2:
                            pred = raw_pred[0][0]
                        elif len(raw_pred.shape) == 2:
                            pred = raw_pred[0][0]
                        else:
                            pred = raw_pred[0]
                    else:
                        # Standard model prediction
                        pred = model.predict(X, verbose=0)[0][0]
                    
                    # Store prediction and confidence
                    predictions[model_type] = pred
                    confidences[model_type] = abs(pred)
                    
                    # Log successful prediction for debugging
                    logger.debug(f"Successful prediction from {model_type}: {pred:.4f} (confidence: {abs(pred):.4f})")
                    
                except Exception as e:
                    errors[model_type] = str(e)
                    logger.error(f"Error generating prediction for {model_type}: {e}")
                    logger.error(traceback.format_exc())
        
        # Calculate weighted ensemble prediction
        weighted_sum = 0
        weight_total = 0
        
        for model_type, pred in predictions.items():
            if model_type in adjusted_weights:
                weight = adjusted_weights[model_type]
                weighted_sum += pred * weight
                weight_total += weight
        
        # Ensemble prediction
        if weight_total > 0:
            ensemble_prediction = weighted_sum / weight_total
        else:
            ensemble_prediction = 0
        
        # Calculate ensemble confidence
        confidence_factors = []
        
        # Factor 1: Agreement between models
        if predictions:
            sign_agreement = sum(1 if pred > 0 else -1 for pred in predictions.values())
            sign_agreement = abs(sign_agreement) / len(predictions)
            confidence_factors.append(sign_agreement)
        
        # Factor 2: Average confidence of individual models
        if confidences:
            avg_confidence = sum(confidences.values()) / len(confidences)
            confidence_factors.append(min(1.0, avg_confidence * 2))  # Scale to 0-1
        
        # Factor 3: Regime volatility factor
        volatile_regimes = ["volatile", "volatile_trending", "volatile_ranging"]
        volatility_factor = 0.7 if any(regime in self.current_regime for regime in volatile_regimes) else 1.0
        confidence_factors.append(volatility_factor)
        
        # Calculate final confidence
        ensemble_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Create details
        details = {
            "regime": self.current_regime,
            "model_predictions": predictions,
            "model_confidences": confidences,
            "adjusted_weights": adjusted_weights,
            "confidence_factors": confidence_factors,
            "errors": errors,
            "models_loaded": len(self.models),
            "models_used": len(predictions),
            "models_features": {model_type: len(features) for model_type, features in self.feature_names.items()} if hasattr(self, 'feature_names') else {},
            "data_columns": list(market_data.columns) if market_data is not None else []
        }
        
        return ensemble_prediction, ensemble_confidence, details
    
    def calculate_position_sizing_confidence(self, ensemble_confidence, market_data=None, additional_factors=None):
        """
        Calculate an enhanced confidence score specifically for position sizing
        
        This enhanced confidence considers additional market factors and historical performance
        to provide a more nuanced confidence score suitable for dynamic position sizing.
        
        Args:
            ensemble_confidence (float): Base confidence from the prediction ensemble (0.0-1.0)
            market_data (pd.DataFrame, optional): Recent market data for additional analysis
            additional_factors (dict, optional): Additional factors to consider:
                - recent_win_rate (float): Recent win rate of the model
                - price_action_strength (float): Strength of recent price action
                - atr_volatility (float): Current ATR volatility
                - trend_strength (float): Strength of the current trend
                
        Returns:
            tuple: (position_confidence, sizing_details)
        """
        # Start with the base confidence from ensemble prediction
        confidence_factors = [ensemble_confidence]
        sizing_details = {"base_ensemble_confidence": ensemble_confidence}
        
        # Factor weights for final blending
        weights = {
            "ensemble_confidence": 0.35,   # Base confidence is most important
            "market_regime": 0.20,         # Market regime is a strong factor
            "price_action": 0.15,          # Recent price action
            "volatility": 0.15,            # Market volatility consideration
            "win_rate": 0.15               # Recent model performance
        }
        sizing_details["factor_weights"] = weights
        
        # Apply market regime adjustment
        regime_factor = self._calculate_regime_confidence_factor()
        confidence_factors.append(regime_factor)
        sizing_details["regime_factor"] = regime_factor
        sizing_details["current_regime"] = self.current_regime
        
        # Calculate price action strength if market data is available
        price_action_factor = 0.75  # Default to moderate confidence
        if market_data is not None and len(market_data) > 5:
            try:
                # Extract price data
                closes = market_data['close'].values
                
                # Calculate directional price movement strength
                price_changes = np.diff(closes[-10:])
                directional_strength = np.abs(np.sum(price_changes)) / np.sum(np.abs(price_changes)) if np.sum(np.abs(price_changes)) > 0 else 0
                
                # Get recent momentum using RSI
                rsi_periods = 14
                if len(closes) > rsi_periods:
                    delta = np.diff(closes[-rsi_periods-1:])
                    gains = delta * 0
                    losses = delta * 0
                    gains[delta > 0] = delta[delta > 0]
                    losses[-delta > 0] = -delta[-delta > 0]
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Convert RSI to factor (higher near extremes, lower in middle)
                        rsi_factor = abs(rsi - 50) / 50  # 0 at RSI 50, 1 at RSI 0 or 100
                    else:
                        rsi_factor = 0.5
                else:
                    rsi_factor = 0.5
                
                # Combine factors
                price_action_factor = 0.5 + (0.25 * directional_strength + 0.25 * rsi_factor)
                price_action_factor = min(1.0, max(0.5, price_action_factor))
                
                # Add factor to details
                sizing_details["price_action_details"] = {
                    "directional_strength": float(directional_strength),
                    "rsi_factor": float(rsi_factor)
                }
            except Exception as e:
                logger.warning(f"Error calculating price action strength: {e}")
        
        confidence_factors.append(price_action_factor)
        sizing_details["price_action_factor"] = price_action_factor
        
        # Apply volatility adjustment if provided
        volatility_factor = 0.75  # Default to moderate confidence
        if additional_factors and 'atr_volatility' in additional_factors:
            try:
                atr_volatility = additional_factors['atr_volatility']
                
                # Inverse relationship - higher volatility = lower confidence
                # Typical ATR/Price ratios range from 0.005 (0.5%) to 0.03 (3%)
                # Scale from 1.0 to 0.5 as volatility increases
                volatility_factor = max(0.5, 1.0 - (atr_volatility * 25))
                
                sizing_details["volatility_details"] = {
                    "atr_volatility": atr_volatility
                }
            except Exception as e:
                logger.warning(f"Error applying volatility adjustment: {e}")
        
        confidence_factors.append(volatility_factor)
        sizing_details["volatility_factor"] = volatility_factor
        
        # Apply historical win rate adjustment if provided
        win_rate_factor = 0.75  # Default to moderate confidence
        if additional_factors and 'recent_win_rate' in additional_factors:
            try:
                win_rate = additional_factors['recent_win_rate']
                
                # Scale win rate to confidence factor (0.5 to 1.0)
                # A win rate of 50% = 0.5 factor, 100% = 1.0 factor
                win_rate_factor = 0.5 + (0.5 * win_rate)
                
                sizing_details["win_rate_details"] = {
                    "recent_win_rate": win_rate
                }
            except Exception as e:
                logger.warning(f"Error applying win rate adjustment: {e}")
        
        confidence_factors.append(win_rate_factor)
        sizing_details["win_rate_factor"] = win_rate_factor
        
        # Apply additional custom factors if provided
        if additional_factors:
            for factor_name, factor_value in additional_factors.items():
                if factor_name not in ['recent_win_rate', 'atr_volatility'] and isinstance(factor_value, (int, float)):
                    sizing_details[f"additional_{factor_name}"] = factor_value
        
        # Calculate the weighted position confidence
        weighted_confidence = 0.0
        weights_sum = 0.0
        
        # Map factors to their weights
        factor_mapping = [
            ("ensemble_confidence", ensemble_confidence),
            ("market_regime", regime_factor),
            ("price_action", price_action_factor),
            ("volatility", volatility_factor),
            ("win_rate", win_rate_factor)
        ]
        
        for factor_name, factor_value in factor_mapping:
            if factor_name in weights:
                weight = weights[factor_name]
                weighted_confidence += factor_value * weight
                weights_sum += weight
        
        # Normalize if needed
        if weights_sum > 0:
            position_confidence = weighted_confidence / weights_sum
        else:
            # Fallback to simple average
            position_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Ensure confidence is within valid range
        position_confidence = min(1.0, max(0.5, position_confidence))
        
        # Final confidence and details
        sizing_details["position_confidence"] = position_confidence
        sizing_details["confidence_factors"] = confidence_factors
        
        logger.debug(f"Enhanced position sizing confidence: {position_confidence:.4f} (base: {ensemble_confidence:.4f})")
        
        return position_confidence, sizing_details
        
    def _calculate_regime_confidence_factor(self):
        """
        Calculate a confidence factor based on the current market regime
        
        Returns:
            float: Regime-based confidence factor
        """
        # Define confidence levels for different market regimes
        regime_confidence = {
            "normal_trending_up": 0.90,    # High confidence in normal uptrend
            "normal_trending_down": 0.85,  # Good confidence in normal downtrend
            "normal_ranging": 0.70,        # Moderate confidence in normal ranging
            "volatile_trending_up": 0.75,  # Reduced confidence in volatile uptrend
            "volatile_trending_down": 0.70, # Lower confidence in volatile downtrend
            "volatile_ranging": 0.60,      # Low confidence in volatile ranging
            "volatile": 0.65,              # Low confidence in general volatility
            "unknown": 0.70                # Default moderate confidence
        }
        
        # Get confidence factor for current regime
        return regime_confidence.get(self.current_regime, 0.70)
    
    def update_performance(self, prediction_outcomes, trade_details=None):
        """
        Update performance history based on prediction outcomes and optionally trade details
        
        Args:
            prediction_outcomes (dict): Dictionary with model_type -> outcome pairs
                where outcome is 1 for correct prediction, -1 for incorrect, 0 for neutral
            trade_details (dict, optional): Dictionary with additional trade information like:
                - actual_price_change (float): The actual price change that occurred
                - predicted_direction (int): The predicted direction (1 for up, -1 for down)
                - predicted_magnitude (float): The predicted magnitude of movement
                - market_regime (str): The detected market regime
                - timestamp (str): ISO timestamp of the prediction
        """
        timestamp = datetime.now().isoformat() if trade_details is None or 'timestamp' not in trade_details else trade_details.get('timestamp')
        market_regime = self.current_regime if trade_details is None or 'market_regime' not in trade_details else trade_details.get('market_regime')
        
        # Add context information to log messages
        context_info = f"[{timestamp}] Market regime: {market_regime}"
        
        # Data to save for performance analysis - extends beyond just the outcome score
        performance_data = {}
        
        # Process each model's outcome
        for model_type, outcome in prediction_outcomes.items():
            if model_type in self.models:  # Only update for models that exist
                # Initialize model's performance history if it doesn't exist
                if model_type not in self.performance_history:
                    self.performance_history[model_type] = []
                
                # Create detailed performance entry instead of just the score
                entry = {
                    'outcome': outcome,  # The simple score (1, 0, -1)
                    'timestamp': timestamp,
                    'market_regime': market_regime,
                }
                
                # Add trade details if available
                if trade_details is not None:
                    entry.update({
                        'actual_price_change': trade_details.get('actual_price_change', None),
                        'predicted_direction': trade_details.get('predicted_direction', None),
                        'predicted_magnitude': trade_details.get('predicted_magnitude', None),
                        'weight': self.weights.get(model_type, 0),
                    })
                
                # Store the detailed entry
                performance_data[model_type] = entry
                
                # Also keep the simple outcome in the historical record
                self.performance_history[model_type].append(outcome)
                
                # Log individual model performance
                accuracy_msg = "CORRECT" if outcome == 1 else ("INCORRECT" if outcome == -1 else "NEUTRAL")
                logger.info(f"{context_info} | Model {model_type} prediction: {accuracy_msg}")
                
                # Keep history at a reasonable size
                max_history = 100
                if len(self.performance_history[model_type]) > max_history:
                    self.performance_history[model_type] = self.performance_history[model_type][-max_history:]
        
        # Calculate aggregate performance metrics
        if prediction_outcomes:
            correct = sum(1 for outcome in prediction_outcomes.values() if outcome == 1)
            incorrect = sum(1 for outcome in prediction_outcomes.values() if outcome == -1)
            neutral = sum(1 for outcome in prediction_outcomes.values() if outcome == 0)
            total = len(prediction_outcomes)
            
            if total > 0:
                accuracy = correct / total if total > 0 else 0
                logger.info(f"{context_info} | Ensemble prediction summary: {correct}/{total} correct ({accuracy:.2%}), {incorrect} incorrect, {neutral} neutral")
        
        # Update weights based on performance
        self.weights = self._adjust_weights_by_performance(self.weights)
        
        # Log weight changes
        weight_str = ", ".join([f"{m}: {w:.4f}" for m, w in sorted(self.weights.items(), key=lambda x: x[1], reverse=True)])
        logger.info(f"{context_info} | Updated model weights: {weight_str}")
        
        # Save updated weights
        self._save_weights()
        
        # Detailed performance info can be returned for external logging if needed
        return performance_data
    
    def _save_weights(self):
        """Save current ensemble weights to disk"""
        weights_file = os.path.join(ENSEMBLE_DIR, f"{self.ticker_symbol}_weights.json")
        
        try:
            with open(weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.info(f"Saved ensemble weights to {weights_file}")
        except Exception as e:
            logger.error(f"Error saving ensemble weights: {e}")
            traceback.print_exc()
    
    def get_status(self):
        """
        Get current status and information about the ensemble
        
        Returns:
            dict: Status information
        """
        # Get model information
        model_info = {}
        for model_type, model in self.models.items():
            input_shape = model.input_shape
            # Add performance information if available
            perf_history = self.performance_history.get(model_type, [])
            recent_perf = perf_history[-10:] if perf_history else []
            
            model_info[model_type] = {
                "input_shape": str(input_shape),
                "weight": self.weights.get(model_type, 0),
                "recent_performance": sum(recent_perf) / len(recent_perf) if recent_perf else None,
                "performance_samples": len(recent_perf)
            }
        
        # Calculate model usage stats
        total_params = sum(model.count_params() for model in self.models.values())
        
        return {
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "model_count": len(self.models),
            "total_parameters": total_params,
            "current_regime": self.current_regime,
            "models": model_info,
            "weights": self.weights
        }
        
    def auto_prune_models(self, performance_threshold=0.0, min_samples=5, keep_minimum=2):
        """
        Automatically detect and prune underperforming models from the ensemble
        
        This helps maintain high model quality by removing models that consistently
        make incorrect predictions.
        
        Args:
            performance_threshold (float): Minimum performance score to keep a model
                                          (range: -1 to 1, where negative means more wrong than right)
            min_samples (int): Minimum number of performance samples required before pruning
            keep_minimum (int): Minimum number of models to keep regardless of performance
            
        Returns:
            tuple: (list of pruned models, list of kept models)
        """
        logger.info(f"Auto-pruning models with performance threshold: {performance_threshold}")
        
        models_to_prune = []
        models_to_keep = []
        model_performances = {}
        
        # Check performance of each model
        for model_type, history in self.performance_history.items():
            # Skip if not loaded or not in current weights
            if model_type not in self.models or model_type not in self.weights:
                continue
                
            # Skip if not enough performance samples
            if len(history) < min_samples:
                logger.info(f"Skipping model {model_type} - insufficient performance samples ({len(history)}/{min_samples})")
                models_to_keep.append(model_type)
                continue
                
            # Calculate average performance (recent values get more weight)
            recent_history = history[-min_samples:]
            
            # Calculate weighted average (newer outcomes have higher weight)
            weighted_sum = sum(score * (i + 1) for i, score in enumerate(recent_history))
            weight_sum = sum(i + 1 for i in range(len(recent_history)))
            avg_performance = weighted_sum / weight_sum if weight_sum > 0 else 0
            
            logger.info(f"Model {model_type} performance: {avg_performance:.4f} ({len(history)} samples)")
            
            # Store performance for sorting
            model_performances[model_type] = avg_performance
        
        # If we don't have enough models to evaluate, keep all
        if len(model_performances) <= keep_minimum:
            logger.info(f"Not enough models to prune, keeping all {len(self.models)} models")
            return [], list(self.models.keys())
        
        # Sort models by performance (descending)
        sorted_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)
        
        # Mark models for pruning or keeping
        for i, (model_type, performance) in enumerate(sorted_models):
            # Always keep top-performing models regardless of threshold
            if i < keep_minimum:
                logger.info(f"Keeping top-performing model {model_type} with score {performance:.4f}")
                models_to_keep.append(model_type)
            # For remaining models, apply performance threshold
            elif performance < performance_threshold:
                logger.warning(f"Marking model {model_type} for pruning - performance below threshold: {performance:.4f} < {performance_threshold}")
                models_to_prune.append(model_type)
            else:
                models_to_keep.append(model_type)
        
        # Remove pruned models from the ensemble
        for model_type in models_to_prune:
            if model_type in self.models:
                logger.info(f"Pruning model {model_type} from ensemble")
                del self.models[model_type]
                
                # Also clean up related data
                if model_type in self.norm_params:
                    del self.norm_params[model_type]
                if model_type in self.feature_names:
                    del self.feature_names[model_type]
                    
                # Update weights by removing the pruned model's weight
                if model_type in self.weights:
                    del self.weights[model_type]
        
        # Rebalance weights for remaining models
        if models_to_prune and models_to_keep:
            self._normalize_weights()
            self._save_weights()
            
        # Log detailed status after pruning
        if models_to_prune:
            logger.info(f"Pruned {len(models_to_prune)} models, {len(models_to_keep)} models remaining")
            logger.info(f"Pruned models: {models_to_prune}")
            logger.info(f"Kept models: {models_to_keep}")
            
            # Log new weight distribution
            weight_str = ", ".join([f"{m}: {self.weights.get(m, 0):.4f}" for m in models_to_keep])
            logger.info(f"New weight distribution: {weight_str}")
        else:
            logger.info("No models pruned - all models performing adequately")
            
        return models_to_prune, models_to_keep


def create_hybrid_model(input_shape, use_attention=True, use_transformer=True):
    """
    Create a hybrid model that combines CNN, LSTM/GRU, and optionally
    attention and transformer components
    
    Args:
        input_shape (tuple): Shape of input data
        use_attention (bool): Whether to include attention mechanism
        use_transformer (bool): Whether to include transformer layers
        
    Returns:
        Model: Hybrid model
    """
    inputs = Input(shape=input_shape)
    
    # CNN branch
    cnn = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(128, 3, padding='same', activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(64, activation='relu')(cnn)
    cnn = Dropout(0.2)(cnn)
    
    # LSTM branch
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = BatchNormalization()(lstm)
    
    # GRU branch
    gru = GRU(64, return_sequences=True)(inputs)
    gru = BatchNormalization()(gru)
    
    # Combine LSTM and GRU
    recurrent = tf.keras.layers.Concatenate()([lstm, gru])
    
    # Add attention mechanism if requested
    if use_attention:
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(recurrent, recurrent)
        attention = BatchNormalization()(attention)
        recurrent = Add()([recurrent, attention])
    
    # Add transformer components if requested
    if use_transformer:
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(recurrent)
        attention_output = MultiHeadAttention(
            key_dim=32, num_heads=4, dropout=0.1
        )(x, x)
        attention_output = Dropout(0.1)(attention_output)
        out1 = Add()([recurrent, attention_output])
        
        # Feed Forward
        x = LayerNormalization(epsilon=1e-6)(out1)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64)(x)
        transformer_output = Add()([out1, x])
        
        recurrent = transformer_output
    
    # Global pooling
    recurrent_features = tf.keras.layers.GlobalAveragePooling1D()(recurrent)
    
    # Combine CNN and Recurrent features
    combined = tf.keras.layers.Concatenate()([cnn, recurrent_features])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)
    
    # Output layer
    outputs = Dense(1, activation='tanh')(combined)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_hybrid_model(X_train, y_train, X_val, y_val, 
                       use_attention=True, use_transformer=True,
                       batch_size=32, epochs=50):
    """
    Train a hybrid model
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        use_attention (bool): Whether to use attention mechanism
        use_transformer (bool): Whether to use transformer layers
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for training
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info("Training hybrid model...")
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Create model
        model = create_hybrid_model(input_shape, use_attention, use_transformer)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, "hybrid/model.h5"),
                monitor='val_loss',
                save_best_only=True,
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
        
        logger.info("Hybrid model trained successfully.")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training hybrid model: {e}")
        traceback.print_exc()
        return None, None


def main():
    """Test the ensemble model"""
    # Create ensemble
    ensemble = DynamicWeightedEnsemble()
    
    # Get status
    status = ensemble.get_status()
    logger.info(f"Ensemble Status: {status}")
    
    # If we have models, test a prediction
    if ensemble.models:
        # Create dummy data for testing
        dummy_data = pd.DataFrame({
            f"open_{ensemble.timeframe}": [100, 101, 102, 103, 104] * 12,
            f"high_{ensemble.timeframe}": [105, 106, 107, 108, 109] * 12,
            f"low_{ensemble.timeframe}": [95, 96, 97, 98, 99] * 12,
            f"close_{ensemble.timeframe}": [101, 102, 103, 104, 105] * 12,
            f"volume_{ensemble.timeframe}": [1000, 1100, 1200, 1300, 1400] * 12,
            f"volatility_{ensemble.timeframe}": [0.01, 0.02, 0.01, 0.03, 0.02] * 12,
            f"ema9_{ensemble.timeframe}": [101, 102, 103, 104, 105] * 12,
            f"ema21_{ensemble.timeframe}": [100, 101, 102, 103, 104] * 12
        })
        
        # Make a prediction
        pred, conf, details = ensemble.predict(dummy_data)
        logger.info(f"Prediction: {pred}, Confidence: {conf}")
        logger.info(f"Details: {details}")
        
        # Update performance
        outcomes = {model_type: 1 for model_type in ensemble.models.keys()}
        ensemble.update_performance(outcomes)
    else:
        logger.warning("No models available for prediction")


if __name__ == "__main__":
    main()