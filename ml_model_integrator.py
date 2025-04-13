#!/usr/bin/env python3
"""
ML Model Integrator for Kraken Trading Bot

This module integrates trained ML models with the trading bot's strategy system,
allowing the models to contribute signals to the trading decision process.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
DATA_DIR = "historical_data"
DEFAULT_TRADING_PAIR = "SOL/USD"
DEFAULT_TIMEFRAME = "1h"
MODEL_TYPES = ["tcn", "cnn", "lstm"]

class ModelSignalGenerator:
    """
    Class for generating trading signals using trained ML models
    """
    def __init__(self, trading_pair=DEFAULT_TRADING_PAIR, timeframe=DEFAULT_TIMEFRAME):
        """
        Initialize the model signal generator
        
        Args:
            trading_pair (str): Trading pair to generate signals for
            timeframe (str): Timeframe to use
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.models = {}
        self.feature_columns = None
        self.weights = {"tcn": 0.4, "cnn": 0.3, "lstm": 0.3}  # Default weights
        self.performance = {"tcn": 0.0, "cnn": 0.0, "lstm": 0.0}  # Performance tracking
        self.sequence_length = 60  # Default sequence length for models
        self.initialized = False
        self.recent_data = None
        self.recent_predictions = {}
        self.signal_log = []
        self.last_update_time = 0
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            # Check if models exist
            for model_type in MODEL_TYPES:
                model_path = f"{MODELS_DIR}/{model_type}/{self.trading_pair.replace('/', '')}-{self.timeframe}.h5"
                if os.path.exists(model_path):
                    logger.info(f"Loading {model_type} model from {model_path}")
                    self.models[model_type] = load_model(model_path)
                else:
                    logger.warning(f"{model_type} model not found at {model_path}")
            
            # Load feature columns
            feature_file = f"{DATA_DIR}/ml_datasets/{self.trading_pair.replace('/', '')}-{self.timeframe}-features.json"
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Loaded feature columns from {feature_file}")
            else:
                logger.warning(f"Feature columns file not found at {feature_file}")
                # Use default feature columns
                self.feature_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'ema9', 'ema21', 'ema50', 'ema100',
                    'rsi', 'macd', 'signal', 'macd_hist',
                    'upper_band', 'middle_band', 'lower_band',
                    'atr', 'volatility'
                ]
            
            # Load model performance info if available
            info_path = f"{MODELS_DIR}/{self.trading_pair.replace('/', '')}-{self.timeframe}-models-info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    models_info = json.load(f)
                
                # Update performance based on validation accuracy
                for model_type, info in models_info.items():
                    if 'val_accuracy' in info:
                        self.performance[model_type] = info['val_accuracy']
                
                # Update weights based on performance
                total_perf = sum(self.performance.values())
                if total_perf > 0:
                    for model_type in self.weights:
                        if model_type in self.performance:
                            self.weights[model_type] = self.performance[model_type] / total_perf
            
            self.initialized = len(self.models) > 0
            logger.info(f"Model initialization complete. Loaded {len(self.models)} models.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            traceback.print_exc()
    
    def preprocess_market_data(self, market_data):
        """
        Preprocess market data for model input
        
        Args:
            market_data (pd.DataFrame): Market data with OHLCV and indicators
            
        Returns:
            numpy.ndarray: Preprocessed data ready for model input
        """
        try:
            # Check if we have the necessary columns
            missing_columns = [col for col in self.feature_columns if col not in market_data.columns]
            if missing_columns:
                logger.warning(f"Missing columns in market data: {missing_columns}")
                # Fill missing columns with zeros
                for col in missing_columns:
                    market_data[col] = 0.0
            
            # Select and order columns according to feature_columns
            features = market_data[self.feature_columns].values
            
            # Create sequence
            if len(features) < self.sequence_length:
                logger.warning(f"Insufficient data points: {len(features)} < {self.sequence_length}")
                # Pad with zeros if not enough data
                pad_length = self.sequence_length - len(features)
                pad_array = np.zeros((pad_length, len(self.feature_columns)))
                features = np.vstack((pad_array, features))
            
            # Take the most recent sequence_length data points
            features = features[-self.sequence_length:]
            
            # Reshape for model input [batch_size, sequence_length, n_features]
            features = np.expand_dims(features, axis=0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing market data: {e}")
            traceback.print_exc()
            return None
    
    def update_model_weights(self, model_performance=None):
        """
        Update model weights based on recent performance
        
        Args:
            model_performance (dict, optional): Dictionary with performance metrics for each model
        """
        if model_performance:
            # Update performance metrics
            for model_type, perf in model_performance.items():
                if model_type in self.performance:
                    # Blend new performance with existing (70% new, 30% existing)
                    self.performance[model_type] = 0.7 * perf + 0.3 * self.performance[model_type]
        
        # Recalculate weights based on performance
        total_perf = sum(self.performance.values())
        if total_perf > 0:
            for model_type in self.weights:
                if model_type in self.performance:
                    self.weights[model_type] = self.performance[model_type] / total_perf
        
        logger.info(f"Updated model weights: {self.weights}")
    
    def generate_signals(self, market_data):
        """
        Generate trading signals from ML models
        
        Args:
            market_data (pd.DataFrame): Market data with OHLCV and indicators
            
        Returns:
            tuple: (signal_direction, signal_strength, signal_details)
        """
        if not self.initialized:
            logger.warning("Models not initialized, cannot generate signals")
            return "neutral", 0.0, {}
        
        # Throttle updates to once per minute
        current_time = time.time()
        if current_time - self.last_update_time < 60 and self.recent_predictions:
            # Use cached predictions
            logger.info("Using cached predictions")
            return self._calculate_ensemble_signal(self.recent_predictions)
        
        try:
            # Preprocess data
            processed_data = self.preprocess_market_data(market_data)
            if processed_data is None:
                return "neutral", 0.0, {}
            
            # Generate predictions from each model
            predictions = {}
            for model_type, model in self.models.items():
                pred = model.predict(processed_data, verbose=0)
                predictions[model_type] = float(pred[0][0])  # Extract scalar value
                logger.info(f"{model_type.upper()} model prediction: {predictions[model_type]:.4f}")
            
            # Cache predictions
            self.recent_predictions = predictions
            self.last_update_time = current_time
            
            # Calculate ensemble signal
            return self._calculate_ensemble_signal(predictions)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            traceback.print_exc()
            return "neutral", 0.0, {}
    
    def _calculate_ensemble_signal(self, predictions):
        """
        Calculate ensemble signal from model predictions
        
        Args:
            predictions (dict): Predictions from each model
            
        Returns:
            tuple: (signal_direction, signal_strength, signal_details)
        """
        # Calculate weighted average prediction
        weighted_pred = 0.0
        for model_type, pred in predictions.items():
            weight = self.weights.get(model_type, 0.0)
            weighted_pred += pred * weight
        
        # Determine signal direction and strength
        signal_strength = abs(weighted_pred)
        if weighted_pred > 0.15:  # Threshold for buy signal
            signal_direction = "buy"
        elif weighted_pred < -0.15:  # Threshold for sell signal
            signal_direction = "sell"
        else:
            signal_direction = "neutral"
        
        # Normalize signal strength to 0-1 range
        signal_strength = min(max(signal_strength, 0.0), 1.0)
        
        # Prepare signal details
        signal_details = {
            "predictions": predictions,
            "weights": self.weights,
            "weighted_prediction": weighted_pred,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log signal
        self.signal_log.append({
            "timestamp": datetime.now().isoformat(),
            "direction": signal_direction,
            "strength": signal_strength,
            "predictions": predictions,
            "weighted_prediction": weighted_pred
        })
        # Trim log to keep only 100 most recent entries
        if len(self.signal_log) > 100:
            self.signal_log = self.signal_log[-100:]
        
        logger.info(f"ML Ensemble Signal: {signal_direction.upper()} with strength {signal_strength:.4f}")
        
        return signal_direction, signal_strength, signal_details

    def update_performance(self, trade_result):
        """
        Update model performance metrics based on trade results
        
        Args:
            trade_result (dict): Trade result information
        """
        if 'pnl' not in trade_result or 'models_used' not in trade_result:
            logger.warning("Incomplete trade result information, cannot update performance")
            return
        
        # Extract information
        pnl = trade_result['pnl']
        models_used = trade_result['models_used']
        
        # Calculate performance delta based on PnL
        perf_delta = 0.1 if pnl > 0 else -0.1  # Simple binary approach
        
        # Update performance for models that were used
        model_performance = {}
        for model_type in models_used:
            if model_type in self.performance:
                # Update performance based on trade result
                updated_perf = max(0.0, self.performance[model_type] + perf_delta)
                model_performance[model_type] = updated_perf
        
        # Update weights
        self.update_model_weights(model_performance)
        
        logger.info(f"Updated model performance metrics based on trade with PnL {pnl}")

def get_market_data_with_indicators(ohlc_data):
    """
    Process OHLC data and add technical indicators
    
    Args:
        ohlc_data (pd.DataFrame): OHLC data
        
    Returns:
        pd.DataFrame: Processed DataFrame with indicators
    """
    df = ohlc_data.copy()
    
    # Calculate EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Calculate Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Calculate price volatility (standard deviation of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Remove NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def main():
    """Test the ML model integrator"""
    logger.info("Testing ML model integrator")
    
    # Load sample market data
    try:
        # Create sample data if no real data available
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        ohlc_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Add indicators
        market_data = get_market_data_with_indicators(ohlc_data)
        
        # Initialize signal generator
        signal_generator = ModelSignalGenerator()
        
        # Generate signals
        direction, strength, details = signal_generator.generate_signals(market_data)
        
        logger.info(f"Generated signal: {direction} with strength {strength}")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()