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
MODEL_TYPES = ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid"]

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
    
    def _load_models(self):
        """Load all available trained models"""
        self.models = {}
        self.norm_params = {}
        self.feature_names = {}
        
        for model_type in MODEL_TYPES:
            model_path = os.path.join(MODELS_DIR, model_type, f"{self.ticker_symbol}.h5")
            norm_path = os.path.join(MODELS_DIR, model_type, "norm_params.json")
            features_path = os.path.join(MODELS_DIR, model_type, "feature_names.json")
            
            if os.path.exists(model_path) and os.path.exists(norm_path) and os.path.exists(features_path):
                try:
                    # Load model
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
                    traceback.print_exc()
            else:
                logger.warning(f"{model_type} model not found at {model_path}")
        
        logger.info(f"Loaded {len(self.models)} models for {self.ticker_symbol}")
    
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
                "hybrid": 2.0    # Hybrid captures multiple perspectives
            },
            "volatile_trending": {
                "tcn": 1.5,
                "cnn": 1.3,
                "lstm": 1.2,
                "gru": 1.2,
                "bilstm": 1.5,
                "attention": 1.6,
                "transformer": 1.7,
                "hybrid": 2.0
            },
            "volatile_ranging": {
                "tcn": 1.7,
                "cnn": 1.5,
                "lstm": 1.0,
                "gru": 1.0,
                "bilstm": 1.3,
                "attention": 1.5,
                "transformer": 1.5,
                "hybrid": 2.0
            },
            "normal_trending": {
                "tcn": 1.2,
                "cnn": 1.5,
                "lstm": 1.5,
                "gru": 1.5,
                "bilstm": 1.3,
                "attention": 1.2,
                "transformer": 1.2,
                "hybrid": 1.5
            },
            "normal_ranging": {
                "tcn": 1.0,
                "cnn": 1.2,
                "lstm": 1.2,
                "gru": 1.2,
                "bilstm": 1.0,
                "attention": 1.0,
                "transformer": 1.0,
                "hybrid": 1.2
            },
            "normal": {
                "tcn": 1.0,
                "cnn": 1.0,
                "lstm": 1.0,
                "gru": 1.0,
                "bilstm": 1.0,
                "attention": 1.0,
                "transformer": 1.0,
                "hybrid": 1.0
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
        
        for model_type, model in self.models.items():
            try:
                # Get the right columns based on feature names
                if model_type in self.feature_names:
                    feature_names = self.feature_names[model_type]
                    
                    # Extract relevant columns if they exist
                    X = []
                    for feature in feature_names:
                        if feature in market_data.columns:
                            X.append(market_data[feature].values)
                        else:
                            # Try with and without timeframe suffix
                            base_name = feature.split('_')[0]
                            alternate_feature = f"{base_name}_{self.timeframe}"
                            
                            if alternate_feature in market_data.columns:
                                X.append(market_data[alternate_feature].values)
                            else:
                                logger.warning(f"Feature {feature} not found in market data")
                                # Use zeros as placeholder
                                X.append(np.zeros_like(market_data.index.values, dtype=float))
                    
                    # Stack arrays to create input tensor
                    X = np.column_stack(X)
                    
                    # Create sequences for recurrent models
                    input_shape = model.input_shape[1:]
                    sequence_length = input_shape[0]
                    
                    # Create sequences
                    sequences = []
                    for i in range(len(X) - sequence_length + 1):
                        sequences.append(X[i:i+sequence_length])
                    
                    # Convert to numpy array
                    X_sequences = np.array(sequences)
                    
                    # Normalize
                    if model_type in self.norm_params:
                        mean = np.array(self.norm_params[model_type]['mean'])
                        std = np.array(self.norm_params[model_type]['std'])
                        std = np.where(std == 0, 1, std)  # Avoid division by zero
                        X_sequences = (X_sequences - mean) / std
                    
                    preprocessed_data[model_type] = X_sequences
            except Exception as e:
                logger.error(f"Error preprocessing data for {model_type}: {e}")
                traceback.print_exc()
        
        return preprocessed_data
    
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
        
        for model_type, model in self.models.items():
            if model_type in preprocessed_data:
                try:
                    # Get the last sequence for prediction
                    X = preprocessed_data[model_type][-1:]
                    
                    # Generate prediction
                    pred = model.predict(X, verbose=0)[0][0]
                    
                    # Store prediction and confidence
                    predictions[model_type] = pred
                    confidences[model_type] = abs(pred)
                except Exception as e:
                    logger.error(f"Error generating prediction for {model_type}: {e}")
                    traceback.print_exc()
        
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
            "confidence_factors": confidence_factors
        }
        
        return ensemble_prediction, ensemble_confidence, details
    
    def update_performance(self, prediction_outcomes):
        """
        Update performance history based on prediction outcomes
        
        Args:
            prediction_outcomes (dict): Dictionary with model_type -> outcome pairs
                where outcome is 1 for correct prediction, -1 for incorrect, 0 for neutral
        """
        for model_type, outcome in prediction_outcomes.items():
            if model_type in self.performance_history:
                self.performance_history[model_type].append(outcome)
                
                # Keep history at a reasonable size
                max_history = 100
                if len(self.performance_history[model_type]) > max_history:
                    self.performance_history[model_type] = self.performance_history[model_type][-max_history:]
        
        # Update weights based on performance
        self.weights = self._adjust_weights_by_performance(self.weights)
        
        # Save updated weights
        self._save_weights()
    
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
            model_info[model_type] = {
                "input_shape": str(input_shape),
                "weight": self.weights.get(model_type, 0)
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