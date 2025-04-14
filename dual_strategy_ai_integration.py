#!/usr/bin/env python3
"""
Dual Strategy AI Integration Module

This module provides advanced AI integration for combining ARIMA and Adaptive
strategies using deep learning techniques to achieve 90% win rate and 1000% returns.
It serves as the central hub connecting the ML models with both trading strategies.

Features:
1. AI-powered signal arbitration between ARIMA and Adaptive strategies
2. Dynamic confidence-based position sizing
3. Adaptive leverage optimization
4. Market regime detection for optimal strategy selection
5. Cross-strategy feature fusion

This module is designed to be used with the enhanced_strategy_training.py system
and expects models trained on combined strategy features.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dual_strategy_ai.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "ml_enhanced_config.json"
MODELS_DIR = "models"
REGIME_TYPES = ["bullish", "bearish", "sideways", "volatile"]

class DualStrategyAI:
    """
    Advanced AI system for integrating ARIMA and Adaptive strategies
    with machine learning for optimal trading performance.
    """
    def __init__(
        self,
        trading_pair: str,
        timeframe: str = "1h",
        config_path: str = CONFIG_PATH,
        max_leverage: int = 125,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the Dual Strategy AI system
        
        Args:
            trading_pair: Trading pair (e.g., "SOL/USD")
            timeframe: Timeframe for analysis
            config_path: Path to ML configuration file
            max_leverage: Maximum allowed leverage
            confidence_threshold: Minimum confidence threshold for trades
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.config_path = config_path
        self.max_leverage = max_leverage
        self.confidence_threshold = confidence_threshold
        
        # Create filename version of the pair
        self.pair_filename = trading_pair.replace("/", "")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Load ML models
        self.models = self._load_models()
        
        # Initialize feature preprocessing
        self._initialize_feature_preprocessing()
        
        # Market regime state
        self.current_regime = "neutral"
        self.regime_history = []
        
        # Performance tracking
        self.prediction_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logger.info(f"Dual Strategy AI initialized for {trading_pair} on {timeframe}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ML configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _initialize_parameters(self) -> None:
        """Initialize parameters from configuration"""
        # Get global settings
        global_settings = self.config.get("global_settings", {})
        
        # Update confidence threshold if specified in config
        if "ml_confidence_threshold" in global_settings:
            self.confidence_threshold = global_settings["ml_confidence_threshold"]
        
        # Get asset-specific settings
        asset_settings = self.config.get("asset_specific_settings", {}).get(self.trading_pair, {})
        
        # Get strategy integration settings
        self.strategy_integration = self.config.get("strategy_integration", {})
        self.arima_weight = self.strategy_integration.get("arima_weight", 0.5)
        self.adaptive_weight = self.strategy_integration.get("adaptive_weight", 0.5)
        
        # Get risk management settings
        self.risk_management = self.config.get("risk_management", {})
        self.position_sizing = self.risk_management.get("position_sizing", {})
        self.dynamic_leverage = self.risk_management.get("dynamic_leverage", {})
        
        # Asset-specific risk parameters
        self.risk_params = asset_settings.get("risk_params", {})
        if "max_leverage" in self.risk_params:
            self.max_leverage = min(self.risk_params["max_leverage"], self.max_leverage)
    
    def _load_models(self) -> Dict[str, Any]:
        """
        Load pre-trained ML models
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        try:
            # Construct model directory path
            pair_model_dir = os.path.join(MODELS_DIR, self.pair_filename)
            
            if not os.path.exists(pair_model_dir):
                logger.warning(f"Model directory {pair_model_dir} not found")
                return models
            
            # List model directories for this pair
            model_dirs = [d for d in os.listdir(pair_model_dir) if os.path.isdir(os.path.join(pair_model_dir, d))]
            
            # Sort by timestamp (newest first)
            model_dirs.sort(reverse=True)
            
            # Load the latest model
            if model_dirs:
                latest_model_dir = os.path.join(pair_model_dir, model_dirs[0])
                
                # Check if we have a TensorFlow model
                if os.path.exists(os.path.join(latest_model_dir, "saved_model.pb")):
                    try:
                        # Load the model
                        model = tf.keras.models.load_model(latest_model_dir)
                        models["main"] = model
                        logger.info(f"Loaded TensorFlow model from {latest_model_dir}")
                        
                        # Load feature scaler if available
                        scaler_path = os.path.join(latest_model_dir, "feature_scaler.pkl")
                        if os.path.exists(scaler_path):
                            import joblib
                            models["feature_scaler"] = joblib.load(scaler_path)
                            logger.info(f"Loaded feature scaler from {scaler_path}")
                    except Exception as e:
                        logger.error(f"Error loading TensorFlow model: {e}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return models
    
    def _initialize_feature_preprocessing(self) -> None:
        """Initialize feature preprocessing"""
        # Load feature settings from config
        self.feature_settings = self.config.get("feature_settings", {})
        
        # List of technical indicators to use
        self.technical_indicators = self.feature_settings.get("technical_indicators", [])
        
        # Normalization method
        self.normalization = self.feature_settings.get("normalization", "robust_scaler")
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detect the current market regime
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Market regime type (bullish, bearish, sideways, volatile)
        """
        try:
            # Check if we have enough data
            if len(market_data) < 50:
                logger.warning("Not enough data to detect market regime")
                return "neutral"
            
            # Get latest data (most recent 50 candles)
            data = market_data.iloc[-50:].copy()
            
            # Calculate volatility
            returns = data["close"].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend (using EMA9 vs EMA21)
            if "ema9" not in data.columns or "ema21" not in data.columns:
                # Calculate EMAs if not present
                data["ema9"] = data["close"].ewm(span=9, adjust=False).mean()
                data["ema21"] = data["close"].ewm(span=21, adjust=False).mean()
            
            # Check trend direction
            ema9_current = data["ema9"].iloc[-1]
            ema21_current = data["ema21"].iloc[-1]
            ema9_prev = data["ema9"].iloc[-10]
            ema21_prev = data["ema21"].iloc[-10]
            
            # Detect regime
            high_volatility_threshold = 0.003  # 0.3% per candle is high for 1h timeframe
            
            if volatility > high_volatility_threshold:
                regime = "volatile"
            elif ema9_current > ema21_current and ema9_prev > ema21_prev:
                regime = "bullish"
            elif ema9_current < ema21_current and ema9_prev < ema21_prev:
                regime = "bearish" 
            else:
                regime = "sideways"
            
            # Update regime history
            self.regime_history.append(regime)
            if len(self.regime_history) > 10:
                self.regime_history.pop(0)
            
            # Set current regime
            self.current_regime = regime
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "neutral"
    
    def preprocess_features(self, market_data: pd.DataFrame, arima_signal: Dict, adaptive_signal: Dict) -> np.ndarray:
        """
        Preprocess features for ML prediction
        
        Args:
            market_data: DataFrame with market data
            arima_signal: Dictionary with ARIMA strategy signal info
            adaptive_signal: Dictionary with Adaptive strategy signal info
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Create a copy of the market data
            data = market_data.copy()
            
            # Add strategy features
            data["arima_prediction"] = 1 if arima_signal.get("signal") == "buy" else (-1 if arima_signal.get("signal") == "sell" else 0)
            data["arima_strength"] = arima_signal.get("strength", 0)
            data["arima_forecast"] = arima_signal.get("forecast", 0)
            
            data["adaptive_prediction"] = 1 if adaptive_signal.get("signal") == "buy" else (-1 if adaptive_signal.get("signal") == "sell" else 0)
            data["adaptive_strength"] = adaptive_signal.get("strength", 0)
            data["adaptive_volatility"] = adaptive_signal.get("volatility", 0)
            
            # Add strategy interaction features
            data["strategy_agreement"] = np.sign(data["arima_prediction"] * data["adaptive_prediction"])
            data["strategy_combined_strength"] = data["arima_strength"] * data["adaptive_strength"] * data["strategy_agreement"]
            
            # Ensure we have all required technical indicators
            self._ensure_technical_indicators(data)
            
            # Get final features only (use last row)
            features = data.iloc[-1:].copy()
            
            # Scale features if we have a scaler
            if "feature_scaler" in self.models:
                # Select only numeric columns
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                features[numeric_cols] = self.models["feature_scaler"].transform(features[numeric_cols])
            
            # Convert to numpy array
            feature_array = features.values
            
            # Reshape for model input (adding batch and sequence dimensions)
            if "main" in self.models:
                # Get the input shape expected by the model
                input_shape = self.models["main"].input_shape
                # Reshape accordingly
                if len(input_shape) == 3:  # [batch_size, timesteps, features]
                    timesteps = input_shape[1]
                    if timesteps > 1:
                        # We need a sequence, but only have one sample
                        # Repeat the sample to create a sequence
                        feature_array = np.repeat(feature_array, timesteps, axis=0)
                        feature_array = feature_array.reshape(1, timesteps, feature_array.shape[1])
                    else:
                        feature_array = feature_array.reshape(1, 1, feature_array.shape[1])
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return np.array([])
    
    def _ensure_technical_indicators(self, data: pd.DataFrame) -> None:
        """
        Ensure all required technical indicators are present
        
        Args:
            data: DataFrame with market data
        """
        # List of basic indicators to calculate if missing
        basic_indicators = {
            "rsi": (self._calculate_rsi, {"data": data["close"]}),
            "ema9": (self._calculate_ema, {"data": data["close"], "window": 9}),
            "ema21": (self._calculate_ema, {"data": data["close"], "window": 21}),
            "ema50": (self._calculate_ema, {"data": data["close"], "window": 50}),
            "ema100": (self._calculate_ema, {"data": data["close"], "window": 100}),
            "atr": (self._calculate_atr, {"df": data}),
            "bb_width": (self._calculate_bb_width, {"data": data["close"]}),
            "volatility": (self._calculate_volatility, {"data": data["close"]}),
        }
        
        # Calculate missing indicators
        for indicator, (func, kwargs) in basic_indicators.items():
            if indicator not in data.columns:
                try:
                    data[indicator] = func(**kwargs)
                except Exception as e:
                    logger.error(f"Error calculating {indicator}: {e}")
                    # Fill with zeros as fallback
                    data[indicator] = 0
    
    # Technical indicator calculation helper methods
    def _calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for a price series"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate EMA for a price series"""
        return data.ewm(span=window, adjust=False).mean()
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR for price data"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _calculate_bb_width(self, data: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Band width"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return (upper_band - lower_band) / sma
    
    def _calculate_volatility(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate price volatility"""
        return data.pct_change().rolling(window=window).std()
    
    def get_prediction(
        self, 
        market_data: pd.DataFrame, 
        arima_signal: Dict[str, Any], 
        adaptive_signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get AI prediction by combining ARIMA and Adaptive signals with ML
        
        Args:
            market_data: DataFrame with market data
            arima_signal: Dictionary with ARIMA strategy info 
                (signal, strength, forecast)
            adaptive_signal: Dictionary with Adaptive strategy info
                (signal, strength, volatility)
                
        Returns:
            Dictionary with prediction details
        """
        try:
            # Detect market regime
            regime = self.detect_market_regime(market_data)
            
            # Special case: no ML model available
            if "main" not in self.models:
                # Fall back to weighted combination of strategies
                return self._combine_strategies_without_ml(arima_signal, adaptive_signal, regime)
            
            # Preprocess features
            features = self.preprocess_features(market_data, arima_signal, adaptive_signal)
            
            if len(features) == 0:
                logger.error("Failed to preprocess features")
                return self._combine_strategies_without_ml(arima_signal, adaptive_signal, regime)
            
            # Get prediction from model
            raw_prediction = self.models["main"].predict(features, verbose=0)
            
            # Process predictions based on model output shape
            output_shape = raw_prediction.shape
            
            if len(output_shape) == 2 and output_shape[1] >= 5:
                # Multi-output model: direction, leverage, stop loss, etc.
                direction = raw_prediction[0, 0]  # Direction prediction
                direction_thresh = raw_prediction[0, 1]  # Thresholded direction
                pct_change = raw_prediction[0, 2]  # Percentage change prediction
                vol_adj = raw_prediction[0, 3]  # Volatility-adjusted return
                leverage = raw_prediction[0, 4]  # Leverage recommendation
                
                # Determine signal based on direction_thresh
                if direction_thresh > 0.5:
                    signal = "buy"
                elif direction_thresh < -0.5:
                    signal = "sell"
                else:
                    signal = "neutral"
                
                # Determine confidence
                confidence = abs(direction)
                
                # Apply regime-specific adjustments
                regime_adjustments = self._get_regime_adjustments(regime)
                leverage = min(leverage * regime_adjustments.get("leverage_factor", 1.0), self.max_leverage)
                confidence = min(confidence * regime_adjustments.get("confidence_factor", 1.0), 1.0)
                
            else:
                # Simple model: just direction
                direction = raw_prediction[0, 0] if len(raw_prediction.shape) > 1 else raw_prediction[0]
                
                # Determine signal based on direction
                if direction > 0.6:
                    signal = "buy"
                elif direction < 0.4:
                    signal = "sell"
                else:
                    signal = "neutral"
                
                # Convert to normalized direction (-1 to 1)
                normalized_direction = (direction - 0.5) * 2
                
                # Determine confidence (0 to 1)
                confidence = abs(normalized_direction)
                
                # Simplistic leverage based on confidence
                leverage = self.max_leverage * confidence if confidence > self.confidence_threshold else 0
                
                # Percentage change prediction (naive)
                pct_change = normalized_direction * 0.01  # 1% prediction as placeholder
                vol_adj = normalized_direction  # Simple placeholder
            
            # Create result dictionary
            result = {
                "signal": signal,
                "confidence": confidence,
                "direction": direction,
                "pct_change": pct_change,
                "vol_adj": vol_adj,
                "leverage": leverage,
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            }
            
            # Track for performance measurement
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            # Fall back to weighted combination of strategies
            return self._combine_strategies_without_ml(arima_signal, adaptive_signal, regime)
    
    def _combine_strategies_without_ml(
        self, 
        arima_signal: Dict[str, Any], 
        adaptive_signal: Dict[str, Any],
        regime: str
    ) -> Dict[str, Any]:
        """
        Combine ARIMA and Adaptive strategies without ML
        
        Args:
            arima_signal: Dictionary with ARIMA strategy info
            adaptive_signal: Dictionary with Adaptive strategy info
            regime: Detected market regime
            
        Returns:
            Dictionary with combined prediction
        """
        # Get signals
        arima_sig = arima_signal.get("signal", "neutral")
        adaptive_sig = adaptive_signal.get("signal", "neutral")
        
        # Get strengths
        arima_strength = arima_signal.get("strength", 0)
        adaptive_strength = adaptive_signal.get("strength", 0)
        
        # Convert signals to numeric
        arima_value = 1 if arima_sig == "buy" else (-1 if arima_sig == "sell" else 0)
        adaptive_value = 1 if adaptive_sig == "buy" else (-1 if adaptive_sig == "sell" else 0)
        
        # Get regime-specific adjustments
        regime_adjustments = self._get_regime_adjustments(regime)
        
        # Apply weights based on regime
        arima_weight = self.arima_weight * regime_adjustments.get("arima_factor", 1.0)
        adaptive_weight = self.adaptive_weight * regime_adjustments.get("adaptive_factor", 1.0)
        
        # Normalize weights
        total_weight = arima_weight + adaptive_weight
        if total_weight > 0:
            arima_weight = arima_weight / total_weight
            adaptive_weight = adaptive_weight / total_weight
        else:
            arima_weight = adaptive_weight = 0.5
        
        # Combine signals
        combined_value = (arima_value * arima_weight * arima_strength) + (adaptive_value * adaptive_weight * adaptive_strength)
        
        # Determine final signal
        if combined_value > 0.3:
            signal = "buy"
        elif combined_value < -0.3:
            signal = "sell"
        else:
            signal = "neutral"
        
        # Determine confidence
        confidence = min(abs(combined_value), 1.0)
        
        # Calculate simple leverage based on confidence
        leverage = self.max_leverage * confidence if confidence > self.confidence_threshold else 0
        
        # Return result
        return {
            "signal": signal,
            "confidence": confidence,
            "direction": (combined_value + 1) / 2,  # Normalize to 0-1
            "pct_change": combined_value * 0.01,  # Simple 1% prediction
            "vol_adj": combined_value,
            "leverage": leverage,
            "regime": regime,
            "arima_contribution": arima_value * arima_weight * arima_strength,
            "adaptive_contribution": adaptive_value * adaptive_weight * adaptive_strength,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """
        Get regime-specific adjustments from config
        
        Args:
            regime: Market regime type
            
        Returns:
            Dictionary with adjustment factors
        """
        # Default adjustments
        default_adjustments = {
            "arima_factor": 1.0,
            "adaptive_factor": 1.0,
            "leverage_factor": 1.0,
            "confidence_factor": 1.0
        }
        
        # Get asset-specific regime weights
        asset_settings = self.config.get("asset_specific_settings", {}).get(self.trading_pair, {})
        regime_weights = asset_settings.get("regime_weights", {})
        
        # Get weight for this regime
        regime_weight = regime_weights.get(regime, 1.0)
        
        # Apply regime-specific logic
        if regime == "bullish":
            return {
                "arima_factor": 0.8,
                "adaptive_factor": 1.2,
                "leverage_factor": regime_weight,
                "confidence_factor": regime_weight
            }
        elif regime == "bearish":
            return {
                "arima_factor": 1.2,
                "adaptive_factor": 0.8,
                "leverage_factor": regime_weight,
                "confidence_factor": regime_weight
            }
        elif regime == "volatile":
            return {
                "arima_factor": 0.7,
                "adaptive_factor": 1.3,
                "leverage_factor": regime_weight * 0.8,  # Reduce leverage in volatile markets
                "confidence_factor": regime_weight
            }
        elif regime == "sideways":
            return {
                "arima_factor": 1.1,
                "adaptive_factor": 0.9,
                "leverage_factor": regime_weight * 0.7,  # Reduce leverage in sideways markets
                "confidence_factor": regime_weight * 0.9
            }
        else:
            return default_adjustments
    
    def adjust_position_size(
        self,
        base_position_size: float,
        confidence: float,
        atr: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Adjust position size based on confidence and market conditions
        
        Args:
            base_position_size: Base position size
            confidence: Prediction confidence (0-1)
            atr: Average True Range
            volatility: Market volatility
            
        Returns:
            Tuple of (adjusted position size, adjustment factor)
        """
        try:
            # Starting with the base position size
            position_size = base_position_size
            
            # Get position sizing method from config
            sizing_method = self.position_sizing.get("method", "fixed")
            
            # Apply the appropriate method
            if sizing_method == "kelly":
                # Kelly Criterion sizing (simplified)
                win_rate = self.get_win_rate()
                edge = win_rate - (1 - win_rate)  # Edge based on historical win rate
                
                if edge <= 0:
                    edge = 0.1  # Minimum edge
                
                # Kelly fraction from config
                kelly_fraction = self.position_sizing.get("kelly_fraction", 0.5)
                
                # Kelly position size
                kelly_size = edge * kelly_fraction
                
                # Apply confidence factor
                adjustment_factor = kelly_size * confidence
                position_size = base_position_size * adjustment_factor
                
            elif sizing_method == "volatility":
                # Volatility-based sizing
                # Lower position size when volatility is high
                if volatility > 0:
                    vol_factor = 0.02 / volatility  # Target 2% volatility
                    vol_factor = max(0.5, min(2.0, vol_factor))  # Cap between 0.5x and 2x
                    
                    # Apply confidence
                    adjustment_factor = vol_factor * confidence
                    position_size = base_position_size * adjustment_factor
                else:
                    adjustment_factor = confidence
                    position_size = base_position_size * confidence
            
            else:
                # Fixed sizing with confidence adjustment
                adjustment_factor = confidence
                position_size = base_position_size * confidence
            
            # Apply regime-specific adjustments
            regime_adjustments = self._get_regime_adjustments(self.current_regime)
            position_size *= regime_adjustments.get("leverage_factor", 1.0)
            
            # Apply maximum position size constraint
            max_position_pct = self.position_sizing.get("max_position_size", 0.25)
            max_position = base_position_size * max_position_pct * 2  # 2x as upper limit
            position_size = min(position_size, max_position)
            
            # Ensure position size is positive
            position_size = max(0, position_size)
            
            # Calculate final adjustment factor
            final_adjustment_factor = position_size / base_position_size if base_position_size > 0 else 0
            
            return position_size, final_adjustment_factor
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {e}")
            return base_position_size, 1.0
    
    def get_stop_loss_levels(
        self,
        entry_price: float,
        position_type: str,
        atr: float,
        confidence: float
    ) -> Tuple[float, float, float]:
        """
        Calculate optimal stop loss and take profit levels
        
        Args:
            entry_price: Entry price
            position_type: Position type ('long' or 'short')
            atr: Average True Range
            confidence: Prediction confidence
            
        Returns:
            Tuple of (stop loss price, take profit price, trailing stop distance)
        """
        try:
            # Get risk parameters
            stop_loss_pct = self.risk_params.get("stop_loss_pct", 0.03)
            take_profit_pct = self.risk_params.get("take_profit_pct", 0.05)
            trailing_stop_pct = self.risk_params.get("trailing_stop_pct", 0.02)
            
            # Adjust based on confidence
            if confidence < self.confidence_threshold:
                # Tighter stops for lower confidence trades
                stop_loss_pct *= (0.5 + 0.5 * confidence)
                trailing_stop_pct *= (0.7 + 0.3 * confidence)
            else:
                # Wider stops for higher confidence trades
                stop_loss_pct *= (0.8 + 0.4 * confidence)
                trailing_stop_pct *= (0.6 + 0.6 * confidence)
                # Larger take profit for high confidence
                take_profit_pct *= (0.8 + 0.6 * confidence)
            
            # Adjust based on ATR
            atr_multiplier = 2.0
            atr_stop = atr * atr_multiplier
            atr_take_profit = atr * atr_multiplier * 1.5  # 1.5x risk/reward
            
            # Calculate percentages and ATR-based levels
            if position_type.lower() == "long":
                stop_loss_price_pct = entry_price * (1 - stop_loss_pct)
                stop_loss_price_atr = entry_price - atr_stop
                take_profit_price_pct = entry_price * (1 + take_profit_pct)
                take_profit_price_atr = entry_price + atr_take_profit
            else:  # short
                stop_loss_price_pct = entry_price * (1 + stop_loss_pct)
                stop_loss_price_atr = entry_price + atr_stop
                take_profit_price_pct = entry_price * (1 - take_profit_pct)
                take_profit_price_atr = entry_price - atr_take_profit
            
            # Use the more conservative stop loss (closer to entry)
            if position_type.lower() == "long":
                stop_loss_price = max(stop_loss_price_pct, stop_loss_price_atr)
                take_profit_price = min(take_profit_price_pct, take_profit_price_atr)
            else:
                stop_loss_price = min(stop_loss_price_pct, stop_loss_price_atr)
                take_profit_price = max(take_profit_price_pct, take_profit_price_atr)
            
            # Calculate trailing stop distance
            trailing_stop_distance = entry_price * trailing_stop_pct
            
            return stop_loss_price, take_profit_price, trailing_stop_distance
            
        except Exception as e:
            logger.error(f"Error calculating stop loss levels: {e}")
            # Default percentages
            if position_type.lower() == "long":
                return entry_price * 0.97, entry_price * 1.05, entry_price * 0.02
            else:
                return entry_price * 1.03, entry_price * 0.95, entry_price * 0.02
    
    def get_leverage_recommendation(
        self,
        confidence: float,
        volatility: float,
        win_streak: int = 0
    ) -> float:
        """
        Get leverage recommendation based on prediction confidence
        
        Args:
            confidence: Prediction confidence (0-1)
            volatility: Market volatility
            win_streak: Current win streak (consecutive winning trades)
            
        Returns:
            Recommended leverage
        """
        try:
            # Get dynamic leverage settings
            base_leverage = self.dynamic_leverage.get("base_leverage", 3.0)
            max_leverage = min(self.max_leverage, self.dynamic_leverage.get("max_leverage", 20.0))
            confidence_multiplier = self.dynamic_leverage.get("confidence_multiplier", 1.5)
            volatility_divider = self.dynamic_leverage.get("volatility_divider", 2.0)
            
            # Start with base leverage
            leverage = base_leverage
            
            # Apply confidence factor
            if confidence > self.confidence_threshold:
                confidence_factor = 1.0 + (confidence - self.confidence_threshold) * confidence_multiplier
                leverage *= confidence_factor
            else:
                # Reduce leverage for low confidence
                confidence_factor = confidence / self.confidence_threshold
                leverage *= confidence_factor
            
            # Apply volatility adjustment
            if volatility > 0:
                volatility_factor = 1.0 / (1.0 + volatility * volatility_divider)
                leverage *= volatility_factor
            
            # Apply win streak bonus
            if win_streak > 0:
                streak_bonus = min(1.0 + (win_streak * 0.05), 1.5)  # Up to 50% bonus
                leverage *= streak_bonus
            
            # Apply regime-specific adjustments
            regime_adjustments = self._get_regime_adjustments(self.current_regime)
            leverage *= regime_adjustments.get("leverage_factor", 1.0)
            
            # Cap at maximum leverage
            leverage = min(leverage, max_leverage)
            
            # Round to nearest integer
            leverage = round(leverage)
            
            # Ensure minimum of 1
            leverage = max(1, leverage)
            
            return leverage
            
        except Exception as e:
            logger.error(f"Error calculating leverage recommendation: {e}")
            return 1  # Safest default
    
    def get_win_rate(self, window: int = 20) -> float:
        """
        Get historical win rate
        
        Args:
            window: Number of recent predictions to consider
            
        Returns:
            Win rate (0-1)
        """
        if self.total_predictions == 0:
            return 0.5  # Default when no data
        
        if self.prediction_history and len(self.prediction_history) >= 2:
            # Calculate win rate from recent predictions
            recent_predictions = self.prediction_history[-min(window, len(self.prediction_history)):]
            
            # Count successful predictions
            correct = 0
            total = 0
            
            for i in range(1, len(recent_predictions)):
                # Check if previous prediction was correct
                prev = recent_predictions[i-1]
                current = recent_predictions[i]
                
                # Get the predicted direction and actual price change
                if prev["signal"] == "buy":
                    predicted_direction = 1
                elif prev["signal"] == "sell":
                    predicted_direction = -1
                else:
                    continue  # Skip neutral signals
                
                # Calculate actual direction
                if "pct_change" in current:
                    actual_direction = np.sign(current["pct_change"])
                    
                    # Count as correct if directions match
                    if predicted_direction == actual_direction or actual_direction == 0:
                        correct += 1
                    
                    total += 1
            
            # Calculate win rate
            if total > 0:
                return correct / total
        
        # Return overall win rate if not enough recent data
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.5
    
    def update_prediction_performance(self, was_correct: bool) -> None:
        """
        Update prediction performance tracking
        
        Args:
            was_correct: Whether the prediction was correct
        """
        if was_correct:
            self.correct_predictions += 1
        self.total_predictions += 1
    
    def get_recommendation(
        self,
        current_price: float,
        market_data: pd.DataFrame,
        arima_signal: Dict[str, Any],
        adaptive_signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get a comprehensive trading recommendation
        
        Args:
            current_price: Current price
            market_data: DataFrame with market data
            arima_signal: Dictionary with ARIMA strategy info
            adaptive_signal: Dictionary with Adaptive strategy info
            
        Returns:
            Dictionary with trading recommendation
        """
        try:
            # Get prediction
            prediction = self.get_prediction(market_data, arima_signal, adaptive_signal)
            
            # Extract key values
            signal = prediction["signal"]
            confidence = prediction["confidence"]
            regime = prediction["regime"]
            leverage = self.get_leverage_recommendation(confidence, prediction.get("volatility", 0))
            
            # Get ATR for calculating stops
            atr = market_data["atr"].iloc[-1] if "atr" in market_data else 0
            if atr == 0:
                atr = current_price * 0.01  # Use 1% as fallback
            
            # Determine position type
            position_type = "long" if signal == "buy" else "short" if signal == "sell" else "none"
            
            # Calculate stop levels if we have a position
            stop_loss = take_profit = trailing_stop = None
            if position_type != "none":
                stop_loss, take_profit, trailing_stop = self.get_stop_loss_levels(
                    current_price, position_type, atr, confidence
                )
            
            # Adjust position size (using 1.0 as base)
            volatility = market_data["volatility"].iloc[-1] if "volatility" in market_data else 0
            position_size, size_factor = self.adjust_position_size(1.0, confidence, atr, volatility)
            
            # Create recommendation
            recommendation = {
                "signal": signal,
                "confidence": confidence,
                "regime": regime,
                "leverage": leverage,
                "position_size_factor": size_factor,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "risk_reward_ratio": ((take_profit - current_price) / (current_price - stop_loss)) if position_type == "long" and stop_loss and take_profit else
                                     ((current_price - take_profit) / (stop_loss - current_price)) if position_type == "short" and stop_loss and take_profit else None,
                "arima_contribution": prediction.get("arima_contribution"),
                "adaptive_contribution": prediction.get("adaptive_contribution"),
                "timestamp": datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting recommendation: {e}")
            return {
                "signal": "neutral",
                "confidence": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the AI system
        
        Returns:
            Dictionary with AI system status
        """
        return {
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "model_loaded": "main" in self.models,
            "current_regime": self.current_regime,
            "win_rate": self.get_win_rate(),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "ml_confidence_threshold": self.confidence_threshold,
            "max_leverage": self.max_leverage,
            "arima_weight": self.arima_weight,
            "adaptive_weight": self.adaptive_weight,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Test function to demonstrate usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Strategy AI")
    parser.add_argument("--pair", type=str, default="SOL/USD", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--max-leverage", type=int, default=125, help="Maximum leverage")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()
    
    # Create AI system
    ai = DualStrategyAI(
        trading_pair=args.pair,
        timeframe=args.timeframe,
        max_leverage=args.max_leverage,
        confidence_threshold=args.confidence
    )
    
    # Print status
    status = ai.get_status()
    print(f"AI Status: {json.dumps(status, indent=2)}")
    
    # Try loading some sample data
    pair_filename = args.pair.replace("/", "")
    data_path = f"historical_data/{pair_filename}_{args.timeframe}.csv"
    
    if os.path.exists(data_path):
        # Load data
        data = pd.read_csv(data_path)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Create sample signals
        arima_signal = {"signal": "sell", "strength": 0.8, "forecast": -0.5}
        adaptive_signal = {"signal": "neutral", "strength": 0.3, "volatility": 0.001}
        
        # Get recommendation
        recommendation = ai.get_recommendation(
            current_price=data["close"].iloc[-1],
            market_data=data.tail(100),
            arima_signal=arima_signal,
            adaptive_signal=adaptive_signal
        )
        
        print(f"Recommendation: {json.dumps(recommendation, indent=2)}")
    else:
        print(f"No data found at {data_path}")

if __name__ == "__main__":
    main()