#!/usr/bin/env python3
"""
ML Model Integrator for Kraken Trading Bot

This module integrates the advanced ensemble model with the trading bot,
providing predictions and signals that can be used by trading strategies.
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
from tensorflow.keras.models import load_model

# Import advanced models
from advanced_ensemble_model import DynamicWeightedEnsemble
from attention_gru_model import load_attention_gru_model, AttentionLayer

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
DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to generate a signal

class MLModelIntegrator:
    """
    Integrates ML models with the trading bot by providing predictions
    and signals that can be used by trading strategies.
    """
    def __init__(self, trading_pair="SOL/USD", timeframe="1h", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the ML model integrator
        
        Args:
            trading_pair (str): Trading pair to model
            timeframe (str): Timeframe for analysis
            confidence_threshold (float): Minimum confidence threshold to generate signals
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.ensemble = DynamicWeightedEnsemble(trading_pair, timeframe)
        self.last_prediction = None
        self.last_confidence = None
        self.last_details = None
        self.last_prediction_time = None
        self.prediction_count = 0
        self.correct_predictions = 0
    
    def get_model_status(self):
        """
        Get status of ML models
        
        Returns:
            dict: Status information about loaded models
        """
        return self.ensemble.get_status()
    
    def predict(self, market_data):
        """
        Generate prediction from ML models
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (prediction, confidence, signal, details)
        """
        # Generate ensemble prediction
        prediction, confidence, details = self.ensemble.predict(market_data)
        
        # Store prediction for reference
        self.last_prediction = prediction
        self.last_confidence = confidence
        self.last_details = details
        self.last_prediction_time = datetime.now()
        
        # Increment prediction count
        self.prediction_count += 1
        
        # Generate signal based on prediction and confidence
        if confidence >= self.confidence_threshold:
            if prediction > 0.1:  # Use threshold to avoid noise
                signal = "BUY"
            elif prediction < -0.1:
                signal = "SELL"
            else:
                signal = "NEUTRAL"
        else:
            signal = "NEUTRAL"
        
        # Log prediction
        logger.info(f"ML Prediction: {prediction:.4f}, Confidence: {confidence:.4f}, Signal: {signal}")
        logger.info(f"Market Regime: {details['regime']}")
        
        return prediction, confidence, signal, details
    
    def calculate_signal_strength(self, prediction, confidence):
        """
        Calculate signal strength based on prediction and confidence
        
        Args:
            prediction (float): Prediction value (-1 to 1)
            confidence (float): Confidence value (0 to 1)
            
        Returns:
            float: Signal strength (0 to 1)
        """
        # Scale prediction to 0-1 range (from -1 to 1)
        prediction_abs = abs(prediction)
        
        # Calculate signal strength as a combination of prediction and confidence
        strength = (prediction_abs * 0.7) + (confidence * 0.3)
        
        # Scale to 0-1 range, with 1 being strongest
        strength = min(1.0, max(0.0, strength))
        
        return strength
    
    def update_performance(self, actual_direction):
        """
        Update model performance based on actual market direction
        
        Args:
            actual_direction (int): Actual market direction (1 for up, -1 for down, 0 for neutral)
            
        Returns:
            float: Current accuracy
        """
        if self.last_prediction is None:
            return 0.0
        
        # Determine if prediction was correct
        predicted_direction = 1 if self.last_prediction > 0 else (-1 if self.last_prediction < 0 else 0)
        
        # Update performance for each model
        model_outcomes = {}
        for model_type, pred in self.last_details.get('model_predictions', {}).items():
            model_direction = 1 if pred > 0 else (-1 if pred < 0 else 0)
            if model_direction == actual_direction:
                model_outcomes[model_type] = 1  # Correct
            elif model_direction == 0 or actual_direction == 0:
                model_outcomes[model_type] = 0  # Neutral
            else:
                model_outcomes[model_type] = -1  # Incorrect
        
        # Update ensemble with model outcomes
        self.ensemble.update_performance(model_outcomes)
        
        # Update overall performance
        if predicted_direction == actual_direction:
            self.correct_predictions += 1
        
        # Calculate accuracy
        accuracy = self.correct_predictions / self.prediction_count if self.prediction_count > 0 else 0.0
        
        logger.info(f"Updated performance: {self.correct_predictions}/{self.prediction_count} = {accuracy:.4f}")
        
        return accuracy
    
    def get_recommendation(self, current_price=None, current_position=None):
        """
        Get trading recommendation based on current prediction
        
        Args:
            current_price (float): Current market price
            current_position (str): Current position (LONG, SHORT, NONE)
            
        Returns:
            dict: Trading recommendation
        """
        if self.last_prediction is None:
            return {
                "action": "HOLD",
                "reason": "No prediction available",
                "confidence": 0.0,
                "regime": "unknown"
            }
        
        # Convert prediction to action
        if current_position in [None, "NONE"]:
            # No current position, consider opening new position
            if self.last_prediction > 0.2 and self.last_confidence > self.confidence_threshold:
                action = "BUY"
                reason = f"ML predicts upward movement ({self.last_prediction:.4f}) with high confidence ({self.last_confidence:.4f})"
            elif self.last_prediction < -0.2 and self.last_confidence > self.confidence_threshold:
                action = "SELL"
                reason = f"ML predicts downward movement ({self.last_prediction:.4f}) with high confidence ({self.last_confidence:.4f})"
            else:
                action = "HOLD"
                reason = f"ML prediction ({self.last_prediction:.4f}) or confidence ({self.last_confidence:.4f}) too weak to act"
        
        elif current_position == "LONG":
            # Currently long, consider closing position
            if self.last_prediction < -0.1 and self.last_confidence > self.confidence_threshold:
                action = "CLOSE_LONG"
                reason = f"ML predicts downward movement ({self.last_prediction:.4f}) with high confidence ({self.last_confidence:.4f})"
            else:
                action = "HOLD_LONG"
                reason = f"ML prediction ({self.last_prediction:.4f}) suggests maintaining long position"
        
        elif current_position == "SHORT":
            # Currently short, consider closing position
            if self.last_prediction > 0.1 and self.last_confidence > self.confidence_threshold:
                action = "CLOSE_SHORT"
                reason = f"ML predicts upward movement ({self.last_prediction:.4f}) with high confidence ({self.last_confidence:.4f})"
            else:
                action = "HOLD_SHORT"
                reason = f"ML prediction ({self.last_prediction:.4f}) suggests maintaining short position"
        
        else:
            action = "HOLD"
            reason = f"Unknown position state: {current_position}"
        
        # Get current market regime
        regime = self.last_details.get('regime', 'unknown') if self.last_details else 'unknown'
        
        # Calculate signal strength
        strength = self.calculate_signal_strength(self.last_prediction, self.last_confidence)
        
        return {
            "action": action,
            "reason": reason,
            "prediction": float(self.last_prediction),
            "confidence": float(self.last_confidence),
            "strength": float(strength),
            "regime": regime,
            "timestamp": self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }


def main():
    """Test the ML model integrator"""
    # Create integrator
    integrator = MLModelIntegrator()
    
    # Get model status
    status = integrator.get_model_status()
    logger.info(f"Model Status: {status}")
    
    # If we have models, test a prediction
    if status.get('model_count', 0) > 0:
        # Create dummy data for testing
        dummy_data = pd.DataFrame({
            f"open_{integrator.timeframe}": [100, 101, 102, 103, 104] * 12,
            f"high_{integrator.timeframe}": [105, 106, 107, 108, 109] * 12,
            f"low_{integrator.timeframe}": [95, 96, 97, 98, 99] * 12,
            f"close_{integrator.timeframe}": [101, 102, 103, 104, 105] * 12,
            f"volume_{integrator.timeframe}": [1000, 1100, 1200, 1300, 1400] * 12,
            f"volatility_{integrator.timeframe}": [0.01, 0.02, 0.01, 0.03, 0.02] * 12,
            f"ema9_{integrator.timeframe}": [101, 102, 103, 104, 105] * 12,
            f"ema21_{integrator.timeframe}": [100, 101, 102, 103, 104] * 12
        })
        
        # Make a prediction
        prediction, confidence, signal, details = integrator.predict(dummy_data)
        logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Signal: {signal}")
        
        # Get recommendation
        recommendation = integrator.get_recommendation(current_price=103.5, current_position="NONE")
        logger.info(f"Recommendation: {recommendation}")
        
        # Update performance
        accuracy = integrator.update_performance(1)  # Assume market went up
        logger.info(f"Accuracy: {accuracy}")
    else:
        logger.warning("No models available for prediction")


if __name__ == "__main__":
    main()