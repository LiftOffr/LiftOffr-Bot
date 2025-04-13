#!/usr/bin/env python3
"""
ML Strategy Integrator for Kraken Trading Bot

This module integrates ML model predictions with existing trading strategies,
allowing the ensemble model to influence trading decisions based on market conditions.
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

# Import ensemble model
from advanced_ensemble_model import DynamicWeightedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MLStrategyIntegrator:
    """
    Integrates ML model predictions with existing trading strategies.
    
    This class serves as a bridge between the ML ensemble model and 
    traditional trading strategies, providing signal augmentation and
    risk management based on ML predictions.
    """
    
    def __init__(self, trading_pair="SOL/USD", timeframe="1h", 
                influence_weight=0.5, confidence_threshold=0.6):
        """
        Initialize the ML strategy integrator
        
        Args:
            trading_pair (str): Trading pair to analyze
            timeframe (str): Timeframe for analysis
            influence_weight (float): Weight of ML predictions in final decision (0.0-1.0)
            confidence_threshold (float): Minimum confidence required for ML to influence decision
        """
        self.trading_pair = trading_pair
        self.ticker_symbol = trading_pair.replace('/', '')
        self.timeframe = timeframe
        self.influence_weight = influence_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialize ensemble model
        self.ensemble = DynamicWeightedEnsemble(trading_pair=trading_pair, timeframe=timeframe)
        
        # Prediction history
        self.prediction_history = []
        self.max_history_size = 100  # Maximum number of predictions to store
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logger.info(f"ML Strategy Integrator initialized for {trading_pair} on {timeframe} timeframe")
        logger.info(f"ML influence weight: {influence_weight}, confidence threshold: {confidence_threshold}")
    
    def get_ml_prediction(self, market_data):
        """
        Get prediction from the ensemble model
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (prediction, confidence, details)
                prediction: float between -1 (strong sell) and 1 (strong buy)
                confidence: float between 0 and 1
                details: dict with detailed prediction information
        """
        try:
            # Generate prediction from ensemble
            prediction, confidence, details = self.ensemble.predict(market_data)
            
            # Convert prediction from probability to -1 to 1 scale
            # 0.5 is neutral (0), 1.0 is strong buy (1), 0.0 is strong sell (-1)
            scaled_prediction = (prediction - 0.5) * 2
            
            # Store prediction in history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': scaled_prediction,
                'confidence': confidence,
                'details': details
            })
            
            # Trim history if needed
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history = self.prediction_history[-self.max_history_size:]
            
            logger.info(f"ML prediction: {scaled_prediction:.4f} with confidence {confidence:.4f}")
            
            return scaled_prediction, confidence, details
        
        except Exception as e:
            logger.error(f"Error generating ML prediction: {e}")
            traceback.print_exc()
            return 0.0, 0.0, {}
    
    def integrate_with_strategy_signal(self, strategy_signal, strategy_strength, market_data):
        """
        Integrate ML prediction with strategy signal
        
        Args:
            strategy_signal (str): Signal from traditional strategy ("buy", "sell", "neutral")
            strategy_strength (float): Strength of strategy signal (0.0-1.0)
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (integrated_signal, integrated_strength, details)
        """
        # Get ML prediction
        ml_prediction, ml_confidence, ml_details = self.get_ml_prediction(market_data)
        
        # Check if ML confidence is sufficient to influence decision
        if ml_confidence < self.confidence_threshold:
            logger.info(f"ML confidence ({ml_confidence:.4f}) below threshold ({self.confidence_threshold}), using strategy signal only")
            return strategy_signal, strategy_strength, {
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "ml_influence": 0.0,
                "reason": "Low ML confidence"
            }
        
        # Convert strategy signal to numerical value
        if strategy_signal.lower() == "buy":
            strategy_value = 1.0
        elif strategy_signal.lower() == "sell":
            strategy_value = -1.0
        else:  # neutral
            strategy_value = 0.0
        
        # Calculate integrated signal value
        integrated_value = (strategy_value * (1 - self.influence_weight)) + (ml_prediction * self.influence_weight)
        
        # Calculate integrated strength
        integrated_strength = (strategy_strength * (1 - self.influence_weight)) + (ml_confidence * self.influence_weight)
        
        # Determine signal type from integrated value
        if integrated_value > 0.2:
            integrated_signal = "buy"
        elif integrated_value < -0.2:
            integrated_signal = "sell"
        else:
            integrated_signal = "neutral"
        
        # Prepare details
        details = {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "strategy_signal": strategy_signal,
            "strategy_strength": strategy_strength,
            "integrated_value": integrated_value,
            "integrated_strength": integrated_strength,
            "ml_influence": self.influence_weight,
            "ml_details": ml_details
        }
        
        logger.info(f"Integrated signal: {integrated_signal.upper()} (value: {integrated_value:.4f}, strength: {integrated_strength:.4f})")
        logger.info(f"Integration details: Strategy {strategy_signal.upper()} ({strategy_strength:.2f}) + ML {ml_prediction:.2f} ({ml_confidence:.2f})")
        
        return integrated_signal, integrated_strength, details
    
    def update_prediction_performance(self, was_correct):
        """
        Update prediction performance statistics
        
        Args:
            was_correct (bool): Whether the prediction was correct
        """
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        
        # Update ensemble model performance
        if len(self.prediction_history) > 0:
            last_prediction = self.prediction_history[-1]
            
            # Create outcome value (1 for correct, -1 for incorrect)
            outcome = 1 if was_correct else -1
            
            # Get model contributions from last prediction
            if 'details' in last_prediction and 'model_contributions' in last_prediction['details']:
                model_outcomes = {}
                
                for model_type, contribution in last_prediction['details']['model_contributions'].items():
                    # Determine if model was correct based on contribution direction
                    # and actual outcome
                    model_direction = 1 if contribution > 0.5 else -1
                    actual_direction = 1 if was_correct else -1
                    
                    # Model is correct if directions match
                    model_correct = (model_direction == actual_direction)
                    model_outcomes[model_type] = 1 if model_correct else -1
                
                # Update ensemble model performance
                self.ensemble.update_performance(model_outcomes)
    
    def get_prediction_statistics(self):
        """
        Get statistics about prediction performance
        
        Returns:
            dict: Statistics about prediction performance
        """
        accuracy = 0.0
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
        
        # Calculate recent accuracy (last 20 predictions)
        recent_correct = 0
        recent_total = min(20, self.total_predictions)
        
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": accuracy,
            "recent_accuracy": recent_correct / max(1, recent_total)
        }
    
    def adjust_position_size(self, base_position_size, market_data):
        """
        Adjust position size based on ML prediction confidence
        
        Args:
            base_position_size (float): Base position size from strategy
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (adjusted_position_size, adjustment_factor)
        """
        # Get ML prediction
        ml_prediction, ml_confidence, _ = self.get_ml_prediction(market_data)
        
        # Only adjust if confidence is high enough
        if ml_confidence < self.confidence_threshold:
            return base_position_size, 1.0
        
        # Calculate adjustment factor based on prediction strength and confidence
        # Strong predictions with high confidence can increase position size up to 50%
        # Weak predictions with high confidence can decrease position size up to 50%
        prediction_strength = abs(ml_prediction)
        adjustment_factor = 1.0 + (prediction_strength * ml_confidence * 0.5)
        
        # If prediction is counter to current position, reduce size
        if ml_prediction < 0:  # Bearish prediction
            adjustment_factor = 1.0 / adjustment_factor
        
        # Adjust position size
        adjusted_position_size = base_position_size * adjustment_factor
        
        logger.info(f"Position size adjusted: {base_position_size:.2f} → {adjusted_position_size:.2f} (factor: {adjustment_factor:.2f})")
        
        return adjusted_position_size, adjustment_factor
    
    def recommend_stop_loss(self, base_stop_loss, entry_price, position_type, market_data):
        """
        Recommend stop loss level based on ML prediction volatility assessment
        
        Args:
            base_stop_loss (float): Base stop loss from strategy
            entry_price (float): Entry price of position
            position_type (str): Position type ("long" or "short")
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (recommended_stop_loss, adjustment_factor, details)
        """
        try:
            # Get ML prediction
            _, ml_confidence, ml_details = self.get_ml_prediction(market_data)
            
            # Extract volatility assessment if available
            volatility_assessment = ml_details.get('volatility_assessment', 0.5)
            
            # Calculate stop loss distance from entry price
            base_distance = abs(entry_price - base_stop_loss)
            
            # Adjust stop loss distance based on volatility assessment
            # Higher volatility = wider stop loss, lower volatility = tighter stop loss
            adjustment_factor = 0.8 + (volatility_assessment * 0.4)  # 0.8 to 1.2
            adjusted_distance = base_distance * adjustment_factor
            
            # Calculate recommended stop loss
            if position_type.lower() == "long":
                recommended_stop_loss = entry_price - adjusted_distance
            else:  # short
                recommended_stop_loss = entry_price + adjusted_distance
            
            details = {
                "base_stop_loss": base_stop_loss,
                "base_distance": base_distance,
                "volatility_assessment": volatility_assessment,
                "adjustment_factor": adjustment_factor,
                "adjusted_distance": adjusted_distance
            }
            
            logger.info(f"Stop loss adjusted: {base_stop_loss:.2f} → {recommended_stop_loss:.2f} (factor: {adjustment_factor:.2f})")
            
            return recommended_stop_loss, adjustment_factor, details
        
        except Exception as e:
            logger.error(f"Error recommending stop loss: {e}")
            return base_stop_loss, 1.0, {"error": str(e)}
    
    def analyze_market_regime(self, market_data):
        """
        Analyze current market regime using ensemble model
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            tuple: (regime, confidence, details)
        """
        try:
            # Detect market regime
            regime = self.ensemble.detect_market_regime(market_data)
            
            # Get regime properties from ensemble
            if hasattr(self.ensemble, 'current_regime_properties'):
                regime_properties = self.ensemble.current_regime_properties
            else:
                regime_properties = {
                    'volatility': market_data['volatility'].iloc[-1] if 'volatility' in market_data.columns else 0.0,
                    'trend_strength': 0.0,
                    'confidence': 0.7
                }
            
            # Extract confidence
            confidence = regime_properties.get('confidence', 0.7)
            
            details = {
                'regime': regime,
                'properties': regime_properties
            }
            
            logger.info(f"Market regime: {regime} with confidence {confidence:.2f}")
            
            return regime, confidence, details
        
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return "unknown", 0.0, {"error": str(e)}
    
    def get_status(self):
        """
        Get current status of the ML strategy integrator
        
        Returns:
            dict: Status information
        """
        # Get ensemble status
        ensemble_status = self.ensemble.get_status()
        
        # Get prediction statistics
        prediction_stats = self.get_prediction_statistics()
        
        # Prepare status
        status = {
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "influence_weight": self.influence_weight,
            "confidence_threshold": self.confidence_threshold,
            "prediction_statistics": prediction_stats,
            "ensemble": ensemble_status,
            "recent_predictions": self.prediction_history[-5:] if len(self.prediction_history) > 0 else []
        }
        
        return status

def main():
    """Test the ML strategy integrator"""
    # Create integrator
    integrator = MLStrategyIntegrator(trading_pair="SOL/USD", timeframe="1h")
    
    # Load sample market data
    file_path = os.path.join("historical_data", "SOLUSD_1h.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate basic indicators
        df['return'] = df['close'].pct_change()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['volatility'] = df['return'].rolling(window=20).std()
        
        # Get latest data
        market_data = df.tail(100)
        
        # Test integration with strategy signal
        signal, strength, details = integrator.integrate_with_strategy_signal(
            strategy_signal="buy",
            strategy_strength=0.7,
            market_data=market_data
        )
        
        logger.info(f"Integrated signal: {signal}, strength: {strength}")
        logger.info(f"Details: {json.dumps(details, indent=2)}")
        
        # Test position size adjustment
        adjusted_size, factor = integrator.adjust_position_size(
            base_position_size=1000.0,
            market_data=market_data
        )
        
        logger.info(f"Adjusted position size: {adjusted_size} (factor: {factor})")
        
        # Test stop loss recommendation
        stop_loss, factor, sl_details = integrator.recommend_stop_loss(
            base_stop_loss=market_data['close'].iloc[-1] * 0.95,
            entry_price=market_data['close'].iloc[-1],
            position_type="long",
            market_data=market_data
        )
        
        logger.info(f"Recommended stop loss: {stop_loss} (factor: {factor})")
        
        # Test market regime analysis
        regime, confidence, regime_details = integrator.analyze_market_regime(market_data)
        
        logger.info(f"Market regime: {regime} with confidence {confidence}")
        
        # Get status
        status = integrator.get_status()
        logger.info(f"Status: {json.dumps(status, indent=2)}")
    
    else:
        logger.error(f"Historical data file not found: {file_path}")

if __name__ == "__main__":
    main()