#!/usr/bin/env python3
"""
ML Live Trading Integration

This module integrates ML models into the live trading system, providing real-time
predictions and trading signal generation based on advanced machine learning models.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import ML components
from model_collaboration_integrator import ModelCollaborationIntegrator

# ML model function placeholders
def load_model(path): return {"model": "dummy"}
def preprocess_data(df): return np.zeros((10, 10)), ["feature"]
def predict_price_movement(model, data): return {"direction": "up", "confidence": 0.75}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class MLLiveTradingIntegration:
    """
    Integration of ML models with live trading system
    """
    
    def __init__(
        self,
        assets: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        model_dir: str = "models",
        use_extreme_leverage: bool = False,
        log_level: str = "INFO"
    ):
        """
        Initialize the ML live trading integration
        
        Args:
            assets: List of assets to trade
            model_dir: Directory containing trained models
            use_extreme_leverage: Whether to use extreme leverage settings
            log_level: Logging level
        """
        self.assets = assets
        self.model_dir = model_dir
        self.use_extreme_leverage = use_extreme_leverage
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Initialize models
        self.models = {}
        
        # Load models
        self._load_models()
        
        logger.info(f"ML Live Trading Integration initialized for assets: {assets}")
    
    def _load_models(self):
        """Load ML models for all assets"""
        for asset in self.assets:
            asset_key = asset.replace("/", "")
            
            # Load transformer model
            transformer_path = os.path.join(self.model_dir, "transformer", f"{asset_key}_transformer.h5")
            if os.path.exists(transformer_path):
                try:
                    self.models[f"{asset_key}_transformer"] = load_model(transformer_path)
                    logger.info(f"Loaded transformer model for {asset}")
                except Exception as e:
                    logger.error(f"Error loading transformer model for {asset}: {e}")
            
            # Load TCN model
            tcn_path = os.path.join(self.model_dir, "tcn", f"{asset_key}_tcn.h5")
            if os.path.exists(tcn_path):
                try:
                    self.models[f"{asset_key}_tcn"] = load_model(tcn_path)
                    logger.info(f"Loaded TCN model for {asset}")
                except Exception as e:
                    logger.error(f"Error loading TCN model for {asset}: {e}")
            
            # Load LSTM model
            lstm_path = os.path.join(self.model_dir, "lstm", f"{asset_key}_lstm.h5")
            if os.path.exists(lstm_path):
                try:
                    self.models[f"{asset_key}_lstm"] = load_model(lstm_path)
                    logger.info(f"Loaded LSTM model for {asset}")
                except Exception as e:
                    logger.error(f"Error loading LSTM model for {asset}: {e}")
    
    def preprocess_market_data(
        self,
        data: Dict[str, Any],
        asset: str
    ) -> np.ndarray:
        """
        Preprocess market data for ML prediction
        
        Args:
            data: Market data
            asset: Asset to preprocess for
            
        Returns:
            ndarray: Preprocessed data
        """
        try:
            # In a real implementation, this would convert market data to the format
            # expected by the ML models
            
            # For now, just create a dummy array
            preprocessed, features = preprocess_data(data)
            
            logger.debug(f"Preprocessed market data for {asset} with {len(features)} features")
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error preprocessing market data for {asset}: {e}")
            return np.zeros((10, 10))  # Dummy data in case of error
    
    def predict(
        self,
        asset: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make prediction using ML models
        
        Args:
            asset: Asset to predict for
            market_data: Market data
            
        Returns:
            Dict: Prediction
        """
        asset_key = asset.replace("/", "")
        
        # Check if we have models for this asset
        if not any(key.startswith(asset_key) for key in self.models):
            logger.warning(f"No models available for {asset}")
            return None
        
        try:
            # Preprocess data
            preprocessed_data = self.preprocess_market_data(market_data, asset)
            
            # Make predictions with each model
            predictions = {}
            
            if f"{asset_key}_transformer" in self.models:
                transformer_pred = predict_price_movement(
                    self.models[f"{asset_key}_transformer"],
                    preprocessed_data
                )
                predictions["transformer"] = transformer_pred
            
            if f"{asset_key}_tcn" in self.models:
                tcn_pred = predict_price_movement(
                    self.models[f"{asset_key}_tcn"],
                    preprocessed_data
                )
                predictions["tcn"] = tcn_pred
            
            if f"{asset_key}_lstm" in self.models:
                lstm_pred = predict_price_movement(
                    self.models[f"{asset_key}_lstm"],
                    preprocessed_data
                )
                predictions["lstm"] = lstm_pred
            
            # Combine predictions
            combined = self._combine_predictions(predictions)
            
            # Add asset and timestamp
            combined["asset"] = asset
            combined["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"ML prediction for {asset}: direction={combined['direction']}, confidence={combined['confidence']:.2f}")
            return combined
            
        except Exception as e:
            logger.error(f"Error making prediction for {asset}: {e}")
            return {
                "asset": asset,
                "direction": "neutral",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _combine_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: Predictions from different models
            
        Returns:
            Dict: Combined prediction
        """
        if not predictions:
            return {
                "direction": "neutral",
                "confidence": 0.0
            }
        
        # Count votes for each direction
        direction_votes = {"up": 0, "down": 0, "neutral": 0}
        direction_confidence = {"up": 0.0, "down": 0.0, "neutral": 0.0}
        
        for model, pred in predictions.items():
            direction = pred.get("direction", "neutral")
            confidence = pred.get("confidence", 0.5)
            
            direction_votes[direction] += 1
            direction_confidence[direction] += confidence
        
        # Find the direction with the most votes
        max_votes = 0
        final_direction = "neutral"
        
        for direction, votes in direction_votes.items():
            if votes > max_votes:
                max_votes = votes
                final_direction = direction
            elif votes == max_votes and direction_confidence[direction] > direction_confidence[final_direction]:
                final_direction = direction
        
        # Calculate average confidence for the winning direction
        avg_confidence = direction_confidence[final_direction] / max(1, direction_votes[final_direction])
        
        return {
            "direction": final_direction,
            "confidence": avg_confidence,
            "model_predictions": predictions
        }
    
    def generate_trading_signal(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal from ML prediction
        
        Args:
            prediction: ML prediction
            
        Returns:
            Dict: Trading signal
        """
        if not prediction:
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "strategy": "MLStrategy",
                "pair": "UNKNOWN/USD"
            }
        
        # Extract prediction details
        direction = prediction.get("direction", "neutral")
        confidence = prediction.get("confidence", 0.0)
        asset = prediction.get("asset", "UNKNOWN/USD")
        
        # Map direction to signal type
        if direction == "up":
            signal_type = "BUY"
        elif direction == "down":
            signal_type = "SELL"
        else:
            signal_type = "NEUTRAL"
        
        # Calculate signal strength based on confidence
        strength = min(1.0, max(0.3, confidence))
        
        signal = {
            "signal_type": signal_type,
            "strength": strength,
            "confidence": confidence,
            "strategy": "MLStrategy",
            "pair": asset,
            "params": {
                "ml_confidence": confidence
            }
        }
        
        logger.info(f"Generated {signal_type} signal for {asset} with strength {strength:.2f}")
        return signal
    
    def calculate_position_parameters(
        self,
        signal: Dict[str, Any],
        available_capital: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate position parameters based on ML signal
        
        Args:
            signal: Trading signal
            available_capital: Available capital
            current_price: Current price
            
        Returns:
            Dict: Position parameters
        """
        # Extract signal details
        signal_type = signal.get("signal_type", "NEUTRAL")
        confidence = signal.get("confidence", 0.0)
        asset = signal.get("pair", "UNKNOWN/USD")
        
        # Skip neutral signals
        if signal_type == "NEUTRAL":
            return {
                "size": 0.0,
                "leverage": 1.0,
                "margin_pct": 0.0
            }
        
        # Determine base position size (% of capital)
        base_size = 0.05  # 5% of capital
        
        # Scale size based on confidence
        size_factor = min(1.0, max(0.2, confidence))
        position_size = base_size * size_factor
        
        # Determine leverage based on confidence and extreme settings
        if self.use_extreme_leverage:
            # Asset-specific leverage ranges
            leverage_ranges = {
                "SOL/USD": (20, 125),
                "ETH/USD": (15, 100),
                "BTC/USD": (12, 85)
            }
            
            # Get range for this asset (or default)
            min_lev, max_lev = leverage_ranges.get(asset, (5, 20))
            
            # Scale leverage based on confidence
            leverage_range = max_lev - min_lev
            leverage = min_lev + (leverage_range * confidence)
        else:
            # Normal leverage range (1-10x)
            leverage = 1.0 + (9.0 * confidence)
        
        # Ensure minimum leverage
        leverage = max(1.0, leverage)
        
        # Calculate margin percentage
        margin_pct = 1.0 / leverage
        
        # Calculate actual position size
        position_value = available_capital * position_size
        position_size_units = position_value / current_price
        
        params = {
            "size": position_size,
            "size_units": position_size_units,
            "leverage": leverage,
            "margin_pct": margin_pct,
            "confidence": confidence
        }
        
        logger.info(f"Calculated position parameters for {asset}: size={position_size:.2%}, leverage={leverage:.1f}x")
        return params
    
    def integrate_with_bot_manager(self, bot_manager):
        """
        Integrate with bot manager for live trading
        
        Args:
            bot_manager: Bot manager instance
        """
        # Register with bot manager
        logger.info("Integrating ML trading with bot manager")
        
        # Placeholder for real implementation with actual bot manager
    
    def run_ml_trading_iteration(
        self,
        market_data: Dict[str, Dict[str, Any]],
        bot_manager
    ):
        """
        Run a single ML trading iteration
        
        Args:
            market_data: Market data for all assets
            bot_manager: Bot manager instance
        """
        logger.info("Running ML trading iteration")
        
        # Process each asset
        for asset in self.assets:
            try:
                # Get market data for this asset
                asset_data = market_data.get(asset)
                
                if not asset_data:
                    logger.warning(f"No market data available for {asset}")
                    continue
                
                # Make prediction
                prediction = self.predict(asset, asset_data)
                
                if not prediction:
                    logger.warning(f"Failed to generate prediction for {asset}")
                    continue
                
                # Generate trading signal
                signal = self.generate_trading_signal(prediction)
                
                # Register signal with bot manager
                if bot_manager:
                    bot_manager.register_signal(
                        strategy_name="MLStrategy",
                        signal_type=signal["signal_type"],
                        strength=signal["strength"],
                        pair=asset,
                        price=asset_data.get("close"),
                        params=signal.get("params", {})
                    )
                
            except Exception as e:
                logger.error(f"Error processing {asset} in ML trading iteration: {e}")
    
    def run_ml_trading_loop(
        self,
        bot_manager,
        interval: int = 60,
        max_iterations: Optional[int] = None
    ):
        """
        Run ML trading loop
        
        Args:
            bot_manager: Bot manager instance
            interval: Seconds between iterations
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting ML trading loop with interval {interval}s")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                # Get current market data (in a real implementation, this would
                # be fetched from exchange or price feed)
                market_data = self._get_current_market_data()
                
                # Run trading iteration
                self.run_ml_trading_iteration(market_data, bot_manager)
                
                # Increment iteration counter
                iteration += 1
                
                # Sleep until next iteration
                logger.debug(f"Completed iteration {iteration}, sleeping for {interval}s")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("ML trading loop stopped by user")
        except Exception as e:
            logger.error(f"Error in ML trading loop: {e}")
    
    def _get_current_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current market data
        
        Returns:
            Dict: Market data by asset
        """
        # In a real implementation, this would fetch current market data
        # from the exchange or price feed
        
        # For now, just return dummy data
        data = {}
        
        for asset in self.assets:
            data[asset] = {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "timestamp": datetime.now().isoformat()
            }
        
        return data

def main():
    """Run ML live trading integration"""
    parser = argparse.ArgumentParser(description='Run ML live trading integration')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading pairs to trade')
    
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory containing trained models')
    
    parser.add_argument('--extreme-leverage', action='store_true',
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                      help='Maximum number of iterations (None for infinite)')
    
    args = parser.parse_args()
    
    # Initialize ML trading integration
    ml_trading = MLLiveTradingIntegration(
        assets=args.assets,
        model_dir=args.model_dir,
        use_extreme_leverage=args.extreme_leverage
    )
    
    # In a real implementation, this would get a real bot manager instance
    bot_manager = None
    
    # Run trading loop
    ml_trading.run_ml_trading_loop(
        bot_manager=bot_manager,
        interval=args.interval,
        max_iterations=args.max_iterations
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())