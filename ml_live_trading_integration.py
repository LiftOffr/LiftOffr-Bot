#!/usr/bin/env python3
"""
ML Live Trading Integration

This module provides integration of trained ML models with the trading bot,
loading models and providing predictions for trading decisions.
"""

import os
import sys
import json
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class MLLiveTradingIntegration:
    """
    ML Live Trading Integration
    
    This class loads trained ML models and provides predictions
    for live trading, with confidence-based position sizing.
    
    It handles:
    1. Loading trained ML models
    2. Preprocessing market data for predictions
    3. Generating trading signals with confidence scores
    4. Providing position sizing recommendations
    5. Tracking model performance
    """
    
    def __init__(
        self,
        trading_pairs: List[str] = ["SOL/USD"],
        models_path: str = "models",
        use_extreme_leverage: bool = True
    ):
        """
        Initialize the ML live trading integration
        
        Args:
            trading_pairs: List of trading pairs to support
            models_path: Path to trained models
            use_extreme_leverage: Whether to use extreme leverage settings
        """
        self.trading_pairs = trading_pairs
        self.models_path = models_path
        self.use_extreme_leverage = use_extreme_leverage
        
        # Model tracking
        self.loaded_models = {pair: {} for pair in trading_pairs}
        self.performance_metrics = {pair: {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "trades_history": []
        } for pair in trading_pairs}
        
        # Position sizing configurations
        self.position_sizing_configs = self._load_position_sizing_configs()
        
        # Load models for each trading pair
        self._load_models()
        
        logger.info(f"ML Live Trading Integration initialized for {len(trading_pairs)} pairs")
        logger.info(f"Using extreme leverage: {use_extreme_leverage}")
    
    def _load_position_sizing_configs(self) -> Dict[str, Any]:
        """
        Load position sizing configurations
        
        Returns:
            Dict: Position sizing configurations by pair
        """
        configs = {}
        
        for pair in self.trading_pairs:
            pair_filename = pair.replace("/", "")
            config_path = f"{self.models_path}/ensemble/{pair_filename}_position_sizing.json"
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        configs[pair] = json.load(f)
                    logger.info(f"Loaded position sizing configuration for {pair}")
                except Exception as e:
                    logger.error(f"Error loading position sizing configuration for {pair}: {e}")
                    configs[pair] = self._get_default_position_sizing_config(pair)
            else:
                logger.warning(f"No position sizing configuration found for {pair}, using defaults")
                configs[pair] = self._get_default_position_sizing_config(pair)
        
        return configs
    
    def _get_default_position_sizing_config(self, pair: str) -> Dict[str, Any]:
        """
        Get default position sizing configuration for a pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Dict: Default position sizing configuration
        """
        # Default leverage settings
        if "SOL" in pair:
            leverage_settings = {
                "min": 20.0,
                "default": 35.0,
                "max": 125.0,
                "confidence_threshold": 0.65
            }
        elif "ETH" in pair:
            leverage_settings = {
                "min": 15.0,
                "default": 30.0,
                "max": 100.0,
                "confidence_threshold": 0.70
            }
        elif "BTC" in pair:
            leverage_settings = {
                "min": 12.0,
                "default": 25.0,
                "max": 85.0,
                "confidence_threshold": 0.75
            }
        else:
            leverage_settings = {
                "min": 10.0,
                "default": 20.0,
                "max": 50.0,
                "confidence_threshold": 0.70
            }
        
        return {
            "asset": pair,
            "timestamp": datetime.now().isoformat(),
            "leverage_settings": leverage_settings,
            "position_sizing": {
                "confidence_thresholds": [0.65, 0.70, 0.80, 0.90],
                "size_multipliers": [0.3, 0.5, 0.8, 1.0]
            }
        }
    
    def _load_models(self) -> None:
        """Load trained models for all trading pairs"""
        try:
            for pair in self.trading_pairs:
                pair_filename = pair.replace("/", "")
                
                # Load ensemble configuration
                ensemble_path = f"{self.models_path}/ensemble/{pair_filename}_ensemble.json"
                
                if os.path.exists(ensemble_path):
                    try:
                        with open(ensemble_path, "r") as f:
                            ensemble_config = json.load(f)
                        
                        # In a real implementation, we would load the actual models
                        # For this prototype, we'll just track the configuration
                        
                        # Store model information
                        self.loaded_models[pair] = {
                            "ensemble": ensemble_config,
                            "models": ensemble_config.get("models", []),
                            "weights": ensemble_config.get("weights", {}),
                            "loaded_at": datetime.now().isoformat()
                        }
                        
                        logger.info(f"Loaded ensemble configuration for {pair}: "
                                  f"{len(ensemble_config.get('models', []))} models")
                        
                    except Exception as e:
                        logger.error(f"Error loading ensemble configuration for {pair}: {e}")
                else:
                    logger.warning(f"No ensemble configuration found for {pair}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_loaded_models_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dict: Information about loaded models
        """
        return self.loaded_models
    
    def predict(self, pair: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for a trading pair
        
        Args:
            pair: Trading pair
            market_data: Market data for the pair
            
        Returns:
            Dict: Prediction result
        """
        try:
            if pair not in self.trading_pairs:
                logger.warning(f"Prediction requested for unknown pair: {pair}")
                return {
                    "signal": 0,
                    "confidence": 0.0,
                    "position_sizing": {},
                    "details": {"error": "Unknown trading pair"}
                }
            
            if pair not in self.loaded_models or not self.loaded_models[pair]:
                logger.warning(f"No models loaded for {pair}")
                return {
                    "signal": 0,
                    "confidence": 0.0,
                    "position_sizing": {},
                    "details": {"error": "No models loaded"}
                }
            
            # In a real implementation, we would:
            # 1. Preprocess the market data for model input
            # 2. Generate predictions from each model
            # 3. Combine predictions using ensemble weights
            
            # For this prototype, we'll simulate a prediction
            
            # Randomize prediction (in a real implementation, this would be model-based)
            random_value = np.random.random()
            
            # Simulate different behavior for different pairs
            if "SOL" in pair:
                # SOL/USD - Bullish bias
                if random_value > 0.7:
                    signal = 1  # BUY
                    confidence = 0.75 + (random_value - 0.7) * 0.5  # 0.75-1.0
                elif random_value < 0.3:
                    signal = -1  # SELL
                    confidence = 0.65 + random_value * 0.25  # 0.65-0.725
                else:
                    signal = 0  # HOLD
                    confidence = 0.5 + random_value * 0.2  # 0.5-0.7
            elif "ETH" in pair:
                # ETH/USD - Neutral
                if random_value > 0.6:
                    signal = 1  # BUY
                    confidence = 0.7 + (random_value - 0.6) * 0.5  # 0.7-0.9
                elif random_value < 0.4:
                    signal = -1  # SELL
                    confidence = 0.7 + (0.4 - random_value) * 0.5  # 0.7-0.9
                else:
                    signal = 0  # HOLD
                    confidence = 0.6 + (random_value - 0.4) * 0.2  # 0.6-0.64
            else:
                # BTC/USD - Slightly bearish
                if random_value > 0.65:
                    signal = 1  # BUY
                    confidence = 0.7 + (random_value - 0.65) * 0.6  # 0.7-0.91
                elif random_value < 0.45:
                    signal = -1  # SELL
                    confidence = 0.75 + (0.45 - random_value) * 0.5  # 0.75-0.97
                else:
                    signal = 0  # HOLD
                    confidence = 0.6 + (random_value - 0.45) * 0.3  # 0.6-0.66
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(pair, signal, confidence, market_data)
            
            # Prepare prediction result
            prediction = {
                "signal": signal,
                "confidence": confidence,
                "position_sizing": position_sizing,
                "details": {
                    "timestamp": datetime.now().isoformat(),
                    "models_used": self.loaded_models[pair].get("models", []),
                    "latest_price": market_data.get("close", [])[-1] if market_data.get("close") else None,
                    "regime": market_data.get("regime", "unknown")
                }
            }
            
            logger.info(f"Generated prediction for {pair}: "
                      f"{'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'}, "
                      f"Confidence={confidence:.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {pair}: {e}")
            return {
                "signal": 0,
                "confidence": 0.0,
                "position_sizing": {},
                "details": {"error": str(e)}
            }
    
    def _calculate_position_sizing(
        self, 
        pair: str, 
        signal: int, 
        confidence: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate position sizing based on prediction confidence
        
        Args:
            pair: Trading pair
            signal: Prediction signal (1 for BUY, -1 for SELL, 0 for HOLD)
            confidence: Prediction confidence (0.0-1.0)
            market_data: Market data
            
        Returns:
            Dict: Position sizing recommendation
        """
        try:
            # If signal is HOLD or confidence is below threshold, return empty sizing
            if signal == 0:
                return {
                    "size": 0.0,
                    "leverage": 1.0,
                    "confidence": confidence,
                    "risk_level": "none"
                }
            
            # Get position sizing configuration
            config = self.position_sizing_configs.get(pair, self._get_default_position_sizing_config(pair))
            leverage_settings = config.get("leverage_settings", {})
            position_settings = config.get("position_sizing", {})
            
            # Check confidence threshold
            confidence_threshold = leverage_settings.get("confidence_threshold", 0.65)
            if confidence < confidence_threshold:
                return {
                    "size": 0.0,
                    "leverage": 1.0,
                    "confidence": confidence,
                    "risk_level": "below_threshold"
                }
            
            # Calculate size based on confidence
            size = 0.3  # Default size
            confidence_thresholds = position_settings.get("confidence_thresholds", [0.65, 0.75, 0.85, 0.95])
            size_multipliers = position_settings.get("size_multipliers", [0.3, 0.5, 0.8, 1.0])
            
            # Find appropriate size multiplier
            for i, threshold in enumerate(confidence_thresholds):
                if confidence >= threshold and i < len(size_multipliers):
                    size = size_multipliers[i]
            
            # Calculate leverage based on confidence
            min_leverage = leverage_settings.get("min", 5.0)
            default_leverage = leverage_settings.get("default", 20.0)
            max_leverage = leverage_settings.get("max", 50.0)
            
            if not self.use_extreme_leverage:
                # Cap leverage at more conservative levels if extreme leverage is disabled
                min_leverage = min(min_leverage, 5.0)
                default_leverage = min(default_leverage, 10.0)
                max_leverage = min(max_leverage, 20.0)
            
            # Scale leverage based on confidence
            # Higher confidence = higher leverage
            leverage_range = max_leverage - min_leverage
            confidence_factor = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
            leverage = min_leverage + leverage_range * confidence_factor
            
            # Cap leverage at max allowed
            leverage = min(leverage, max_leverage)
            
            # Determine risk level
            if confidence > 0.9:
                risk_level = "very_high"
            elif confidence > 0.8:
                risk_level = "high"
            elif confidence > 0.7:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "size": size,
                "leverage": leverage,
                "confidence": confidence,
                "risk_level": risk_level
            }
            
        except Exception as e:
            logger.error(f"Error calculating position sizing for {pair}: {e}")
            return {
                "size": 0.0,
                "leverage": 1.0,
                "confidence": confidence,
                "risk_level": "error"
            }
    
    def update_performance(self, pair: str, trade_data: Dict[str, Any]) -> None:
        """
        Update model performance based on trade outcome
        
        Args:
            pair: Trading pair
            trade_data: Trade data
        """
        try:
            if pair not in self.trading_pairs:
                logger.warning(f"Performance update for unknown pair: {pair}")
                return
            
            # Extract trade data
            entry_price = trade_data.get("entry_price", 0.0)
            exit_price = trade_data.get("exit_price", 0.0)
            profit_loss = trade_data.get("profit_loss", 0.0)
            direction = trade_data.get("direction", 0)  # 1 for long, -1 for short
            leverage = trade_data.get("leverage", 1.0)
            timestamp = trade_data.get("timestamp", datetime.now().isoformat())
            
            # Update performance metrics
            metrics = self.performance_metrics[pair]
            
            # Increment total trades
            metrics["total_trades"] += 1
            
            # Track trade history
            metrics["trades_history"].append({
                "timestamp": timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_loss": profit_loss,
                "direction": direction,
                "leverage": leverage
            })
            
            # Limit history to last 1000 trades
            if len(metrics["trades_history"]) > 1000:
                metrics["trades_history"] = metrics["trades_history"][-1000:]
            
            # Calculate total P&L and win rate
            total_pnl = sum(trade["profit_loss"] for trade in metrics["trades_history"])
            profitable_trades = sum(1 for trade in metrics["trades_history"] if trade["profit_loss"] > 0)
            
            metrics["total_pnl"] = total_pnl
            metrics["win_rate"] = profitable_trades / len(metrics["trades_history"]) if metrics["trades_history"] else 0.0
            
            # Calculate average profit and loss
            profits = [trade["profit_loss"] for trade in metrics["trades_history"] if trade["profit_loss"] > 0]
            losses = [trade["profit_loss"] for trade in metrics["trades_history"] if trade["profit_loss"] < 0]
            
            metrics["avg_profit"] = sum(profits) / len(profits) if profits else 0.0
            metrics["avg_loss"] = sum(losses) / len(losses) if losses else 0.0
            
            logger.info(f"Updated performance for {pair}: "
                      f"Win Rate={metrics['win_rate']:.2f}, "
                      f"Total P&L={metrics['total_pnl']:.2f}%, "
                      f"Trades={metrics['total_trades']}")
            
        except Exception as e:
            logger.error(f"Error updating performance for {pair}: {e}")
    
    def get_performance_metrics(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Args:
            pair: Trading pair, if None returns metrics for all pairs
            
        Returns:
            Dict: Performance metrics
        """
        if pair is not None:
            if pair not in self.performance_metrics:
                return {}
            return self.performance_metrics[pair]
        else:
            return self.performance_metrics

def main():
    """Test the ML live trading integration"""
    try:
        # Initialize ML live trading integration
        integration = MLLiveTradingIntegration(
            trading_pairs=["SOL/USD", "ETH/USD", "BTC/USD"],
            use_extreme_leverage=True
        )
        
        # Create sample market data
        market_data = {
            "close": [130.0, 131.5, 132.8, 133.4, 134.2, 133.8, 133.5],
            "high": [130.5, 132.0, 133.2, 133.8, 134.5, 134.2, 133.8],
            "low": [129.5, 130.8, 132.0, 132.9, 133.6, 133.2, 133.0],
            "volume": [10000, 12000, 11000, 9500, 10500, 9800, 9200],
            "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(7, 0, -1)],
            "regime": "volatile"
        }
        
        # Generate predictions
        for pair in integration.trading_pairs:
            prediction = integration.predict(pair, market_data)
            
            print(f"\nPrediction for {pair}:")
            print(f"Signal: {'BUY' if prediction['signal'] == 1 else 'SELL' if prediction['signal'] == -1 else 'HOLD'}")
            print(f"Confidence: {prediction['confidence']:.2f}")
            print(f"Position Sizing: {prediction['position_sizing']}")
            
            # Simulate a trade outcome
            trade_data = {
                "entry_price": market_data["close"][-2],
                "exit_price": market_data["close"][-1],
                "profit_loss": (market_data["close"][-1] - market_data["close"][-2]) / market_data["close"][-2] * 100 * prediction["position_sizing"]["leverage"] * (prediction["signal"]),
                "direction": prediction["signal"],
                "leverage": prediction["position_sizing"]["leverage"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Update performance
            integration.update_performance(pair, trade_data)
        
        # Get performance metrics
        metrics = integration.get_performance_metrics()
        
        print("\nPerformance Metrics:")
        for pair, pair_metrics in metrics.items():
            print(f"{pair}: Win Rate={pair_metrics['win_rate']:.2f}, "
                 f"Total P&L={pair_metrics['total_pnl']:.2f}%, "
                 f"Trades={pair_metrics['total_trades']}")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()