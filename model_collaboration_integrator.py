#!/usr/bin/env python3
"""
Model Collaboration Integrator

This module provides a central hub for ML model collaboration, 
connecting predictions from various models to trading decisions.
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
        logging.FileHandler('model_collaboration.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelCollaborationIntegrator:
    """
    Model Collaboration Integrator
    
    This class provides a central hub for ML model collaboration, 
    connecting predictions from various models and strategies to make
    cohesive trading decisions.
    
    It handles:
    1. Market regime detection
    2. Strategy weighting based on market conditions
    3. Performance tracking and adaptation
    4. Model consensus building
    5. Signal filtering and amplification
    """
    
    def __init__(
        self,
        trading_pairs: List[str] = ["SOL/USD"],
        config_path: str = "models/ensemble",
        enable_adaptive_weights: bool = True
    ):
        """
        Initialize the model collaboration integrator
        
        Args:
            trading_pairs: List of trading pairs to support
            config_path: Path to configuration files
            enable_adaptive_weights: Whether to adapt weights based on performance
        """
        self.trading_pairs = trading_pairs
        self.config_path = config_path
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # Market regimes and conditions
        self.market_regimes = {pair: "unknown" for pair in trading_pairs}
        self.market_conditions = {pair: {} for pair in trading_pairs}
        
        # Strategy weights and performance tracking
        self.strategy_weights = self._initialize_strategy_weights()
        self.performance_history = {pair: {} for pair in trading_pairs}
        self.model_performance = {pair: {} for pair in trading_pairs}
        
        # Signal generation
        self.latest_signals = {pair: {} for pair in trading_pairs}
        self.signal_history = {pair: [] for pair in trading_pairs}
        
        logger.info(f"Model Collaboration Integrator initialized for {len(trading_pairs)} pairs")
        logger.info(f"Adaptive weights: {enable_adaptive_weights}")
    
    def _initialize_strategy_weights(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Initialize strategy weights from configuration or defaults
        
        Returns:
            Dict: Strategy weights by pair and market regime
        """
        strategy_weights = {}
        
        # For each trading pair
        for pair in self.trading_pairs:
            pair_weights = {}
            pair_filename = pair.replace("/", "")
            
            # Try to load weights from config file
            weights_path = f"{self.config_path}/{pair_filename}_weights.json"
            
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, "r") as f:
                        pair_weights = json.load(f)
                    logger.info(f"Loaded strategy weights for {pair}")
                except Exception as e:
                    logger.error(f"Error loading weights for {pair}: {e}")
                    pair_weights = self._get_default_weights()
            else:
                logger.warning(f"No weights config found for {pair}, using defaults")
                pair_weights = self._get_default_weights()
            
            strategy_weights[pair] = pair_weights
        
        return strategy_weights
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Get default strategy weights by market regime
        
        Returns:
            Dict: Default strategy weights
        """
        return {
            "trending": {
                "ml_transformer": 0.30,
                "ml_tcn": 0.25,
                "ml_lstm": 0.20,
                "arima": 0.15,
                "adaptive": 0.10,
            },
            "ranging": {
                "ml_transformer": 0.25,
                "ml_tcn": 0.20,
                "ml_lstm": 0.15,
                "arima": 0.15,
                "adaptive": 0.25,
            },
            "volatile": {
                "ml_transformer": 0.35,
                "ml_tcn": 0.30,
                "ml_lstm": 0.25,
                "arima": 0.05,
                "adaptive": 0.05,
            },
            "unknown": {
                "ml_transformer": 0.25,
                "ml_tcn": 0.25,
                "ml_lstm": 0.20,
                "arima": 0.15,
                "adaptive": 0.15,
            }
        }
    
    def update_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Update market regime assessment based on current market data
        
        Args:
            market_data: Market data for various trading pairs
            
        Returns:
            Dict: Updated market regimes
        """
        try:
            updated_regimes = {}
            
            # For each trading pair
            for pair, data in market_data.items():
                if pair not in self.trading_pairs:
                    continue
                
                # Default to unknown
                regime = "unknown"
                
                # Extract key metrics from market data
                # This is a simplified example - real implementation would be more sophisticated
                
                # For simplicity, use random values for now
                # In a real implementation, analyze volatility, trend strength, etc.
                regime_probabilities = {
                    "trending": np.random.random() * 0.5 + 0.3,  # 0.3-0.8
                    "ranging": np.random.random() * 0.4 + 0.2,   # 0.2-0.6
                    "volatile": np.random.random() * 0.3 + 0.1   # 0.1-0.4
                }
                
                # Normalize
                total_prob = sum(regime_probabilities.values())
                if total_prob > 0:
                    regime_probabilities = {k: v / total_prob for k, v in regime_probabilities.items()}
                
                # Select the regime with highest probability
                regime = max(regime_probabilities, key=regime_probabilities.get)
                
                # Store market conditions
                self.market_conditions[pair] = {
                    "regime": regime,
                    "regime_probabilities": regime_probabilities,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update regime
                self.market_regimes[pair] = regime
                updated_regimes[pair] = regime
                
                logger.info(f"Updated market regime for {pair}: {regime}")
            
            return updated_regimes
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
            return {pair: "unknown" for pair in self.trading_pairs}
    
    def register_performance(
        self, 
        pair: str, 
        strategy: str, 
        prediction: int, 
        outcome: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register prediction performance for a strategy
        
        Args:
            pair: Trading pair
            strategy: Strategy name
            prediction: The prediction (1 for up, -1 for down, 0 for neutral)
            outcome: The actual outcome (1 for correct, -1 for incorrect, 0 for neutral)
            details: Optional additional details
        """
        try:
            # Initialize performance tracking for this pair and strategy
            if pair not in self.performance_history:
                self.performance_history[pair] = {}
            
            if strategy not in self.performance_history[pair]:
                self.performance_history[pair][strategy] = []
            
            # Add performance record
            record = {
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "outcome": outcome,
                "regime": self.market_regimes.get(pair, "unknown")
            }
            
            if details:
                record.update(details)
            
            self.performance_history[pair][strategy].append(record)
            
            # Limit history to last 1000 records
            if len(self.performance_history[pair][strategy]) > 1000:
                self.performance_history[pair][strategy] = self.performance_history[pair][strategy][-1000:]
            
            # Update performance metrics
            self._update_performance_metrics(pair, strategy)
            
            # If adaptive weights are enabled, adjust weights based on performance
            if self.enable_adaptive_weights:
                self._adapt_strategy_weights(pair)
            
        except Exception as e:
            logger.error(f"Error registering performance for {pair} - {strategy}: {e}")
    
    def _update_performance_metrics(self, pair: str, strategy: str) -> None:
        """
        Update performance metrics for a strategy
        
        Args:
            pair: Trading pair
            strategy: Strategy name
        """
        try:
            if pair not in self.model_performance:
                self.model_performance[pair] = {}
            
            if strategy not in self.model_performance[pair]:
                self.model_performance[pair][strategy] = {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0,
                    "neutral_predictions": 0,
                    "accuracy": 0.0,
                    "by_regime": {
                        "trending": {"correct": 0, "total": 0, "accuracy": 0.0},
                        "ranging": {"correct": 0, "total": 0, "accuracy": 0.0},
                        "volatile": {"correct": 0, "total": 0, "accuracy": 0.0},
                        "unknown": {"correct": 0, "total": 0, "accuracy": 0.0}
                    }
                }
            
            # Get performance history
            history = self.performance_history[pair][strategy]
            
            # Count predictions and outcomes
            total = len(history)
            correct = sum(1 for record in history if record["outcome"] == 1)
            incorrect = sum(1 for record in history if record["outcome"] == -1)
            neutral = sum(1 for record in history if record["outcome"] == 0)
            
            # Count by regime
            by_regime = {
                "trending": {"correct": 0, "total": 0},
                "ranging": {"correct": 0, "total": 0},
                "volatile": {"correct": 0, "total": 0},
                "unknown": {"correct": 0, "total": 0}
            }
            
            for record in history:
                regime = record.get("regime", "unknown")
                if regime not in by_regime:
                    regime = "unknown"
                
                by_regime[regime]["total"] += 1
                if record["outcome"] == 1:
                    by_regime[regime]["correct"] += 1
            
            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate accuracy by regime
            for regime in by_regime:
                regime_total = by_regime[regime]["total"]
                regime_correct = by_regime[regime]["correct"]
                by_regime[regime]["accuracy"] = regime_correct / regime_total if regime_total > 0 else 0.0
            
            # Update metrics
            self.model_performance[pair][strategy] = {
                "total_predictions": total,
                "correct_predictions": correct,
                "incorrect_predictions": incorrect,
                "neutral_predictions": neutral,
                "accuracy": accuracy,
                "by_regime": by_regime
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics for {pair} - {strategy}: {e}")
    
    def _adapt_strategy_weights(self, pair: str) -> None:
        """
        Adapt strategy weights based on performance
        
        Args:
            pair: Trading pair
        """
        try:
            # Get current market regime
            regime = self.market_regimes.get(pair, "unknown")
            
            # Get current weights
            if pair not in self.strategy_weights:
                self.strategy_weights[pair] = self._get_default_weights()
            
            if regime not in self.strategy_weights[pair]:
                self.strategy_weights[pair][regime] = self._get_default_weights()[regime]
            
            weights = self.strategy_weights[pair][regime]
            
            # Get strategies with performance data
            strategies = [strategy for strategy in weights.keys() 
                         if strategy in self.model_performance.get(pair, {})]
            
            if len(strategies) < 2:
                logger.info(f"Not enough strategies with performance data for {pair}")
                return
            
            # Calculate performance-adjusted weights
            performance = {}
            for strategy in strategies:
                metrics = self.model_performance[pair][strategy]
                
                # Use regime-specific accuracy if available
                if (regime in metrics["by_regime"] and 
                    metrics["by_regime"][regime]["total"] > 5):
                    accuracy = metrics["by_regime"][regime]["accuracy"]
                else:
                    accuracy = metrics["accuracy"]
                
                # Apply a minimum threshold
                performance[strategy] = max(0.1, accuracy)
            
            # Normalize
            total_performance = sum(performance.values())
            if total_performance > 0:
                adjusted_weights = {strategy: perf / total_performance 
                                   for strategy, perf in performance.items()}
                
                # Blend with current weights (80% new, 20% old)
                for strategy in strategies:
                    weights[strategy] = 0.8 * adjusted_weights[strategy] + 0.2 * weights[strategy]
                
                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {strategy: weight / total_weight 
                              for strategy, weight in weights.items()}
                
                # Update weights
                self.strategy_weights[pair][regime] = weights
                
                logger.info(f"Adapted strategy weights for {pair} - {regime}")
            
        except Exception as e:
            logger.error(f"Error adapting strategy weights for {pair}: {e}")
    
    def get_strategy_weights(self, pair: str, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get strategy weights for a trading pair
        
        Args:
            pair: Trading pair
            regime: Optional market regime, defaults to current
            
        Returns:
            Dict: Strategy weights
        """
        try:
            if pair not in self.strategy_weights:
                return self._get_default_weights()["unknown"]
            
            if regime is None:
                regime = self.market_regimes.get(pair, "unknown")
            
            if regime not in self.strategy_weights[pair]:
                regime = "unknown"
            
            return self.strategy_weights[pair][regime]
            
        except Exception as e:
            logger.error(f"Error getting strategy weights for {pair}: {e}")
            return self._get_default_weights()["unknown"]
    
    def get_performance_metrics(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Args:
            pair: Optional trading pair, if None returns all pairs
            
        Returns:
            Dict: Performance metrics
        """
        try:
            if pair is not None:
                if pair not in self.model_performance:
                    return {}
                return self.model_performance[pair]
            else:
                return self.model_performance
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def register_signal(
        self, 
        pair: str, 
        strategy: str, 
        signal: int, 
        confidence: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a trading signal from a strategy
        
        Args:
            pair: Trading pair
            strategy: Strategy name
            signal: Trading signal (1 for BUY, -1 for SELL, 0 for HOLD)
            confidence: Confidence score (0.0-1.0)
            details: Optional additional details
        """
        try:
            # Store signal
            if pair not in self.latest_signals:
                self.latest_signals[pair] = {}
            
            self.latest_signals[pair][strategy] = {
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            if details:
                self.latest_signals[pair][strategy].update(details)
            
            # Add to history
            if pair not in self.signal_history:
                self.signal_history[pair] = []
            
            record = {
                "strategy": strategy,
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "regime": self.market_regimes.get(pair, "unknown")
            }
            
            if details:
                record.update(details)
            
            self.signal_history[pair].append(record)
            
            # Limit history to last 1000 signals
            if len(self.signal_history[pair]) > 1000:
                self.signal_history[pair] = self.signal_history[pair][-1000:]
            
            logger.info(f"Registered signal for {pair} - {strategy}: "
                      f"{'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'}, "
                      f"Confidence={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error registering signal for {pair} - {strategy}: {e}")
    
    def get_weighted_signal(self, pair: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Get weighted trading signal for a trading pair
        
        Args:
            pair: Trading pair
            
        Returns:
            tuple: (signal, confidence, details)
                signal: Trading signal (1 for BUY, -1 for SELL, 0 for HOLD)
                confidence: Confidence score (0.0-1.0)
                details: Additional details
        """
        try:
            if pair not in self.latest_signals or not self.latest_signals[pair]:
                return 0, 0.0, {"error": "No signals available"}
            
            # Get current market regime
            regime = self.market_regimes.get(pair, "unknown")
            
            # Get strategy weights
            weights = self.get_strategy_weights(pair, regime)
            
            # Calculate weighted signal
            weighted_sum = 0.0
            total_weight = 0.0
            signal_count = {"buy": 0, "sell": 0, "hold": 0}
            confidence_sum = 0.0
            
            for strategy, signal_data in self.latest_signals[pair].items():
                if strategy not in weights:
                    continue
                
                signal = signal_data["signal"]
                confidence = signal_data["confidence"]
                weight = weights.get(strategy, 0.0)
                
                # Count signals
                if signal == 1:
                    signal_count["buy"] += 1
                elif signal == -1:
                    signal_count["sell"] += 1
                else:
                    signal_count["hold"] += 1
                
                # Add to weighted sum
                weighted_sum += signal * confidence * weight
                total_weight += weight
                confidence_sum += confidence * weight
            
            # Calculate final signal and confidence
            if total_weight > 0:
                weighted_signal = weighted_sum / total_weight
                confidence = confidence_sum / total_weight
                
                # Determine signal based on weighted value and threshold
                if weighted_signal > 0.3:
                    signal = 1  # BUY
                elif weighted_signal < -0.3:
                    signal = -1  # SELL
                else:
                    signal = 0  # HOLD
                
                return signal, confidence, {
                    "weighted_value": weighted_signal,
                    "signal_count": signal_count,
                    "regime": regime,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return 0, 0.0, {"error": "No valid signals with weights"}
            
        except Exception as e:
            logger.error(f"Error calculating weighted signal for {pair}: {e}")
            return 0, 0.0, {"error": str(e)}
    
    def save_weights(self) -> bool:
        """
        Save strategy weights to disk
        
        Returns:
            bool: Whether save was successful
        """
        try:
            # Create directories if needed
            os.makedirs(self.config_path, exist_ok=True)
            
            # Save weights for each pair
            for pair in self.trading_pairs:
                if pair not in self.strategy_weights:
                    continue
                
                pair_filename = pair.replace("/", "")
                weights_path = f"{self.config_path}/{pair_filename}_weights.json"
                
                with open(weights_path, "w") as f:
                    json.dump(self.strategy_weights[pair], f, indent=4)
                
                logger.info(f"Saved strategy weights for {pair}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            return False

def main():
    """Test the model collaboration integrator"""
    try:
        # Initialize model collaboration integrator
        collaborator = ModelCollaborationIntegrator(
            trading_pairs=["SOL/USD", "ETH/USD", "BTC/USD"],
            enable_adaptive_weights=True
        )
        
        # Create sample market data
        market_data = {
            "SOL/USD": {
                "close": [120.0, 122.5, 121.8, 123.4, 125.2, 126.5, 127.8, 128.2, 129.5, 130.1],
                "volume": [10000, 12000, 8000, 9500, 11000, 14000, 15000, 13000, 12500, 11500],
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10, 0, -1)]
            },
            "ETH/USD": {
                "close": [2800.0, 2820.5, 2815.8, 2830.4, 2850.2, 2860.5, 2870.8, 2880.2, 2890.5, 2900.1],
                "volume": [5000, 5200, 4800, 5100, 5300, 5500, 5600, 5400, 5200, 5100],
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10, 0, -1)]
            }
        }
        
        # Update market regime
        regimes = collaborator.update_market_regime(market_data)
        
        # Register some signals
        collaborator.register_signal("SOL/USD", "ml_transformer", 1, 0.85, {"price": 130.1})
        collaborator.register_signal("SOL/USD", "ml_tcn", 1, 0.75, {"price": 130.1})
        collaborator.register_signal("SOL/USD", "ml_lstm", 0, 0.60, {"price": 130.1})
        collaborator.register_signal("SOL/USD", "arima", 1, 0.70, {"price": 130.1})
        collaborator.register_signal("SOL/USD", "adaptive", -1, 0.65, {"price": 130.1})
        
        # Register some performance
        collaborator.register_performance("SOL/USD", "ml_transformer", 1, 1, {"price": 130.1})
        collaborator.register_performance("SOL/USD", "ml_tcn", 1, 1, {"price": 130.1})
        collaborator.register_performance("SOL/USD", "ml_lstm", 0, 0, {"price": 130.1})
        collaborator.register_performance("SOL/USD", "arima", 1, -1, {"price": 130.1})
        collaborator.register_performance("SOL/USD", "adaptive", -1, -1, {"price": 130.1})
        
        # Get weighted signal
        signal, confidence, details = collaborator.get_weighted_signal("SOL/USD")
        
        # Print results
        print("\nMarket Regimes:")
        for pair, regime in regimes.items():
            print(f"{pair}: {regime}")
        
        print("\nWeighted Signal:")
        print(f"SOL/USD: {'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'}, "
              f"Confidence={confidence:.2f}, Details={details}")
        
        print("\nStrategy Weights:")
        for pair in collaborator.trading_pairs:
            weights = collaborator.get_strategy_weights(pair)
            print(f"{pair}: {weights}")
        
        print("\nPerformance Metrics:")
        metrics = collaborator.get_performance_metrics("SOL/USD")
        for strategy, perf in metrics.items():
            print(f"{strategy}: Accuracy={perf['accuracy']:.2f}, "
                 f"Total={perf['total_predictions']}")
        
        # Save weights
        collaborator.save_weights()
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()