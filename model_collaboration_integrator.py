#!/usr/bin/env python3
"""
Model Collaboration Integrator

This module provides the integration layer between ML models and trading strategies,
enabling collaborative decision making based on market regimes and strategy performance.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

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
    Integrator for ML models and trading strategies that facilitates
    collaborative decision making based on market regimes.
    """
    
    def __init__(
        self,
        config_path: str = "models/ensemble/strategy_ensemble_weights.json",
        strategies: Optional[List[str]] = None,
        enable_adaptive_weights: bool = True,
        learning_rate: float = 0.01,
        log_level: str = "INFO"
    ):
        """
        Initialize the model collaboration integrator
        
        Args:
            config_path: Path to collaboration configuration
            strategies: List of strategy names (if None, load from config)
            enable_adaptive_weights: Whether to allow adaptive weights
            learning_rate: Rate at which to update strategy weights
            log_level: Logging level
        """
        self.config_path = config_path
        self.strategies = strategies
        self.enable_adaptive_weights = enable_adaptive_weights
        self.learning_rate = learning_rate
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Load configuration
        self.config = self._load_config()
        
        # Set default strategy weights if not found in config
        if "strategy_weights" not in self.config:
            self.config["strategy_weights"] = self._initialize_strategy_weights()
        
        # Set default regime if not provided
        self.current_regime = self.config.get("default_regime", "neutral")
        
        # Initialize performance tracking
        if "performance" not in self.config:
            self.config["performance"] = {}
        
        logger.info(f"Model Collaboration Integrator initialized with {len(self.config['strategy_weights'])} strategies")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load collaboration configuration
        
        Returns:
            Dict: Collaboration configuration
        """
        default_config = {
            "strategy_weights": {},
            "default_regime": "neutral",
            "regimes": ["trending_bullish", "trending_bearish", "volatile", "neutral", "ranging"],
            "performance": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Create directory for config if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Load from file if it exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded collaboration configuration from {self.config_path}")
            else:
                # Create default config file
                config = default_config
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Created default collaboration configuration at {self.config_path}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading collaboration configuration: {e}")
            return default_config
    
    def _initialize_strategy_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize strategy weights
        
        Returns:
            Dict: Strategy weights by regime
        """
        # Get list of regimes
        regimes = self.config.get("regimes", ["trending_bullish", "trending_bearish", "volatile", "neutral", "ranging"])
        
        # Get list of strategies
        strategies = self.strategies or ["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"]
        
        # Initialize weights
        weights = {}
        
        for regime in regimes:
            regime_weights = {}
            
            # Set initial weights based on regime
            for strategy in strategies:
                if "ARIMA" in strategy:
                    # ARIMA works well in ranging and trending markets
                    if regime in ["ranging", "trending_bullish", "trending_bearish"]:
                        regime_weights[strategy] = 0.3
                    else:
                        regime_weights[strategy] = 0.2
                elif "Adaptive" in strategy:
                    # Adaptive works well in volatile markets
                    if regime in ["volatile"]:
                        regime_weights[strategy] = 0.3
                    else:
                        regime_weights[strategy] = 0.2
                elif "Integrated" in strategy:
                    # Integrated works well in all markets
                    regime_weights[strategy] = 0.25
                elif "ML" in strategy:
                    # ML strategy starts with a conservative weight
                    regime_weights[strategy] = 0.25
                else:
                    # Unknown strategy gets a default weight
                    regime_weights[strategy] = 0.1
            
            # Normalize weights to sum to 1.0
            total_weight = sum(regime_weights.values())
            for strategy in regime_weights:
                regime_weights[strategy] /= total_weight
            
            weights[regime] = regime_weights
        
        return weights
    
    def update_market_regime(self, regime: str):
        """
        Update current market regime
        
        Args:
            regime: Market regime
        """
        # Validate regime
        regimes = self.config.get("regimes", ["trending_bullish", "trending_bearish", "volatile", "neutral", "ranging"])
        
        if regime not in regimes:
            logger.warning(f"Unknown regime: {regime}, defaulting to neutral")
            regime = "neutral"
        
        self.current_regime = regime
        logger.info(f"Updated market regime to {regime}")
    
    def get_strategy_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get strategy weights for a specific regime
        
        Args:
            regime: Market regime (None for current)
            
        Returns:
            Dict: Strategy weights
        """
        if regime is None:
            regime = self.current_regime
        
        # Get weights for regime
        weights = self.config.get("strategy_weights", {}).get(regime, {})
        
        if not weights:
            logger.warning(f"No weights found for regime: {regime}, using defaults")
            weights = self._initialize_strategy_weights().get(regime, {})
        
        return weights
    
    def register_performance(
        self,
        strategy: str,
        outcome: float,
        signal_type: str,
        regime: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Register strategy performance for adaptive weight adjustment
        
        Args:
            strategy: Strategy name
            outcome: Performance outcome (-1 to 1, where 1 is best)
            signal_type: Signal type (BUY, SELL, NEUTRAL)
            regime: Market regime
            details: Additional performance details
        """
        if not self.enable_adaptive_weights:
            return
        
        # Validate outcome
        outcome = max(-1.0, min(1.0, outcome))
        
        # Update performance tracking
        if strategy not in self.config["performance"]:
            self.config["performance"][strategy] = []
        
        # Add performance record
        performance_record = {
            "strategy": strategy,
            "outcome": outcome,
            "signal_type": signal_type,
            "regime": regime,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.config["performance"][strategy].append(performance_record)
        
        # Limit history size
        max_history = 1000
        if len(self.config["performance"][strategy]) > max_history:
            self.config["performance"][strategy] = self.config["performance"][strategy][-max_history:]
        
        # Update weights based on performance
        self._update_weights_based_on_performance(strategy, outcome, regime)
        
        # Save updated config
        self._save_config()
        
        logger.info(f"Registered performance for {strategy} in {regime} regime: {outcome:.2f}")
    
    def _update_weights_based_on_performance(
        self,
        strategy: str,
        outcome: float,
        regime: str
    ):
        """
        Update strategy weights based on performance
        
        Args:
            strategy: Strategy name
            outcome: Performance outcome (-1 to 1, where 1 is best)
            regime: Market regime
        """
        if strategy not in self.config["strategy_weights"].get(regime, {}):
            logger.warning(f"Strategy {strategy} not found in weights for regime {regime}")
            return
        
        # Get current weights
        weights = self.config["strategy_weights"].get(regime, {})
        
        # Calculate weight adjustment
        adjustment = outcome * self.learning_rate
        
        # Update weight for this strategy
        weights[strategy] = max(0.05, min(0.95, weights[strategy] + adjustment))
        
        # Adjust other weights to maintain sum of 1.0
        total_weight = sum(weights.values())
        scaling_factor = 1.0 / total_weight
        
        for s in weights:
            weights[s] *= scaling_factor
        
        # Update config
        self.config["strategy_weights"][regime] = weights
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config["last_updated"] = datetime.datetime.now().isoformat()
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving collaboration configuration: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all strategies
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        for strategy, performances in self.config.get("performance", {}).items():
            if not performances:
                continue
            
            # Calculate overall metrics
            outcomes = [p.get("outcome", 0.0) for p in performances]
            metrics[strategy] = {
                "mean_outcome": sum(outcomes) / len(outcomes),
                "count": len(outcomes),
                "positive_rate": sum(1 for o in outcomes if o > 0) / len(outcomes),
                "negative_rate": sum(1 for o in outcomes if o < 0) / len(outcomes)
            }
            
            # Calculate regime-specific metrics
            regime_metrics = {}
            for regime in self.config.get("regimes", []):
                regime_outcomes = [p.get("outcome", 0.0) for p in performances if p.get("regime") == regime]
                
                if regime_outcomes:
                    regime_metrics[regime] = {
                        "mean_outcome": sum(regime_outcomes) / len(regime_outcomes),
                        "count": len(regime_outcomes),
                        "positive_rate": sum(1 for o in regime_outcomes if o > 0) / len(regime_outcomes),
                        "negative_rate": sum(1 for o in regime_outcomes if o < 0) / len(regime_outcomes)
                    }
            
            metrics[strategy]["regime_metrics"] = regime_metrics
        
        return metrics
    
    def arbitrate_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Arbitrate between multiple competing signals
        
        Args:
            signals: Signals by strategy
            regime: Market regime (None for current)
            
        Returns:
            Dict: Arbitrated signal
        """
        if not signals:
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "strategy": "Collaboration",
                "confidence": 0.0
            }
        
        # Use current regime if not specified
        if regime is None:
            regime = self.current_regime
        
        # Get strategy weights for this regime
        weights = self.get_strategy_weights(regime)
        
        # Calculate weighted signals
        buy_score = 0.0
        sell_score = 0.0
        neutral_score = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        weighted_params = {}
        
        for strategy, signal in signals.items():
            # Get strategy weight
            weight = weights.get(strategy, 0.1)
            
            # Get signal properties
            signal_type = signal.get("signal_type", "NEUTRAL")
            strength = signal.get("strength", 0.0)
            
            # Optional confidence
            confidence = signal.get("confidence", strength)
            
            # Calculate weighted contribution
            weighted_value = weight * strength
            
            # Add to signal type scores
            if signal_type == "BUY":
                buy_score += weighted_value
            elif signal_type == "SELL":
                sell_score += weighted_value
            else:  # NEUTRAL
                neutral_score += weighted_value
            
            # Add to weighted strength and confidence
            weighted_strength += weight * strength
            weighted_confidence += weight * confidence
            
            # Merge params (for leverage, position size, etc.)
            for key, value in signal.get("params", {}).items():
                if key in weighted_params:
                    weighted_params[key] = weighted_params[key] + (weight * value)
                else:
                    weighted_params[key] = weight * value
        
        # Determine final signal type
        final_type = "NEUTRAL"
        max_score = max(buy_score, sell_score, neutral_score)
        
        if max_score == buy_score and buy_score > neutral_score:
            final_type = "BUY"
        elif max_score == sell_score and sell_score > neutral_score:
            final_type = "SELL"
        
        # Build final signal
        final_signal = {
            "signal_type": final_type,
            "strength": weighted_strength,
            "strategy": "Collaboration",
            "confidence": weighted_confidence,
            "params": weighted_params,
            "regime": regime,
            "component_signals": signals
        }
        
        logger.info(f"Arbitrated signals: BUY={buy_score:.2f}, SELL={sell_score:.2f}, NEUTRAL={neutral_score:.2f} → {final_type}")
        return final_signal
    
    def integrate_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        ml_prediction: Optional[Dict[str, Any]] = None,
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Integrate strategy signals with ML prediction
        
        Args:
            signals: Signals by strategy
            ml_prediction: ML prediction (if available)
            regime: Market regime (None for current)
            
        Returns:
            Dict: Integrated signal
        """
        # If ML prediction is available, add it to signals
        if ml_prediction:
            ml_signal = {
                "signal_type": ml_prediction.get("signal_type", "NEUTRAL"),
                "strength": ml_prediction.get("strength", 0.0),
                "confidence": ml_prediction.get("confidence", 0.0),
                "strategy": "MLStrategy",
                "params": ml_prediction.get("params", {})
            }
            
            signals["MLStrategy"] = ml_signal
        
        # Arbitrate between signals
        return self.arbitrate_signals(signals, regime)
    
    def resolve_conflicts(
        self,
        signals: Dict[str, Dict[str, Any]],
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between strategy signals
        
        Args:
            signals: Signals by strategy
            regime: Market regime (None for current)
            
        Returns:
            Dict: Resolved signal
        """
        # Simple wrapper around arbitrate_signals for clearer API
        return self.arbitrate_signals(signals, regime)
    
    def get_collaborative_confidence(
        self,
        signal: Dict[str, Any]
    ) -> float:
        """
        Get collaborative confidence score for a signal
        
        Args:
            signal: Trading signal
            
        Returns:
            float: Confidence score (0-1)
        """
        # Extract from arbitrated signal if available
        if "confidence" in signal:
            return signal["confidence"]
        
        # Otherwise calculate from strength
        return signal.get("strength", 0.0)

def main():
    """Test the model collaboration integrator"""
    logging.basicConfig(level=logging.INFO)
    
    # Create integrator
    integrator = ModelCollaborationIntegrator(
        enable_adaptive_weights=True
    )
    
    # Create test signals
    signals = {
        "ARIMAStrategy": {
            "signal_type": "BUY",
            "strength": 0.7,
            "strategy": "ARIMAStrategy",
            "pair": "SOL/USD",
            "params": {"leverage": 5.0}
        },
        "AdaptiveStrategy": {
            "signal_type": "NEUTRAL",
            "strength": 0.3,
            "strategy": "AdaptiveStrategy",
            "pair": "SOL/USD",
            "params": {"leverage": 0.0}
        },
        "IntegratedStrategy": {
            "signal_type": "BUY",
            "strength": 0.6,
            "strategy": "IntegratedStrategy",
            "pair": "SOL/USD",
            "params": {"leverage": 10.0}
        }
    }
    
    # Arbitrate signals
    arbitrated = integrator.arbitrate_signals(signals)
    
    # Print result
    print(json.dumps(arbitrated, indent=2))
    
    # Register performance
    integrator.register_performance(
        strategy="ARIMAStrategy",
        outcome=0.5,
        signal_type="BUY",
        regime="trending_bullish"
    )
    
    # Print updated weights
    print("Updated weights:")
    print(json.dumps(integrator.get_strategy_weights("trending_bullish"), indent=2))

if __name__ == "__main__":
    main()