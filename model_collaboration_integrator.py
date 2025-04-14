#!/usr/bin/env python3
"""
Model Collaboration Integrator

This module integrates different trading models and strategies to work together collaboratively,
enabling them to make better decisions as a team rather than individually.

It implements advanced signal arbitration, strategy weight optimization, and regime-specific
collaboration mechanisms.
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

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
    Integrates multiple trading models and strategies to work collaboratively
    """
    
    def __init__(
        self,
        config_path: str = "models/ensemble/strategy_ensemble_weights.json",
        strategies: List[str] = None,
        enable_adaptive_weights: bool = True,
        weight_update_frequency: int = 24,  # Hours
        min_performance_samples: int = 10,
        performance_memory_length: int = 100,
        log_level: str = "INFO"
    ):
        """
        Initialize the model collaboration integrator
        
        Args:
            config_path: Path to strategy weights configuration
            strategies: List of strategies to integrate (None for all)
            enable_adaptive_weights: Whether to adapt weights based on performance
            weight_update_frequency: Hours between weight updates
            min_performance_samples: Minimum samples before adapting weights
            performance_memory_length: Maximum performance samples to keep
            log_level: Logging level
        """
        self.config_path = config_path
        self.strategies = strategies
        self.enable_adaptive_weights = enable_adaptive_weights
        self.weight_update_frequency = weight_update_frequency
        self.min_performance_samples = min_performance_samples
        self.performance_memory_length = performance_memory_length
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Load strategy weights
        self.strategy_weights = self._load_strategy_weights()
        
        # Performance tracking
        self.strategy_performance = {}
        self.last_weight_update = datetime.datetime.now() - datetime.timedelta(hours=weight_update_frequency)
        
        # Current market regime
        self.current_regime = "neutral"
        
        logger.info(f"Model Collaboration Integrator initialized with {len(self.strategy_weights)} strategies")
    
    def _load_strategy_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load strategy weights from config file
        
        Returns:
            Dict: Strategy weights by market regime
        """
        default_weights = {
            "trending_bullish": {
                "ARIMAStrategy": 0.4,
                "AdaptiveStrategy": 0.2,
                "IntegratedStrategy": 0.3,
                "MLStrategy": 0.1
            },
            "trending_bearish": {
                "ARIMAStrategy": 0.4,
                "AdaptiveStrategy": 0.2,
                "IntegratedStrategy": 0.3,
                "MLStrategy": 0.1
            },
            "volatile": {
                "ARIMAStrategy": 0.1,
                "AdaptiveStrategy": 0.3,
                "IntegratedStrategy": 0.4,
                "MLStrategy": 0.2
            },
            "neutral": {
                "ARIMAStrategy": 0.25,
                "AdaptiveStrategy": 0.25,
                "IntegratedStrategy": 0.25,
                "MLStrategy": 0.25
            },
            "ranging": {
                "ARIMAStrategy": 0.2,
                "AdaptiveStrategy": 0.4,
                "IntegratedStrategy": 0.2,
                "MLStrategy": 0.2
            }
        }
        
        try:
            # Create directory for config if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Load from file if it exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    weights = json.load(f)
                logger.info(f"Loaded strategy weights from {self.config_path}")
            else:
                # Create default weights file
                weights = default_weights
                with open(self.config_path, 'w') as f:
                    json.dump(weights, f, indent=4)
                logger.info(f"Created default strategy weights at {self.config_path}")
            
            # Filter to specified strategies if provided
            if self.strategies:
                for regime in weights:
                    weights[regime] = {k: v for k, v in weights[regime].items() if k in self.strategies}
                    # Renormalize weights
                    total = sum(weights[regime].values())
                    if total > 0:
                        weights[regime] = {k: v / total for k, v in weights[regime].items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error loading strategy weights: {e}")
            return default_weights
    
    def _save_strategy_weights(self):
        """Save current strategy weights to config file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.strategy_weights, f, indent=4)
            logger.info(f"Saved strategy weights to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving strategy weights: {e}")
    
    def update_market_regime(self, regime: str):
        """
        Update the current market regime
        
        Args:
            regime: Current market regime
        """
        if regime not in self.strategy_weights:
            logger.warning(f"Unknown market regime: {regime}, using 'neutral'")
            regime = "neutral"
        
        self.current_regime = regime
        logger.info(f"Market regime updated to: {regime}")
    
    def register_performance(
        self,
        strategy: str,
        outcome: float,
        signal_type: str,
        regime: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Register strategy performance for adaptation
        
        Args:
            strategy: Strategy name
            outcome: Performance outcome (-1 to 1, where 1 is perfect)
            signal_type: Signal type (BUY, SELL, NEUTRAL)
            regime: Market regime during the signal
            details: Optional performance details
        """
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        # Add performance record
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "outcome": outcome,
            "signal_type": signal_type,
            "regime": regime,
            "details": details or {}
        }
        
        # Add to performance history, keep most recent samples
        self.strategy_performance[strategy].append(record)
        if len(self.strategy_performance[strategy]) > self.performance_memory_length:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-self.performance_memory_length:]
        
        logger.debug(f"Registered performance for {strategy}: {outcome:.2f} ({signal_type} in {regime})")
        
        # Check if we should update weights
        self._check_weight_update()
    
    def _check_weight_update(self):
        """Check if weights should be updated based on performance"""
        if not self.enable_adaptive_weights:
            return
        
        # Check if enough time has passed since last update
        now = datetime.datetime.now()
        hours_since_update = (now - self.last_weight_update).total_seconds() / 3600
        
        if hours_since_update >= self.weight_update_frequency:
            # Check if we have enough performance samples
            has_enough_samples = all(
                len(samples) >= self.min_performance_samples
                for samples in self.strategy_performance.values()
            )
            
            if has_enough_samples:
                self._update_weights_by_performance()
                self.last_weight_update = now
    
    def _update_weights_by_performance(self):
        """Update strategy weights based on performance"""
        logger.info("Updating strategy weights based on performance")
        
        # Calculate performance scores by regime
        regime_scores = {}
        
        for regime in self.strategy_weights:
            regime_scores[regime] = {}
            
            for strategy, samples in self.strategy_performance.items():
                # Filter samples for this regime
                regime_samples = [s for s in samples if s["regime"] == regime]
                
                if regime_samples:
                    # Calculate average performance
                    avg_performance = sum(s["outcome"] for s in regime_samples) / len(regime_samples)
                    
                    # Convert to positive score (add 1 to shift from -1..1 to 0..2)
                    # and square it to emphasize differences
                    score = ((avg_performance + 1) / 2) ** 2
                    
                    regime_scores[regime][strategy] = max(0.01, score)  # Ensure minimum weight
                else:
                    # Keep existing weight if no samples for this regime
                    regime_scores[regime][strategy] = self.strategy_weights[regime].get(strategy, 0.25)
        
        # Update weights based on scores
        for regime in self.strategy_weights:
            if regime in regime_scores:
                scores = regime_scores[regime]
                
                # Normalize scores to sum to 1
                total_score = sum(scores.values())
                if total_score > 0:
                    self.strategy_weights[regime] = {
                        strategy: score / total_score
                        for strategy, score in scores.items()
                    }
        
        # Save updated weights
        self._save_strategy_weights()
        
        logger.info("Strategy weights updated based on performance")
    
    def get_strategy_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get strategy weights for a specific regime
        
        Args:
            regime: Market regime (None for current regime)
            
        Returns:
            Dict: Strategy weights
        """
        if regime is None:
            regime = self.current_regime
        
        if regime not in self.strategy_weights:
            logger.warning(f"Unknown market regime: {regime}, using 'neutral'")
            regime = "neutral"
        
        return self.strategy_weights[regime]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all strategies
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        for strategy, samples in self.strategy_performance.items():
            if samples:
                # Overall performance
                overall_perf = sum(s["outcome"] for s in samples) / len(samples)
                
                # Performance by regime
                regime_perf = {}
                for regime in self.strategy_weights:
                    regime_samples = [s for s in samples if s["regime"] == regime]
                    if regime_samples:
                        avg = sum(s["outcome"] for s in regime_samples) / len(regime_samples)
                        regime_perf[regime] = avg
                
                # Performance by signal type
                signal_perf = {}
                for signal in ["BUY", "SELL", "NEUTRAL"]:
                    signal_samples = [s for s in samples if s["signal_type"] == signal]
                    if signal_samples:
                        avg = sum(s["outcome"] for s in signal_samples) / len(signal_samples)
                        signal_perf[signal] = avg
                
                metrics[strategy] = {
                    "overall": overall_perf,
                    "by_regime": regime_perf,
                    "by_signal": signal_perf,
                    "samples": len(samples)
                }
        
        return metrics
    
    def arbitrate_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Arbitrate between competing signals from different strategies
        
        Args:
            signals: Dictionary of strategy signals
            regime: Market regime (None for current regime)
            
        Returns:
            Dict: Arbitrated signal
        """
        if not signals:
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "strategy": "Collaborative",
                "details": {
                    "reason": "No signals provided"
                }
            }
        
        # Use specified regime or current regime
        if regime is None:
            regime = self.current_regime
        
        # Get strategy weights for this regime
        weights = self.get_strategy_weights(regime)
        
        # Filter to strategies in the weights
        filtered_signals = {
            strategy: signal
            for strategy, signal in signals.items()
            if strategy in weights
        }
        
        if not filtered_signals:
            logger.warning(f"No signals from weighted strategies, using raw signals")
            filtered_signals = signals
        
        # Calculate signal scores
        buy_score = 0.0
        sell_score = 0.0
        neutral_score = 0.0
        total_weight = 0.0
        
        signal_details = {}
        
        for strategy, signal in filtered_signals.items():
            # Get strategy weight
            weight = weights.get(strategy, 0.0)
            
            # Skip if weight is zero
            if weight <= 0.0:
                continue
            
            # Get signal type and strength
            signal_type = signal.get("signal_type", "NEUTRAL")
            strength = signal.get("strength", 0.5)
            confidence = signal.get("confidence", strength)
            
            # Calculate weighted score
            weighted_score = weight * confidence
            
            # Add to appropriate score
            if signal_type == "BUY":
                buy_score += weighted_score
            elif signal_type == "SELL":
                sell_score += weighted_score
            else:  # NEUTRAL
                neutral_score += weighted_score
            
            # Add to total weight
            total_weight += weight
            
            # Record signal details
            signal_details[strategy] = {
                "type": signal_type,
                "strength": strength,
                "confidence": confidence,
                "weight": weight,
                "weighted_score": weighted_score
            }
        
        # Normalize scores if there were any weighted signals
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            neutral_score /= total_weight
        
        # Determine final signal type and strength
        if buy_score > sell_score and buy_score > neutral_score:
            signal_type = "BUY"
            strength = buy_score
        elif sell_score > buy_score and sell_score > neutral_score:
            signal_type = "SELL"
            strength = sell_score
        else:
            signal_type = "NEUTRAL"
            strength = neutral_score
        
        # Calculate confidence based on dominance of the winning signal
        total_score = buy_score + sell_score + neutral_score
        if total_score > 0:
            if signal_type == "BUY":
                confidence = buy_score / total_score
            elif signal_type == "SELL":
                confidence = sell_score / total_score
            else:
                confidence = neutral_score / total_score
        else:
            confidence = 0.0
        
        # Create arbitrated signal
        result = {
            "signal_type": signal_type,
            "strength": strength,
            "confidence": confidence,
            "strategy": "Collaborative",
            "regime": regime,
            "details": {
                "buy_score": buy_score,
                "sell_score": sell_score,
                "neutral_score": neutral_score,
                "signals": signal_details,
                "weights": weights
            }
        }
        
        logger.info(f"Arbitrated signal: {signal_type} (strength: {strength:.2f}, confidence: {confidence:.2f})")
        return result
    
    def calculate_position_sizing(
        self,
        signal: Dict[str, Any],
        base_size: float,
        max_size: float,
        min_size: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size based on signal confidence
        
        Args:
            signal: Arbitrated signal
            base_size: Base position size
            max_size: Maximum position size
            min_size: Minimum position size
            
        Returns:
            Tuple: (position_size, details)
        """
        # Extract signal properties
        signal_type = signal.get("signal_type", "NEUTRAL")
        confidence = signal.get("confidence", 0.0)
        
        # No position for neutral signals
        if signal_type == "NEUTRAL":
            return 0.0, {"reason": "NEUTRAL signal"}
        
        # Calculate size based on confidence
        size_range = max_size - min_size
        position_size = min_size + (size_range * confidence)
        
        # Ensure size is within limits
        position_size = max(min_size, min(max_size, position_size))
        
        details = {
            "base_size": base_size,
            "confidence": confidence,
            "min_size": min_size,
            "max_size": max_size,
            "final_size": position_size,
            "sizing_explanation": f"Position sized {position_size:.2f} based on confidence {confidence:.2f}"
        }
        
        logger.info(f"Calculated position size: {position_size:.2f} for {signal_type} signal (confidence: {confidence:.2f})")
        return position_size, details