#!/usr/bin/env python3
"""
Model Collaboration Integrator

This module provides a central hub for connecting ML predictions to trading signals.
"""

import os
import sys
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

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
    
    A central hub for connecting ML predictions to trading signals.
    This integrator arbitrates between:
    1. Multiple ML models (transformer, TCN, LSTM)
    2. Traditional strategies (ARIMA, Adaptive)
    3. Different assets (SOL/USD, ETH/USD, BTC/USD)
    """
    
    def __init__(
        self,
        assets: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        strategies: List[str] = ["MLStrategy", "ARIMAStrategy", "AdaptiveStrategy"],
        use_ml_position_sizing: bool = False
    ):
        """
        Initialize the model collaboration integrator
        
        Args:
            assets: Trading assets to support
            strategies: Trading strategies to arbitrate
            use_ml_position_sizing: Whether to use ML for position sizing
        """
        self.assets = assets
        self.strategies = strategies
        self.use_ml_position_sizing = use_ml_position_sizing
        
        # Store the latest predictions and signals
        self.ml_predictions = {}
        self.strategy_signals = {}
        self.arbitrated_signals = {}
        
        # Track performance for adaptive weighting
        self.performance_history = {}
        self.strategy_weights = self._initialize_strategy_weights()
        
        # Configure market regime detection
        self.market_regimes = {}
        self.regime_weights = self._initialize_regime_weights()
        
        logger.info(f"Model Collaboration Integrator initialized with {len(assets)} assets and {len(strategies)} strategies")
    
    def _initialize_strategy_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize strategy weights
        
        Returns:
            Dict: Strategy weights by asset and strategy
        """
        weights = {}
        
        for asset in self.assets:
            asset_weights = {}
            
            # Set default weights based on strategy type
            for strategy in self.strategies:
                if strategy == "MLStrategy":
                    asset_weights[strategy] = 0.60  # 60% weight to ML
                elif strategy == "ARIMAStrategy":
                    asset_weights[strategy] = 0.25  # 25% weight to ARIMA
                elif strategy == "AdaptiveStrategy":
                    asset_weights[strategy] = 0.15  # 15% weight to Adaptive
                else:
                    asset_weights[strategy] = 0.10  # 10% weight to other strategies
            
            weights[asset] = asset_weights
        
        return weights
    
    def _initialize_regime_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize regime weights for different market conditions
        
        Returns:
            Dict: Regime weights by market regime and strategy
        """
        weights = {
            "trending_up": {
                "MLStrategy": 0.60,
                "ARIMAStrategy": 0.30,
                "AdaptiveStrategy": 0.10
            },
            "trending_down": {
                "MLStrategy": 0.65,
                "ARIMAStrategy": 0.25,
                "AdaptiveStrategy": 0.10
            },
            "volatile": {
                "MLStrategy": 0.70,
                "ARIMAStrategy": 0.20,
                "AdaptiveStrategy": 0.10
            },
            "sideways": {
                "MLStrategy": 0.55,
                "ARIMAStrategy": 0.25,
                "AdaptiveStrategy": 0.20
            },
            "uncertain": {
                "MLStrategy": 0.50,
                "ARIMAStrategy": 0.30,
                "AdaptiveStrategy": 0.20
            }
        }
        
        return weights
    
    def detect_market_regime(
        self,
        asset: str,
        market_data: Dict[str, Any]
    ) -> str:
        """
        Detect the current market regime
        
        Args:
            asset: Trading asset
            market_data: Market data
            
        Returns:
            str: Market regime
        """
        try:
            # In a real implementation, this would analyze the market data
            # to determine the current market regime.
            
            # For demonstration, randomly choose a regime with a bias toward volatile
            regimes = ["trending_up", "trending_down", "volatile", "sideways", "uncertain"]
            weights = [0.2, 0.2, 0.3, 0.2, 0.1]  # Bias toward volatile
            
            regime = random.choices(regimes, weights=weights, k=1)[0]
            
            # Store the detected regime
            self.market_regimes[asset] = regime
            
            logger.info(f"Detected {regime} market regime for {asset}")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime for {asset}: {e}")
            return "uncertain"
    
    def register_ml_prediction(
        self,
        asset: str,
        prediction: Dict[str, Any]
    ) -> None:
        """
        Register an ML prediction
        
        Args:
            asset: Trading asset
            prediction: ML prediction
        """
        self.ml_predictions[asset] = prediction
        logger.info(f"Registered ML prediction for {asset}")
    
    def register_strategy_signal(
        self,
        asset: str,
        strategy: str,
        signal: Dict[str, Any]
    ) -> None:
        """
        Register a strategy signal
        
        Args:
            asset: Trading asset
            strategy: Strategy name
            signal: Strategy signal
        """
        # Initialize if needed
        if asset not in self.strategy_signals:
            self.strategy_signals[asset] = {}
        
        self.strategy_signals[asset][strategy] = signal
        logger.info(f"Registered {signal.get('signal_type', 'UNKNOWN')} signal from {strategy} for {asset}")
    
    def _combine_signals(
        self,
        asset: str,
        market_regime: str
    ) -> Dict[str, Any]:
        """
        Combine signals from all strategies for an asset
        
        Args:
            asset: Trading asset
            market_regime: Current market regime
            
        Returns:
            Dict: Combined signal
        """
        if asset not in self.strategy_signals or not self.strategy_signals[asset]:
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "asset": asset,
                "strategy": "Combined",
                "params": {}
            }
        
        # Get regime weights
        regime_weights = self.regime_weights.get(market_regime, {})
        
        # Get asset-specific weights
        asset_weights = self.strategy_weights.get(asset, {})
        
        # Combine weights with preference to the regime
        combined_weights = {}
        for strategy in self.strategies:
            regime_weight = regime_weights.get(strategy, 0.5)
            asset_weight = asset_weights.get(strategy, 0.5)
            combined_weights[strategy] = (regime_weight * 0.7) + (asset_weight * 0.3)
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(combined_weights.values())
        if weight_sum > 0:
            combined_weights = {k: v / weight_sum for k, v in combined_weights.items()}
        
        # Initialize vote counters
        signal_votes = {
            "BUY": {"weight": 0.0, "count": 0, "strength": 0.0, "confidence": 0.0},
            "SELL": {"weight": 0.0, "count": 0, "strength": 0.0, "confidence": 0.0},
            "NEUTRAL": {"weight": 0.0, "count": 0, "strength": 0.0, "confidence": 0.0}
        }
        
        # Gather votes
        for strategy, signal in self.strategy_signals[asset].items():
            signal_type = signal.get("signal_type", "NEUTRAL")
            strength = signal.get("strength", 0.0)
            confidence = signal.get("confidence", 0.5)
            
            weight = combined_weights.get(strategy, 0.0)
            
            # Add vote
            signal_votes[signal_type]["weight"] += weight
            signal_votes[signal_type]["count"] += 1
            signal_votes[signal_type]["strength"] += strength * weight
            signal_votes[signal_type]["confidence"] += confidence * weight
        
        # Determine the combined signal
        best_signal = "NEUTRAL"
        best_weight = 0.0
        
        for signal_type, vote in signal_votes.items():
            if vote["weight"] > best_weight:
                best_weight = vote["weight"]
                best_signal = signal_type
        
        # Calculate strength and confidence
        strength = 0.0
        confidence = 0.5
        
        if signal_votes[best_signal]["count"] > 0:
            strength = signal_votes[best_signal]["strength"] / signal_votes[best_signal]["weight"] if signal_votes[best_signal]["weight"] > 0 else 0.0
            confidence = signal_votes[best_signal]["confidence"] / signal_votes[best_signal]["weight"] if signal_votes[best_signal]["weight"] > 0 else 0.5
        
        # Build combined signal
        combined_signal = {
            "signal_type": best_signal,
            "strength": strength,
            "confidence": confidence,
            "asset": asset,
            "strategy": "Combined",
            "market_regime": market_regime,
            "params": {
                "strategy_weights": combined_weights,
                "signal_votes": signal_votes
            }
        }
        
        return combined_signal
    
    def arbitrate_signals(
        self,
        asset: str,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Arbitrate between strategy signals for an asset
        
        Args:
            asset: Trading asset
            market_data: Optional market data
            
        Returns:
            Dict: Arbitrated signal
        """
        try:
            # Detect market regime if we have market data
            market_regime = "uncertain"
            if market_data:
                market_regime = self.detect_market_regime(asset, market_data)
            elif asset in self.market_regimes:
                market_regime = self.market_regimes[asset]
            
            # Combine signals
            combined_signal = self._combine_signals(asset, market_regime)
            
            # Store the arbitrated signal
            self.arbitrated_signals[asset] = combined_signal
            
            logger.info(f"Arbitrated {combined_signal['signal_type']} signal for {asset} in {market_regime} regime")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error arbitrating signals for {asset}: {e}")
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "asset": asset,
                "strategy": "Combined",
                "params": {}
            }
    
    def update_strategy_performance(
        self,
        asset: str,
        strategy: str,
        trade_result: Dict[str, Any]
    ) -> None:
        """
        Update strategy performance
        
        Args:
            asset: Trading asset
            strategy: Strategy name
            trade_result: Trade result
        """
        try:
            # Initialize if needed
            if asset not in self.performance_history:
                self.performance_history[asset] = {}
            
            if strategy not in self.performance_history[asset]:
                self.performance_history[asset][strategy] = []
            
            # Add trade result to performance history
            self.performance_history[asset][strategy].append(trade_result)
            
            # Limit history size
            max_history = 100
            if len(self.performance_history[asset][strategy]) > max_history:
                self.performance_history[asset][strategy] = self.performance_history[asset][strategy][-max_history:]
            
            # Update strategy weights based on performance
            self._update_strategy_weights(asset)
            
            logger.info(f"Updated performance for {strategy} on {asset}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance for {strategy} on {asset}: {e}")
    
    def _update_strategy_weights(self, asset: str) -> None:
        """
        Update strategy weights based on performance
        
        Args:
            asset: Trading asset
        """
        # Initialize if needed
        if asset not in self.strategy_weights:
            self.strategy_weights[asset] = {}
            for strategy in self.strategies:
                self.strategy_weights[asset][strategy] = 1.0 / len(self.strategies)
        
        if asset not in self.performance_history:
            return
        
        # Calculate performance scores
        performance_scores = {}
        
        for strategy in self.strategies:
            if strategy not in self.performance_history[asset] or not self.performance_history[asset][strategy]:
                performance_scores[strategy] = 0.0
                continue
            
            # Consider only recent trades
            recent_trades = self.performance_history[asset][strategy][-20:]
            
            # Calculate score based on profitability
            profit_sum = sum(trade.get("profit_pct", 0.0) for trade in recent_trades)
            win_count = sum(1 for trade in recent_trades if trade.get("profit_pct", 0.0) > 0)
            
            # Calculate win rate and average profit
            win_rate = win_count / len(recent_trades) if recent_trades else 0.0
            avg_profit = profit_sum / len(recent_trades) if recent_trades else 0.0
            
            # Calculate score (higher is better)
            score = (win_rate * 0.6) + (avg_profit * 0.4 * 10)  # Scale profit
            performance_scores[strategy] = max(0.1, score)  # Minimum weight of 0.1
        
        # Normalize scores to get weights
        score_sum = sum(performance_scores.values())
        if score_sum > 0:
            new_weights = {strategy: score / score_sum for strategy, score in performance_scores.items()}
            
            # Apply gradual changes (70% new, 30% old) to avoid drastic shifts
            for strategy in self.strategies:
                old_weight = self.strategy_weights[asset].get(strategy, 0.0)
                new_weight = new_weights.get(strategy, 0.0)
                self.strategy_weights[asset][strategy] = (new_weight * 0.7) + (old_weight * 0.3)
            
            # Re-normalize to ensure they sum to 1.0
            weight_sum = sum(self.strategy_weights[asset].values())
            if weight_sum > 0:
                self.strategy_weights[asset] = {k: v / weight_sum for k, v in self.strategy_weights[asset].items()}
            
            logger.info(f"Updated strategy weights for {asset}: {self.strategy_weights[asset]}")
    
    def get_enhanced_position_size(
        self,
        asset: str,
        signal: Dict[str, Any],
        base_position_size: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get enhanced position size based on ML confidence
        
        Args:
            asset: Trading asset
            signal: Trading signal
            base_position_size: Base position size
            
        Returns:
            Tuple[float, Dict]: Enhanced position size and details
        """
        if not self.use_ml_position_sizing:
            return base_position_size, {
                "position_sizing": "base",
                "position_size": base_position_size
            }
        
        try:
            # Get confidence from signal
            confidence = signal.get("confidence", 0.5)
            
            # Get market regime
            market_regime = signal.get("market_regime", "uncertain")
            
            # Adjust base size based on market regime
            regime_factors = {
                "trending_up": 1.5,    # Increase size in uptrends
                "trending_down": 1.2,  # Moderately increase in downtrends
                "volatile": 1.8,       # Largest increase in volatile markets
                "sideways": 0.8,       # Reduce size in sideways markets
                "uncertain": 0.5       # Smallest size in uncertain markets
            }
            
            regime_factor = regime_factors.get(market_regime, 1.0)
            
            # Adjust by confidence
            # Scale from 0.5-1.0 range to 0.0-1.0 range
            confidence_adjusted = (confidence - 0.5) * 2.0 if confidence > 0.5 else 0.0
            
            # Calculate confidence factor (0.5-1.5 range)
            confidence_factor = 0.5 + confidence_adjusted
            
            # Combine factors
            combined_factor = regime_factor * confidence_factor
            
            # Apply to base position size (max: 2.5x, min: 0.25x)
            enhanced_size = base_position_size * combined_factor
            enhanced_size = max(0.25 * base_position_size, min(2.5 * base_position_size, enhanced_size))
            
            details = {
                "position_sizing": "enhanced",
                "base_size": base_position_size,
                "confidence": confidence,
                "confidence_factor": confidence_factor,
                "market_regime": market_regime,
                "regime_factor": regime_factor,
                "combined_factor": combined_factor,
                "position_size": enhanced_size
            }
            
            logger.info(f"Enhanced position size for {asset}: {enhanced_size:.2f} (base: {base_position_size:.2f}, factor: {combined_factor:.2f})")
            return enhanced_size, details
            
        except Exception as e:
            logger.error(f"Error calculating enhanced position size for {asset}: {e}")
            return base_position_size, {
                "position_sizing": "base",
                "position_size": base_position_size
            }

def main():
    """Test the model collaboration integrator"""
    integrator = ModelCollaborationIntegrator()
    
    # Register ML prediction
    integrator.register_ml_prediction("SOL/USD", {
        "direction": "BUY",
        "confidence": 0.85,
        "prediction": 1.2
    })
    
    # Register strategy signals
    integrator.register_strategy_signal("SOL/USD", "MLStrategy", {
        "signal_type": "BUY",
        "strength": 0.85,
        "confidence": 0.85
    })
    
    integrator.register_strategy_signal("SOL/USD", "ARIMAStrategy", {
        "signal_type": "BUY",
        "strength": 0.70,
        "confidence": 0.70
    })
    
    integrator.register_strategy_signal("SOL/USD", "AdaptiveStrategy", {
        "signal_type": "NEUTRAL",
        "strength": 0.30,
        "confidence": 0.50
    })
    
    # Arbitrate signals
    signal = integrator.arbitrate_signals("SOL/USD")
    
    # Get enhanced position size
    position_size, details = integrator.get_enhanced_position_size("SOL/USD", signal, 1000.0)
    
    # Print results
    print(f"Arbitrated Signal: {signal['signal_type']} with {signal['confidence']:.2f} confidence")
    print(f"Enhanced Position Size: ${position_size:.2f}")
    print(f"Details: {json.dumps(details, indent=2)}")

if __name__ == "__main__":
    main()