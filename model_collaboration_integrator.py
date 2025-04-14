#!/usr/bin/env python3
"""
Model Collaboration Integrator

This module implements a system to integrate the outputs of multiple ML models
and trading strategies, ensuring they work together optimally rather than competing.
It uses a dynamic arbitration mechanism that adjusts how signals are weighted
based on market conditions and recent performance.

Key features:
1. Signal arbitration with dynamic weighting
2. Model specialization by market regime
3. Feedback-based performance optimization
4. Strategy conflict resolution
5. Ensemble decision making with confidence scoring
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

# Local imports
from market_context import detect_market_regime, analyze_market_context
from dynamic_position_sizing_ml import calculate_dynamic_leverage, get_optimal_trade_parameters_ml

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

# Constants
SIGNAL_TYPES = ["BUY", "SELL", "NEUTRAL"]
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider a signal strong

class ModelCollaborationIntegrator:
    """
    Integrates multiple trading models to ensure collaborative decision making
    """
    
    def __init__(
        self,
        config_path: str = "models/ensemble/strategy_ensemble_weights.json",
        performance_history_path: str = "models/performance_history.json",
        default_timeframe: str = "1h",
        lookback_window: int = 20,
        enable_adaptive_weights: bool = True
    ):
        """
        Initialize the model collaboration integrator
        
        Args:
            config_path: Path to ensemble weights configuration
            performance_history_path: Path to performance history data
            default_timeframe: Default timeframe for analysis
            lookback_window: Number of past decisions to consider for adaptation
            enable_adaptive_weights: Whether to adapt weights based on performance
        """
        self.config_path = config_path
        self.performance_history_path = performance_history_path
        self.default_timeframe = default_timeframe
        self.lookback_window = lookback_window
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # Load configuration
        self.weights = self._load_weights()
        self.performance_history = self._load_performance_history()
        
        # Initialize decision history
        self.decision_history = []
        
        # Current market regime (will be updated during execution)
        self.current_regime = "neutral"
        
        # Track strategy specializations
        self.strategy_specializations = {
            "ARIMAStrategy": ["trending", "range_bound"],
            "AdaptiveStrategy": ["volatile", "range_bound"],
            "IntegratedStrategy": ["volatile", "trending"],
            "MLStrategy": ["all"]  # ML strategy works in all regimes
        }
        
        logger.info("Model Collaboration Integrator initialized")
    
    def _load_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load ensemble weights from configuration file
        
        Returns:
            Dict: Strategy weights by market regime
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Weights configuration not found at {self.config_path}, using defaults")
                return self._create_default_weights()
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return self._create_default_weights()
    
    def _create_default_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Create default weights when configuration is not available
        
        Returns:
            Dict: Default strategy weights by market regime
        """
        regimes = ["volatile_trending_up", "volatile_trending_down", 
                  "normal_trending_up", "normal_trending_down", "neutral"]
        
        strategies = ["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"]
        
        weights = {}
        
        for regime in regimes:
            weights[regime] = {}
            for strategy in strategies:
                weights[regime][strategy] = 0.25  # Equal weights by default
        
        # Adjust weights based on strategy specializations
        for regime in regimes:
            if "volatile" in regime:
                # Boost volatile specialists in volatile regimes
                weights[regime]["IntegratedStrategy"] = 0.35
                weights[regime]["AdaptiveStrategy"] = 0.30
                weights[regime]["MLStrategy"] = 0.25
                weights[regime]["ARIMAStrategy"] = 0.10
            
            elif "trending" in regime:
                # Boost trend specialists in trending regimes
                weights[regime]["ARIMAStrategy"] = 0.30
                weights[regime]["IntegratedStrategy"] = 0.30
                weights[regime]["MLStrategy"] = 0.30
                weights[regime]["AdaptiveStrategy"] = 0.10
            
            elif regime == "neutral":
                # Boost range specialists in neutral regime
                weights[regime]["AdaptiveStrategy"] = 0.35
                weights[regime]["ARIMAStrategy"] = 0.25
                weights[regime]["MLStrategy"] = 0.25
                weights[regime]["IntegratedStrategy"] = 0.15
        
        return weights
    
    def _load_performance_history(self) -> Dict[str, List]:
        """
        Load performance history from file
        
        Returns:
            Dict: Performance history by strategy
        """
        try:
            if os.path.exists(self.performance_history_path):
                with open(self.performance_history_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Performance history not found at {self.performance_history_path}, starting fresh")
                return self._initialize_performance_history()
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return self._initialize_performance_history()
    
    def _initialize_performance_history(self) -> Dict[str, List]:
        """
        Initialize empty performance history
        
        Returns:
            Dict: Empty performance history
        """
        strategies = ["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"]
        regimes = ["volatile_trending_up", "volatile_trending_down", 
                  "normal_trending_up", "normal_trending_down", "neutral"]
        
        history = {}
        
        for strategy in strategies:
            history[strategy] = []
        
        history["ensemble"] = []
        history["regimes"] = {}
        
        for regime in regimes:
            history["regimes"][regime] = {}
            for strategy in strategies:
                history["regimes"][regime][strategy] = []
        
        return history
    
    def _save_performance_history(self):
        """Save the current performance history to disk"""
        try:
            with open(self.performance_history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=4)
            logger.info(f"Performance history saved to {self.performance_history_path}")
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detect the current market regime
        
        Args:
            market_data: Market data for regime detection
            
        Returns:
            str: Detected market regime
        """
        try:
            regime = detect_market_regime(market_data)
            logger.info(f"Detected market regime: {regime}")
            self.current_regime = regime
            return regime
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "neutral"  # Default to neutral if detection fails
    
    def update_strategy_performance(
        self,
        strategy: str,
        was_correct: bool,
        signal_type: str,
        confidence: float,
        regime: str
    ):
        """
        Update performance history for a strategy
        
        Args:
            strategy: Strategy name
            was_correct: Whether the strategy's signal was correct
            signal_type: Type of signal (BUY, SELL, NEUTRAL)
            confidence: Signal confidence (0-1)
            regime: Market regime when the signal was generated
        """
        # Create performance record
        record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "correct": was_correct,
            "signal_type": signal_type,
            "confidence": confidence,
            "regime": regime,
            "score": confidence if was_correct else -confidence
        }
        
        # Add to strategy history
        if strategy in self.performance_history:
            self.performance_history[strategy].append(record)
            
            # Limit history length to prevent unbounded growth
            if len(self.performance_history[strategy]) > 1000:
                self.performance_history[strategy] = self.performance_history[strategy][-1000:]
        
        # Add to regime-specific history
        if regime in self.performance_history["regimes"]:
            if strategy in self.performance_history["regimes"][regime]:
                self.performance_history["regimes"][regime][strategy].append(record)
                
                # Limit history length
                if len(self.performance_history["regimes"][regime][strategy]) > 200:
                    self.performance_history["regimes"][regime][strategy] = self.performance_history["regimes"][regime][strategy][-200:]
        
        logger.debug(f"Updated {strategy} performance in {regime} regime. Was correct: {was_correct}")
    
    def update_ensemble_performance(
        self,
        was_correct: bool,
        signal_type: str,
        confidence: float,
        regime: str,
        contributing_strategies: Dict[str, float]
    ):
        """
        Update performance history for the ensemble
        
        Args:
            was_correct: Whether the ensemble's signal was correct
            signal_type: Type of signal (BUY, SELL, NEUTRAL)
            confidence: Signal confidence (0-1)
            regime: Market regime when the signal was generated
            contributing_strategies: Strategies that contributed to the decision with their weights
        """
        # Create performance record
        record = {
            "timestamp": datetime.now().isoformat(),
            "correct": was_correct,
            "signal_type": signal_type,
            "confidence": confidence,
            "regime": regime,
            "contributing_strategies": contributing_strategies,
            "score": confidence if was_correct else -confidence
        }
        
        # Add to ensemble history
        self.performance_history["ensemble"].append(record)
        
        # Limit history length
        if len(self.performance_history["ensemble"]) > 1000:
            self.performance_history["ensemble"] = self.performance_history["ensemble"][-1000:]
        
        logger.debug(f"Updated ensemble performance in {regime} regime. Was correct: {was_correct}")
        
        # Periodically save performance history
        if len(self.performance_history["ensemble"]) % 10 == 0:
            self._save_performance_history()
    
    def _adapt_weights_by_performance(self, regime: str) -> Dict[str, float]:
        """
        Adapt strategy weights based on recent performance
        
        Args:
            regime: Current market regime
            
        Returns:
            Dict: Updated weights for the current regime
        """
        if not self.enable_adaptive_weights:
            return self.weights.get(regime, self._create_default_weights()[regime])
        
        # Get current weights
        current_weights = self.weights.get(regime, self._create_default_weights()[regime])
        
        # Get recent performance by strategy for this regime
        recent_performance = {}
        
        for strategy in current_weights.keys():
            # Check if we have regime-specific performance data
            if (regime in self.performance_history["regimes"] and 
                strategy in self.performance_history["regimes"][regime] and
                len(self.performance_history["regimes"][regime][strategy]) > 0):
                
                # Get most recent records
                records = self.performance_history["regimes"][regime][strategy]
                recent = records[-min(self.lookback_window, len(records)):]
                
                # Calculate score
                scores = [r["score"] for r in recent]
                if scores:
                    recent_performance[strategy] = sum(scores) / len(scores)
                else:
                    recent_performance[strategy] = 0.0
            
            else:
                # If no regime-specific data, use overall performance
                if strategy in self.performance_history:
                    records = self.performance_history[strategy]
                    if records:
                        recent = records[-min(self.lookback_window, len(records)):]
                        scores = [r["score"] for r in recent]
                        if scores:
                            recent_performance[strategy] = sum(scores) / len(scores)
                        else:
                            recent_performance[strategy] = 0.0
                    else:
                        recent_performance[strategy] = 0.0
                else:
                    recent_performance[strategy] = 0.0
        
        # Normalize scores to prevent extreme weights
        min_score = min(recent_performance.values()) if recent_performance else 0
        max_score = max(recent_performance.values()) if recent_performance else 0
        
        # Adjust for all negative scores
        if max_score <= 0:
            # All strategies are performing poorly, keep weights relatively equal
            normalized_scores = {s: 1.0 for s in recent_performance.keys()}
        else:
            # Normalize to [0, 1] range, but keep some weight for all strategies
            score_range = max_score - min_score
            if score_range > 0:
                normalized_scores = {
                    s: max(0.1, (score - min_score) / score_range) 
                    for s, score in recent_performance.items()
                }
            else:
                normalized_scores = {s: 1.0 for s in recent_performance.keys()}
        
        # Calculate new weights
        total = sum(normalized_scores.values())
        if total > 0:
            new_weights = {s: score / total for s, score in normalized_scores.items()}
        else:
            new_weights = {s: 1.0 / len(normalized_scores) for s in normalized_scores.keys()}
        
        # Blend with current weights (70% new, 30% old) for stability
        final_weights = {}
        for strategy in current_weights.keys():
            if strategy in new_weights:
                final_weights[strategy] = 0.7 * new_weights[strategy] + 0.3 * current_weights[strategy]
            else:
                final_weights[strategy] = current_weights[strategy]
        
        # Ensure weights sum to 1.0
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {s: w / total for s, w in final_weights.items()}
        
        logger.info(f"Adapted weights for {regime} regime: {final_weights}")
        
        # Update stored weights
        self.weights[regime] = final_weights
        
        return final_weights
    
    def integrate_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        market_data: pd.DataFrame,
        current_positions: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate signals from multiple strategies into a single decision
        
        Args:
            signals: Dictionary of signals from different strategies
            market_data: Current market data
            current_positions: Current open positions
            
        Returns:
            Dict: Integrated signal decision
        """
        # Detect market regime
        regime = self.detect_market_regime(market_data)
        
        # Get adapted weights for current regime
        weights = self._adapt_weights_by_performance(regime)
        
        # Calculate weighted signals
        buy_score = 0.0
        sell_score = 0.0
        neutral_score = 0.0
        contributing_strategies = {}
        
        for strategy, signal in signals.items():
            # Skip if strategy not in weights (likely a new strategy)
            if strategy not in weights:
                weights[strategy] = 1.0 / (len(weights) + 1)
                
                # Renormalize
                total = sum(weights.values())
                weights = {s: w / total for s, w in weights.items()}
            
            strategy_weight = weights[strategy]
            signal_type = signal.get("signal", "NEUTRAL").upper()
            signal_strength = signal.get("strength", 0.5)
            
            # Adjust weight based on confidence
            effective_weight = strategy_weight * signal_strength
            
            # Record contribution
            contributing_strategies[strategy] = effective_weight
            
            # Add to score
            if signal_type == "BUY":
                buy_score += effective_weight
            elif signal_type == "SELL":
                sell_score += effective_weight
            else:  # NEUTRAL
                neutral_score += effective_weight
        
        # Determine final signal
        if buy_score > sell_score and buy_score > neutral_score:
            signal_type = "BUY"
            confidence = buy_score / (buy_score + sell_score + neutral_score)
        elif sell_score > buy_score and sell_score > neutral_score:
            signal_type = "SELL"
            confidence = sell_score / (buy_score + sell_score + neutral_score)
        else:
            signal_type = "NEUTRAL"
            confidence = neutral_score / (buy_score + sell_score + neutral_score)
        
        # Consider current positions when making the decision
        if current_positions:
            # If already in a position, require higher confidence to reverse
            for position in current_positions.values():
                position_type = position.get("type", "")
                # If long position and sell signal, check confidence
                if position_type == "LONG" and signal_type == "SELL":
                    if confidence < CONFIDENCE_THRESHOLD + 0.1:  # Higher threshold for reversals
                        signal_type = "NEUTRAL"
                        confidence = 0.5
                        logger.info("Signal not strong enough to reverse long position")
                
                # If short position and buy signal, check confidence
                elif position_type == "SHORT" and signal_type == "BUY":
                    if confidence < CONFIDENCE_THRESHOLD + 0.1:  # Higher threshold for reversals
                        signal_type = "NEUTRAL"
                        confidence = 0.5
                        logger.info("Signal not strong enough to reverse short position")
        
        # Get optimal trade parameters for this decision
        direction = "long" if signal_type == "BUY" else "short" if signal_type == "SELL" else "neutral"
        
        # Get market context for position sizing
        context = analyze_market_context(market_data)
        
        # Default parameters
        params = {
            'leverage': 1.0,
            'margin_pct': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'market_regime': regime,
            'confidence': confidence
        }
        
        # Get asset from market data if available
        asset = None
        if hasattr(market_data, 'name'):
            asset = market_data.name
        
        # If signal is actionable, get optimal parameters
        if signal_type != "NEUTRAL" and confidence >= CONFIDENCE_THRESHOLD:
            try:
                if hasattr(market_data, 'to_dict'):
                    data_dict = market_data.to_dict()
                else:
                    data_dict = market_data
                
                # Use ML-enhanced position sizing
                params = get_optimal_trade_parameters_ml(
                    data_dict,
                    direction,
                    signal_strength=confidence,
                    ml_confidence=confidence,
                    asset=asset or "SOL/USD"
                )
            except Exception as e:
                logger.error(f"Error getting optimal trade parameters: {e}")
        
        # Create integrated decision
        decision = {
            "signal": signal_type,
            "confidence": confidence,
            "regime": regime,
            "contributing_strategies": contributing_strategies,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to decision history
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        logger.info(f"Integrated signal: {signal_type} with confidence {confidence:.2f} in {regime} regime")
        return decision
    
    def analyze_strategy_conflicts(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conflicts between strategies
        
        Args:
            signals: Dictionary of signals from different strategies
            
        Returns:
            Dict: Analysis of conflicts
        """
        # Count signals by type
        counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        strategies_by_signal = {"BUY": [], "SELL": [], "NEUTRAL": []}
        
        for strategy, signal in signals.items():
            signal_type = signal.get("signal", "NEUTRAL").upper()
            counts[signal_type] += 1
            strategies_by_signal[signal_type].append(strategy)
        
        # Determine if there's a conflict
        conflict = counts["BUY"] > 0 and counts["SELL"] > 0
        
        # Calculate conflict severity (0-1)
        total_strategies = sum(counts.values())
        agreement_score = max(counts.values()) / total_strategies if total_strategies > 0 else 0
        conflict_severity = 1.0 - agreement_score
        
        # Identify strongest signals
        strongest_buy = None
        strongest_sell = None
        
        for strategy, signal in signals.items():
            signal_type = signal.get("signal", "NEUTRAL").upper()
            strength = signal.get("strength", 0.5)
            
            if signal_type == "BUY" and (strongest_buy is None or strength > strongest_buy[1]):
                strongest_buy = (strategy, strength)
            elif signal_type == "SELL" and (strongest_sell is None or strength > strongest_sell[1]):
                strongest_sell = (strategy, strength)
        
        analysis = {
            "conflict": conflict,
            "conflict_severity": conflict_severity,
            "agreement_score": agreement_score,
            "strongest_buy": strongest_buy[0] if strongest_buy else None,
            "strongest_sell": strongest_sell[0] if strongest_sell else None,
            "strategies_by_signal": strategies_by_signal
        }
        
        logger.debug(f"Strategy conflict analysis: {analysis}")
        return analysis
    
    def resolve_conflicts(
        self,
        signals: Dict[str, Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resolve conflicts between strategies based on specialization and performance
        
        Args:
            signals: Dictionary of signals from different strategies
            market_data: Current market data
            
        Returns:
            Dict: Modified signals with conflicts resolved
        """
        # First analyze conflicts
        analysis = self.analyze_strategy_conflicts(signals)
        
        # If no conflict, return original signals
        if not analysis["conflict"]:
            return signals
        
        # Detect market regime
        regime = self.current_regime
        
        # Get strategies specialized for this regime
        specialized_strategies = []
        for strategy, specializations in self.strategy_specializations.items():
            if "all" in specializations:
                specialized_strategies.append(strategy)
                continue
                
            for spec in specializations:
                if spec in regime:
                    specialized_strategies.append(strategy)
                    break
        
        # Resolve conflicts by favoring specialized strategies
        modified_signals = signals.copy()
        
        if analysis["conflict_severity"] > 0.4:  # Significant conflict
            logger.info(f"Resolving strategy conflict in {regime} regime")
            
            # Favor specialized strategies with stronger signals
            strongest_specialized = None
            strongest_strength = 0.0
            
            for strategy, signal in signals.items():
                if strategy in specialized_strategies:
                    strength = signal.get("strength", 0.5)
                    if strength > strongest_strength:
                        strongest_specialized = strategy
                        strongest_strength = strength
            
            if strongest_specialized and strongest_strength > CONFIDENCE_THRESHOLD:
                winning_signal = signals[strongest_specialized]["signal"].upper()
                
                # Reduce confidence of conflicting signals
                for strategy, signal in modified_signals.items():
                    if strategy != strongest_specialized:
                        signal_type = signal.get("signal", "NEUTRAL").upper()
                        # If signal conflicts with the winning signal
                        if (winning_signal == "BUY" and signal_type == "SELL") or (winning_signal == "SELL" and signal_type == "BUY"):
                            # Reduce strength
                            modified_signals[strategy]["strength"] = signal.get("strength", 0.5) * 0.6
                            logger.debug(f"Reduced strength of conflicting {strategy} signal")
            
            # If no specialized strategy found, then use MLStrategy if available
            elif "MLStrategy" in signals:
                ml_signal = signals["MLStrategy"]["signal"].upper()
                ml_strength = signals["MLStrategy"].get("strength", 0.5)
                
                if ml_strength > CONFIDENCE_THRESHOLD:
                    # Reduce confidence of conflicting signals
                    for strategy, signal in modified_signals.items():
                        if strategy != "MLStrategy":
                            signal_type = signal.get("signal", "NEUTRAL").upper()
                            # If signal conflicts with ML signal
                            if (ml_signal == "BUY" and signal_type == "SELL") or (ml_signal == "SELL" and signal_type == "BUY"):
                                # Reduce strength
                                modified_signals[strategy]["strength"] = signal.get("strength", 0.5) * 0.6
                                logger.debug(f"Reduced strength of conflicting {strategy} signal")
        
        return modified_signals
    
    def get_collaborative_confidence(
        self,
        decision: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate a collaborative confidence score for the decision
        
        Args:
            decision: Integrated signal decision
            market_data: Current market data
            
        Returns:
            float: Collaborative confidence score (0-1)
        """
        signal_type = decision["signal"]
        base_confidence = decision["confidence"]
        regime = decision["regime"]
        
        # If neutral, confidence is always as provided
        if signal_type == "NEUTRAL":
            return base_confidence
        
        # Consider recent performance
        recent_ensemble_performance = 0.0
        ensemble_history = self.performance_history.get("ensemble", [])
        if ensemble_history:
            recent = ensemble_history[-min(self.lookback_window, len(ensemble_history)):]
            scores = [r["score"] for r in recent]
            if scores:
                recent_ensemble_performance = sum(scores) / len(scores)
        
        # Consider market conditions
        market_confidence = 0.5  # Neutral by default
        
        try:
            context = analyze_market_context(market_data)
            market_confidence = context.get("confidence", 0.5)
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
        
        # Calculate collaborative confidence
        # Weights: 50% signal confidence, 30% recent performance, 20% market conditions
        collaborative_confidence = (
            0.5 * base_confidence +
            0.3 * max(0, min(1, 0.5 + recent_ensemble_performance)) +
            0.2 * market_confidence
        )
        
        # Ensure in range 0-1
        collaborative_confidence = max(0, min(1, collaborative_confidence))
        
        logger.debug(f"Collaborative confidence: {collaborative_confidence:.2f} (base: {base_confidence:.2f})")
        return collaborative_confidence

def main():
    """Run a test of the model collaboration integrator"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model collaboration integrator')
    parser.add_argument('--asset', type=str, default="SOL/USD",
                      help='Asset to analyze')
    parser.add_argument('--timeframe', type=str, default="1h",
                      help='Timeframe to use')
    parser.add_argument('--data-dir', type=str, default="historical_data",
                      help='Directory with historical data')
    
    args = parser.parse_args()
    
    # Initialize the integrator
    integrator = ModelCollaborationIntegrator()
    
    # Load some test data
    asset_clean = args.asset.replace("/", "")
    file_path = os.path.join(args.data_dir, f"{asset_clean}_{args.timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Historical data file not found: {file_path}")
        return
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} historical data points for {args.asset} ({args.timeframe})")
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return
    
    # Create some test signals
    test_signals = {
        "ARIMAStrategy": {
            "signal": "BUY",
            "strength": 0.7,
            "price": df['close'].iloc[-1],
            "timestamp": datetime.now().isoformat()
        },
        "AdaptiveStrategy": {
            "signal": "NEUTRAL",
            "strength": 0.5,
            "price": df['close'].iloc[-1],
            "timestamp": datetime.now().isoformat()
        },
        "IntegratedStrategy": {
            "signal": "SELL",
            "strength": 0.8,
            "price": df['close'].iloc[-1],
            "timestamp": datetime.now().isoformat()
        },
        "MLStrategy": {
            "signal": "BUY",
            "strength": 0.6,
            "price": df['close'].iloc[-1],
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Detect market regime
    regime = integrator.detect_market_regime(df)
    logger.info(f"Detected market regime: {regime}")
    
    # Resolve conflicts
    resolved_signals = integrator.resolve_conflicts(test_signals, df)
    logger.info("Resolved signals:")
    for strategy, signal in resolved_signals.items():
        logger.info(f"  {strategy}: {signal['signal']} (strength: {signal['strength']:.2f})")
    
    # Integrate signals
    decision = integrator.integrate_signals(resolved_signals, df)
    logger.info(f"Integrated decision: {decision['signal']} with confidence {decision['confidence']:.2f}")
    
    # Calculate collaborative confidence
    collab_confidence = integrator.get_collaborative_confidence(decision, df)
    logger.info(f"Collaborative confidence: {collab_confidence:.2f}")
    
    # Test updating performance
    integrator.update_strategy_performance("ARIMAStrategy", True, "BUY", 0.7, regime)
    integrator.update_strategy_performance("MLStrategy", True, "BUY", 0.6, regime)
    integrator.update_strategy_performance("AdaptiveStrategy", False, "NEUTRAL", 0.5, regime)
    integrator.update_strategy_performance("IntegratedStrategy", False, "SELL", 0.8, regime)
    
    integrator.update_ensemble_performance(True, "BUY", 0.7, regime, 
                                         {"ARIMAStrategy": 0.35, "MLStrategy": 0.30, 
                                          "AdaptiveStrategy": 0.15, "IntegratedStrategy": 0.20})
    
    logger.info("Updated performance history")
    
    # Save performance history
    integrator._save_performance_history()

if __name__ == "__main__":
    main()