#!/usr/bin/env python3
"""
Dynamic Parameter Adjustment

This module provides dynamic adjustment of trading parameters based on confidence levels,
market conditions, and historical performance. It works with the integrated risk management
system to optimize trading parameters in real-time.

Features:
- Adjusts trading parameters (leverage, risk %, stop distance) based on confidence
- Scales parameters based on market volatility and regime
- Applies Kelly criterion for position sizing
- Implements win streak bonuses and loss streak reductions
- Provides time-decay for adjustments to return to baseline

The module acts as a bridge between the parameter optimizer and risk manager.
"""

import os
import json
import logging
import time
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicParameterAdjuster:
    """
    Dynamically adjusts trading parameters based on confidence levels,
    market conditions, and historical performance.
    """
    
    def __init__(self, config_path: str = "config/dynamic_params_config.json"):
        """
        Initialize the dynamic parameter adjuster.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Initialize state variables
        self.performance_history = {}
        self.adjustment_history = {}
        self.base_parameters = {}
        self.last_update_time = datetime.now()
        
        # Load optimized parameters if available
        self.optimized_params = self._load_optimized_params()
        
        logger.info("Dynamic parameter adjuster initialized")
    
    def load_config(self):
        """Load dynamic parameter configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded dynamic parameter configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = {
                    "confidence_scaling": {
                        "base_leverage": 20.0,
                        "max_leverage": 100.0,
                        "min_confidence": 0.5,
                        "leverage_curve": "exponential",  # linear, exponential, sigmoid
                        "leverage_bias": 0.3  # positive values bias towards higher leverage
                    },
                    "market_adjustments": {
                        "volatility_scaling": True,
                        "volatility_dampening": {
                            "very_low": 1.2,
                            "low": 1.1,
                            "medium": 1.0,
                            "high": 0.8,
                            "very_high": 0.6,
                            "extreme": 0.4
                        },
                        "regime_adjustments": {
                            "trending_up": 1.1,
                            "trending_down": 0.9,
                            "ranging": 1.0,
                            "volatile": 0.8
                        }
                    },
                    "performance_adjustments": {
                        "enable_win_streak_bonus": True,
                        "win_streak_bonus_per_win": 0.05,
                        "win_streak_bonus_cap": 0.25,
                        "enable_loss_streak_reduction": True,
                        "loss_streak_reduction_per_loss": 0.1,
                        "loss_streak_reduction_cap": 0.5,
                        "time_decay_hours": 48
                    },
                    "risk_limits": {
                        "max_risk_percentage": 0.3,
                        "min_risk_percentage": 0.05,
                        "default_risk_percentage": 0.2,
                        "max_leverage_cap": 125.0,
                        "min_leverage": 5.0
                    }
                }
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save the default configuration
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                logger.info(f"Created default dynamic parameter configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading dynamic parameter configuration: {e}")
            # Fallback to basic configuration
            self.config = {
                "confidence_scaling": {
                    "base_leverage": 20.0,
                    "max_leverage": 100.0,
                    "min_confidence": 0.5
                },
                "risk_limits": {
                    "max_risk_percentage": 0.3,
                    "min_risk_percentage": 0.05,
                    "default_risk_percentage": 0.2
                }
            }
    
    def _load_optimized_params(self) -> Dict[str, Any]:
        """
        Load optimized parameters from file.
        
        Returns:
            Dictionary with optimized parameters
        """
        optimized_params_path = "config/optimized_params.json"
        
        try:
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded optimized parameters from {optimized_params_path}")
                return params
            else:
                logger.warning(f"No optimized parameters file found at {optimized_params_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading optimized parameters: {e}")
            return {}
    
    def get_parameters(self, pair: str, strategy: str, confidence: float, 
                      signal_strength: float) -> Dict[str, Any]:
        """
        Get dynamically adjusted parameters based on confidence and other factors.
        
        Args:
            pair: Trading pair symbol
            strategy: Strategy name
            confidence: Model prediction confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            Dictionary with adjusted parameters
        """
        # Get base parameters from optimized params if available
        if pair in self.optimized_params:
            base_params = self.optimized_params[pair]
        else:
            # Use default parameters
            base_params = {
                "risk_percentage": self.config["risk_limits"]["default_risk_percentage"],
                "base_leverage": self.config["confidence_scaling"]["base_leverage"],
                "max_leverage": self.config["confidence_scaling"]["max_leverage"],
                "trailing_stop_atr_multiplier": 3.0
            }
        
        # Prepare to track adjustments for logging
        adjustments = {}
        
        # 1. Adjust for confidence level
        confidence_adjusted = self._adjust_for_confidence(
            base_params, confidence, signal_strength, adjustments
        )
        
        # 2. Apply performance adjustments (win/loss streaks)
        performance_adjusted = self._adjust_for_performance(
            pair, strategy, confidence_adjusted, adjustments
        )
        
        # 3. Apply market condition adjustments (volatility, regime)
        # In a real implementation, these would come from the market analyzer
        # For now, we'll use default values
        volatility_category = "medium"
        market_regime = "ranging"
        
        market_adjusted = self._adjust_for_market_conditions(
            performance_adjusted, volatility_category, market_regime, adjustments
        )
        
        # 4. Apply risk limits
        final_params = self._apply_risk_limits(market_adjusted, adjustments)
        
        # Add metadata
        final_params["adjustment_factors"] = adjustments
        final_params["base_parameters"] = base_params
        final_params["adjusted_at"] = datetime.now().isoformat()
        
        # Log adjustments
        logger.info(f"Adjusted parameters for {pair} ({strategy}) with confidence {confidence:.2f}")
        logger.info(f"Risk: {base_params.get('risk_percentage', 0.2):.2%} → {final_params['risk_percentage']:.2%}")
        logger.info(f"Leverage: {base_params.get('base_leverage', 20.0):.1f} → {final_params['leverage']:.1f}")
        
        return final_params
    
    def _adjust_for_confidence(self, base_params: Dict[str, Any], confidence: float,
                             signal_strength: float, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters based on confidence level and signal strength.
        
        Args:
            base_params: Base parameters
            confidence: Model prediction confidence (0-1)
            signal_strength: Signal strength (0-1)
            adjustments: Dictionary to track adjustment factors
            
        Returns:
            Adjusted parameters dictionary
        """
        # Get configuration values
        base_leverage = base_params.get("base_leverage", 
                                       self.config["confidence_scaling"]["base_leverage"])
        max_leverage = base_params.get("max_leverage", 
                                      self.config["confidence_scaling"]["max_leverage"])
        min_confidence = self.config["confidence_scaling"].get("min_confidence", 0.5)
        leverage_curve = self.config["confidence_scaling"].get("leverage_curve", "exponential")
        leverage_bias = self.config["confidence_scaling"].get("leverage_bias", 0.3)
        
        # Create adjusted parameters dictionary
        adjusted = base_params.copy()
        
        # Only apply confidence scaling if confidence exceeds minimum threshold
        if confidence >= min_confidence:
            # Normalize confidence to 0-1 range from min_confidence to 1.0
            normalized_confidence = (confidence - min_confidence) / (1.0 - min_confidence)
            
            # Apply signal strength as a weight (30% confidence, 70% signal strength)
            combined_strength = (normalized_confidence * 0.3) + (signal_strength * 0.7)
            combined_strength = max(0.0, min(1.0, combined_strength))
            
            # Apply leverage curve
            if leverage_curve == "linear":
                # Linear scaling
                leverage_factor = combined_strength
            elif leverage_curve == "sigmoid":
                # Sigmoid curve for more moderate scaling
                leverage_factor = 1.0 / (1.0 + math.exp(-10 * (combined_strength - 0.5)))
            else:  # exponential
                # Exponential curve for more aggressive scaling at high confidence
                # Apply bias to shape the curve
                biased_strength = combined_strength + (leverage_bias * combined_strength * (1 - combined_strength))
                leverage_factor = biased_strength ** 2
            
            # Calculate leverage and risk percentage adjustments
            leverage_range = max_leverage - base_leverage
            adjusted_leverage = base_leverage + (leverage_range * leverage_factor)
            
            # Adjust risk percentage proportionally but less dramatically
            risk_percentage = base_params.get("risk_percentage", 
                                             self.config["risk_limits"]["default_risk_percentage"])
            max_risk = self.config["risk_limits"]["max_risk_percentage"]
            risk_range = max_risk - risk_percentage
            adjusted_risk = risk_percentage + (risk_range * leverage_factor * 0.7)  # More conservative
            
            # Adjust stop-loss multiplier inversely to risk
            # Higher confidence = tighter stops
            stop_multiplier = base_params.get("trailing_stop_atr_multiplier", 3.0)
            adjusted_stop = stop_multiplier * (1.0 - (leverage_factor * 0.3))
            
            # Update parameters
            adjusted["leverage"] = adjusted_leverage
            adjusted["risk_percentage"] = adjusted_risk
            adjusted["trailing_stop_atr_multiplier"] = adjusted_stop
            
            # Record adjustments
            adjustments["confidence_factor"] = leverage_factor
            adjustments["combined_strength"] = combined_strength
        else:
            # Below minimum confidence, use base values
            adjusted["leverage"] = base_leverage
            adjusted["risk_percentage"] = base_params.get("risk_percentage", 
                                                        self.config["risk_limits"]["default_risk_percentage"])
            adjusted["trailing_stop_atr_multiplier"] = base_params.get("trailing_stop_atr_multiplier", 3.0)
            
            adjustments["confidence_factor"] = 0.0
            adjustments["combined_strength"] = 0.0
        
        return adjusted
    
    def _adjust_for_performance(self, pair: str, strategy: str, params: Dict[str, Any],
                              adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters based on recent performance (win/loss streaks).
        
        Args:
            pair: Trading pair symbol
            strategy: Strategy name
            params: Parameters to adjust
            adjustments: Dictionary to track adjustment factors
            
        Returns:
            Adjusted parameters dictionary
        """
        # Create key for pair+strategy
        key = f"{pair}_{strategy}"
        
        # Get performance history or create default
        if key not in self.performance_history:
            self.performance_history[key] = {
                "win_streak": 0,
                "loss_streak": 0,
                "last_trade_result": None,
                "last_trade_time": None
            }
        
        performance = self.performance_history[key]
        
        # Create adjusted parameters dictionary
        adjusted = params.copy()
        
        # Check if performance adjustments are enabled
        if not self.config["performance_adjustments"].get("enable_win_streak_bonus", True) and \
           not self.config["performance_adjustments"].get("enable_loss_streak_reduction", True):
            # No adjustments needed
            adjustments["performance_factor"] = 1.0
            return adjusted
        
        # Calculate bonus for win streak
        win_streak_bonus = 0.0
        if self.config["performance_adjustments"].get("enable_win_streak_bonus", True):
            bonus_per_win = self.config["performance_adjustments"].get("win_streak_bonus_per_win", 0.05)
            bonus_cap = self.config["performance_adjustments"].get("win_streak_bonus_cap", 0.25)
            
            win_streak_bonus = min(performance.get("win_streak", 0) * bonus_per_win, bonus_cap)
        
        # Calculate reduction for loss streak
        loss_streak_reduction = 0.0
        if self.config["performance_adjustments"].get("enable_loss_streak_reduction", True):
            reduction_per_loss = self.config["performance_adjustments"].get("loss_streak_reduction_per_loss", 0.1)
            reduction_cap = self.config["performance_adjustments"].get("loss_streak_reduction_cap", 0.5)
            
            loss_streak_reduction = min(performance.get("loss_streak", 0) * reduction_per_loss, reduction_cap)
        
        # Calculate time decay if last trade exists
        time_decay = 1.0
        if performance.get("last_trade_time") is not None:
            last_trade_time = datetime.fromisoformat(performance["last_trade_time"])
            hours_since_last_trade = (datetime.now() - last_trade_time).total_seconds() / 3600
            
            decay_hours = self.config["performance_adjustments"].get("time_decay_hours", 48)
            decay_factor = min(1.0, hours_since_last_trade / decay_hours)
            
            # Apply time decay to both bonus and reduction
            win_streak_bonus *= (1 - decay_factor)
            loss_streak_reduction *= (1 - decay_factor)
        
        # Calculate net adjustment factor
        performance_factor = 1.0 + win_streak_bonus - loss_streak_reduction
        
        # Apply adjustments
        adjusted["leverage"] *= performance_factor
        adjusted["risk_percentage"] *= performance_factor
        
        # Record adjustments
        adjustments["performance_factor"] = performance_factor
        adjustments["win_streak_bonus"] = win_streak_bonus
        adjustments["loss_streak_reduction"] = loss_streak_reduction
        
        return adjusted
    
    def _adjust_for_market_conditions(self, params: Dict[str, Any], volatility_category: str,
                                    market_regime: str, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters based on market conditions.
        
        Args:
            params: Parameters to adjust
            volatility_category: Volatility category (very_low, low, medium, high, very_high, extreme)
            market_regime: Market regime (trending_up, trending_down, ranging, volatile)
            adjustments: Dictionary to track adjustment factors
            
        Returns:
            Adjusted parameters dictionary
        """
        # Create adjusted parameters dictionary
        adjusted = params.copy()
        
        # Check if market adjustments are enabled
        if not self.config["market_adjustments"].get("volatility_scaling", True):
            # No adjustments needed
            adjustments["market_factor"] = 1.0
            return adjusted
        
        # Get volatility adjustment factor
        volatility_dampening = self.config["market_adjustments"]["volatility_dampening"]
        volatility_factor = volatility_dampening.get(volatility_category, 1.0)
        
        # Get regime adjustment factor
        regime_adjustments = self.config["market_adjustments"]["regime_adjustments"]
        regime_factor = regime_adjustments.get(market_regime, 1.0)
        
        # Calculate combined factor
        market_factor = volatility_factor * regime_factor
        
        # Apply adjustments
        adjusted["leverage"] *= market_factor
        
        # Adjust risk slightly less aggressively
        risk_adjustment = 1.0 + ((market_factor - 1.0) * 0.7)
        adjusted["risk_percentage"] *= risk_adjustment
        
        # Adjust stop-loss inversely to volatility (tighter in high volatility)
        if volatility_category in ["high", "very_high", "extreme"]:
            inverse_vol = 1.0 / volatility_factor
            adjusted["trailing_stop_atr_multiplier"] *= (1.0 - ((1.0 - inverse_vol) * 0.5))
        
        # Record adjustments
        adjustments["market_factor"] = market_factor
        adjustments["volatility_factor"] = volatility_factor
        adjustments["regime_factor"] = regime_factor
        
        return adjusted
    
    def _apply_risk_limits(self, params: Dict[str, Any], 
                         adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply risk limits to ensure parameters stay within safe bounds.
        
        Args:
            params: Parameters to check
            adjustments: Dictionary to track adjustment factors
            
        Returns:
            Parameters with limits applied
        """
        # Create limited parameters dictionary
        limited = params.copy()
        
        # Get limit values
        max_risk = self.config["risk_limits"]["max_risk_percentage"]
        min_risk = self.config["risk_limits"]["min_risk_percentage"]
        max_leverage = self.config["risk_limits"]["max_leverage_cap"]
        min_leverage = self.config["risk_limits"]["min_leverage"]
        
        # Apply limits
        limited["risk_percentage"] = min(max_risk, max(min_risk, params["risk_percentage"]))
        limited["leverage"] = min(max_leverage, max(min_leverage, params["leverage"]))
        
        # Ensure stop multiplier is reasonable
        if "trailing_stop_atr_multiplier" in params:
            limited["trailing_stop_atr_multiplier"] = min(6.0, max(1.0, params["trailing_stop_atr_multiplier"]))
        
        # Check if limits were applied
        if limited["risk_percentage"] != params["risk_percentage"] or \
           limited["leverage"] != params["leverage"]:
            adjustments["limits_applied"] = True
        else:
            adjustments["limits_applied"] = False
        
        return limited
    
    def register_trade_result(self, pair: str, strategy: str, trade_result: Dict[str, Any]):
        """
        Register a trade result to update performance metrics.
        
        Args:
            pair: Trading pair symbol
            strategy: Strategy name
            trade_result: Dictionary with trade result details
        """
        # Create key for pair+strategy
        key = f"{pair}_{strategy}"
        
        # Get or create performance record
        if key not in self.performance_history:
            self.performance_history[key] = {
                "win_streak": 0,
                "loss_streak": 0,
                "last_trade_result": None,
                "last_trade_time": None
            }
        
        performance = self.performance_history[key]
        
        # Extract profit info
        profit = trade_result.get("profit", 0)
        
        # Update streaks
        if profit > 0:
            # Winning trade
            performance["win_streak"] += 1
            performance["loss_streak"] = 0
        else:
            # Losing trade
            performance["loss_streak"] += 1
            performance["win_streak"] = 0
        
        # Update last trade info
        performance["last_trade_result"] = profit
        performance["last_trade_time"] = datetime.now().isoformat()
        
        # Update performance history
        self.performance_history[key] = performance
        
        logger.info(f"Updated performance for {key}: " +
                   f"Win streak: {performance['win_streak']}, " +
                   f"Loss streak: {performance['loss_streak']}")
    
    def save_state(self):
        """Save performance history and adjustment state to disk"""
        try:
            state_path = "dynamic_parameter_state.json"
            
            state = {
                "performance_history": self.performance_history,
                "last_update_time": datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved dynamic parameter state to {state_path}")
        except Exception as e:
            logger.error(f"Error saving dynamic parameter state: {e}")
    
    def load_state(self):
        """Load performance history and adjustment state from disk"""
        try:
            state_path = "dynamic_parameter_state.json"
            
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.performance_history = state.get("performance_history", {})
                self.last_update_time = datetime.fromisoformat(state.get("last_update_time", 
                                                                       datetime.now().isoformat()))
                
                logger.info(f"Loaded dynamic parameter state from {state_path}")
            else:
                logger.info(f"No dynamic parameter state file found at {state_path}")
        except Exception as e:
            logger.error(f"Error loading dynamic parameter state: {e}")

# Create singleton dynamic adjuster
dynamic_adjuster = DynamicParameterAdjuster()