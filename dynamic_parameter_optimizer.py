#!/usr/bin/env python3
"""
Dynamic Parameter Optimizer

This module allows for dynamic modification of trading parameters based on confidence levels
and market conditions to maximize returns for each trade. Parameters are optimized
for each cryptocurrency pair independently, but the system learns from cross-asset patterns.

Features:
- Dynamically adjusts risk percentage based on signal strength and model confidence
- Modifies leverage based on volatility and prediction confidence
- Adaptively tunes entry/exit thresholds for each cryptocurrency
- Maintains per-asset parameter sets optimized through backtesting
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameter boundaries
MIN_RISK_PERCENTAGE = 0.05  # 5% minimum risk per trade
MAX_RISK_PERCENTAGE = 0.40  # 40% maximum risk per trade
MIN_LEVERAGE = 5.0
MAX_LEVERAGE = 125.0
MIN_CONFIDENCE_THRESHOLD = 0.55
MAX_CONFIDENCE_THRESHOLD = 0.95
MIN_SIGNAL_STRENGTH = 0.45
MAX_SIGNAL_STRENGTH = 0.95

class DynamicParameterOptimizer:
    """
    Dynamically optimizes trading parameters based on confidence levels,
    market conditions, and historical performance.
    """
    
    def __init__(self, base_config_path: str = "config/ml_config.json"):
        """
        Initialize the dynamic parameter optimizer.
        
        Args:
            base_config_path: Path to the base ML configuration file
        """
        self.base_config_path = base_config_path
        self.optimized_params = {}
        self.performance_history = {}
        self.last_optimization_time = {}
        
        # Load base configuration
        self.load_base_config()
        
        # Load optimized parameters if available
        self.load_optimized_params()
        
        # Initialize optimization histories
        self.optimization_history = {}

    def load_base_config(self) -> None:
        """Load the base ML configuration"""
        try:
            with open(self.base_config_path, 'r') as f:
                self.base_config = json.load(f)
            logger.info(f"Loaded base configuration from {self.base_config_path}")
        except Exception as e:
            logger.error(f"Error loading base configuration: {e}")
            self.base_config = {
                "pairs": ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"],
                "base_leverage": 20.0,
                "max_leverage": 125.0,
                "risk_percentage": 0.20,
                "confidence_threshold": 0.65,
            }
            logger.warning("Using default configuration")

    def load_optimized_params(self) -> None:
        """Load previously optimized parameters if available"""
        optimized_params_path = "config/optimized_params.json"
        try:
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path, 'r') as f:
                    self.optimized_params = json.load(f)
                logger.info(f"Loaded optimized parameters from {optimized_params_path}")
            else:
                logger.info("No optimized parameters found, will create when optimization is performed")
        except Exception as e:
            logger.error(f"Error loading optimized parameters: {e}")
            self.optimized_params = {}

    def save_optimized_params(self) -> None:
        """Save optimized parameters to file"""
        optimized_params_path = "config/optimized_params.json"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(optimized_params_path), exist_ok=True)
            
            with open(optimized_params_path, 'w') as f:
                json.dump(self.optimized_params, f, indent=2)
            logger.info(f"Saved optimized parameters to {optimized_params_path}")
        except Exception as e:
            logger.error(f"Error saving optimized parameters: {e}")

    def get_pair_params(self, pair: str) -> Dict[str, Any]:
        """
        Get optimized parameters for a specific trading pair.
        If not available, use the base configuration.
        
        Args:
            pair: Trading pair symbol (e.g., "SOL/USD")
            
        Returns:
            Dictionary of optimized parameters for the pair
        """
        if pair in self.optimized_params:
            return self.optimized_params[pair]
        else:
            # Create default parameters based on base configuration
            default_params = {
                "risk_percentage": self.base_config.get("risk_percentage", 0.20),
                "base_leverage": self.base_config.get("base_leverage", 20.0),
                "max_leverage": self.base_config.get("max_leverage", 125.0),
                "confidence_threshold": self.base_config.get("confidence_threshold", 0.65),
                "signal_strength_threshold": self.base_config.get("min_signal_strength", 0.60),
                "trailing_stop_atr_multiplier": self.base_config.get("trailing_stop_atr_multiplier", 3.0),
                "exit_multiplier": self.base_config.get("exit_multiplier", 1.5),
                "drawdown_limit_percentage": self.base_config.get("drawdown_limit_percentage", 4.0),
                "strategy_weights": {
                    "arima": self.base_config.get("optimization", {}).get("strategy_weight", {}).get("arima", 0.3),
                    "adaptive": self.base_config.get("optimization", {}).get("strategy_weight", {}).get("adaptive", 0.7)
                }
            }
            self.optimized_params[pair] = default_params
            return default_params

    def optimize_parameters_for_pair(self, pair: str, backtest_results: Dict) -> Dict[str, Any]:
        """
        Optimize parameters for a specific pair based on backtest results.
        
        Args:
            pair: Trading pair symbol
            backtest_results: Results from backtesting with metrics
            
        Returns:
            Optimized parameters for the pair
        """
        logger.info(f"Optimizing parameters for {pair}")
        
        # Extract metrics from backtest results
        profit_factor = backtest_results.get("profit_factor", 1.0)
        win_rate = backtest_results.get("win_rate", 0.5)
        max_drawdown = backtest_results.get("max_drawdown", 0.2)
        avg_profit_per_trade = backtest_results.get("avg_profit_per_trade", 0.0)
        total_return = backtest_results.get("total_return", 0.0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
        
        # Get current parameters
        current_params = self.get_pair_params(pair)
        
        # Adaptive risk percentage based on win rate and profit factor
        risk_weight = (win_rate * 0.5) + (min(profit_factor, 3.0) / 3.0 * 0.5)
        new_risk_percentage = MIN_RISK_PERCENTAGE + risk_weight * (MAX_RISK_PERCENTAGE - MIN_RISK_PERCENTAGE)
        
        # Adaptive leverage based on volatility and max drawdown
        max_drawdown_factor = max(0.1, 1.0 - max_drawdown)
        volatility = backtest_results.get("volatility", 0.05)
        volatility_factor = max(0.1, 1.0 - min(volatility * 10, 0.9))
        leverage_factor = max_drawdown_factor * volatility_factor
        
        new_base_leverage = MIN_LEVERAGE + leverage_factor * (current_params["base_leverage"] - MIN_LEVERAGE)
        new_max_leverage = new_base_leverage + leverage_factor * (current_params["max_leverage"] - current_params["base_leverage"])
        
        # Clamp values to allowed ranges
        new_base_leverage = min(MAX_LEVERAGE, max(MIN_LEVERAGE, new_base_leverage))
        new_max_leverage = min(MAX_LEVERAGE, max(new_base_leverage, new_max_leverage))
        
        # Adaptive confidence threshold based on precision metrics
        precision = backtest_results.get("precision", 0.5)
        new_confidence_threshold = MIN_CONFIDENCE_THRESHOLD + precision * (MAX_CONFIDENCE_THRESHOLD - MIN_CONFIDENCE_THRESHOLD)
        
        # Adaptive signal strength threshold
        signal_quality = (win_rate + precision) / 2.0
        new_signal_strength = MIN_SIGNAL_STRENGTH + signal_quality * (MAX_SIGNAL_STRENGTH - MIN_SIGNAL_STRENGTH)
        
        # Adaptive trailing stop based on price action
        avg_win_loss_ratio = backtest_results.get("avg_win_loss_ratio", 1.0)
        new_trailing_stop = current_params["trailing_stop_atr_multiplier"]
        if avg_win_loss_ratio < 1.5:
            # Tighten stops if wins are not much bigger than losses
            new_trailing_stop *= 0.9
        elif avg_win_loss_ratio > 2.5:
            # Loosen stops to let winners run
            new_trailing_stop *= 1.1
        
        # Adaptive exit multiplier
        new_exit_multiplier = current_params["exit_multiplier"]
        if win_rate > 0.6:
            # For high win rate, be more aggressive with exits
            new_exit_multiplier *= 0.95
        elif win_rate < 0.4:
            # For low win rate, be more patient with exits
            new_exit_multiplier *= 1.05
        
        # Strategy weights
        arima_win_rate = backtest_results.get("arima_win_rate", 0.5)
        adaptive_win_rate = backtest_results.get("adaptive_win_rate", 0.5)
        total_win_rate = arima_win_rate + adaptive_win_rate
        
        if total_win_rate > 0:
            arima_weight = arima_win_rate / total_win_rate
            adaptive_weight = adaptive_win_rate / total_win_rate
        else:
            arima_weight = 0.5
            adaptive_weight = 0.5
        
        # Save the optimized parameters
        optimized_params = {
            "risk_percentage": new_risk_percentage,
            "base_leverage": new_base_leverage,
            "max_leverage": new_max_leverage,
            "confidence_threshold": new_confidence_threshold,
            "signal_strength_threshold": new_signal_strength,
            "trailing_stop_atr_multiplier": new_trailing_stop,
            "exit_multiplier": new_exit_multiplier,
            "drawdown_limit_percentage": current_params["drawdown_limit_percentage"],
            "strategy_weights": {
                "arima": arima_weight,
                "adaptive": adaptive_weight
            },
            "optimization_time": datetime.now().isoformat(),
            "optimization_metrics": {
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "avg_profit_per_trade": avg_profit_per_trade,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio
            }
        }
        
        self.optimized_params[pair] = optimized_params
        self.save_optimized_params()
        
        # Log the optimization results
        logger.info(f"Optimized parameters for {pair}:")
        logger.info(f"  Risk %: {optimized_params['risk_percentage']:.2f}")
        logger.info(f"  Leverage: {optimized_params['base_leverage']:.1f} - {optimized_params['max_leverage']:.1f}")
        logger.info(f"  Confidence Threshold: {optimized_params['confidence_threshold']:.2f}")
        logger.info(f"  Signal Strength: {optimized_params['signal_strength_threshold']:.2f}")
        logger.info(f"  Strategy Weights: ARIMA={optimized_params['strategy_weights']['arima']:.2f}, Adaptive={optimized_params['strategy_weights']['adaptive']:.2f}")
        
        return optimized_params

    def get_dynamic_parameters(self, pair: str, confidence: float, signal_strength: float, 
                               volatility: float = None, market_regime: str = None) -> Dict[str, float]:
        """
        Get dynamically adjusted parameters for a specific trade based on confidence
        and current market conditions.
        
        Args:
            pair: Trading pair symbol (e.g., "SOL/USD")
            confidence: Model prediction confidence (0.0-1.0)
            signal_strength: Signal strength from strategy (0.0-1.0)
            volatility: Current market volatility (optional)
            market_regime: Current market regime (trending, ranging, volatile) (optional)
            
        Returns:
            Dictionary of dynamically adjusted parameters for this specific trade
        """
        # Get the base optimized parameters for this pair
        base_params = self.get_pair_params(pair)
        
        # Adjust risk percentage based on confidence and signal strength
        confidence_factor = max(0.0, min(1.0, (confidence - base_params["confidence_threshold"]) / 
                                        (1.0 - base_params["confidence_threshold"])))
        signal_factor = max(0.0, min(1.0, (signal_strength - base_params["signal_strength_threshold"]) / 
                                    (1.0 - base_params["signal_strength_threshold"])))
        
        # Combined strength factor
        combined_factor = (confidence_factor * 0.6) + (signal_factor * 0.4)
        
        # Dynamically adjust risk percentage
        dynamic_risk = base_params["risk_percentage"] * (1.0 + combined_factor)
        dynamic_risk = max(MIN_RISK_PERCENTAGE, min(MAX_RISK_PERCENTAGE, dynamic_risk))
        
        # Dynamically adjust leverage based on confidence
        leverage_range = base_params["max_leverage"] - base_params["base_leverage"]
        dynamic_leverage = base_params["base_leverage"] + (leverage_range * combined_factor)
        
        # If volatility is provided, adjust leverage further
        if volatility is not None:
            # Reduce leverage in high volatility environments
            vol_factor = max(0.5, 1.0 - (volatility * 5.0))
            dynamic_leverage *= vol_factor
        
        # If market regime is provided, make further adjustments
        if market_regime == "volatile":
            # Reduce risk and leverage in volatile markets
            dynamic_risk *= 0.8
            dynamic_leverage *= 0.8
        elif market_regime == "trending":
            # Increase risk slightly in trending markets
            dynamic_risk *= 1.1
        
        # Clamp leverage to allowed range
        dynamic_leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, dynamic_leverage))
        
        # Dynamically adjust trailing stop
        if confidence > 0.8:
            # High confidence - give more room
            trailing_stop = base_params["trailing_stop_atr_multiplier"] * 1.2
        elif confidence < 0.65:
            # Low confidence - tighter stops
            trailing_stop = base_params["trailing_stop_atr_multiplier"] * 0.8
        else:
            trailing_stop = base_params["trailing_stop_atr_multiplier"]
        
        # Return the dynamically adjusted parameters
        return {
            "risk_percentage": dynamic_risk,
            "leverage": dynamic_leverage,
            "trailing_stop_atr_multiplier": trailing_stop,
            "exit_multiplier": base_params["exit_multiplier"],
            "confidence_factor": confidence_factor,
            "signal_factor": signal_factor,
            "combined_factor": combined_factor
        }

    def update_performance(self, pair: str, trade_result: Dict) -> None:
        """
        Update performance history for a pair after a trade.
        
        Args:
            pair: Trading pair symbol
            trade_result: Dictionary with trade results
        """
        if pair not in self.performance_history:
            self.performance_history[pair] = []
            
        self.performance_history[pair].append(trade_result)
        
        # If we have enough trades, evaluate if reoptimization is needed
        if len(self.performance_history[pair]) >= 10:
            self._check_for_reoptimization(pair)
            
    def _check_for_reoptimization(self, pair: str) -> bool:
        """
        Check if parameters should be reoptimized based on recent performance.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if reoptimization is needed, False otherwise
        """
        # Get the most recent trades
        recent_trades = self.performance_history[pair][-10:]
        profitable_trades = sum(1 for trade in recent_trades if trade.get('profit_loss', 0) > 0)
        win_rate = profitable_trades / len(recent_trades)
        
        # If win rate is below 40%, consider reoptimization
        if win_rate < 0.4:
            logger.info(f"Low win rate ({win_rate:.2f}) for {pair}, reoptimization recommended")
            return True
            
        # If we haven't optimized in a while, consider reoptimization
        last_opt_time = self.last_optimization_time.get(pair)
        if last_opt_time:
            time_since_opt = datetime.now() - datetime.fromisoformat(last_opt_time)
            if time_since_opt.days > 7:  # More than a week
                logger.info(f"More than a week since last optimization for {pair}, reoptimization recommended")
                return True
                
        return False
        
    def optimize_all_pairs(self, backtest_results: Dict[str, Dict]) -> None:
        """
        Optimize parameters for all trading pairs based on backtest results.
        
        Args:
            backtest_results: Dictionary mapping pair symbols to backtest results
        """
        for pair, results in backtest_results.items():
            self.optimize_parameters_for_pair(pair, results)
            self.last_optimization_time[pair] = datetime.now().isoformat()
            
        self.save_optimized_params()
        logger.info("Optimization complete for all pairs")


# Example usage
if __name__ == "__main__":
    # Example backtest results
    backtest_results = {
        "SOL/USD": {
            "profit_factor": 2.5,
            "win_rate": 0.75,
            "max_drawdown": 0.12,
            "avg_profit_per_trade": 0.034,
            "total_return": 0.87,
            "sharpe_ratio": 1.9,
            "precision": 0.78,
            "avg_win_loss_ratio": 2.1,
            "arima_win_rate": 0.72,
            "adaptive_win_rate": 0.78,
            "volatility": 0.04
        },
        "BTC/USD": {
            "profit_factor": 1.8,
            "win_rate": 0.65,
            "max_drawdown": 0.18,
            "avg_profit_per_trade": 0.021,
            "total_return": 0.56,
            "sharpe_ratio": 1.3,
            "precision": 0.67,
            "avg_win_loss_ratio": 1.6,
            "arima_win_rate": 0.63,
            "adaptive_win_rate": 0.66,
            "volatility": 0.03
        }
    }
    
    # Initialize the optimizer
    optimizer = DynamicParameterOptimizer()
    
    # Optimize parameters for all pairs
    optimizer.optimize_all_pairs(backtest_results)
    
    # Get dynamic parameters for a specific trade
    params = optimizer.get_dynamic_parameters("SOL/USD", confidence=0.85, signal_strength=0.78, volatility=0.04)
    print("Dynamic parameters for SOL/USD trade:")
    for key, value in params.items():
        print(f"  {key}: {value}")