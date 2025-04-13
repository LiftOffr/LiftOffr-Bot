"""
Dynamic Position Sizing Module for ML-Enhanced Trading Bot

This module implements adaptive position sizing based on ML model confidence,
signal strength, market volatility, and trailing stop parameters.

Features:
- Scale position size based on ML ensemble prediction confidence
- Adjust size based on signal strength from different strategy components
- Consider market volatility for risk-aware position sizing
- Use trailing stop parameters to optimize risk-reward positioning
- Implement dynamic maximum exposure limits based on market conditions
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class DynamicPositionSizer:
    """
    Dynamically calculate optimal position size based on multiple factors
    including ML confidence, signal strength, and market conditions.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the position sizer with configuration parameters
        
        Args:
            config_file (str, optional): Path to JSON configuration file
        """
        # Default configuration 
        self.config = {
            'base_position_size': 0.35,  # Base position size (35% of available capital)
            'min_position_size': 0.15,   # Minimum position size (15% of available capital)
            'max_position_size': 0.50,   # Maximum position size (50% of available capital)
            'confidence_weight': 0.40,   # Weight given to ML confidence
            'signal_weight': 0.30,       # Weight given to signal strength
            'volatility_weight': 0.20,   # Weight given to volatility
            'trend_weight': 0.10,        # Weight given to trend strength
            'max_portfolio_exposure': 0.80,  # Maximum portfolio exposure (80%)
            'volatility_scaling_factor': 1.5, # Volatility scaling factor
            'regime_allocation': {       # Allocation adjustments for different market regimes
                'volatile_trending_up': 1.2,
                'volatile_trending_down': 0.8,
                'normal_trending_up': 1.1,
                'normal_trending_down': 0.9,
                'neutral': 1.0
            }
        }
        
        # Override defaults with configuration file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                logger.info(f"Loaded position sizing configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Performance tracking
        self.performance_history = []
        
    def calculate_position_size(self, 
                               available_capital,
                               ml_confidence, 
                               signal_strength, 
                               market_volatility, 
                               market_regime,
                               atr_value=None,
                               current_exposure=0.0,
                               trailing_stop_distance=None):
        """
        Calculate optimal position size based on multiple factors
        
        Args:
            available_capital (float): Available capital for trading
            ml_confidence (float): ML model prediction confidence (0.0 to 1.0)
            signal_strength (float): Signal strength from strategy (0.0 to 1.0)
            market_volatility (float): Current market volatility metric
            market_regime (str): Current market regime
            atr_value (float, optional): Current ATR value
            current_exposure (float): Current portfolio exposure (0.0 to 1.0)
            trailing_stop_distance (float, optional): Trailing stop distance
            
        Returns:
            tuple: (position_size_pct, position_size_value, reasoning)
        """
        # Start with base position size
        base_size = self.config['base_position_size']
        
        # Adjust for ML confidence (higher confidence = larger position)
        # Scale from 0.7-1.0 confidence to 0.0-1.0 adjustment factor
        confidence_factor = max(0, min(1, (ml_confidence - 0.7) * 3.33))
        confidence_adjustment = confidence_factor * self.config['confidence_weight']
        
        # Adjust for signal strength (stronger signal = larger position)
        signal_adjustment = signal_strength * self.config['signal_weight']
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_factor = 1.0
        if market_volatility > 0:
            # Normalize volatility - lower factor for higher volatility
            volatility_factor = max(0.2, min(1.0, 
                                            1.0 / (market_volatility * self.config['volatility_scaling_factor'])))
        volatility_adjustment = volatility_factor * self.config['volatility_weight']
        
        # Adjust for market regime
        regime_factor = self.config['regime_allocation'].get(market_regime, 1.0)
        
        # Calculate position size percentage with all adjustments
        position_size_pct = base_size * (1.0 + confidence_adjustment + signal_adjustment + volatility_adjustment)
        position_size_pct *= regime_factor
        
        # Further adjust based on trailing stop distance if provided
        # Tighter stops = slightly smaller positions
        if trailing_stop_distance and trailing_stop_distance > 0:
            # Normalize trailing stop (smaller stops = smaller size adjustment)
            trailing_stop_factor = max(0.8, min(1.2, 0.01 / trailing_stop_distance))
            position_size_pct *= trailing_stop_factor
        
        # Apply ATR-based adjustment if provided
        # Higher ATR = slightly smaller position size
        if atr_value and atr_value > 0:
            # Base ATR value for typical SOL/USD market
            base_atr = 0.2
            atr_factor = min(1.2, max(0.8, base_atr / atr_value))
            position_size_pct *= atr_factor
        
        # Ensure position size is within configured limits
        position_size_pct = max(self.config['min_position_size'], 
                              min(self.config['max_position_size'], position_size_pct))
        
        # Check portfolio exposure constraints
        remaining_exposure = max(0, self.config['max_portfolio_exposure'] - current_exposure)
        position_size_pct = min(position_size_pct, remaining_exposure)
        
        # Calculate absolute position size
        position_size_value = available_capital * position_size_pct
        
        # Generate reasoning
        reasoning = {
            'base_size': base_size,
            'ml_confidence_adjustment': confidence_adjustment,
            'signal_strength_adjustment': signal_adjustment,
            'volatility_adjustment': volatility_adjustment,
            'regime_factor': regime_factor,
            'final_position_size_pct': position_size_pct,
            'position_value': position_size_value,
            'available_capital': available_capital
        }
        
        if trailing_stop_distance:
            reasoning['trailing_stop_factor'] = trailing_stop_factor
        
        if atr_value:
            reasoning['atr_factor'] = atr_factor
        
        logger.info(f"Position size calculated: {position_size_pct:.2%} ({position_size_value:.2f})")
        
        return position_size_pct, position_size_value, reasoning
    
    def track_performance(self, timestamp, position_size, trade_result, parameters):
        """
        Track performance of position sizing decisions
        
        Args:
            timestamp (datetime): Trade timestamp
            position_size (float): Position size used
            trade_result (float): Profit/loss from trade
            parameters (dict): Parameters used for position sizing
        """
        performance_entry = {
            'timestamp': timestamp,
            'position_size': position_size,
            'trade_result': trade_result,
            'risk_adjusted_return': trade_result / position_size if position_size > 0 else 0,
            'parameters': parameters
        }
        
        self.performance_history.append(performance_entry)
        
        # Periodically save performance history
        if len(self.performance_history) % 10 == 0:
            self._save_performance_history()
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        try:
            df = pd.DataFrame(self.performance_history)
            os.makedirs('optimization_results', exist_ok=True)
            filename = f"optimization_results/position_sizing_performance_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Position sizing performance history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def optimize_configuration(self):
        """
        Analyze performance history and optimize configuration parameters
        
        Returns:
            dict: Optimized configuration
        """
        if len(self.performance_history) < 20:
            logger.info("Not enough performance data for optimization")
            return self.config
            
        try:
            # Convert performance history to DataFrame
            df = pd.DataFrame(self.performance_history)
            
            # Calculate risk-adjusted return
            df['risk_adjusted_return'] = df.apply(
                lambda row: row['trade_result'] / row['position_size'] 
                            if row['position_size'] > 0 else 0, axis=1)
            
            # Find optimal configuration based on performance
            # Simple implementation: group by parameter buckets and find best results
            # In a real implementation, this would use more sophisticated optimization techniques
            
            # Optimize base position size
            base_sizes = np.linspace(0.15, 0.50, 8)
            best_base_size = self._find_best_parameter(df, 'parameters.base_size', base_sizes)
            
            # Optimize weights
            confidence_weights = np.linspace(0.2, 0.6, 5)
            best_confidence_weight = self._find_best_parameter(df, 'parameters.ml_confidence_adjustment', confidence_weights)
            
            signal_weights = np.linspace(0.1, 0.5, 5)
            best_signal_weight = self._find_best_parameter(df, 'parameters.signal_strength_adjustment', signal_weights)
            
            # Update configuration
            optimized_config = self.config.copy()
            optimized_config['base_position_size'] = best_base_size
            optimized_config['confidence_weight'] = best_confidence_weight
            optimized_config['signal_weight'] = best_signal_weight
            
            logger.info("Position sizing configuration optimized")
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error optimizing configuration: {e}")
            return self.config
            
    def _find_best_parameter(self, df, param_name, param_values):
        """
        Find best parameter value based on performance
        
        Args:
            df (pd.DataFrame): Performance data
            param_name (str): Parameter name
            param_values (list): List of parameter values to test
            
        Returns:
            float: Best parameter value
        """
        best_value = param_values[0]
        best_return = -float('inf')
        
        for value in param_values:
            # Find trades with parameter value close to test value
            matching_trades = df[
                (df[param_name] >= value * 0.9) & 
                (df[param_name] <= value * 1.1)
            ]
            
            if len(matching_trades) >= 5:  # Need minimum sample size
                avg_return = matching_trades['risk_adjusted_return'].mean()
                if avg_return > best_return:
                    best_return = avg_return
                    best_value = value
        
        return best_value
    
    def save_configuration(self, filename='config/position_sizing_config.json'):
        """
        Save current configuration to file
        
        Args:
            filename (str): Output filename
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Position sizing configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

# Sample usage
if __name__ == "__main__":
    # Example usage
    position_sizer = DynamicPositionSizer()
    
    # Simulate different market conditions
    test_conditions = [
        {
            'scenario': 'High confidence in bullish market',
            'available_capital': 10000,
            'ml_confidence': 0.95,
            'signal_strength': 0.85,
            'market_volatility': 0.01,
            'market_regime': 'normal_trending_up',
            'atr_value': 0.18,
            'current_exposure': 0.2,
            'trailing_stop_distance': 0.01
        },
        {
            'scenario': 'Moderate confidence in volatile market',
            'available_capital': 10000,
            'ml_confidence': 0.82,
            'signal_strength': 0.6,
            'market_volatility': 0.03,
            'market_regime': 'volatile_trending_up',
            'atr_value': 0.28,
            'current_exposure': 0.35,
            'trailing_stop_distance': 0.015
        },
        {
            'scenario': 'Low confidence in bearish market',
            'available_capital': 10000,
            'ml_confidence': 0.75,
            'signal_strength': 0.4,
            'market_volatility': 0.02,
            'market_regime': 'normal_trending_down',
            'atr_value': 0.22,
            'current_exposure': 0.5,
            'trailing_stop_distance': 0.0085
        }
    ]
    
    # Test each scenario
    for scenario in test_conditions:
        print(f"\nTesting: {scenario['scenario']}")
        pct, value, reasoning = position_sizer.calculate_position_size(
            scenario['available_capital'],
            scenario['ml_confidence'],
            scenario['signal_strength'],
            scenario['market_volatility'],
            scenario['market_regime'],
            scenario['atr_value'],
            scenario['current_exposure'],
            scenario['trailing_stop_distance']
        )
        print(f"Position Size: {pct:.2%} (${value:.2f})")
        print(f"Reasoning: {json.dumps(reasoning, indent=2)}")