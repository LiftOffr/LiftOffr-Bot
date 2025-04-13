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
- Dynamically adjust margin percentages and leverage based on model confidence
- Allow ML-guided limit order price adjustments based on predicted price movements
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
    
    The class also provides functionality for adjusting margin percentages,
    leverage levels, and limit order prices based on ML predictions.
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

def calculate_dynamic_margin_percent(ml_confidence, signal_strength, market_volatility=None, base_margin=0.2):
    """
    Calculate an optimal margin percentage based on ML model confidence
    
    This function adjusts the margin percentage (amount of leverage) based on
    the confidence of ML predictions and other factors. Higher confidence leads
    to higher leverage use.
    
    Args:
        ml_confidence (float): ML model prediction confidence (0.0 to 1.0)
        signal_strength (float): Signal strength from strategy (0.0 to 1.0)
        market_volatility (float, optional): Market volatility metric
        base_margin (float): Base margin percentage (default: 0.2 or 20%)
        
    Returns:
        float: Dynamic margin percentage (0.0 to 1.0)
    """
    # Start with base margin
    margin_pct = base_margin
    
    # Only adjust margin if confidence is high enough
    if ml_confidence >= 0.75:
        # Scale from 0.75-1.0 confidence to 0.0-1.0 adjustment factor
        confidence_factor = (ml_confidence - 0.75) * 4.0
        
        # Higher confidence allows for up to 50% more margin
        confidence_adjustment = confidence_factor * 0.5 * base_margin
        
        # Signal strength provides an additional factor
        signal_adjustment = signal_strength * 0.3 * base_margin
        
        # Combine adjustments
        margin_pct += confidence_adjustment + signal_adjustment
        
        # If market volatility is provided, adjust margin inversely
        if market_volatility is not None and market_volatility > 0:
            # Reduce margin in more volatile markets
            volatility_factor = max(0.5, min(1.0, 1.0 / (market_volatility * 5.0)))
            margin_pct *= volatility_factor
    
    # Cap margin at reasonable bounds (10%-50%)
    margin_pct = max(0.1, min(0.5, margin_pct))
    
    return margin_pct

def calculate_dynamic_leverage(ml_confidence, market_regime, atr_value=None, base_leverage=3.0):
    """
    Calculate an optimal leverage level based on ML model confidence and market conditions
    
    This function adjusts the leverage based on prediction confidence and market regime.
    
    Args:
        ml_confidence (float): ML model prediction confidence (0.0 to 1.0)
        market_regime (str): Current market regime
        atr_value (float, optional): Current ATR value
        base_leverage (float): Base leverage level
        
    Returns:
        float: Dynamic leverage level
    """
    # Start with base leverage
    leverage = base_leverage
    
    # Adjust for ML confidence (minimum confidence of 0.75 required for any increase)
    if ml_confidence >= 0.75:
        confidence_adjustment = (ml_confidence - 0.75) * 4.0 * base_leverage
        leverage += confidence_adjustment
    
    # Adjust for market regime
    regime_factors = {
        'volatile_trending_up': 0.7,     # Reduce leverage in volatile markets
        'volatile_trending_down': 0.6,   # Reduce even more in volatile down markets
        'normal_trending_up': 1.2,       # Increase in normal uptrends
        'normal_trending_down': 0.8,     # Slightly reduce in downtrends
        'neutral': 1.0                  # No change in neutral markets
    }
    
    regime_factor = regime_factors.get(market_regime, 1.0)
    leverage *= regime_factor
    
    # If ATR is provided, adjust leverage inversely with volatility
    if atr_value is not None and atr_value > 0:
        # Base ATR value for typical SOL/USD market
        base_atr = 0.2
        if atr_value > base_atr:
            # Higher ATR = reduce leverage
            atr_factor = max(0.6, base_atr / atr_value)
            leverage *= atr_factor
    
    # Cap leverage at reasonable bounds (1x-10x)
    leverage = max(1.0, min(10.0, round(leverage, 1)))
    
    return leverage

def adjust_limit_order_price(base_price, direction, ml_confidence, price_prediction=None, atr_value=None):
    """
    Adjust limit order prices based on ML predictions
    
    This function can modify limit order prices to optimize execution probability
    based on ML predictions of price movements.
    
    Args:
        base_price (float): Base price for the limit order
        direction (int): Order direction (1 for buy, -1 for sell)
        ml_confidence (float): ML model prediction confidence
        price_prediction (float, optional): ML predicted price target
        atr_value (float, optional): Current ATR value
        
    Returns:
        float: Adjusted limit order price
    """
    # If no ML prediction or low confidence, return base price
    if price_prediction is None or ml_confidence < 0.8:
        return base_price
    
    # Calculate basic price adjustment
    # Higher confidence = more aggressive adjustment toward the predicted price
    confidence_factor = (ml_confidence - 0.8) * 5.0  # Scale 0.8-1.0 to 0-1.0
    
    # Determine how much to adjust based on ATR
    # Default to 0.1% of price if ATR not available
    adjustment_basis = atr_value if atr_value is not None else (base_price * 0.001)
    
    # Determine adjustment direction based on order direction and prediction
    price_diff = price_prediction - base_price
    
    # For buy orders: if prediction is higher, we can be more aggressive (lower price)
    # For sell orders: if prediction is lower, we can be more aggressive (higher price)
    if (direction == 1 and price_diff > 0) or (direction == -1 and price_diff < 0):
        # Predicted price movement favorable to our order
        # We can be more aggressive with our limit price
        max_adjustment = adjustment_basis * 0.5  # Max 50% of ATR
    else:
        # Predicted price movement unfavorable to our order
        # We need to be more conservative with our limit price
        max_adjustment = adjustment_basis * 0.25  # Max 25% of ATR
    
    # Calculate actual adjustment
    actual_adjustment = max_adjustment * confidence_factor * direction * -1
    
    # Apply adjustment to base price
    adjusted_price = base_price + actual_adjustment
    
    return adjusted_price

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
        
        # Also calculate dynamic margin and leverage
        margin_pct = calculate_dynamic_margin_percent(
            scenario['ml_confidence'],
            scenario['signal_strength'],
            scenario['market_volatility']
        )
        
        leverage = calculate_dynamic_leverage(
            scenario['ml_confidence'],
            scenario['market_regime'],
            scenario['atr_value']
        )
        
        print(f"Margin: {margin_pct:.2%} | Leverage: {leverage:.1f}x")
        
        # Test limit order price adjustment
        base_price = 100.0
        price_prediction = 102.0 if scenario['market_regime'].endswith('up') else 98.0
        
        buy_price = adjust_limit_order_price(
            base_price, 
            1, 
            scenario['ml_confidence'], 
            price_prediction,
            scenario['atr_value']
        )
        
        sell_price = adjust_limit_order_price(
            base_price, 
            -1, 
            scenario['ml_confidence'], 
            price_prediction,
            scenario['atr_value']
        )
        
        print(f"Original price: ${base_price:.2f} | ML adjusted buy: ${buy_price:.2f} | ML adjusted sell: ${sell_price:.2f}")
        print(f"Reasoning: {json.dumps(reasoning, indent=2)}")