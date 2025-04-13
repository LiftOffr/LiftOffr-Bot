"""
Dynamic Position Sizing with ML Confidence Integration

This module implements adaptive position sizing based on ML prediction confidence
and trailing stop parameters. Position sizes scale with prediction confidence and 
are inversely related to volatility measures.

Features:
- Confidence-based position sizing (higher confidence = larger position)
- Volatility-adjusted positions (higher volatility = smaller position)
- Trailing stop position adjustment (tighter stops = larger position)
- Market regime-adaptive sizing (trending = larger, volatile = smaller)
- Historical performance-based scaling
- Advanced position sizing with ATR-based risk management
- Enhanced ensemble confidence integration
"""

import logging
import numpy as np
from typing import Dict, Union, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class MLConfidencePositionSizer:
    """
    Position sizer that dynamically adjusts trade size based on ML prediction confidence
    and market conditions.
    """
    
    def __init__(
        self,
        base_position_pct: float = 0.20,
        max_position_pct: float = 0.35,
        min_position_pct: float = 0.10,
        min_confidence: float = 0.55,
        max_confidence: float = 0.95,
        volatility_scaling: bool = True,
        trailing_stop_scaling: bool = True
    ):
        """
        Initialize the ML confidence-based position sizer
        
        Args:
            base_position_pct: Base position size as percentage of capital
            max_position_pct: Maximum position size allowed
            min_position_pct: Minimum position size for valid signals
            min_confidence: Minimum ML confidence required for base position size
            max_confidence: ML confidence level that would warrant maximum position
            volatility_scaling: Whether to scale position size inversely with volatility
            trailing_stop_scaling: Whether to adjust position based on trailing stop distance
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.volatility_scaling = volatility_scaling
        self.trailing_stop_scaling = trailing_stop_scaling
        
        logger.info(f"Initialized ML confidence-based position sizer: "
                   f"base={base_position_pct:.1%}, max={max_position_pct:.1%}, "
                   f"min_confidence={min_confidence:.2f}")
    
    def calculate_position_size(
        self,
        capital: float,
        ml_confidence: float,
        atr: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        market_volatility: Optional[float] = None,
        max_drawdown_target: float = 0.02  # Target max drawdown per trade
    ) -> Tuple[float, Dict]:
        """
        Calculate position size based on ML confidence and market conditions
        
        Args:
            capital: Available capital for trading
            ml_confidence: Confidence score from ML models (0.0-1.0)
            atr: Average True Range (volatility measure)
            trailing_stop_pct: Trailing stop percentage
            market_volatility: General market volatility measure (normalized 0-1)
            max_drawdown_target: Maximum drawdown target per trade
            
        Returns:
            tuple: (position_size_usd, calculation_details)
        """
        # Start with base calculation from confidence
        confidence_factor = self._calculate_confidence_factor(ml_confidence)
        position_pct = self.base_position_pct * confidence_factor
        
        # Apply volatility scaling if enabled and data available
        volatility_factor = 1.0
        if self.volatility_scaling and market_volatility is not None:
            volatility_factor = self._calculate_volatility_factor(market_volatility)
            position_pct *= volatility_factor
        
        # Apply trailing stop scaling if enabled and data available
        trailing_stop_factor = 1.0
        if self.trailing_stop_scaling and trailing_stop_pct is not None:
            trailing_stop_factor = self._calculate_trailing_stop_factor(trailing_stop_pct)
            position_pct *= trailing_stop_factor
        
        # Apply ATR-based position sizing if available
        atr_factor = 1.0
        atr_size = None
        if atr is not None:
            atr_factor, atr_size = self._calculate_atr_position(
                capital, atr, max_drawdown_target
            )
        
        # Calculate final position size
        position_pct = min(position_pct, self.max_position_pct)
        position_pct = max(position_pct, self.min_position_pct)
        
        # If ATR-based sizing suggests a smaller position, use it
        position_size = capital * position_pct
        if atr_size is not None:
            position_size = min(position_size, atr_size)
        
        # Compile calculation details
        details = {
            "position_pct": position_pct,
            "position_size": position_size,
            "confidence_factor": confidence_factor,
            "volatility_factor": volatility_factor,
            "trailing_stop_factor": trailing_stop_factor,
            "atr_factor": atr_factor,
            "atr_position_size": atr_size
        }
        
        logger.debug(f"Position size calculation: {position_size:.2f} USD ({position_pct:.1%} of capital)")
        return position_size, details
    
    def _calculate_confidence_factor(self, ml_confidence: float) -> float:
        """
        Calculate a scaling factor based on ML prediction confidence
        
        Args:
            ml_confidence: Confidence score from ML models (0.0-1.0)
            
        Returns:
            float: Confidence scaling factor
        """
        # No position if below minimum confidence
        if ml_confidence < self.min_confidence:
            return 0.0
        
        # Linear scaling between min and max confidence
        if ml_confidence >= self.max_confidence:
            return 2.0  # Double the base position at maximum confidence
        
        # Scale between 1.0 and 2.0 based on confidence
        confidence_range = self.max_confidence - self.min_confidence
        confidence_factor = 1.0 + (ml_confidence - self.min_confidence) / confidence_range
        
        return confidence_factor
    
    def _calculate_volatility_factor(self, market_volatility: float) -> float:
        """
        Calculate a scaling factor based on market volatility
        
        Args:
            market_volatility: Market volatility measure (normalized 0-1)
            
        Returns:
            float: Volatility scaling factor (lower when volatility is high)
        """
        # Scale inversely with volatility - reduce size in high volatility
        # 1.0 at 0% volatility, 0.5 at 100% volatility
        return 1.0 - (0.5 * market_volatility)
    
    def _calculate_trailing_stop_factor(self, trailing_stop_pct: float) -> float:
        """
        Calculate a scaling factor based on trailing stop percentage
        
        Args:
            trailing_stop_pct: Trailing stop as percentage
            
        Returns:
            float: Trailing stop scaling factor
        """
        # For tighter trailing stops, increase position size
        # For wider trailing stops, decrease position size
        # Base assumption: 1% trailing stop is "standard"
        standard_stop = 0.01  # 1%
        
        if trailing_stop_pct <= 0.0:
            return 1.0
        
        # Inverse relationship - tighter stops allow larger positions
        return min(1.5, standard_stop / trailing_stop_pct)
    
    def _calculate_atr_position(
        self, capital: float, atr: float, max_drawdown_target: float
    ) -> Tuple[float, float]:
        """
        Calculate position size based on ATR to limit risk per trade
        
        Args:
            capital: Available capital
            atr: Average True Range
            max_drawdown_target: Maximum acceptable drawdown per trade
            
        Returns:
            tuple: (atr_factor, position_size)
        """
        if atr <= 0:
            return 1.0, None
        
        # Calculate maximum position size based on risk tolerance
        risk_amount = capital * max_drawdown_target  # Maximum dollar risk
        atr_multiples = 1.5  # Allow for price to move 1.5 x ATR against position
        max_position = risk_amount / (atr * atr_multiples)
        
        # Calculate the factor as a proportion of base position
        base_position = capital * self.base_position_pct
        atr_factor = max_position / base_position if base_position > 0 else 1.0
        
        return atr_factor, max_position


class AdaptivePositionSizer:
    """
    Advanced position sizer that combines multiple factors including:
    - Market regime (trending, ranging, volatile)
    - Signal strength from multiple sources
    - Historical win rate in current conditions
    - Risk-adjusted position sizing
    """
    
    def __init__(
        self,
        base_position_pct: float = 0.20,
        max_position_pct: float = 0.35,
        model_performance_window: int = 50,  # Use last 50 trades for performance
        enable_regime_adaptation: bool = True,
        enable_performance_scaling: bool = True
    ):
        """
        Initialize the adaptive position sizer
        
        Args:
            base_position_pct: Base position size as percentage of capital
            max_position_pct: Maximum position size allowed
            model_performance_window: Number of past trades to consider for performance
            enable_regime_adaptation: Whether to adapt to market regimes
            enable_performance_scaling: Whether to scale based on model performance
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.model_performance_window = model_performance_window
        self.enable_regime_adaptation = enable_regime_adaptation
        self.enable_performance_scaling = enable_performance_scaling
        
        # Regime-specific position sizing
        self.regime_position_pct = {
            'normal_trending_up': base_position_pct * 1.2,     # More confident in trends
            'normal_trending_down': base_position_pct * 1.0,
            'volatile_trending_up': base_position_pct * 0.8,   # More cautious in volatility
            'volatile_trending_down': base_position_pct * 0.7,
            'ranging': base_position_pct * 0.6                 # Smallest in ranges
        }
        
        # Historical performance by model and regime
        self.model_performance = {}
        
        logger.info(f"Initialized adaptive position sizer: base={base_position_pct:.1%}, "
                   f"max={max_position_pct:.1%}")
    
    def calculate_position_size(
        self,
        capital: float,
        ml_predictions: Dict[str, Dict],
        current_regime: str,
        atr: Optional[float] = None,
        risk_params: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate position size based on multiple ML models, their confidence, 
        historical performance, and market regime
        
        Args:
            capital: Available capital for trading
            ml_predictions: Dictionary with model predictions and confidence
            current_regime: Current market regime
            atr: Average True Range (volatility measure)
            risk_params: Additional risk parameters
            
        Returns:
            tuple: (position_size_usd, calculation_details)
        """
        # Start with regime-based position sizing
        if current_regime in self.regime_position_pct and self.enable_regime_adaptation:
            position_pct = self.regime_position_pct[current_regime]
        else:
            position_pct = self.base_position_pct
        
        # Factor in ML model confidences and historical performance
        avg_confidence = self._calculate_weighted_confidence(ml_predictions)
        
        # Apply confidence scaling - scale from 60% to 150% of base
        confidence_scaling = 0.6 + (avg_confidence * 0.9)
        position_pct *= confidence_scaling
        
        # Apply model performance scaling if enabled
        performance_scaling = 1.0
        if self.enable_performance_scaling:
            performance_scaling = self._calculate_performance_factor(
                ml_predictions, current_regime
            )
            position_pct *= performance_scaling
        
        # Apply ATR-based risk control
        risk_adjusted_position = capital * position_pct
        atr_position = None
        
        if atr is not None and risk_params is not None:
            max_drawdown_target = risk_params.get('max_drawdown_target', 0.02)
            atr_multiple = risk_params.get('atr_multiple', 1.5)
            
            if atr > 0:
                # Calculate position size based on ATR risk control
                risk_amount = capital * max_drawdown_target
                atr_position = risk_amount / (atr * atr_multiple)
        
        # Use the more conservative of the two approaches
        if atr_position is not None:
            position_size = min(risk_adjusted_position, atr_position)
        else:
            position_size = risk_adjusted_position
        
        # Ensure we don't exceed maximum position size
        position_size = min(position_size, capital * self.max_position_pct)
        final_position_pct = position_size / capital if capital > 0 else 0
        
        # Prepare calculation details for logging and analysis
        details = {
            "position_pct": final_position_pct,
            "position_size": position_size,
            "regime": current_regime,
            "avg_confidence": avg_confidence,
            "confidence_scaling": confidence_scaling,
            "performance_scaling": performance_scaling,
            "atr_position": atr_position
        }
        
        logger.debug(f"Adaptive position size: {position_size:.2f} USD "
                    f"({final_position_pct:.1%} of capital) in {current_regime} regime")
        
        return position_size, details
    
    def _calculate_weighted_confidence(self, ml_predictions: Dict[str, Dict]) -> float:
        """
        Calculate weighted average confidence from multiple models
        
        Args:
            ml_predictions: Dictionary with model predictions and confidence
            
        Returns:
            float: Weighted average confidence
        """
        if not ml_predictions:
            return 0.5  # Default to neutral confidence
        
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for model_name, prediction in ml_predictions.items():
            if 'confidence' in prediction and 'weight' in prediction:
                confidence = prediction['confidence']
                weight = prediction['weight']
                
                weighted_confidence += confidence * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return 0.5
    
    def _calculate_performance_factor(
        self, ml_predictions: Dict[str, Dict], current_regime: str
    ) -> float:
        """
        Calculate performance scaling factor based on historical model performance
        in the current market regime
        
        Args:
            ml_predictions: Dictionary with model predictions
            current_regime: Current market regime
            
        Returns:
            float: Performance scaling factor
        """
        if not self.model_performance:
            return 1.0
        
        # Get weights for each model
        model_weights = {
            model_name: pred.get('weight', 1.0) 
            for model_name, pred in ml_predictions.items()
        }
        
        # Calculate weighted performance factor
        total_weight = sum(model_weights.values())
        if total_weight == 0:
            return 1.0
        
        weighted_performance = 0.0
        for model_name, weight in model_weights.items():
            # Get historical performance for this model and regime
            if model_name in self.model_performance:
                model_perf = self.model_performance[model_name]
                regime_win_rate = model_perf.get(current_regime, 0.5)  # Default 50% win rate
                
                # Scale between 0.8 (at 50% win rate) to 1.5 (at 100% win rate)
                model_factor = 0.8 + ((regime_win_rate - 0.5) * 1.4)
                weighted_performance += model_factor * weight
        
        performance_factor = weighted_performance / total_weight
        
        # Constrain between 0.7 and 1.5
        return max(0.7, min(1.5, performance_factor))
    
    def update_model_performance(
        self, model_name: str, regime: str, trade_result: float, is_win: bool
    ) -> None:
        """
        Update the historical performance records for a model in a specific regime
        
        Args:
            model_name: Name of the ML model
            regime: Market regime during the trade
            trade_result: P&L result of the trade
            is_win: Whether the trade was profitable
        """
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'normal_trending_up': 0.5,
                'normal_trending_down': 0.5,
                'volatile_trending_up': 0.5,
                'volatile_trending_down': 0.5,
                'ranging': 0.5,
                'trades': []
            }
        
        # Add this trade to the performance history
        model_perf = self.model_performance[model_name]
        model_perf['trades'].append({
            'regime': regime,
            'result': trade_result,
            'win': is_win,
            'timestamp': pd.Timestamp.now()
        })
        
        # Limit history to the specified window
        if len(model_perf['trades']) > self.model_performance_window:
            model_perf['trades'] = model_perf['trades'][-self.model_performance_window:]
        
        # Recalculate win rates for each regime
        regime_trades = {}
        for trade in model_perf['trades']:
            trade_regime = trade['regime']
            if trade_regime not in regime_trades:
                regime_trades[trade_regime] = []
            regime_trades[trade_regime].append(trade)
        
        for r, trades in regime_trades.items():
            win_count = sum(1 for t in trades if t['win'])
            win_rate = win_count / len(trades) if trades else 0.5
            model_perf[r] = win_rate
        
        logger.debug(f"Updated performance for {model_name} in {regime} regime: {model_perf[regime]:.2f}")


class EnhancedEnsemblePositionSizer:
    """
    Advanced position sizer that integrates with the DynamicWeightedEnsemble model
    to leverage enhanced confidence calculations for more effective position sizing.
    
    This position sizer utilizes the sophisticated confidence calculation methods from
    the ensemble model to better adjust position sizes based on market conditions,
    prediction confidence, and historical performance.
    """
    
    def __init__(
        self,
        base_position_pct: float = 0.20,
        max_position_pct: float = 0.35,
        min_position_pct: float = 0.10,
        min_confidence: float = 0.60,
        confidence_scaling_factor: float = 1.5,
        apply_atr_limits: bool = True,
        apply_volatility_scaling: bool = True
    ):
        """
        Initialize the enhanced ensemble position sizer
        
        Args:
            base_position_pct: Base position size as percentage of capital
            max_position_pct: Maximum position size allowed as percentage of capital
            min_position_pct: Minimum position size for valid signals
            min_confidence: Minimum confidence required to take a position
            confidence_scaling_factor: How much to scale positions based on confidence
            apply_atr_limits: Whether to apply ATR-based risk limits
            apply_volatility_scaling: Whether to scale positions based on volatility
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.min_confidence = min_confidence
        self.confidence_scaling_factor = confidence_scaling_factor
        self.apply_atr_limits = apply_atr_limits
        self.apply_volatility_scaling = apply_volatility_scaling
        
        logger.info(f"Initialized Enhanced Ensemble Position Sizer: "
                   f"base={base_position_pct:.1%}, max={max_position_pct:.1%}, "
                   f"min_confidence={min_confidence:.2f}")
    
    def calculate_position_size(
        self,
        capital: float,
        ensemble_model,
        market_data: pd.DataFrame,
        prediction: float,
        confidence: float,
        direction: int,  # 1 for long, -1 for short
        atr: Optional[float] = None,
        current_price: Optional[float] = None,
        additional_factors: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate position size using the enhanced confidence calculation from the ensemble model
        
        Args:
            capital: Available capital for trading
            ensemble_model: Instance of DynamicWeightedEnsemble model
            market_data: Recent market data for additional analysis
            prediction: The ensemble prediction value
            confidence: The base confidence score from the model
            direction: Trade direction (1 for long, -1 for short)
            atr: Average True Range (volatility measure)
            current_price: Current price of the asset
            additional_factors: Additional factors to consider for sizing
            
        Returns:
            tuple: (position_size_usd, calculation_details)
        """
        # Use the advanced confidence calculation from ensemble model
        enhanced_confidence, sizing_details = ensemble_model.calculate_position_sizing_confidence(
            confidence, market_data, additional_factors
        )
        
        # Check if confidence meets minimum threshold
        if enhanced_confidence < self.min_confidence:
            logger.info(f"Confidence {enhanced_confidence:.2f} below minimum threshold {self.min_confidence:.2f}")
            return 0.0, {"reason": "confidence_below_threshold", "confidence": enhanced_confidence}
        
        # Calculate position size as percentage of capital based on confidence
        # Scale from base_position_pct to max_position_pct based on confidence
        confidence_range = 1.0 - self.min_confidence
        if confidence_range > 0:
            # Scale confidence factor from 1.0 to confidence_scaling_factor
            confidence_scale = 1.0 + (self.confidence_scaling_factor - 1.0) * (enhanced_confidence - self.min_confidence) / confidence_range
        else:
            confidence_scale = 1.0
            
        position_pct = self.base_position_pct * confidence_scale
        
        # Apply directional bias if specified in additional factors
        trend_bias = 1.0
        if additional_factors and 'trend_direction' in additional_factors:
            trend_direction = additional_factors['trend_direction']  # 1 for uptrend, -1 for downtrend, 0 for ranging
            
            # Increase position size if trading in direction of trend
            if trend_direction * direction > 0:
                trend_bias = 1.1  # 10% increase when trading with the trend
            # Decrease position size if trading against the trend
            elif trend_direction * direction < 0:
                trend_bias = 0.8  # 20% decrease when trading against the trend
                
        position_pct *= trend_bias
        
        # Apply volatility scaling if enabled and ATR is available
        volatility_factor = 1.0
        if self.apply_volatility_scaling and atr is not None and current_price is not None:
            # Calculate normalized volatility as ATR / Price
            normalized_volatility = atr / current_price if current_price > 0 else 0
            
            # Scale inversely with volatility (higher volatility = smaller position)
            # Typical ATR/Price ratios range from 0.005 (0.5%) to 0.03 (3%)
            # Scale from 1.0 to 0.6 as volatility increases
            volatility_factor = max(0.6, 1.0 - normalized_volatility * 15)
            position_pct *= volatility_factor
            
        # Apply ATR-based position sizing if enabled and ATR is available
        atr_position = None
        if self.apply_atr_limits and atr is not None:
            max_drawdown_target = additional_factors.get('max_drawdown_target', 0.02) if additional_factors else 0.02
            atr_multiple = additional_factors.get('atr_multiple', 1.5) if additional_factors else 1.5
            
            # Calculate position size based on ATR risk control
            # This limits the position size so that a move of atr_multiple * ATR
            # would result in a loss of max_drawdown_target * capital
            if atr > 0 and current_price is not None and current_price > 0:
                risk_amount = capital * max_drawdown_target
                atr_position = risk_amount / (atr * atr_multiple)
        
        # Calculate final position size
        position_pct = min(position_pct, self.max_position_pct)
        position_pct = max(position_pct, self.min_position_pct)
        
        position_size = capital * position_pct
        
        # Apply ATR-based limit if it's more conservative
        if atr_position is not None:
            position_size = min(position_size, atr_position)
        
        # Prepare calculation details
        details = {
            "position_pct": position_pct,
            "position_size": position_size,
            "enhanced_confidence": enhanced_confidence,
            "base_confidence": confidence,
            "confidence_scale": confidence_scale,
            "volatility_factor": volatility_factor,
            "trend_bias": trend_bias,
            "atr_position": atr_position,
            "direction": direction,
            "sizing_details": sizing_details
        }
        
        if additional_factors:
            details["additional_factors"] = additional_factors
        
        logger.info(f"Enhanced position sizing: {position_size:.2f} USD ({position_pct:.1%} of capital) "
                   f"with confidence {enhanced_confidence:.2f}")
        
        return position_size, details
    
    def calculate_trailing_stop(
        self,
        position_size: float,
        capital: float,
        confidence: float,
        atr: Optional[float] = None,
        base_trailing_stop_pct: float = 0.01,
        direction: int = 1  # 1 for long, -1 for short
    ) -> Tuple[float, Dict]:
        """
        Calculate trailing stop percentage based on position size and confidence
        
        Args:
            position_size: Current position size in USD
            capital: Total available capital
            confidence: ML model confidence (0.0-1.0)
            atr: Average True Range (volatility measure)
            base_trailing_stop_pct: Base trailing stop percentage
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            tuple: (trailing_stop_pct, calculation_details)
        """
        # Calculate position percentage
        position_pct = position_size / capital if capital > 0 else 0
        
        # Adjust trailing stop based on confidence
        # Higher confidence = tighter stops (smaller percentage)
        confidence_factor = max(0.7, min(1.3, 2.0 - confidence))
        trailing_stop_pct = base_trailing_stop_pct * confidence_factor
        
        # Adjust trailing stop based on ATR if available (for volatility-aware stops)
        atr_stop = None
        atr_factor = 1.0
        if atr is not None:
            # Use ATR-based trailing stop as a minimum sensible value
            # Typical is 2-3x ATR for a trailing stop
            atr_multiple = 2.0  
            current_price = position_size / (position_pct * capital / base_trailing_stop_pct) if position_pct > 0 else 0
            
            if current_price > 0:
                atr_stop = (atr * atr_multiple) / current_price
                atr_factor = atr_stop / base_trailing_stop_pct if base_trailing_stop_pct > 0 else 1.0
                
                # Use the larger of percentage-based or ATR-based stop
                trailing_stop_pct = max(trailing_stop_pct, atr_stop)
        
        # Adjust trailing stop based on position size relative to max allowed
        # Larger positions should have tighter stops
        position_size_factor = 1.0
        if position_pct > 0 and self.max_position_pct > 0:
            rel_position_pct = position_pct / self.max_position_pct
            # Scale from 1.2 (small position) to 0.8 (large position)
            position_size_factor = 1.2 - 0.4 * min(1.0, rel_position_pct)
            trailing_stop_pct *= position_size_factor
        
        # Prepare calculation details
        details = {
            "base_trailing_stop_pct": base_trailing_stop_pct,
            "confidence": confidence,
            "confidence_factor": confidence_factor,
            "position_pct": position_pct,
            "position_size_factor": position_size_factor,
            "atr_stop": atr_stop,
            "atr_factor": atr_factor,
            "trailing_stop_pct": trailing_stop_pct,
            "direction": direction
        }
        
        logger.debug(f"Calculated trailing stop: {trailing_stop_pct:.2%}")
        return trailing_stop_pct, details