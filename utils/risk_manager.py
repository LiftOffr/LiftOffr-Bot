#!/usr/bin/env python3
"""
Advanced Risk Management System

This module provides sophisticated risk management capabilities to prevent
liquidations and large losses while maximizing profits. It dynamically adjusts
position sizing, leverage, and stop-loss levels based on multiple factors:

1. Portfolio drawdown monitoring and prevention
2. Volatility-based position sizing with Kelly criterion
3. Dynamic stop-loss management with volatility-adjusted trailing stops
4. Multi-timeframe risk assessment to identify potential adverse price moves
5. Trade correlation detection for portfolio risk measurement
6. Profit-protecting strategy with ratcheting stops
"""

import os
import json
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Enumeration of risk levels for position sizing"""
    VERY_LOW = auto()    # 25% of standard risk
    LOW = auto()         # 50% of standard risk
    MODERATE = auto()    # 75% of standard risk
    STANDARD = auto()    # 100% of base risk percentage
    ELEVATED = auto()    # 110% of standard risk
    HIGH = auto()        # 125% of standard risk
    AGGRESSIVE = auto()  # 150% of standard risk, only for highest confidence trades

class RiskManager:
    """
    Advanced risk management system for protecting capital while maximizing returns.
    """
    
    def __init__(self, config_path: str = "config/risk_config.json"):
        """
        Initialize the risk manager.
        
        Args:
            config_path: Path to risk configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Initialize state variables
        self.current_positions = {}
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.historical_drawdowns = []
        self.trade_history = []
        self.volatility_metrics = {}
        self.correlation_matrix = None
        self.position_weight_used = 0.0
        
        # Risk metrics
        self.daily_risk_used = 0.0
        self.max_daily_risk = self.config.get("max_daily_risk_percentage", 5.0) / 100.0
        self.current_open_risk = 0.0
        
        # Ratcheting stop tracking
        self.profit_ratchets = {}
    
    def load_config(self):
        """Load risk management configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded risk configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = {
                    "max_drawdown_percentage": 15.0,  # Maximum allowed drawdown (15%)
                    "max_position_size_percentage": 25.0,  # Maximum single position size (25% of portfolio)
                    "max_leverage_mapping": {  # Maximum leverage based on volatility
                        "very_low": 125.0,      # <1% daily volatility
                        "low": 100.0,           # 1-2% daily volatility
                        "medium": 75.0,         # 2-3% daily volatility
                        "high": 50.0,           # 3-5% daily volatility
                        "very_high": 25.0,      # >5% daily volatility
                        "extreme": 10.0         # Extreme market conditions
                    },
                    "stop_loss_volatility_multipliers": {  # ATR multipliers for stop-loss
                        "very_low": 4.0,        # Low volatility = wider stops
                        "low": 3.5,
                        "medium": 3.0,
                        "high": 2.5,
                        "very_high": 2.0,       # High volatility = tighter stops
                        "extreme": 1.5          # Extreme volatility = very tight stops
                    },
                    "profit_ratchet_levels": [  # Levels where stop-loss is moved to lock profits
                        {"profit_percentage": 1.0, "stop_to_breakeven": True},
                        {"profit_percentage": 2.0, "lock_percentage": 0.5},
                        {"profit_percentage": 3.0, "lock_percentage": 1.0},
                        {"profit_percentage": 5.0, "lock_percentage": 2.0},
                        {"profit_percentage": 10.0, "lock_percentage": 5.0}
                    ],
                    "kelly_fraction": 0.3,      # Conservative Kelly criterion multiplier
                    "max_correlated_exposure": 35.0,  # Maximum exposure to correlated assets
                    "max_daily_risk_percentage": 5.0,  # Maximum portfolio at risk per day
                    "risk_scaling": {
                        "win_streak_bonus_cap": 0.25,  # Cap the bonus at 25%
                        "loss_streak_reduction_cap": 0.5,  # Cap the reduction at 50%
                        "time_decay_hours": 48,  # Hours to normalize after streaks
                        "volatility_dampening": true  # Reduce position size in high vol
                    }
                }
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save the default configuration
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                logger.info(f"Created default risk configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading risk configuration: {e}")
            # Fallback to basic configuration
            self.config = {
                "max_drawdown_percentage": 15.0,
                "max_position_size_percentage": 25.0,
                "max_leverage_mapping": {"medium": 50.0},
                "stop_loss_volatility_multipliers": {"medium": 3.0},
                "profit_ratchet_levels": [{"profit_percentage": 1.0, "stop_to_breakeven": True}],
                "kelly_fraction": 0.3,
                "max_correlated_exposure": 35.0,
                "max_daily_risk_percentage": 5.0
            }
    
    def update_portfolio_metrics(self, portfolio_value: float, positions: Dict[str, Any]):
        """
        Update portfolio metrics for risk assessment.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of current positions
        """
        # Update portfolio value
        self.current_portfolio_value = portfolio_value
        
        # Update peak portfolio value and calculate drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            # Record significant drawdowns for analysis
            if self.current_drawdown > 0.05:  # Record drawdowns > 5%
                self.historical_drawdowns.append({
                    "timestamp": datetime.now().isoformat(),
                    "drawdown": self.current_drawdown,
                    "portfolio_value": portfolio_value,
                    "peak_value": self.peak_portfolio_value
                })
        
        # Update current positions
        self.current_positions = positions
        
        # Calculate position weight used
        total_position_value = sum(pos.get("position_value", 0) for pos in positions.values())
        self.position_weight_used = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Log current metrics
        logger.info(f"Portfolio value: ${portfolio_value:.2f}, Drawdown: {self.current_drawdown:.2%}")
        logger.info(f"Position weight used: {self.position_weight_used:.2%}")
    
    def assess_volatility(self, pair: str, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess price volatility to determine appropriate risk parameters.
        
        Args:
            pair: Trading pair symbol
            price_data: DataFrame with price history
            
        Returns:
            Dictionary with volatility metrics
        """
        # Calculate returns
        price_data['returns'] = price_data['close'].pct_change()
        
        # Calculate metrics for different time periods
        volatility_1d = price_data['returns'].iloc[-24:].std() * np.sqrt(24) if len(price_data) >= 24 else 0
        volatility_3d = price_data['returns'].iloc[-72:].std() * np.sqrt(72) if len(price_data) >= 72 else 0
        volatility_7d = price_data['returns'].iloc[-168:].std() * np.sqrt(168) if len(price_data) >= 168 else 0
        
        # Calculate ATR
        high = price_data['high']
        low = price_data['low']
        close = price_data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()
        
        # Calculate volatility percentile compared to historical
        volatility_90d = price_data['returns'].iloc[-2160:].std() * np.sqrt(2160) if len(price_data) >= 2160 else 0
        volatility_hist = price_data['returns'].dropna().std() * np.sqrt(24)
        vol_percentile = 0.0
        
        if not np.isnan(volatility_hist) and volatility_hist > 0:
            vol_percentile = (volatility_1d / volatility_hist) * 100
        
        # Store metrics
        metrics = {
            "volatility_1d": volatility_1d,
            "volatility_3d": volatility_3d,
            "volatility_7d": volatility_7d,
            "volatility_90d": volatility_90d,
            "atr": float(atr),
            "atr_percentage": float(atr / price_data['close'].iloc[-1]),
            "volatility_percentile": vol_percentile
        }
        
        # Categorize volatility
        if volatility_1d < 0.01:
            metrics["volatility_category"] = "very_low"
        elif volatility_1d < 0.02:
            metrics["volatility_category"] = "low"
        elif volatility_1d < 0.03:
            metrics["volatility_category"] = "medium"
        elif volatility_1d < 0.05:
            metrics["volatility_category"] = "high"
        elif volatility_1d < 0.08:
            metrics["volatility_category"] = "very_high"
        else:
            metrics["volatility_category"] = "extreme"
            
        # Store in volatility metrics
        self.volatility_metrics[pair] = metrics
        
        return metrics
    
    def calculate_position_size(self, pair: str, price: float, strategy: str, 
                                confidence: float, win_rate: float, 
                                portfolio_value: float) -> Dict[str, float]:
        """
        Calculate optimal position size with advanced risk management.
        
        Args:
            pair: Trading pair symbol
            price: Current price
            strategy: Strategy name
            confidence: Signal confidence (0-1)
            win_rate: Historical win rate for this signal type (0-1)
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with position sizing details
        """
        # Get volatility metrics or use default
        vol_metrics = self.volatility_metrics.get(pair, {"volatility_category": "medium", "atr_percentage": 0.02})
        vol_category = vol_metrics["volatility_category"]
        
        # Get maximum allowed leverage based on volatility
        max_leverage = self.config["max_leverage_mapping"].get(vol_category, 50.0)
        
        # Calculate base risk percentage
        risk_percentage = 0.2  # Default 20% of capital at risk for the trade
        
        # Adjust risk based on drawdown
        max_drawdown = self.config["max_drawdown_percentage"] / 100.0
        drawdown_factor = 1.0 - (self.current_drawdown / max_drawdown) if max_drawdown > 0 else 1.0
        drawdown_factor = max(0.25, drawdown_factor)  # Never go below 25% of base risk
        
        # Adjust risk based on confidence and win rate using Kelly criterion
        kelly_criterion = win_rate - ((1.0 - win_rate) / 1.0)  # Assuming 1:1 risk:reward ratio
        kelly_criterion = max(0.05, kelly_criterion)  # Ensure minimum value
        
        # Apply conservative Kelly fraction
        kelly_fraction = self.config["kelly_fraction"]
        kelly_risk = kelly_criterion * kelly_fraction
        
        # Final risk percentage adjusted by Kelly and drawdown
        adjusted_risk = risk_percentage * drawdown_factor * min(1.0, kelly_risk * 5.0)  
        
        # Apply confidence modifier
        confidence_modifier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 based on confidence
        adjusted_risk *= confidence_modifier
        
        # Check for daily risk limit
        remaining_daily_risk = self.max_daily_risk - self.daily_risk_used
        if adjusted_risk > remaining_daily_risk:
            adjusted_risk = remaining_daily_risk
            
        # Calculate position leverage (scale with confidence but cap at max_leverage)
        base_leverage = 20.0  # Default base leverage
        leverage = base_leverage + ((max_leverage - base_leverage) * confidence * confidence)
        leverage = min(max_leverage, leverage)
        
        # Calculate position size
        margin_amount = portfolio_value * adjusted_risk
        position_value = margin_amount * leverage
        position_size = position_value / price
        
        # Apply maximum position size constraint
        max_position_pct = self.config["max_position_size_percentage"] / 100.0
        max_position_value = portfolio_value * max_position_pct
        
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / price
        
        # Update daily risk used
        risk_amount = margin_amount  # The actual capital at risk
        self.daily_risk_used += risk_amount / portfolio_value
        
        # Calculate stop-loss distance based on volatility
        stop_loss_multiplier = self.config["stop_loss_volatility_multipliers"].get(vol_category, 3.0)
        atr = vol_metrics.get("atr", price * 0.02)  # Use 2% as default if ATR not available
        stop_loss_distance = atr * stop_loss_multiplier
        
        # Calculate take_profit distance (asymmetric in favor of profits)
        take_profit_multiplier = stop_loss_multiplier * 1.5  # 1.5x stop distance
        take_profit_distance = atr * take_profit_multiplier
        
        position_info = {
            "position_size": position_size,
            "position_value": position_value,
            "margin_amount": margin_amount,
            "leverage": leverage,
            "risk_percentage": adjusted_risk,
            "stop_loss_distance": stop_loss_distance,
            "take_profit_distance": take_profit_distance,
            "max_leverage": max_leverage,
            "volatility_category": vol_category,
            "kelly_risk": kelly_risk,
            "drawdown_factor": drawdown_factor,
            "confidence_modifier": confidence_modifier
        }
        
        logger.info(f"Calculated position size for {pair} with risk {adjusted_risk:.2%}, "
                  f"leverage {leverage:.1f}x, and size {position_size:.6f}")
        
        return position_info
    
    def calculate_stop_loss_price(self, entry_price: float, direction: str, 
                                  stop_distance: float) -> float:
        """
        Calculate initial stop-loss price.
        
        Args:
            entry_price: Position entry price
            direction: "long" or "short"
            stop_distance: Stop-loss distance in price units
            
        Returns:
            Stop-loss price
        """
        if direction.lower() == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def calculate_take_profit_price(self, entry_price: float, direction: str, 
                                   take_profit_distance: float) -> float:
        """
        Calculate take-profit price.
        
        Args:
            entry_price: Position entry price
            direction: "long" or "short"
            take_profit_distance: Take-profit distance in price units
            
        Returns:
            Take-profit price
        """
        if direction.lower() == "long":
            return entry_price + take_profit_distance
        else:  # short
            return entry_price - take_profit_distance
    
    def update_position_risk(self, position_id: str, current_price: float, 
                            position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update position risk metrics and trailing stop based on current price.
        
        Args:
            position_id: Unique position identifier
            current_price: Current market price
            position_data: Position data including entry price, direction, etc.
            
        Returns:
            Updated position data with new stop-loss and risk metrics
        """
        entry_price = position_data.get("entry_price", 0)
        direction = position_data.get("direction", "long")
        initial_stop_loss = position_data.get("initial_stop_loss", 0)
        current_stop_loss = position_data.get("stop_loss", initial_stop_loss)
        
        # Calculate unrealized profit/loss percentage
        if direction.lower() == "long":
            profit_pct = (current_price / entry_price) - 1.0
        else:  # short
            profit_pct = (entry_price / current_price) - 1.0
            
        profit_pct *= position_data.get("leverage", 1.0)
        
        # Initialize ratchet tracking if not present
        if position_id not in self.profit_ratchets:
            self.profit_ratchets[position_id] = {
                "highest_profit_pct": 0.0,
                "ratchet_stops": [False] * len(self.config["profit_ratchet_levels"]),
                "current_ratchet": -1
            }
        
        # Track highest profit percentage
        ratchet_data = self.profit_ratchets[position_id]
        if profit_pct > ratchet_data["highest_profit_pct"]:
            ratchet_data["highest_profit_pct"] = profit_pct
        
        # Check for profit ratchet levels
        new_stop_loss = current_stop_loss
        highest_pct = ratchet_data["highest_profit_pct"]
        
        for i, level in enumerate(self.config["profit_ratchet_levels"]):
            # Skip already applied ratchets
            if ratchet_data["ratchet_stops"][i]:
                continue
                
            # Check if this ratchet level is reached
            if highest_pct >= level["profit_percentage"] / 100.0:
                # Mark this ratchet as applied
                ratchet_data["ratchet_stops"][i] = True
                ratchet_data["current_ratchet"] = i
                
                # Apply the stop-loss adjustment
                if level.get("stop_to_breakeven", False):
                    # Move stop to breakeven plus a small buffer
                    buffer = (entry_price * 0.001)  # 0.1% buffer
                    if direction.lower() == "long":
                        new_stop_loss = entry_price + buffer
                    else:  # short
                        new_stop_loss = entry_price - buffer
                    
                    logger.info(f"Moving stop to breakeven for {position_id} at profit {highest_pct:.2%}")
                
                elif "lock_percentage" in level:
                    # Lock in a percentage of the profit
                    lock_pct = level["lock_percentage"] / 100.0
                    profit_to_lock = highest_pct * lock_pct
                    
                    if direction.lower() == "long":
                        price_move = entry_price * profit_to_lock / position_data.get("leverage", 1.0)
                        new_stop_loss = entry_price + price_move
                    else:  # short
                        price_move = entry_price * profit_to_lock / position_data.get("leverage", 1.0)
                        new_stop_loss = entry_price - price_move
                    
                    logger.info(f"Locking {lock_pct:.2%} of profit for {position_id} at profit {highest_pct:.2%}")
        
        # Never move the stop-loss in the wrong direction
        if direction.lower() == "long":
            new_stop_loss = max(current_stop_loss, new_stop_loss)
        else:  # short
            new_stop_loss = min(current_stop_loss, new_stop_loss)
        
        # Update position data
        position_data["stop_loss"] = new_stop_loss
        position_data["profit_percentage"] = profit_pct
        position_data["highest_profit_percentage"] = highest_pct
        position_data["ratchet_level"] = ratchet_data["current_ratchet"]
        
        # Calculate remaining risk
        if direction.lower() == "long":
            remaining_risk_pct = abs((new_stop_loss / current_price) - 1.0)
        else:  # short
            remaining_risk_pct = abs((current_price / new_stop_loss) - 1.0)
            
        remaining_risk_pct *= position_data.get("leverage", 1.0)
        position_data["remaining_risk_percentage"] = remaining_risk_pct
        
        return position_data
    
    def check_portfolio_risk_limits(self) -> Dict[str, Any]:
        """
        Check overall portfolio risk limits and return warning status.
        
        Returns:
            Dictionary with risk warnings and status
        """
        warnings = []
        critical_warnings = []
        
        # Check drawdown
        max_drawdown = self.config["max_drawdown_percentage"] / 100.0
        if self.current_drawdown > max_drawdown * 0.7:
            warnings.append(f"Drawdown ({self.current_drawdown:.2%}) approaching maximum ({max_drawdown:.2%})")
        if self.current_drawdown > max_drawdown * 0.9:
            critical_warnings.append(f"Drawdown ({self.current_drawdown:.2%}) near maximum limit ({max_drawdown:.2%})")
        
        # Check exposure
        max_exposure = self.config["max_position_size_percentage"] / 100.0
        if self.position_weight_used > max_exposure * 0.8:
            warnings.append(f"Portfolio exposure ({self.position_weight_used:.2%}) approaching limit ({max_exposure:.2%})")
        if self.position_weight_used > max_exposure * 0.95:
            critical_warnings.append(f"Portfolio exposure ({self.position_weight_used:.2%}) near maximum")
        
        # Check daily risk limit
        if self.daily_risk_used > self.max_daily_risk * 0.8:
            warnings.append(f"Daily risk allocation ({self.daily_risk_used:.2%}) approaching limit ({self.max_daily_risk:.2%})")
        if self.daily_risk_used > self.max_daily_risk * 0.95:
            critical_warnings.append(f"Daily risk allocation ({self.daily_risk_used:.2%}) near maximum")
        
        # Determine status
        status = "normal"
        if warnings:
            status = "warning"
        if critical_warnings:
            status = "critical"
        
        return {
            "status": status,
            "warnings": warnings,
            "critical_warnings": critical_warnings,
            "metrics": {
                "drawdown": self.current_drawdown,
                "max_drawdown": max_drawdown,
                "portfolio_exposure": self.position_weight_used,
                "max_exposure": max_exposure,
                "daily_risk_used": self.daily_risk_used,
                "max_daily_risk": self.max_daily_risk
            }
        }
    
    def reset_daily_risk(self):
        """Reset daily risk allocation (call at start of trading day)"""
        self.daily_risk_used = 0.0
        logger.info("Reset daily risk allocation")
    
    def get_risk_adjusted_parameters(self, pair: str, base_params: Dict[str, Any], 
                                    confidence: float, signal_strength: float) -> Dict[str, Any]:
        """
        Get risk-adjusted parameters based on market conditions and confidence.
        
        Args:
            pair: Trading pair symbol
            base_params: Base parameters from optimizer
            confidence: Model confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            Risk-adjusted parameters
        """
        # Get volatility metrics or use default medium
        vol_metrics = self.volatility_metrics.get(pair, {"volatility_category": "medium"})
        vol_category = vol_metrics["volatility_category"]
        
        # Get base parameters with defaults
        risk_percentage = base_params.get("risk_percentage", 0.2)
        base_leverage = base_params.get("base_leverage", 20.0)
        max_leverage = base_params.get("max_leverage", 50.0)
        stop_loss_multiplier = base_params.get("trailing_stop_atr_multiplier", 3.0)
        
        # Adjust for volatility
        if vol_category == "very_low":
            leverage_factor = 1.2
            risk_factor = 1.1
            stop_factor = 1.2
        elif vol_category == "low":
            leverage_factor = 1.1
            risk_factor = 1.05
            stop_factor = 1.1
        elif vol_category == "medium":
            leverage_factor = 1.0
            risk_factor = 1.0
            stop_factor = 1.0
        elif vol_category == "high":
            leverage_factor = 0.8
            risk_factor = 0.9
            stop_factor = 0.8
        elif vol_category == "very_high":
            leverage_factor = 0.6
            risk_factor = 0.7
            stop_factor = 0.6
        else:  # extreme
            leverage_factor = 0.4
            risk_factor = 0.5
            stop_factor = 0.5
            
        # Adjust for drawdown
        max_drawdown = self.config["max_drawdown_percentage"] / 100.0
        drawdown_factor = 1.0 - (self.current_drawdown / max_drawdown) if max_drawdown > 0 else 1.0
        drawdown_factor = max(0.5, drawdown_factor)  # Never go below 50% of base risk
        
        # Apply drawdown factor to risk and leverage
        risk_factor *= drawdown_factor
        leverage_factor *= drawdown_factor ** 0.5  # Less aggressive reduction for leverage
        
        # Apply confidence and signal strength
        combined_confidence = (confidence * 0.6) + (signal_strength * 0.4)
        
        # Adjust final parameters
        adjusted_risk = risk_percentage * risk_factor
        adjusted_base_leverage = base_leverage * leverage_factor
        adjusted_max_leverage = max_leverage * leverage_factor
        
        # Calculate dynamic leverage based on confidence
        leverage_range = adjusted_max_leverage - adjusted_base_leverage
        adjusted_leverage = adjusted_base_leverage + (leverage_range * combined_confidence)
        
        # Adjust stop-loss multiplier
        adjusted_stop_multiplier = stop_loss_multiplier * stop_factor
        
        # Final adjustments to never exceed limits
        adjusted_risk = min(0.4, max(0.05, adjusted_risk))  # 5-40% risk range
        adjusted_leverage = min(125.0, max(5.0, adjusted_leverage))  # 5-125x leverage range
        
        return {
            "risk_percentage": adjusted_risk,
            "leverage": adjusted_leverage,
            "trailing_stop_atr_multiplier": adjusted_stop_multiplier,
            "volatility_category": vol_category,
            "drawdown_factor": drawdown_factor,
            "leverage_factor": leverage_factor,
            "risk_factor": risk_factor
        }
    
    def calculate_portfolio_correlation(self, price_data: Dict[str, pd.DataFrame]):
        """
        Calculate correlation between trading pairs to manage portfolio risk.
        
        Args:
            price_data: Dictionary mapping pair names to price DataFrames
        """
        # Extract close prices for each pair
        close_data = {}
        for pair, data in price_data.items():
            if "close" in data.columns:
                close_data[pair] = data["close"]
        
        if len(close_data) <= 1:
            return  # Need at least 2 assets for correlation
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(close_data).pct_change().dropna()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        # Log high correlations
        high_corr_pairs = []
        for i, pair1 in enumerate(self.correlation_matrix.index):
            for j, pair2 in enumerate(self.correlation_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    corr = self.correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:  # High correlation threshold
                        high_corr_pairs.append((pair1, pair2, corr))
        
        if high_corr_pairs:
            logger.info("High correlation detected between pairs:")
            for pair1, pair2, corr in high_corr_pairs:
                logger.info(f"  {pair1} and {pair2}: {corr:.2f}")
    
    def adjust_for_correlation(self, pair: str, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust position size for correlation with existing positions.
        
        Args:
            pair: Trading pair symbol
            position_info: Position information
            
        Returns:
            Adjusted position information
        """
        if self.correlation_matrix is None or pair not in self.correlation_matrix.index:
            return position_info
        
        # Check correlation with existing positions
        high_correlations = []
        for existing_pair in self.current_positions:
            if existing_pair in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[pair, existing_pair]
                if abs(corr) > 0.5:  # Moderate correlation threshold
                    high_correlations.append((existing_pair, corr))
        
        # If no high correlations, return original position info
        if not high_correlations:
            return position_info
        
        # Calculate correlation-based adjustment
        # More correlated assets = more risk reduction
        max_corr = max(abs(c[1]) for c in high_correlations)
        reduction_factor = 1.0 - (max_corr - 0.5) * 0.5  # 0.5-1.0 reduction factor
        
        # Apply reduction to position size and margin
        position_info["position_size"] *= reduction_factor
        position_info["position_value"] *= reduction_factor
        position_info["margin_amount"] *= reduction_factor
        
        logger.info(f"Adjusted {pair} position for correlation: {reduction_factor:.2f} factor")
        logger.info(f"Correlated with: {', '.join([f'{p} ({c:.2f})' for p, c in high_correlations])}")
        
        return position_info
    
    def get_risk_level(self, pair: str, confidence: float, win_rate: float,
                      recent_performance: Dict[str, Any]) -> RiskLevel:
        """
        Determine appropriate risk level based on multiple factors.
        
        Args:
            pair: Trading pair symbol
            confidence: Signal confidence (0-1)
            win_rate: Historical win rate (0-1)
            recent_performance: Recent trading performance metrics
            
        Returns:
            RiskLevel enum value
        """
        # Base risk score calculated from confidence and win rate
        base_score = (confidence * 0.6) + (win_rate * 0.4)
        
        # Adjust for recent performance
        win_streak = recent_performance.get("win_streak", 0)
        loss_streak = recent_performance.get("loss_streak", 0)
        
        # Win streaks increase risk appetite slightly (with cap)
        win_bonus = min(0.2, win_streak * 0.04)
        
        # Loss streaks decrease risk appetite more aggressively
        loss_penalty = min(0.5, loss_streak * 0.1)
        
        # Adjust score
        adjusted_score = base_score + win_bonus - loss_penalty
        
        # Adjust for volatility
        vol_metrics = self.volatility_metrics.get(pair, {"volatility_category": "medium"})
        vol_category = vol_metrics["volatility_category"]
        
        vol_adjustment = 0.0
        if vol_category == "very_low":
            vol_adjustment = 0.1
        elif vol_category == "low":
            vol_adjustment = 0.05
        elif vol_category == "high":
            vol_adjustment = -0.05
        elif vol_category == "very_high":
            vol_adjustment = -0.1
        elif vol_category == "extreme":
            vol_adjustment = -0.2
        
        # Apply volatility adjustment
        final_score = adjusted_score + vol_adjustment
        
        # Map score to risk level
        if final_score < 0.3:
            return RiskLevel.VERY_LOW
        elif final_score < 0.45:
            return RiskLevel.LOW
        elif final_score < 0.6:
            return RiskLevel.MODERATE
        elif final_score < 0.75:
            return RiskLevel.STANDARD
        elif final_score < 0.85:
            return RiskLevel.ELEVATED
        elif final_score < 0.95:
            return RiskLevel.HIGH
        else:
            return RiskLevel.AGGRESSIVE
    
    def get_risk_coefficient(self, risk_level: RiskLevel) -> float:
        """
        Get risk coefficient for position sizing based on risk level.
        
        Args:
            risk_level: RiskLevel enum value
            
        Returns:
            Risk coefficient for position sizing
        """
        risk_coefficients = {
            RiskLevel.VERY_LOW: 0.25,
            RiskLevel.LOW: 0.5,
            RiskLevel.MODERATE: 0.75,
            RiskLevel.STANDARD: 1.0,
            RiskLevel.ELEVATED: 1.1,
            RiskLevel.HIGH: 1.25,
            RiskLevel.AGGRESSIVE: 1.5
        }
        
        return risk_coefficients.get(risk_level, 1.0)

# Create singleton risk manager
risk_manager = RiskManager()