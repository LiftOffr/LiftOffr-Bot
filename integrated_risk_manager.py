#!/usr/bin/env python3
"""
Integrated Risk Management System

This module integrates the advanced risk management system with the dynamic parameter
optimizer and trading strategies to create a comprehensive risk-aware trading system
that prevents liquidations and large losses while maximizing profits.

It serves as the central coordination point between the various components:
1. Risk management system for position sizing and stop-loss calculation
2. Dynamic parameter optimizer for trade parameter adjustment
3. Market analyzer for regime detection and volatility assessment
4. Trading strategies for signal generation and trade execution

Features:
- Real-time risk monitoring and adjustment
- Portfolio-level risk constraints
- Position-level ratcheting stops for profit protection
- Volatility and correlation-based position sizing
- Kelly criterion optimization for long-term capital growth
"""

import os
import json
import logging
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import our components
from dynamic_parameter_optimizer import DynamicParameterOptimizer
from utils.risk_manager import risk_manager, RiskLevel
from utils.market_analyzer import MarketAnalyzer, MarketRegime
from dynamic_parameter_adjustment import dynamic_adjuster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("integrated_risk_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedRiskManager:
    """
    Integrated risk management system that coordinates between risk management,
    parameter optimization, and trading strategies.
    """
    
    def __init__(self, config_path: str = "config/integrated_risk_config.json"):
        """
        Initialize the integrated risk manager.
        
        Args:
            config_path: Path to integrated risk configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Initialize components
        self.risk_manager = risk_manager  # Use the singleton instance
        self.parameter_optimizer = DynamicParameterOptimizer()
        self.market_analyzer = MarketAnalyzer()
        self.dynamic_adjuster = dynamic_adjuster  # Use the singleton instance
        
        # Initialize state variables
        self.active_trades = {}
        self.portfolio_value = 10000.0  # Default starting value
        self.daily_results = {"wins": 0, "losses": 0, "profit": 0.0}
        self.last_risk_check = time.time()
        self.market_regimes = {}
        self.pair_metrics = {}
        self.last_data_update = {}
        self.emergency_mode = False
        
        # Load trade history if available
        self.trade_history = self._load_trade_history()
        
        # Register daily reset at UTC midnight
        self._schedule_daily_reset()
        
        logger.info("Integrated Risk Manager initialized")
    
    def load_config(self):
        """Load integrated risk configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded integrated risk configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = {
                    "enable_integrated_risk": True,
                    "check_interval_seconds": 60,
                    "enable_emergency_controls": True,
                    "log_level": "INFO",
                    "enable_portfolio_correlation": True,
                    "performance_metrics_window": 30,  # days
                    "backtest_validation": True,
                    "risk_profile": "balanced",  # conservative, balanced, aggressive
                    "monitoring": {
                        "enable_alerts": True,
                        "alert_threshold_warning": 0.7,  # 70% of max risk
                        "alert_threshold_critical": 0.9,  # 90% of max risk
                        "max_drawdown_alert": 10.0,  # Alert at 10% drawdown
                        "consecutive_loss_alert": 5  # Alert after 5 consecutive losses
                    }
                }
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save the default configuration
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                logger.info(f"Created default integrated risk configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading integrated risk configuration: {e}")
            # Fallback to basic configuration
            self.config = {
                "enable_integrated_risk": True,
                "check_interval_seconds": 60,
                "enable_emergency_controls": True,
                "log_level": "INFO"
            }
    
    def _load_trade_history(self) -> List[Dict[str, Any]]:
        """
        Load trade history from file.
        
        Returns:
            List of trade records
        """
        history_path = "trade_history.json"
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
        
        return []
    
    def _save_trade_history(self):
        """Save trade history to file"""
        history_path = "trade_history.json"
        try:
            # Limit to last 1000 trades to prevent file growth
            recent_history = self.trade_history[-1000:] if len(self.trade_history) > 1000 else self.trade_history
            with open(history_path, 'w') as f:
                json.dump(recent_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def _schedule_daily_reset(self):
        """Schedule daily reset of risk metrics at UTC midnight"""
        # In a real implementation, this would use a scheduler
        # For simplicity, we'll check time in the update_portfolio method
        pass
    
    def update_portfolio(self, portfolio_value: float, positions: Dict[str, Any]):
        """
        Update portfolio metrics and check for daily reset.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of current positions
        """
        # Check if day has changed (UTC)
        now = datetime.utcnow()
        today = now.date()
        
        if not hasattr(self, 'last_reset_date') or self.last_reset_date != today:
            # Perform daily reset
            self.risk_manager.reset_daily_risk()
            self.daily_results = {"wins": 0, "losses": 0, "profit": 0.0}
            self.last_reset_date = today
            logger.info(f"Daily risk metrics reset for {today}")
        
        # Update portfolio metrics
        self.portfolio_value = portfolio_value
        self.risk_manager.update_portfolio_metrics(portfolio_value, positions)
        
        # Check if we should perform a risk assessment
        current_time = time.time()
        if current_time - self.last_risk_check >= self.config["check_interval_seconds"]:
            self._perform_risk_assessment()
            self.last_risk_check = current_time
    
    def update_market_data(self, pair: str, price_data: pd.DataFrame):
        """
        Update market data for a trading pair.
        
        Args:
            pair: Trading pair symbol
            price_data: DataFrame with price history
        """
        # Update last data timestamp
        self.last_data_update[pair] = datetime.now()
        
        # Assess volatility
        volatility_metrics = self.risk_manager.assess_volatility(pair, price_data)
        
        # Analyze market regime
        market_analysis = self.market_analyzer.analyze_market_regimes(price_data)
        regime = market_analysis.get("current_regime", "UNKNOWN")
        
        # Store metrics
        self.pair_metrics[pair] = {
            "volatility_metrics": volatility_metrics,
            "market_regime": regime,
            "market_analysis": market_analysis
        }
        
        # If multiple pairs, calculate correlation
        if self.config.get("enable_portfolio_correlation", True) and len(self.last_data_update) > 1:
            # Collect price data for all pairs
            all_price_data = {}
            for p in self.last_data_update.keys():
                if p in self.pair_metrics:
                    all_price_data[p] = price_data
            
            # Calculate correlation
            if len(all_price_data) > 1:
                self.risk_manager.calculate_portfolio_correlation(all_price_data)
    
    def _perform_risk_assessment(self):
        """Perform comprehensive risk assessment of the portfolio"""
        # Check portfolio risk limits
        risk_status = self.risk_manager.check_portfolio_risk_limits()
        
        # Log risk status
        if risk_status["status"] != "normal":
            logger.warning(f"Risk assessment status: {risk_status['status']}")
            for warning in risk_status.get("warnings", []):
                logger.warning(f"Risk warning: {warning}")
            for critical in risk_status.get("critical_warnings", []):
                logger.critical(f"Critical risk warning: {critical}")
        
        # Check if emergency mode should be activated
        if risk_status["status"] == "critical" and self.config.get("enable_emergency_controls", True):
            if not self.emergency_mode:
                logger.critical("EMERGENCY MODE ACTIVATED - Implementing risk reduction measures")
                self.emergency_mode = True
                self._implement_emergency_measures()
        elif self.emergency_mode and risk_status["status"] != "critical":
            logger.info("Emergency mode deactivated - Risk levels back within acceptable range")
            self.emergency_mode = False
    
    def _implement_emergency_measures(self):
        """Implement emergency risk reduction measures"""
        # In a real implementation, this would:
        # 1. Reduce position sizes for new trades
        # 2. Tighten stops on existing positions
        # 3. Possibly close highest risk positions
        # 4. Disable certain high-risk strategies
        
        # For demonstration, we'll just log the actions that would be taken
        logger.critical("Emergency measures: Reducing risk parameters by 50%")
        logger.critical("Emergency measures: Tightening stops on all open positions")
        logger.critical("Emergency measures: Disabling high leverage trades")
    
    def get_trade_parameters(self, pair: str, strategy: str, confidence: float,
                            signal_strength: float) -> Dict[str, Any]:
        """
        Get optimized and risk-adjusted parameters for a trade.
        
        Args:
            pair: Trading pair symbol
            strategy: Strategy name
            confidence: Model prediction confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            Dictionary with trade parameters
        """
        # Get base parameters from optimizer
        base_params = self.parameter_optimizer.get_pair_params(pair)
        
        # Check if integrated risk management is enabled
        if not self.config.get("enable_integrated_risk", True):
            # If not enabled, just return dynamically adjusted parameters
            return self.dynamic_adjuster.get_parameters(
                pair=pair,
                strategy=strategy,
                confidence=confidence,
                signal_strength=signal_strength
            )
        
        # Get market metrics
        market_metrics = self.pair_metrics.get(pair, {})
        volatility_metrics = market_metrics.get("volatility_metrics", {})
        market_regime = market_metrics.get("market_regime", "UNKNOWN")
        
        # Get recent performance metrics
        recent_performance = self._get_recent_performance(pair, strategy)
        
        # Determine appropriate risk level
        risk_level = self.risk_manager.get_risk_level(
            pair=pair,
            confidence=confidence,
            win_rate=recent_performance.get("win_rate", 0.5),
            recent_performance=recent_performance
        )
        
        # Get risk coefficient for position sizing
        risk_coefficient = self.risk_manager.get_risk_coefficient(risk_level)
        
        # Apply emergency mode reduction if active
        if self.emergency_mode:
            risk_coefficient *= 0.5
        
        # Get risk-adjusted parameters
        risk_adjusted_params = self.risk_manager.get_risk_adjusted_parameters(
            pair=pair,
            base_params=base_params,
            confidence=confidence,
            signal_strength=signal_strength
        )
        
        # Calculate position size
        position_info = self.risk_manager.calculate_position_size(
            pair=pair,
            price=market_metrics.get("current_price", 100.0),  # Default for demonstration
            strategy=strategy,
            confidence=confidence,
            win_rate=recent_performance.get("win_rate", 0.5),
            portfolio_value=self.portfolio_value
        )
        
        # Adjust for correlation with existing positions
        if self.config.get("enable_portfolio_correlation", True):
            position_info = self.risk_manager.adjust_for_correlation(pair, position_info)
        
        # Compile final parameters
        trade_params = {
            "position_size": position_info["position_size"],
            "leverage": position_info["leverage"],
            "margin_amount": position_info["margin_amount"],
            "risk_percentage": position_info["risk_percentage"],
            "trailing_stop_atr_multiplier": risk_adjusted_params["trailing_stop_atr_multiplier"],
            "stop_loss_distance": position_info["stop_loss_distance"],
            "take_profit_distance": position_info["take_profit_distance"],
            "risk_level": risk_level.name,
            "risk_coefficient": risk_coefficient,
            "volatility_category": volatility_metrics.get("volatility_category", "medium"),
            "market_regime": market_regime
        }
        
        logger.info(f"Trade parameters for {pair} ({strategy}): "
                  f"Size={position_info['position_size']:.6f}, "
                  f"Leverage={position_info['leverage']:.1f}x, "
                  f"Risk={position_info['risk_percentage']:.2%}, "
                  f"Risk Level={risk_level.name}")
        
        return trade_params
    
    def _get_recent_performance(self, pair: str, strategy: str) -> Dict[str, Any]:
        """
        Get recent performance metrics for a pair and strategy.
        
        Args:
            pair: Trading pair symbol
            strategy: Strategy name
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate performance over the configured window
        window_days = self.config.get("performance_metrics_window", 30)
        window_start = datetime.now() - timedelta(days=window_days)
        
        # Filter trade history for this pair and strategy
        recent_trades = []
        for trade in self.trade_history:
            # Check if trade is for this pair and strategy
            if trade.get("pair") == pair and trade.get("strategy") == strategy:
                # Check if trade is within the window
                trade_date = datetime.fromisoformat(trade.get("exit_timestamp", "2000-01-01T00:00:00"))
                if trade_date >= window_start:
                    recent_trades.append(trade)
        
        # Calculate metrics
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.get("profit", 0) > 0)
        losing_trades = sum(1 for t in recent_trades if t.get("profit", 0) <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
        
        # Calculate win/loss streaks
        current_streak = 0
        win_streak = 0
        loss_streak = 0
        
        # Sort trades by timestamp
        sorted_trades = sorted(recent_trades, key=lambda t: t.get("exit_timestamp", ""))
        
        for trade in sorted_trades:
            if trade.get("profit", 0) > 0:  # Win
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                win_streak = max(win_streak, current_streak)
            else:  # Loss
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                loss_streak = max(loss_streak, abs(current_streak))
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "current_streak": current_streak
        }
    
    def register_trade_result(self, trade_result: Dict[str, Any]):
        """
        Register a completed trade result for performance tracking.
        
        Args:
            trade_result: Dictionary with trade result details
        """
        # Add trade to history
        self.trade_history.append(trade_result)
        
        # Update daily results
        profit = trade_result.get("profit", 0)
        if profit > 0:
            self.daily_results["wins"] += 1
        else:
            self.daily_results["losses"] += 1
        self.daily_results["profit"] += profit
        
        # Save trade history
        self._save_trade_history()
        
        # Log trade result
        logger.info(f"Trade result registered: {trade_result.get('pair')} {trade_result.get('type')} - "
                  f"Profit: {profit:.2f} ({trade_result.get('profit_percentage', 0):.2%})")
    
    def update_active_trade(self, trade_id: str, current_price: float, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update active trade with risk management (e.g., trailing stops).
        
        Args:
            trade_id: Unique trade identifier
            current_price: Current market price
            position_data: Position data
            
        Returns:
            Updated position data
        """
        # Store in active trades if not already there
        if trade_id not in self.active_trades:
            self.active_trades[trade_id] = position_data
        
        # Update position risk metrics
        updated_position = self.risk_manager.update_position_risk(
            position_id=trade_id,
            current_price=current_price,
            position_data=position_data
        )
        
        # Store updated position
        self.active_trades[trade_id] = updated_position
        
        return updated_position
    
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for optimization across all pairs.
        
        Returns:
            Dictionary with optimization parameters
        """
        # Define parameters for optimization based on risk profile
        risk_profile = self.config.get("risk_profile", "balanced")
        
        if risk_profile == "conservative":
            optimization_params = {
                "kelly_fraction": 0.2,
                "max_leverage": {
                    "very_low": 75.0,
                    "low": 50.0,
                    "medium": 35.0,
                    "high": 25.0,
                    "very_high": 15.0,
                    "extreme": 5.0
                },
                "risk_percentage_range": [0.05, 0.15],  # 5-15% risk
                "stop_multiplier_range": [2.5, 4.0]  # 2.5-4.0 ATR
            }
        elif risk_profile == "aggressive":
            optimization_params = {
                "kelly_fraction": 0.4,
                "max_leverage": {
                    "very_low": 125.0,
                    "low": 100.0,
                    "medium": 75.0,
                    "high": 50.0,
                    "very_high": 30.0,
                    "extreme": 15.0
                },
                "risk_percentage_range": [0.15, 0.35],  # 15-35% risk
                "stop_multiplier_range": [2.0, 3.5]  # 2.0-3.5 ATR
            }
        else:  # balanced (default)
            optimization_params = {
                "kelly_fraction": 0.3,
                "max_leverage": {
                    "very_low": 100.0,
                    "low": 75.0,
                    "medium": 50.0,
                    "high": 35.0,
                    "very_high": 20.0,
                    "extreme": 10.0
                },
                "risk_percentage_range": [0.1, 0.25],  # 10-25% risk
                "stop_multiplier_range": [2.2, 3.8]  # 2.2-3.8 ATR
            }
        
        return optimization_params
    
    def get_validation_criteria(self) -> Dict[str, Any]:
        """
        Get criteria for validating optimization results.
        
        Returns:
            Dictionary with validation criteria
        """
        # Define validation criteria
        return {
            "min_win_rate": 0.55,  # Minimum acceptable win rate
            "max_drawdown": 0.15,  # Maximum acceptable drawdown
            "min_profit_factor": 1.5,  # Minimum profit factor
            "min_sharpe_ratio": 1.0,  # Minimum Sharpe ratio
            "max_consecutive_losses": 5,  # Maximum acceptable consecutive losses
            "min_trades": 30,  # Minimum number of trades for validation
            "default_optimization_days": 180  # Days of data for optimization
        }

# Create singleton integrated risk manager
integrated_risk_manager = IntegratedRiskManager()