#!/usr/bin/env python3
"""
Risk-Enhanced Backtest Engine

This script provides an enhanced backtesting engine that incorporates the advanced 
risk management system for more realistic backtesting results. It simulates:

1. Dynamic position sizing based on trade confidence and volatility
2. Volatility-based leverage adjustments
3. Trailing stops with profit ratcheting
4. Portfolio correlation effects
5. Drawdown protection mechanisms
6. Liquidation prevention

It can be used to evaluate the effectiveness of the risk management system across
different market conditions and stress scenarios.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Import our components
from utils.risk_manager import risk_manager
from utils.data_loader import HistoricalDataLoader
from utils.market_analyzer import MarketAnalyzer
from integrated_risk_manager import integrated_risk_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("risk_enhanced_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskEnhancedBacktester:
    """
    Enhanced backtesting engine with integrated risk management.
    """
    
    def __init__(self, config_path: str = "config/risk_config.json"):
        """
        Initialize the risk-enhanced backtester.
        
        Args:
            config_path: Path to risk configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Initialize components
        self.risk_manager = risk_manager  # Use singleton
        self.integrated_risk_manager = integrated_risk_manager  # Use singleton
        self.data_loader = HistoricalDataLoader()
        self.market_analyzer = MarketAnalyzer()
        
        # Create output directory for results
        self.results_dir = "backtest_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize backtest state variables
        self.reset_backtest_state()
        
        logger.info("Risk-enhanced backtester initialized")
    
    def load_config(self):
        """Load risk configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded risk configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading risk configuration: {e}")
            # Use default configuration
            self.config = {
                "backtest": {
                    "initial_capital": 10000.0,
                    "enable_risk_management": True,
                    "enable_trailing_stops": True,
                    "enable_profit_ratcheting": True,
                    "enable_dynamic_leverage": True,
                    "fee_percentage": 0.0005,  # 0.05% fee (5 basis points)
                    "slippage_percentage": 0.0010,  # 0.1% slippage (10 basis points)
                    "enable_partial_liquidations": True,
                    "enable_stress_scenarios": False
                }
            }
    
    def reset_backtest_state(self):
        """Reset backtest state variables"""
        # Portfolio tracking
        self.initial_capital = self.config.get("backtest", {}).get("initial_capital", 10000.0)
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.drawdowns = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.current_drawdown_duration = 0
        self.drawdown_start_value = self.initial_capital
        
        # Liquidation tracking
        self.liquidations = 0
        self.partial_liquidations = 0
        self.emergency_exits = 0
        
        # Risk tracking
        self.avg_leverage = 0.0
        self.max_leverage = 0.0
        self.risk_adjustments = 0
        self.daily_risk_used = {}
        
        logger.info(f"Backtest state reset with initial capital: ${self.initial_capital:.2f}")
    
    def fetch_historical_data(self, pair: str, timeframe: str = "1h", days: int = 180) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe for candles
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Fetching {days} days of {timeframe} data for {pair}")
        
        try:
            # Load historical data
            data = self.data_loader.fetch_historical_data(
                pair=pair,
                timeframe=timeframe,
                days=days
            )
            
            # Add technical indicators
            data = self.data_loader.add_technical_indicators(data)
            
            logger.info(f"Loaded {len(data)} data points for {pair}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def prepare_stress_scenario(self, data: pd.DataFrame, scenario_type: str) -> pd.DataFrame:
        """
        Prepare a stress test scenario by modifying price data.
        
        Args:
            data: Historical price data
            scenario_type: Type of stress scenario
            
        Returns:
            Modified DataFrame for stress testing
        """
        # Make a copy to avoid modifying original data
        scenario_data = data.copy()
        
        if scenario_type == "flash_crash":
            # Create a flash crash (40-60% drop in a short time)
            crash_idx = len(scenario_data) // 3
            crash_depth = np.random.uniform(0.4, 0.6)
            
            # Create crash over just a few candles
            for i in range(3):
                if crash_idx + i < len(scenario_data):
                    depth = crash_depth * (i + 1) / 3
                    scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - depth)
                    scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - depth * 1.2)
                    scenario_data.loc[scenario_data.index[crash_idx + i], "high"] *= (1 - depth * 0.8)
            
            # Create recovery
            for i in range(3, 10):
                if crash_idx + i < len(scenario_data):
                    recovery = crash_depth * (1 - (i - 3) / 7)
                    scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - recovery)
                    scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - recovery * 1.2)
                    scenario_data.loc[scenario_data.index[crash_idx + i], "high"] *= (1 - recovery * 0.8)
        
        elif scenario_type == "volatility_spike":
            # Increase volatility throughout the dataset
            for i in range(len(scenario_data)):
                if np.random.random() < 0.3:  # 30% of candles
                    # Calculate random volatility multiplier (1.5-3x normal range)
                    vol_mult = np.random.uniform(1.5, 3.0)
                    
                    # Get the normal candle range
                    normal_range = scenario_data["high"].iloc[i] - scenario_data["low"].iloc[i]
                    
                    # Calculate new extended range
                    extended_range = normal_range * vol_mult
                    
                    # Calculate center price (to expand around)
                    center = scenario_data["close"].iloc[i]
                    
                    # Create new high/low with extended range
                    scenario_data.loc[scenario_data.index[i], "high"] = center + (extended_range / 2)
                    scenario_data.loc[scenario_data.index[i], "low"] = center - (extended_range / 2)
        
        elif scenario_type == "sustained_downtrend":
            # Create a sustained downtrend (40% drop over time)
            total_drop = 0.4
            start_idx = len(scenario_data) // 4
            duration = len(scenario_data) // 2
            
            for i in range(duration):
                if start_idx + i < len(scenario_data):
                    # Calculate cumulative drop for this candle
                    drop_pct = total_drop * (i + 1) / duration
                    
                    # Apply drop to prices
                    idx = scenario_data.index[start_idx + i]
                    orig_close = data["close"].iloc[start_idx + i]
                    adjusted_close = orig_close * (1 - drop_pct)
                    
                    scenario_data.loc[idx, "close"] = adjusted_close
                    
                    # Adjust high/low proportionally
                    range_pct = (data["high"].iloc[start_idx + i] - data["low"].iloc[start_idx + i]) / orig_close
                    scenario_data.loc[idx, "high"] = adjusted_close * (1 + range_pct/2)
                    scenario_data.loc[idx, "low"] = adjusted_close * (1 - range_pct/2)
        
        # Recalculate technical indicators for modified data
        scenario_data = self.data_loader.add_technical_indicators(scenario_data)
        
        logger.info(f"Created {scenario_type} stress scenario with {len(scenario_data)} data points")
        
        return scenario_data
    
    def run_backtest(self, pair: str, data: pd.DataFrame, strategy_function, 
                    parameters: Dict[str, Any], stress_scenario: Optional[str] = None, 
                    label: str = "backtest") -> Dict[str, Any]:
        """
        Run a risk-enhanced backtest.
        
        Args:
            pair: Trading pair
            data: Historical price data
            strategy_function: Function that generates trading signals
            parameters: Strategy parameters
            stress_scenario: Optional stress scenario type
            label: Label for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Reset backtest state
        self.reset_backtest_state()
        
        # Prepare stress scenario if requested
        if stress_scenario and self.config.get("backtest", {}).get("enable_stress_scenarios", False):
            data = self.prepare_stress_scenario(data, stress_scenario)
            logger.info(f"Running backtest with {stress_scenario} stress scenario")
        
        logger.info(f"Starting backtest for {pair} with {len(data)} data points")
        
        # Track metrics across backtest
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        breakeven_trades = 0
        total_profit = 0.0
        total_fees = 0.0
        daily_returns = []
        last_date = None
        daily_portfolio_value = self.portfolio_value
        
        # Cache to avoid repeated calculations
        volatility_window = 24  # 1 day for hourly data
        market_regime_cache = {}
        volatility_metrics_cache = {}
        
        # Get fee and slippage parameters
        fee_percentage = self.config.get("backtest", {}).get("fee_percentage", 0.0005)
        slippage_percentage = self.config.get("backtest", {}).get("slippage_percentage", 0.001)
        
        # Enable/disable features
        enable_risk_management = self.config.get("backtest", {}).get("enable_risk_management", True)
        enable_trailing_stops = self.config.get("backtest", {}).get("enable_trailing_stops", True)
        enable_profit_ratcheting = self.config.get("backtest", {}).get("enable_profit_ratcheting", True)
        enable_dynamic_leverage = self.config.get("backtest", {}).get("enable_dynamic_leverage", True)
        
        # For each time step in the data
        for i in range(volatility_window, len(data)):
            # Get current timestamp and price
            timestamp = data.index[i]
            current_candle = {
                "timestamp": timestamp,
                "open": data["open"].iloc[i],
                "high": data["high"].iloc[i],
                "low": data["low"].iloc[i],
                "close": data["close"].iloc[i],
                "volume": data["volume"].iloc[i] if "volume" in data else 0
            }
            current_price = current_candle["close"]
            
            # Calculate daily returns if date changed
            current_date = timestamp.date()
            if last_date is not None and current_date != last_date:
                daily_return = (self.portfolio_value / daily_portfolio_value) - 1
                daily_returns.append(daily_return)
                daily_portfolio_value = self.portfolio_value
                
                # Reset daily risk if using risk management
                if enable_risk_management:
                    self.risk_manager.reset_daily_risk()
            last_date = current_date
            
            # Get historical data up to this point for analysis
            historical_window = data.iloc[i-volatility_window:i+1].copy()
            
            # Analyze market regime (cache by day to avoid redundant calculations)
            day_key = timestamp.strftime("%Y-%m-%d")
            if day_key not in market_regime_cache:
                market_regime = self.market_analyzer.analyze_market_regimes(historical_window)
                market_regime_cache[day_key] = market_regime
            else:
                market_regime = market_regime_cache[day_key]
            
            # Assess volatility (cache by day)
            if day_key not in volatility_metrics_cache:
                volatility_metrics = self.risk_manager.assess_volatility(pair, historical_window)
                volatility_metrics_cache[day_key] = volatility_metrics
            else:
                volatility_metrics = volatility_metrics_cache[day_key]
            
            # Update portfolio value with current prices
            self._update_portfolio_value(current_price)
            
            # Update position risk for open positions
            if enable_trailing_stops and enable_risk_management:
                for trade_id, position in list(self.positions.items()):
                    # Update position with risk management
                    updated_position = self.risk_manager.update_position_risk(
                        position_id=trade_id,
                        current_price=current_price,
                        position_data=position
                    )
                    self.positions[trade_id] = updated_position
                    
                    # Check for stop-loss hits
                    self._check_stops(trade_id, position, current_candle)
            
            # Generate trading signal
            signal = strategy_function(historical_window, parameters)
            
            # Process signal (open, close, or hold positions)
            if signal.get("action") in ["buy", "long", "sell", "short"] and pair not in self.positions:
                # Calculate confidence and signal strength
                confidence = signal.get("confidence", 0.5)
                signal_strength = signal.get("strength", 0.5)
                
                # Calculate position parameters
                if enable_risk_management:
                    # Get trade parameters from risk manager
                    position_params = self.risk_manager.calculate_position_size(
                        pair=pair,
                        price=current_price,
                        strategy=signal.get("strategy", "default"),
                        confidence=confidence,
                        win_rate=winning_trades / max(1, total_trades),
                        portfolio_value=self.portfolio_value
                    )
                    
                    # Apply dynamic leverage if enabled
                    if not enable_dynamic_leverage:
                        position_params["leverage"] = parameters.get("leverage", 20.0)
                else:
                    # Fixed parameters without risk management
                    risk_percentage = parameters.get("risk_percentage", 0.2)
                    leverage = parameters.get("leverage", 20.0)
                    stop_loss_distance = current_price * parameters.get("stop_loss_percentage", 0.05)
                    take_profit_distance = current_price * parameters.get("take_profit_percentage", 0.1)
                    
                    position_params = {
                        "position_size": (self.portfolio_value * risk_percentage * leverage) / current_price,
                        "leverage": leverage,
                        "margin_amount": self.portfolio_value * risk_percentage,
                        "risk_percentage": risk_percentage,
                        "stop_loss_distance": stop_loss_distance,
                        "take_profit_distance": take_profit_distance
                    }
                
                # Open position
                self._open_position(
                    pair=pair,
                    timestamp=timestamp,
                    signal=signal,
                    current_candle=current_candle,
                    position_params=position_params,
                    fee_percentage=fee_percentage,
                    slippage_percentage=slippage_percentage
                )
            
            # Update portfolio history
            self.portfolio_history.append({
                "timestamp": timestamp,
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
                "drawdown": self._calculate_drawdown(),
                "drawdown_duration": self.current_drawdown_duration
            })
        
        # Close any remaining positions at the end of the backtest
        for trade_id, position in list(self.positions.items()):
            self._close_position(
                trade_id=trade_id,
                timestamp=data.index[-1],
                exit_price=data["close"].iloc[-1],
                exit_reason="end_of_backtest",
                fee_percentage=fee_percentage,
                slippage_percentage=slippage_percentage
            )
        
        # Calculate final metrics
        results = self._calculate_backtest_metrics(daily_returns)
        
        # Save results
        self._save_backtest_results(pair, stress_scenario, label, results)
        
        logger.info(f"Backtest completed with {results['total_trades']} trades and {results['total_return']:.2f}% return")
        
        return results
    
    def _open_position(self, pair: str, timestamp, signal: Dict[str, Any], 
                     current_candle: Dict[str, Any], position_params: Dict[str, Any],
                     fee_percentage: float, slippage_percentage: float):
        """
        Open a new position.
        
        Args:
            pair: Trading pair
            timestamp: Current timestamp
            signal: Trading signal
            current_candle: Current price candle
            position_params: Position parameters
            fee_percentage: Fee percentage
            slippage_percentage: Slippage percentage
        """
        # Extract parameters
        position_size = position_params["position_size"]
        leverage = position_params["leverage"]
        margin_amount = position_params["margin_amount"]
        risk_percentage = position_params["risk_percentage"]
        stop_loss_distance = position_params["stop_loss_distance"]
        take_profit_distance = position_params["take_profit_distance"]
        
        # Determine position direction
        direction = "long" if signal.get("action") in ["buy", "long"] else "short"
        
        # Calculate entry price with slippage
        entry_price = current_candle["close"]
        if direction == "long":
            # Buy higher with slippage
            entry_price *= (1 + slippage_percentage)
        else:
            # Sell lower with slippage
            entry_price *= (1 - slippage_percentage)
        
        # Calculate fees
        fee_amount = position_size * entry_price * fee_percentage
        
        # Check if we have enough cash for margin
        if margin_amount + fee_amount > self.cash:
            logger.warning(f"Insufficient cash for trade: " +
                         f"Required={margin_amount + fee_amount:.2f}, Available={self.cash:.2f}")
            return
        
        # Calculate stop-loss and take-profit prices
        if direction == "long":
            stop_loss_price = entry_price - stop_loss_distance
            take_profit_price = entry_price + take_profit_distance
        else:
            stop_loss_price = entry_price + stop_loss_distance
            take_profit_price = entry_price - take_profit_distance
        
        # Generate trade ID
        trade_id = f"{pair}_{timestamp.strftime('%Y%m%d%H%M%S')}_{direction}"
        
        # Create position record
        position = {
            "trade_id": trade_id,
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": timestamp,
            "size": position_size,
            "leverage": leverage,
            "margin_amount": margin_amount,
            "position_value": position_size * entry_price,
            "initial_stop_loss": stop_loss_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "fee_paid": fee_amount,
            "strategy": signal.get("strategy", "default"),
            "signal_confidence": signal.get("confidence", 0.5),
            "risk_percentage": risk_percentage
        }
        
        # Add to positions
        self.positions[trade_id] = position
        
        # Update cash
        self.cash -= (margin_amount + fee_amount)
        
        # Track metrics
        self.total_trades += 1
        self.avg_leverage = ((self.avg_leverage * (self.total_trades - 1)) + leverage) / self.total_trades
        self.max_leverage = max(self.max_leverage, leverage)
        
        logger.info(f"Opened {direction} position at {entry_price:.2f} with " +
                  f"size={position_size:.6f}, leverage={leverage:.1f}x, margin={margin_amount:.2f}")
    
    def _check_stops(self, trade_id: str, position: Dict[str, Any], current_candle: Dict[str, Any]):
        """
        Check if stop-loss or take-profit has been hit.
        
        Args:
            trade_id: Position trade ID
            position: Position data
            current_candle: Current price candle
        """
        # Extract position details
        direction = position["direction"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Check if stop-loss hit
        stop_hit = False
        if direction == "long" and current_candle["low"] <= stop_loss:
            stop_hit = True
            exit_price = max(current_candle["open"], stop_loss)  # Assume we get out at or slightly better than stop
            exit_reason = "stop_loss"
        elif direction == "short" and current_candle["high"] >= stop_loss:
            stop_hit = True
            exit_price = min(current_candle["open"], stop_loss)  # Assume we get out at or slightly better than stop
            exit_reason = "stop_loss"
        
        # Check if take-profit hit
        tp_hit = False
        if direction == "long" and current_candle["high"] >= take_profit:
            tp_hit = True
            exit_price = min(current_candle["high"], take_profit)  # Assume we get out at or slightly worse than target
            exit_reason = "take_profit"
        elif direction == "short" and current_candle["low"] <= take_profit:
            tp_hit = True
            exit_price = max(current_candle["low"], take_profit)  # Assume we get out at or slightly worse than target
            exit_reason = "take_profit"
        
        # Close position if stop-loss or take-profit hit
        if stop_hit or tp_hit:
            self._close_position(
                trade_id=trade_id,
                timestamp=current_candle["timestamp"],
                exit_price=exit_price,
                exit_reason=exit_reason,
                fee_percentage=self.config.get("backtest", {}).get("fee_percentage", 0.0005),
                slippage_percentage=self.config.get("backtest", {}).get("slippage_percentage", 0.001)
            )
            return True
        
        return False
    
    def _close_position(self, trade_id: str, timestamp, exit_price: float, 
                       exit_reason: str, fee_percentage: float, slippage_percentage: float):
        """
        Close a position.
        
        Args:
            trade_id: Position trade ID
            timestamp: Current timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            fee_percentage: Fee percentage
            slippage_percentage: Slippage percentage
        """
        # Check if position exists
        if trade_id not in self.positions:
            logger.warning(f"Tried to close non-existent position: {trade_id}")
            return
        
        # Get position details
        position = self.positions[trade_id]
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]
        leverage = position["leverage"]
        margin_amount = position["margin_amount"]
        
        # Apply slippage to exit price
        if direction == "long":
            # Sell lower with slippage
            exit_price *= (1 - slippage_percentage)
        else:
            # Buy higher with slippage
            exit_price *= (1 + slippage_percentage)
        
        # Calculate profit/loss
        if direction == "long":
            price_change_pct = (exit_price / entry_price) - 1
        else:
            price_change_pct = (entry_price / exit_price) - 1
            
        # Apply leverage
        pnl_pct = price_change_pct * leverage
        
        # Calculate absolute PnL
        pnl_amount = margin_amount * pnl_pct
        
        # Calculate fees
        exit_fee = size * exit_price * fee_percentage
        
        # Check for liquidation
        liquidation = False
        if pnl_pct <= -0.9:  # More than 90% loss indicates liquidation
            liquidation = True
            exit_reason = "liquidation"
            self.liquidations += 1
            logger.warning(f"Position {trade_id} liquidated with {pnl_pct:.2%} loss")
        
        # Return margin plus profit minus fees
        if liquidation:
            # In case of liquidation, we lose all margin
            self.cash += margin_amount * 0.1  # Assume we get back 10% of margin
        else:
            self.cash += margin_amount + pnl_amount - exit_fee
        
        # Record trade result
        trade_result = {
            "trade_id": trade_id,
            "pair": position["pair"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "size": size,
            "leverage": leverage,
            "margin_amount": margin_amount,
            "profit_loss": pnl_amount,
            "profit_loss_percentage": pnl_pct,
            "fees": position["fee_paid"] + exit_fee,
            "exit_reason": exit_reason,
            "strategy": position["strategy"],
            "risk_percentage": position["risk_percentage"]
        }
        
        # Add to trades history
        self.trades.append(trade_result)
        
        # Update metrics
        if pnl_amount > 0:
            self.winning_trades += 1
            self.total_profit += pnl_amount
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl_amount)
        
        # Remove from active positions
        del self.positions[trade_id]
        
        logger.info(f"Closed {direction} position at {exit_price:.2f} with " +
                  f"P/L: {pnl_amount:.2f} ({pnl_pct:.2%}), reason: {exit_reason}")
    
    def _update_portfolio_value(self, current_price: float):
        """
        Update portfolio value based on current positions and price.
        
        Args:
            current_price: Current market price
        """
        # Start with cash value
        portfolio_value = self.cash
        
        # Add unrealized PnL from open positions
        for trade_id, position in self.positions.items():
            entry_price = position["entry_price"]
            size = position["size"]
            leverage = position["leverage"]
            margin_amount = position["margin_amount"]
            direction = position["direction"]
            
            # Calculate unrealized PnL
            if direction == "long":
                price_change_pct = (current_price / entry_price) - 1
            else:
                price_change_pct = (entry_price / current_price) - 1
                
            # Apply leverage
            pnl_pct = price_change_pct * leverage
            pnl_amount = margin_amount * pnl_pct
            
            # Add margin plus unrealized profit
            portfolio_value += margin_amount + pnl_amount
        
        # Update portfolio value
        self.portfolio_value = portfolio_value
        
        # Update peak value and drawdown tracking
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown_duration = 0
            self.drawdown_start_value = portfolio_value
        else:
            self.current_drawdown_duration += 1
        
        # Update maximum drawdown duration
        if self.current_drawdown_duration > self.max_drawdown_duration:
            self.max_drawdown_duration = self.current_drawdown_duration
    
    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown percentage.
        
        Returns:
            Current drawdown as a percentage
        """
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            # Update max drawdown if current drawdown is larger
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                
            # Record significant drawdowns (>5%) with details
            if drawdown > 0.05 and len(self.drawdowns) == 0 or self.drawdowns[-1]["drawdown"] + 0.03 < drawdown:
                self.drawdowns.append({
                    "timestamp": datetime.now().isoformat(),
                    "drawdown": drawdown,
                    "portfolio_value": self.portfolio_value,
                    "peak_value": self.peak_portfolio_value,
                    "duration": self.current_drawdown_duration
                })
                
            return drawdown
        
        return 0.0
    
    def _calculate_backtest_metrics(self, daily_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            Dictionary with backtest metrics
        """
        # Calculate basic metrics
        win_rate = self.winning_trades / max(1, self.total_trades)
        total_return = (self.portfolio_value / self.initial_capital - 1) * 100  # percentage
        profit_factor = self.total_profit / max(0.0001, self.total_loss)  # Avoid division by zero
        
        # Calculate average trade metrics
        avg_profit = self.total_profit / max(1, self.winning_trades)
        avg_loss = self.total_loss / max(1, self.losing_trades)
        
        # Calculate risk-adjusted metrics
        if len(daily_returns) > 0:
            # Sharpe ratio (assuming risk-free rate of 0)
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / max(0.0001, np.std(returns_array)) * np.sqrt(252)
            
            # Sortino ratio (downside deviation only)
            negative_returns = returns_array[returns_array < 0]
            sortino_ratio = np.mean(returns_array) / max(0.0001, np.std(negative_returns)) * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Max drawdown already calculated and tracked
            
            # Calmar ratio
            calmar_ratio = (total_return / 100) / max(0.0001, self.max_drawdown) if self.max_drawdown > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        # Return all metrics
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "liquidations": self.liquidations,
            "avg_leverage": self.avg_leverage,
            "max_leverage": self.max_leverage,
            "risk_adjustments": self.risk_adjustments,
            "final_portfolio_value": self.portfolio_value
        }
    
    def _save_backtest_results(self, pair: str, stress_scenario: Optional[str], 
                             label: str, results: Dict[str, Any]):
        """
        Save backtest results to file.
        
        Args:
            pair: Trading pair
            stress_scenario: Stress scenario type (if applicable)
            label: Backtest label
            results: Backtest results
        """
        # Create filename
        scenario_suffix = f"_{stress_scenario}" if stress_scenario else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pair.replace('/', '_')}_{label}{scenario_suffix}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Create results object with metadata
        full_results = {
            "pair": pair,
            "label": label,
            "stress_scenario": stress_scenario,
            "timestamp": timestamp,
            "metrics": results,
            # Add position history
            "portfolio_history": [{
                "timestamp": str(entry["timestamp"]),
                "portfolio_value": entry["portfolio_value"],
                "cash": entry["cash"],
                "positions": entry["positions"],
                "drawdown": entry["drawdown"],
                "drawdown_duration": entry["drawdown_duration"]
            } for entry in self.portfolio_history],
            # Add trade history
            "trades": [{
                "trade_id": trade["trade_id"],
                "pair": trade["pair"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "entry_time": str(trade["entry_time"]),
                "exit_time": str(trade["exit_time"]),
                "size": trade["size"],
                "leverage": trade["leverage"],
                "profit_loss": trade["profit_loss"],
                "profit_loss_percentage": trade["profit_loss_percentage"],
                "exit_reason": trade["exit_reason"]
            } for trade in self.trades]
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Saved backtest results to {filepath}")
    
    def plot_backtest_results(self, pair: str, results_file: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            pair: Trading pair
            results_file: Optional specific results file to plot
        """
        # If no specific file provided, find latest for this pair
        if results_file is None:
            pair_prefix = pair.replace('/', '_')
            result_files = [f for f in os.listdir(self.results_dir) if f.startswith(pair_prefix) and f.endswith('.json')]
            
            if not result_files:
                logger.warning(f"No backtest results found for {pair}")
                return
                
            # Sort by timestamp (newest first)
            result_files.sort(reverse=True)
            results_file = os.path.join(self.results_dir, result_files[0])
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading results file {results_file}: {e}")
            return
        
        # Extract portfolio history
        portfolio_history = results.get("portfolio_history", [])
        if not portfolio_history:
            logger.warning(f"No portfolio history in results file {results_file}")
            return
            
        # Extract data for plotting
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in portfolio_history]
        portfolio_values = [entry["portfolio_value"] for entry in portfolio_history]
        drawdowns = [entry["drawdown"] for entry in portfolio_history]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot portfolio value
        ax1.plot(timestamps, portfolio_values, 'b-', linewidth=1.5)
        ax1.set_title(f"Backtest Results for {pair}")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True, alpha=0.3)
        
        # Add key metrics as text
        metrics = results.get("metrics", {})
        metrics_text = (
            f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2f}\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Liquidations: {metrics.get('liquidations', 0)}"
        )
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot drawdown
        ax2.fill_between(timestamps, [0] * len(drawdowns), drawdowns, color='red', alpha=0.3)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        # Add trades as markers
        trades = results.get("trades", [])
        for trade in trades:
            entry_time = datetime.fromisoformat(trade["entry_time"])
            exit_time = datetime.fromisoformat(trade["exit_time"])
            pnl = trade["profit_loss"]
            
            # Find portfolio value at these times
            entry_idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - entry_time).total_seconds()))
            exit_idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - exit_time).total_seconds()))
            
            entry_portfolio = portfolio_values[entry_idx]
            exit_portfolio = portfolio_values[exit_idx]
            
            # Plot entry and exit points
            marker_color = 'green' if pnl > 0 else 'red'
            ax1.plot(entry_time, entry_portfolio, 'v', color='blue', markersize=4, alpha=0.7)
            ax1.plot(exit_time, exit_portfolio, 'o', color=marker_color, markersize=4, alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = results_file.replace('.json', '.png')
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved backtest plot to {plot_path}")
        
        plt.close(fig)


def example_strategy(data, parameters):
    """
    Example strategy function for testing.
    
    Args:
        data: Historical price data
        parameters: Strategy parameters
        
    Returns:
        Trading signal dictionary
    """
    # This is just a simplistic example
    close = data["close"].values
    
    # Simple moving average crossover
    short_window = parameters.get("short_window", 5)
    long_window = parameters.get("long_window", 20)
    
    # Calculate moving averages
    if len(close) < long_window:
        return {"action": "hold", "confidence": 0.0, "strength": 0.0}
        
    short_ma = np.mean(close[-short_window:])
    long_ma = np.mean(close[-long_window:])
    
    # Generate signal
    if short_ma > long_ma:
        # Bullish signal
        signal_strength = min(1.0, (short_ma / long_ma - 1) * 20)  # Scale strength
        return {
            "action": "buy", 
            "confidence": 0.6,  # Fixed confidence for example
            "strength": max(0.1, signal_strength),
            "strategy": "example"
        }
    elif short_ma < long_ma:
        # Bearish signal
        signal_strength = min(1.0, (1 - short_ma / long_ma) * 20)  # Scale strength
        return {
            "action": "sell", 
            "confidence": 0.6,  # Fixed confidence for example
            "strength": max(0.1, signal_strength),
            "strategy": "example"
        }
    
    # No signal
    return {"action": "hold", "confidence": 0.0, "strength": 0.0}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Risk-enhanced backtester")
    parser.add_argument("--pair", type=str, default="SOL/USD", help="Trading pair")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--stress", type=str, choices=["flash_crash", "volatility_spike", "sustained_downtrend"], 
                        help="Run stress test scenario")
    parser.add_argument("--plot", action="store_true", help="Plot backtest results")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create backtester
    backtester = RiskEnhancedBacktester()
    
    # Fetch historical data
    data = backtester.fetch_historical_data(args.pair, days=args.days)
    
    if data is None or len(data) == 0:
        logger.error("No historical data available")
        return 1
    
    # Example parameters
    parameters = {
        "short_window": 5,
        "long_window": 20,
        "risk_percentage": 0.2,
        "leverage": 20.0,
        "stop_loss_percentage": 0.05,
        "take_profit_percentage": 0.1
    }
    
    # Run backtest with risk management
    results = backtester.run_backtest(
        pair=args.pair,
        data=data,
        strategy_function=example_strategy,
        parameters=parameters,
        stress_scenario=args.stress,
        label="risk_managed"
    )
    
    # Print summary metrics
    print("\n" + "=" * 80)
    print(f"RISK-ENHANCED BACKTEST RESULTS FOR {args.pair}")
    print("=" * 80)
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Max Drawdown Duration: {results['max_drawdown_duration']} periods")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    print(f"Liquidations: {results['liquidations']}")
    print(f"Average Leverage: {results['avg_leverage']:.2f}x")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print("=" * 80 + "\n")
    
    # Plot results if requested
    if args.plot:
        backtester.plot_backtest_results(args.pair)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())