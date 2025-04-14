#!/usr/bin/env python3
"""
Backtest Engine for Trading Strategies

This module implements a comprehensive backtesting engine for trading strategies.
It supports:
- Multiple trading strategies (ARIMA, Adaptive, ML-based)
- Dynamic parameter adjustment per trade
- Market regime detection
- Performance analytics
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies with various parameter sets
    and providing comprehensive performance metrics.
    """
    
    def __init__(self):
        """Initialize the backtest engine"""
        self.results_dir = "backtest_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Performance metrics
        self.metrics = {}
        
        # Trade history
        self.trades = []
        
        # Current market state
        self.market_state = {}
    
    def run_backtest(self, pair: str, data: pd.DataFrame, parameters: Dict[str, Any],
                    ml_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a backtest with the given parameters and data.
        
        Args:
            pair: Trading pair symbol
            data: Historical price data
            parameters: Dictionary of parameters to test
            ml_models: Optional dictionary of ML models to use
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Running backtest for {pair} with {len(data)} data points")
        
        # Reset trade history and metrics
        self.trades = []
        self.metrics = {}
        
        # Extract parameters
        risk_percentage = parameters.get("risk_percentage", 0.20)
        base_leverage = parameters.get("base_leverage", 20.0)
        max_leverage = parameters.get("max_leverage", 125.0)
        confidence_threshold = parameters.get("confidence_threshold", 0.65)
        signal_strength_threshold = parameters.get("signal_strength_threshold", 0.60)
        trailing_stop_atr_multiplier = parameters.get("trailing_stop_atr_multiplier", 3.0)
        exit_multiplier = parameters.get("exit_multiplier", 1.5)
        strategy_weights = parameters.get("strategy_weights", {"arima": 0.3, "adaptive": 0.7})
        
        # Starting capital for backtest
        starting_capital = 10000.0
        current_capital = starting_capital
        max_capital = starting_capital
        max_drawdown = 0.0
        
        # For recording portfolio value
        portfolio_values = []
        
        # Position tracking
        position = {
            "type": None,  # "long", "short", or None
            "entry_price": 0.0,
            "size": 0.0,
            "leverage": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "strategy": "",
            "timestamp": None
        }
        
        # Performance tracking for strategies
        strategy_performance = {
            "arima": {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0},
            "adaptive": {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
        }
        
        # ARIMA and Adaptive prediction tracking
        arima_predictions = {
            "correct": 0,
            "incorrect": 0,
            "signals": 0
        }
        
        adaptive_predictions = {
            "correct": 0,
            "incorrect": 0,
            "signals": 0
        }
        
        # Prepare the data - ensure it has all necessary columns
        data = self._prepare_data(data)
        
        # Calculate ATR for volatility stops (14-period ATR)
        data = self._calculate_atr(data, period=14)
        
        # Calculate EMAs for trend detection (50 and 100 period)
        data = self._calculate_emas(data)
        
        # Loop through the data points
        for i in range(50, len(data)):  # Start from 50 to have enough data for indicators
            current_row = data.iloc[i]
            previous_row = data.iloc[i-1]
            
            # Update current price and timestamp
            current_price = current_row["close"]
            timestamp = current_row.name if isinstance(current_row.name, datetime) else pd.Timestamp(current_row.name)
            
            # Update portfolio value
            if position["type"] is None:
                # No position, just track capital
                portfolio_values.append((timestamp, current_capital))
            else:
                # Calculate unrealized PnL
                if position["type"] == "long":
                    unrealized_pnl = (current_price / position["entry_price"] - 1) * position["size"] * position["leverage"]
                else:  # short
                    unrealized_pnl = (position["entry_price"] / current_price - 1) * position["size"] * position["leverage"]
                
                current_portfolio_value = current_capital + unrealized_pnl
                portfolio_values.append((timestamp, current_portfolio_value))
                
                # Update maximum capital
                max_capital = max(max_capital, current_portfolio_value)
                
                # Update drawdown
                current_drawdown = (max_capital - current_portfolio_value) / max_capital
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Check for stop loss or take profit
                if self._check_exit_conditions(position, current_price, current_row, trailing_stop_atr_multiplier):
                    # Close the position
                    trade_result = self._close_position(position, current_price, timestamp, "Exit trigger")
                    current_capital = trade_result["capital_after"]
                    
                    # Update strategy performance
                    strategy = position["strategy"]
                    strategy_performance[strategy]["trades"] += 1
                    if trade_result["profit"] > 0:
                        strategy_performance[strategy]["wins"] += 1
                    else:
                        strategy_performance[strategy]["losses"] += 1
                    strategy_performance[strategy]["profit"] += trade_result["profit"]
                    
                    # Reset position
                    position = {
                        "type": None,
                        "entry_price": 0.0,
                        "size": 0.0,
                        "leverage": 0.0,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "strategy": "",
                        "timestamp": None
                    }
            
            # If no position, check for entry signals
            if position["type"] is None:
                # Get prediction signals from different strategies
                arima_signal, arima_confidence = self._get_arima_signal(current_row, previous_row)
                adaptive_signal, adaptive_confidence = self._get_adaptive_signal(current_row, previous_row)
                ml_signal, ml_confidence = self._get_ml_signal(current_row, ml_models)
                
                # Track strategy predictions
                if arima_signal != "neutral":
                    arima_predictions["signals"] += 1
                if adaptive_signal != "neutral":
                    adaptive_predictions["signals"] += 1
                
                # Determine overall signal based on weighted confidence
                if ml_models and ml_confidence >= confidence_threshold:
                    # Use ML signal when available and confidence is high
                    overall_signal = ml_signal
                    signal_confidence = ml_confidence
                    signal_source = "ml"
                else:
                    # Weighted combination of ARIMA and Adaptive
                    arima_weight = strategy_weights.get("arima", 0.3)
                    adaptive_weight = strategy_weights.get("adaptive", 0.7)
                    
                    # Skip neutral signals
                    if arima_signal == "neutral" and adaptive_signal == "neutral":
                        overall_signal = "neutral"
                        signal_confidence = 0.0
                        signal_source = "combined"
                    elif arima_signal == "neutral":
                        overall_signal = adaptive_signal
                        signal_confidence = adaptive_confidence
                        signal_source = "adaptive"
                    elif adaptive_signal == "neutral":
                        overall_signal = arima_signal
                        signal_confidence = arima_confidence
                        signal_source = "arima"
                    # If signals disagree, use the one with higher weight
                    elif arima_signal != adaptive_signal:
                        if arima_confidence * arima_weight > adaptive_confidence * adaptive_weight:
                            overall_signal = arima_signal
                            signal_confidence = arima_confidence
                            signal_source = "arima"
                        else:
                            overall_signal = adaptive_signal
                            signal_confidence = adaptive_confidence
                            signal_source = "adaptive"
                    # If signals agree, combine confidences
                    else:
                        overall_signal = arima_signal  # Same as adaptive_signal
                        signal_confidence = (arima_confidence * arima_weight + 
                                           adaptive_confidence * adaptive_weight)
                        signal_source = "combined"
                
                # Check for entry conditions
                if overall_signal != "neutral" and signal_confidence >= signal_strength_threshold:
                    # Calculate dynamic leverage based on confidence
                    confidence_factor = (signal_confidence - signal_strength_threshold) / (1 - signal_strength_threshold)
                    dynamic_leverage = base_leverage + confidence_factor * (max_leverage - base_leverage)
                    dynamic_leverage = max(5.0, min(125.0, dynamic_leverage))
                    
                    # Calculate position size based on risk percentage
                    # Adjust risk percentage based on confidence
                    dynamic_risk = risk_percentage * (1 + confidence_factor * 0.5)
                    dynamic_risk = max(0.05, min(0.4, dynamic_risk))
                    
                    position_value = current_capital * dynamic_risk
                    position_size = position_value / current_price / dynamic_leverage
                    
                    # Set stop loss based on ATR
                    atr = current_row["atr"]
                    if overall_signal == "long":
                        stop_loss = current_price - (atr * trailing_stop_atr_multiplier)
                        take_profit = current_price + (atr * trailing_stop_atr_multiplier * exit_multiplier)
                    else:  # short
                        stop_loss = current_price + (atr * trailing_stop_atr_multiplier)
                        take_profit = current_price - (atr * trailing_stop_atr_multiplier * exit_multiplier)
                    
                    # Open position
                    position = {
                        "type": overall_signal,
                        "entry_price": current_price,
                        "size": position_size,
                        "leverage": dynamic_leverage,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "strategy": signal_source,
                        "timestamp": timestamp,
                        "initial_atr": atr
                    }
                    
                    # Log the entry
                    self.trades.append({
                        "pair": pair,
                        "timestamp": timestamp,
                        "type": overall_signal,
                        "entry_price": current_price,
                        "size": position_size,
                        "leverage": dynamic_leverage,
                        "strategy": signal_source,
                        "confidence": signal_confidence,
                        "status": "open",
                        "capital_before": current_capital
                    })
            
            # Update trailing stop if position is open
            elif position["type"] is not None:
                atr = current_row["atr"]
                
                # Update trailing stop based on ATR and price movement
                if position["type"] == "long" and current_price > position["entry_price"]:
                    # Move stop loss up for long positions in profit
                    new_stop = current_price - (atr * trailing_stop_atr_multiplier)
                    if new_stop > position["stop_loss"]:
                        position["stop_loss"] = new_stop
                
                elif position["type"] == "short" and current_price < position["entry_price"]:
                    # Move stop loss down for short positions in profit
                    new_stop = current_price + (atr * trailing_stop_atr_multiplier)
                    if new_stop < position["stop_loss"]:
                        position["stop_loss"] = new_stop
        
        # Close any open position at the end of the backtest
        if position["type"] is not None:
            final_price = data.iloc[-1]["close"]
            final_timestamp = data.iloc[-1].name if isinstance(data.iloc[-1].name, datetime) else pd.Timestamp(data.iloc[-1].name)
            
            trade_result = self._close_position(position, final_price, final_timestamp, "End of backtest")
            current_capital = trade_result["capital_after"]
            
            # Update strategy performance
            strategy = position["strategy"]
            strategy_performance[strategy]["trades"] += 1
            if trade_result["profit"] > 0:
                strategy_performance[strategy]["wins"] += 1
            else:
                strategy_performance[strategy]["losses"] += 1
            strategy_performance[strategy]["profit"] += trade_result["profit"]
        
        # Calculate final portfolio value
        final_portfolio_value = current_capital
        
        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.get("profit", 0) > 0)
        losing_trades = sum(1 for trade in self.trades if trade.get("profit", 0) <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade.get("profit", 0) for trade in self.trades)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        total_profit_winning = sum(trade.get("profit", 0) for trade in self.trades if trade.get("profit", 0) > 0)
        total_loss_losing = sum(abs(trade.get("profit", 0)) for trade in self.trades if trade.get("profit", 0) <= 0)
        
        avg_win = total_profit_winning / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss_losing / losing_trades if losing_trades > 0 else 0
        
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        profit_factor = total_profit_winning / total_loss_losing if total_loss_losing > 0 else float('inf')
        
        # Calculate returns and volatility
        returns = [t[1] for t in portfolio_values]
        returns = [returns[i] / returns[i-1] - 1 for i in range(1, len(returns))]
        
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(252 * 24)  # Annualized hourly volatility
            sharpe_ratio = (np.mean(returns) * (252 * 24)) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Calculate strategy-specific metrics
        for strategy, perf in strategy_performance.items():
            if perf["trades"] > 0:
                perf["win_rate"] = perf["wins"] / perf["trades"]
                perf["avg_profit_per_trade"] = perf["profit"] / perf["trades"]
            else:
                perf["win_rate"] = 0
                perf["avg_profit_per_trade"] = 0
        
        # Calculate ARIMA and Adaptive prediction accuracy
        arima_win_rate = (arima_predictions["correct"] / arima_predictions["signals"] 
                         if arima_predictions["signals"] > 0 else 0)
        
        adaptive_win_rate = (adaptive_predictions["correct"] / adaptive_predictions["signals"]
                           if adaptive_predictions["signals"] > 0 else 0)
        
        # Compile all metrics
        self.metrics = {
            "pair": pair,
            "parameter_set": parameters,
            "start_date": data.index[0],
            "end_date": data.index[-1],
            "total_days": (data.index[-1] - data.index[0]).days,
            "starting_capital": starting_capital,
            "final_capital": final_portfolio_value,
            "total_return": (final_portfolio_value / starting_capital) - 1,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_profit_per_trade": avg_profit_per_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "strategy_performance": strategy_performance,
            "arima_win_rate": arima_win_rate,
            "adaptive_win_rate": adaptive_win_rate
        }
        
        # Save backtest results
        self._save_backtest_results(pair, parameters)
        
        return self.metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting by ensuring it has all necessary columns.
        
        Args:
            data: Historical price data
            
        Returns:
            Prepared DataFrame
        """
        # Ensure we have required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure the index is a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by timestamp
        data = data.sort_index()
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range for the data.
        
        Args:
            data: Historical price data
            period: ATR period
            
        Returns:
            DataFrame with ATR column added
        """
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(period).mean()
        
        data["atr"] = atr
        
        return data
    
    def _calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages for the data.
        
        Args:
            data: Historical price data
            
        Returns:
            DataFrame with EMA columns added
        """
        data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema100"] = data["close"].ewm(span=100, adjust=False).mean()
        
        return data
    
    def _get_arima_signal(self, current_row: pd.Series, previous_row: pd.Series) -> Tuple[str, float]:
        """
        Get trading signal from ARIMA strategy.
        
        Args:
            current_row: Current price data
            previous_row: Previous price data
            
        Returns:
            Tuple of (signal, confidence)
        """
        # In a real implementation, this would be based on actual ARIMA predictions
        # Here we'll use a simplified approach based on price movement
        current_price = current_row["close"]
        ema50 = current_row["ema50"]
        ema100 = current_row["ema100"]
        
        # Trend direction
        trend = "neutral"
        if ema50 > ema100:
            trend = "bullish"
        elif ema50 < ema100:
            trend = "bearish"
        
        # Momentum
        prev_price = previous_row["close"]
        price_change = current_price / prev_price - 1
        
        signal = "neutral"
        confidence = 0.5
        
        if trend == "bullish" and price_change > 0:
            signal = "long"
            confidence = 0.6 + abs(price_change) * 10  # Higher change = higher confidence
        elif trend == "bearish" and price_change < 0:
            signal = "short"
            confidence = 0.6 + abs(price_change) * 10
        
        # Cap confidence at 0.95
        confidence = min(0.95, confidence)
        
        return signal, confidence
    
    def _get_adaptive_signal(self, current_row: pd.Series, previous_row: pd.Series) -> Tuple[str, float]:
        """
        Get trading signal from Adaptive strategy.
        
        Args:
            current_row: Current price data
            previous_row: Previous price data
            
        Returns:
            Tuple of (signal, confidence)
        """
        # In a real implementation, this would be based on actual adaptive algorithm
        # Here we'll use a different simplified approach based on price patterns
        current_price = current_row["close"]
        current_high = current_row["high"]
        current_low = current_row["low"]
        
        prev_price = previous_row["close"]
        prev_high = previous_row["high"]
        prev_low = previous_row["low"]
        
        # Volatility tracking
        atr = current_row["atr"]
        avg_price = (current_high + current_low + current_price) / 3
        volatility = atr / avg_price
        
        signal = "neutral"
        confidence = 0.5
        
        # Breakout pattern
        if current_high > prev_high and current_low > prev_low:
            signal = "long"
            confidence = 0.65 + volatility * 10
        elif current_high < prev_high and current_low < prev_low:
            signal = "short"
            confidence = 0.65 + volatility * 10
        
        # Cap confidence at 0.95
        confidence = min(0.95, confidence)
        
        return signal, confidence
    
    def _get_ml_signal(self, current_row: pd.Series, ml_models: Optional[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Get trading signal from ML models.
        
        Args:
            current_row: Current price data
            ml_models: Dictionary of ML models
            
        Returns:
            Tuple of (signal, confidence)
        """
        # In a real implementation, this would use the actual ML models to generate predictions
        # Here we'll return a neutral signal
        return "neutral", 0.0
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float, 
                             current_row: pd.Series, trailing_stop_multiplier: float) -> bool:
        """
        Check if exit conditions are met for the current position.
        
        Args:
            position: Current position details
            current_price: Current price
            current_row: Current price data
            trailing_stop_multiplier: Multiplier for trailing stop
            
        Returns:
            True if exit conditions are met, False otherwise
        """
        # Check for stop loss
        if position["type"] == "long" and current_price <= position["stop_loss"]:
            return True
        elif position["type"] == "short" and current_price >= position["stop_loss"]:
            return True
        
        # Check for take profit
        if position["type"] == "long" and current_price >= position["take_profit"]:
            return True
        elif position["type"] == "short" and current_price <= position["take_profit"]:
            return True
        
        # Check for very strong counter-signals
        # This would be implemented in a real system
        
        return False
    
    def _close_position(self, position: Dict[str, Any], exit_price: float, 
                      timestamp: datetime, exit_reason: str) -> Dict[str, Any]:
        """
        Close an open position and calculate profit/loss.
        
        Args:
            position: Position details
            exit_price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for exit
            
        Returns:
            Dictionary with trade result details
        """
        entry_price = position["entry_price"]
        position_type = position["type"]
        position_size = position["size"]
        leverage = position["leverage"]
        
        # Calculate profit/loss
        if position_type == "long":
            profit_percentage = (exit_price / entry_price - 1) * leverage
        else:  # short
            profit_percentage = (entry_price / exit_price - 1) * leverage
        
        profit = position_size * entry_price * profit_percentage
        
        # Find the corresponding open trade
        for trade in self.trades:
            if (trade["entry_price"] == entry_price and 
                trade["type"] == position_type and 
                trade["status"] == "open"):
                
                # Update the trade record
                trade["exit_price"] = exit_price
                trade["exit_timestamp"] = timestamp
                trade["profit_percentage"] = profit_percentage
                trade["profit"] = profit
                trade["status"] = "closed"
                trade["exit_reason"] = exit_reason
                trade["duration"] = (timestamp - trade["timestamp"]).total_seconds() / 3600  # Duration in hours
                
                # Calculate capital after trade
                capital_before = trade["capital_before"]
                capital_after = capital_before + profit
                trade["capital_after"] = capital_after
                
                break
        
        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_type": position_type,
            "profit_percentage": profit_percentage,
            "profit": profit,
            "exit_reason": exit_reason,
            "capital_after": capital_after
        }
    
    def _save_backtest_results(self, pair: str, parameters: Dict[str, Any]) -> None:
        """
        Save backtest results to file.
        
        Args:
            pair: Trading pair symbol
            parameters: Parameter set used for the backtest
        """
        pair_safe = pair.replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{pair_safe}_backtest_{timestamp}.json"
        
        params_hash = hash(frozenset(parameters.items()))
        
        results = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters,
            "parameters_hash": params_hash,
            "metrics": self.metrics,
            "trades": self.trades
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved backtest results to {filename}")
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def get_best_parameters(self, pair: str, metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Get best parameters for a pair based on a specific metric.
        
        Args:
            pair: Trading pair symbol
            metric: Metric to optimize
            
        Returns:
            Dictionary with best parameters
        """
        pair_safe = pair.replace('/', '_')
        files = [f for f in os.listdir(self.results_dir) if f.startswith(f"{pair_safe}_backtest_")]
        
        if not files:
            logger.warning(f"No backtest results found for {pair}")
            return {}
        
        best_value = float("-inf") if metric in ["sharpe_ratio", "total_return", "profit_factor"] else float("inf")
        best_params = {}
        
        for file in files:
            filepath = os.path.join(self.results_dir, file)
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    
                value = results.get("metrics", {}).get(metric)
                
                if value is None:
                    continue
                    
                if (metric in ["sharpe_ratio", "total_return", "profit_factor"] and value > best_value) or \
                   (metric not in ["sharpe_ratio", "total_return", "profit_factor"] and value < best_value):
                    best_value = value
                    best_params = results.get("parameters", {})
            except Exception as e:
                logger.error(f"Error reading backtest results from {filepath}: {e}")
        
        return best_params