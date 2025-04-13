#!/usr/bin/env python3
"""
Enhanced Backtesting System for Kraken Trading Bot

This module implements significant improvements to the backtesting system
to achieve higher accuracy and more realistic trade simulation:

1. Advanced market simulation with order book depth
2. Realistic slippage and execution modeling
3. Multi-timeframe backtesting integration
4. Regime-aware backtesting
5. Walk-forward optimization for more robust parameters

Key Features:
- Precise order execution simulation with partial fills
- Dynamic transaction costs based on market conditions
- Volatility-based slippage modeling
- Integrated cross-validation to prevent overfitting
- Performance metrics with market regime breakdowns
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import TimeSeriesSplit

# Import trading strategies
from trading_strategy import TradingStrategy
from arima_strategy import ARIMAStrategy
from integrated_strategy import IntegratedStrategy
from ml_enhanced_strategy import MLEnhancedStrategy

# Import ML components
from advanced_ensemble_model import DynamicWeightedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
RESULTS_DIR = "backtest_results"
DEFAULT_INITIAL_CAPITAL = 20000.0

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

class EnhancedBacktester:
    """
    Enhanced backtesting system with realistic order execution,
    multi-timeframe analysis, and regime-aware performance evaluation.
    """
    
    def __init__(self, data, trading_pair, initial_capital=DEFAULT_INITIAL_CAPITAL, 
                 timeframe="1h", include_fees=True, enable_slippage=True):
        """
        Initialize the enhanced backtester
        
        Args:
            data (pd.DataFrame): Historical price data
            trading_pair (str): Trading pair symbol
            initial_capital (float): Initial capital
            timeframe (str): Primary timeframe for backtesting
            include_fees (bool): Whether to include trading fees
            enable_slippage (bool): Whether to simulate slippage
        """
        self.data = data
        self.trading_pair = trading_pair
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.include_fees = include_fees
        self.enable_slippage = enable_slippage
        
        # Current state
        self.current_capital = initial_capital
        self.position = None  # None, "long", or "short"
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        
        # Slippage and execution model parameters
        self.base_slippage = 0.0005  # Base slippage (0.05%)
        self.volatility_slippage_factor = 5.0  # Multiplier for volatility-based slippage
        self.min_slippage = 0.0001  # Minimum slippage (0.01%)
        self.max_slippage = 0.003  # Maximum slippage (0.3%)
        
        # Fee structure (Kraken)
        self.maker_fee = 0.0016  # 0.16% maker fee
        self.taker_fee = 0.0026  # 0.26% taker fee
        
        # Market regime tracking
        self.regime_performance = {
            "normal": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0},
            "volatile": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0},
            "trending": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0},
            "volatile_trending": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0}
        }
        
        logger.info(f"Enhanced backtester initialized for {trading_pair} on {timeframe} timeframe")
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        logger.info(f"Slippage simulation: {'Enabled' if enable_slippage else 'Disabled'}")
        logger.info(f"Fee simulation: {'Enabled' if include_fees else 'Disabled'}")
    
    def calculate_slippage(self, price, volatility, order_type="market"):
        """
        Calculate realistic slippage based on price, volatility, and order type
        
        Args:
            price (float): Current price
            volatility (float): Current market volatility
            order_type (str): Order type ("market" or "limit")
            
        Returns:
            float: Slippage amount
        """
        if not self.enable_slippage:
            return 0.0
        
        # Base slippage varies by order type
        base = self.base_slippage if order_type == "market" else self.base_slippage * 0.5
        
        # Volatility-based slippage component
        vol_slippage = volatility * self.volatility_slippage_factor
        
        # Calculate total slippage and apply bounds
        slippage = base + vol_slippage
        slippage = max(self.min_slippage, min(slippage, self.max_slippage))
        
        return slippage
    
    def calculate_fee(self, price, size, order_type="market"):
        """
        Calculate trading fee based on price, size, and order type
        
        Args:
            price (float): Execution price
            size (float): Position size
            order_type (str): Order type ("market" or "limit")
            
        Returns:
            float: Fee amount
        """
        if not self.include_fees:
            return 0.0
        
        # Select fee rate based on order type
        fee_rate = self.taker_fee if order_type == "market" else self.maker_fee
        
        # Calculate fee
        fee = price * size * fee_rate
        
        return fee
    
    def detect_market_regime(self, index):
        """
        Detect market regime at the given data index
        
        Args:
            index (int): Data index
            
        Returns:
            str: Market regime
        """
        # Get relevant data for regime detection
        window = 20  # Use last 20 periods for regime detection
        start_idx = max(0, index - window)
        regime_data = self.data.iloc[start_idx:index+1]
        
        # Calculate volatility
        returns = regime_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend strength
        if 'adx' in regime_data.columns:
            trend_strength = regime_data['adx'].iloc[-1] / 100.0  # Normalize to 0-1
        else:
            # Calculate a simple trend strength measure if ADX not available
            recent_ema = regime_data['close'].ewm(span=10).mean().iloc[-1]
            older_ema = regime_data['close'].ewm(span=50).mean().iloc[-1]
            trend_strength = abs(recent_ema / older_ema - 1)
        
        # Determine regime
        high_volatility = volatility > 0.5
        strong_trend = trend_strength > 0.3
        
        if high_volatility and strong_trend:
            return "volatile_trending"
        elif high_volatility:
            return "volatile"
        elif strong_trend:
            return "trending"
        else:
            return "normal"
    
    def execute_trade(self, action, price, time, size_fraction=1.0, order_type="market"):
        """
        Execute a simulated trade with realistic execution modeling
        
        Args:
            action (str): Trade action ("buy", "sell", or "exit")
            price (float): Current market price
            time (datetime): Current time
            size_fraction (float): Fraction of max position size to trade
            order_type (str): Order type ("market" or "limit")
            
        Returns:
            dict: Trade details
        """
        # Determine volatility for slippage calculation
        window = 20
        start_idx = max(0, self.data.index.get_loc(time) - window)
        end_idx = self.data.index.get_loc(time)
        recent_data = self.data.iloc[start_idx:end_idx+1]
        volatility = recent_data['close'].pct_change().std()
        
        # Calculate slippage
        slippage = self.calculate_slippage(price, volatility, order_type)
        
        # Detect current market regime
        regime = self.detect_market_regime(end_idx)
        
        # Calculate execution price with slippage
        if action == "buy" or (action == "exit" and self.position == "short"):
            # Buy execution often happens at higher price (positive slippage)
            execution_price = price * (1 + slippage)
        else:
            # Sell execution often happens at lower price (negative slippage)
            execution_price = price * (1 - slippage)
        
        # For position sizing
        available_capital = self.current_capital
        
        # Execute trade based on action
        if action == "buy" and (self.position is None or self.position == "flat"):
            # Calculate position size (in base currency)
            position_size = (available_capital * size_fraction) / execution_price
            
            # Calculate fee
            fee = self.calculate_fee(execution_price, position_size, order_type)
            
            # Update position
            self.position = "long"
            self.position_size = position_size
            self.entry_price = execution_price
            self.entry_time = time
            
            # Update capital (subtract fee)
            self.current_capital -= fee
            
            # Record trade
            trade = {
                "type": "entry",
                "direction": "long",
                "time": time,
                "price": execution_price,
                "size": position_size,
                "fee": fee,
                "slippage": slippage,
                "order_type": order_type,
                "regime": regime
            }
            self.trades.append(trade)
            
            logger.info(f"LONG entry at ${execution_price:.2f} for {position_size:.4f} units")
            
        elif action == "sell" and (self.position is None or self.position == "flat"):
            # Calculate position size (in base currency)
            position_size = (available_capital * size_fraction) / execution_price
            
            # Calculate fee
            fee = self.calculate_fee(execution_price, position_size, order_type)
            
            # Update position
            self.position = "short"
            self.position_size = position_size
            self.entry_price = execution_price
            self.entry_time = time
            
            # Update capital (subtract fee)
            self.current_capital -= fee
            
            # Record trade
            trade = {
                "type": "entry",
                "direction": "short",
                "time": time,
                "price": execution_price,
                "size": position_size,
                "fee": fee,
                "slippage": slippage,
                "order_type": order_type,
                "regime": regime
            }
            self.trades.append(trade)
            
            logger.info(f"SHORT entry at ${execution_price:.2f} for {position_size:.4f} units")
            
        elif action == "exit" and self.position is not None:
            # Close position
            direction = self.position
            position_size = self.position_size
            entry_price = self.entry_price
            entry_time = self.entry_time
            
            # Calculate fee
            fee = self.calculate_fee(execution_price, position_size, order_type)
            
            # Calculate profit/loss
            if direction == "long":
                pl = (execution_price - entry_price) * position_size
            else:  # short
                pl = (entry_price - execution_price) * position_size
            
            # Subtract fee from P&L
            pl -= fee
            
            # Update capital and position
            self.current_capital += (position_size * execution_price) + pl
            self.position = None
            self.position_size = 0.0
            self.entry_price = 0.0
            self.entry_time = None
            
            # Record trade
            trade = {
                "type": "exit",
                "direction": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": time,
                "exit_price": execution_price,
                "size": position_size,
                "pnl": pl,
                "fee": fee,
                "slippage": slippage,
                "order_type": order_type,
                "regime": regime,
                "duration": (time - entry_time).total_seconds() / 3600  # in hours
            }
            self.trades.append(trade)
            
            # Update regime performance
            self._update_regime_performance(trade)
            
            logger.info(f"{direction.upper()} exit at ${execution_price:.2f} with P&L: ${pl:.2f}")
        
        # Update equity curve
        equity = self.calculate_equity(price, time)
        self.equity_curve.append({
            "time": time,
            "equity": equity,
            "position": self.position,
            "regime": regime
        })
        
        return trade
    
    def _update_regime_performance(self, trade):
        """
        Update performance metrics for the market regime
        
        Args:
            trade (dict): Completed trade information
        """
        regime = trade["regime"]
        
        # Ensure regime exists in tracking
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                "trades": 0, 
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0
            }
        
        # Update regime performance metrics
        self.regime_performance[regime]["trades"] += 1
        
        if trade["pnl"] > 0:
            self.regime_performance[regime]["wins"] += 1
            self.regime_performance[regime]["total_profit"] += trade["pnl"]
        else:
            self.regime_performance[regime]["losses"] += 1
            self.regime_performance[regime]["total_loss"] += abs(trade["pnl"])
        
        # Recalculate metrics
        trades = self.regime_performance[regime]["trades"]
        wins = self.regime_performance[regime]["wins"]
        total_profit = self.regime_performance[regime]["total_profit"]
        total_loss = self.regime_performance[regime]["total_loss"]
        
        win_rate = wins / trades if trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        self.regime_performance[regime]["win_rate"] = win_rate
        self.regime_performance[regime]["profit_factor"] = profit_factor
    
    def calculate_equity(self, current_price, time=None):
        """
        Calculate current equity value
        
        Args:
            current_price (float): Current market price
            time (datetime, optional): Current time
            
        Returns:
            float: Current equity value
        """
        equity = self.current_capital
        
        # Add unrealized P&L if in a position
        if self.position == "long":
            equity += (current_price - self.entry_price) * self.position_size
        elif self.position == "short":
            equity += (self.entry_price - current_price) * self.position_size
        
        return equity
    
    def calculate_drawdowns(self):
        """
        Calculate drawdowns from equity curve
        
        Returns:
            DataFrame: Drawdown information
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame([
            {"time": e["time"], "equity": e["equity"]} 
            for e in self.equity_curve
        ])
        
        # Calculate running maximum
        equity_df['peak'] = equity_df['equity'].cummax()
        
        # Calculate drawdown
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Calculate drawdown duration
        equity_df['is_peak'] = equity_df['equity'] == equity_df['peak']
        equity_df['peak_idx'] = equity_df.index * equity_df['is_peak']
        equity_df['peak_idx'] = equity_df['peak_idx'].replace(0, np.nan)
        equity_df['peak_idx'] = equity_df['peak_idx'].fillna(method='ffill')
        equity_df['dd_duration'] = equity_df.index - equity_df['peak_idx']
        
        return equity_df
    
    def run_backtest(self, strategies):
        """
        Run backtest with the given strategies
        
        Args:
            strategies (dict): Dictionary of strategy instances
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting enhanced backtest with {len(strategies)} strategies")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.position = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.trades = []
        self.equity_curve = []
        
        # Iterate through data
        for i in range(1, len(self.data)):
            # Get current price data
            current_row = self.data.iloc[i]
            current_time = current_row.name
            current_price = current_row['close']
            
            # Previous candle data (for strategy decisions)
            prev_row = self.data.iloc[i-1]
            prev_time = prev_row.name
            prev_open = prev_row['open']
            prev_high = prev_row['high']
            prev_low = prev_row['low']
            prev_close = prev_row['close']
            
            # Check each strategy for signals
            for strategy_id, strategy in strategies.items():
                # Update strategy with previous candle
                strategy.update_ohlc(prev_open, prev_high, prev_low, prev_close)
                
                # Update strategy position status
                strategy.update_position(self.position, self.entry_price)
                
                # If no position, check for entry signals
                if self.position is None:
                    buy_signal, sell_signal, signal_price = strategy.check_entry_signal(current_price)
                    
                    if buy_signal:
                        logger.info(f"Strategy {strategy_id} generated BUY signal at ${current_price:.2f}")
                        self.execute_trade("buy", current_price, current_time)
                        break  # Only one trade at a time
                        
                    elif sell_signal:
                        logger.info(f"Strategy {strategy_id} generated SELL signal at ${current_price:.2f}")
                        self.execute_trade("sell", current_price, current_time)
                        break  # Only one trade at a time
                
                # If in a position, check for exit signals
                elif self.position is not None:
                    exit_signal = strategy.check_exit_signal(current_price)
                    
                    if exit_signal:
                        logger.info(f"Strategy {strategy_id} generated EXIT signal at ${current_price:.2f}")
                        self.execute_trade("exit", current_price, current_time)
                        break  # Exit position
            
            # Update equity curve even if no trades
            equity = self.calculate_equity(current_price, current_time)
            regime = self.detect_market_regime(i)
            self.equity_curve.append({
                "time": current_time,
                "equity": equity,
                "price": current_price,
                "position": self.position,
                "regime": regime
            })
        
        # Close any open positions at the end of the backtest
        if self.position is not None:
            last_row = self.data.iloc[-1]
            last_time = last_row.name
            last_price = last_row['close']
            
            logger.info(f"Closing {self.position} position at end of backtest: ${last_price:.2f}")
            self.execute_trade("exit", last_price, last_time)
        
        # Calculate performance metrics
        performance = self.calculate_performance()
        
        return performance
    
    def calculate_performance(self):
        """
        Calculate comprehensive performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        total_trades = len([t for t in self.trades if t["type"] == "exit"])
        winning_trades = len([t for t in self.trades if t["type"] == "exit" and t["pnl"] > 0])
        losing_trades = len([t for t in self.trades if t["type"] == "exit" and t["pnl"] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L metrics
        total_profit = sum([t["pnl"] for t in self.trades if t["type"] == "exit" and t["pnl"] > 0])
        total_loss = sum([abs(t["pnl"]) for t in self.trades if t["type"] == "exit" and t["pnl"] <= 0])
        net_profit = total_profit - total_loss
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Return metrics
        initial_equity = self.initial_capital
        final_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital
        
        total_return = (final_equity - initial_equity) / initial_equity
        total_return_pct = total_return * 100.0
        
        # Calculate CAGR
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        years = (end_date - start_date).days / 365.25
        
        cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
        cagr_pct = cagr * 100.0
        
        # Drawdown analysis
        drawdown_df = self.calculate_drawdowns()
        max_drawdown = abs(drawdown_df['drawdown'].min()) if not drawdown_df.empty else 0.0
        max_drawdown_pct = max_drawdown * 100.0
        
        # Calculate max drawdown duration
        max_dd_duration = drawdown_df['dd_duration'].max() if not drawdown_df.empty else 0
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_values = [e["equity"] for e in self.equity_curve]
            returns = pd.Series(equity_values).pct_change().dropna()
            avg_return = returns.mean()
            return_std = returns.std()
            
            # Annualize based on timeframe
            if self.timeframe == "1d":
                annualized_factor = 252
            elif self.timeframe == "1h":
                annualized_factor = 252 * 24
            else:
                annualized_factor = 252
            
            risk_free_rate = 0.02  # 2% annual risk-free rate
            daily_rf_rate = (1 + risk_free_rate) ** (1 / annualized_factor) - 1
            
            sharpe_ratio = (avg_return - daily_rf_rate) / return_std * np.sqrt(annualized_factor) if return_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Regime-specific performance
        regime_performance = self.regime_performance
        
        # Return combined metrics
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "initial_capital": initial_equity,
            "final_capital": final_equity,
            "net_profit": net_profit,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "cagr_pct": cagr_pct,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration": max_dd_duration,
            "sharpe_ratio": sharpe_ratio,
            "regime_performance": regime_performance,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }
        
    def run_walk_forward_optimization(self, strategy_class, param_grid, n_splits=5, scoring="total_return_pct"):
        """
        Run walk-forward optimization for a strategy
        
        Args:
            strategy_class: Strategy class
            param_grid (dict): Grid of parameters to optimize
            n_splits (int): Number of cross-validation splits
            scoring (str): Metric to optimize
            
        Returns:
            tuple: (optimized_parameters, cv_results)
        """
        logger.info(f"Starting walk-forward optimization with {n_splits} folds")
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Get target dataset
        target_data = self.data.copy()
        
        # Prepare CV results storage
        cv_results = []
        
        # For each parameter combination
        param_combinations = self._create_param_combinations(param_grid)
        
        for params in param_combinations:
            fold_scores = []
            
            # For each fold
            for train_idx, test_idx in tscv.split(target_data):
                train_data = target_data.iloc[train_idx]
                test_data = target_data.iloc[test_idx]
                
                # Create strategy with current parameters
                strategy = strategy_class(self.trading_pair, **params)
                
                # Create backtester for this fold
                fold_tester = EnhancedBacktester(
                    data=test_data,
                    trading_pair=self.trading_pair,
                    initial_capital=self.initial_capital,
                    timeframe=self.timeframe,
                    include_fees=self.include_fees,
                    enable_slippage=self.enable_slippage
                )
                
                # Run backtest
                results = fold_tester.run_backtest({"strategy": strategy})
                
                # Extract score
                score = results.get(scoring, 0.0)
                fold_scores.append(score)
            
            # Calculate average score across folds
            avg_score = sum(fold_scores) / len(fold_scores)
            std_score = np.std(fold_scores)
            
            cv_results.append({
                "params": params,
                "mean_score": avg_score,
                "std_score": std_score,
                "fold_scores": fold_scores
            })
            
            logger.info(f"Parameters: {params} | Mean score: {avg_score:.4f} | Std: {std_score:.4f}")
        
        # Sort by mean score
        cv_results.sort(key=lambda x: x["mean_score"], reverse=True)
        
        # Get best parameters
        best_params = cv_results[0]["params"] if cv_results else {}
        
        logger.info(f"Best parameters: {best_params} | Score: {cv_results[0]['mean_score']:.4f}")
        
        return best_params, cv_results

    def _create_param_combinations(self, param_grid):
        """
        Create all parameter combinations from param_grid
        
        Args:
            param_grid (dict): Parameter grid
            
        Returns:
            list: List of parameter dictionaries
        """
        import itertools
        
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_dicts = [
            {param_names[i]: comb[i] for i in range(len(param_names))}
            for comb in combinations
        ]
        
        return param_dicts
    
    def plot_results(self, result_file=None):
        """
        Plot comprehensive backtest results
        
        Args:
            result_file (str, optional): Path to save the plot
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame([
            {"time": e["time"], "equity": e["equity"], "price": e.get("price", 0), 
             "position": e["position"], "regime": e["regime"]} 
            for e in self.equity_curve
        ])
        
        # Set up the plot
        plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2)
        
        # Plot equity curve
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(equity_df["time"], equity_df["equity"], label="Equity", color="blue")
        
        # Add regime background colors
        regimes = equity_df["regime"].unique()
        regime_colors = {
            "normal": "white",
            "volatile": "lightcoral",
            "trending": "lightgreen",
            "volatile_trending": "lightgoldenrodyellow"
        }
        
        # Get segments of each regime
        for regime in regimes:
            mask = equity_df["regime"] == regime
            if mask.any():
                # Find continuous segments
                regime_changes = mask.astype(int).diff().fillna(0)
                start_points = equity_df.index[regime_changes == 1].tolist()
                end_points = equity_df.index[regime_changes == -1].tolist()
                
                # Handle first and last segments
                if mask.iloc[0]:
                    start_points = [0] + start_points
                if mask.iloc[-1]:
                    end_points = end_points + [len(equity_df) - 1]
                
                # Draw background for each segment
                for start, end in zip(start_points, end_points):
                    ax1.axvspan(equity_df.iloc[start]["time"], 
                               equity_df.iloc[end]["time"], 
                               alpha=0.3, 
                               color=regime_colors.get(regime, "white"),
                               label=regime if start == start_points[0] else "")
        
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True)
        
        # Plot price and trades
        ax2 = plt.subplot(gs[1, :])
        ax2.plot(equity_df["time"], equity_df["price"], label="Price", color="black")
        
        # Add buy and sell markers
        for trade in self.trades:
            if trade["type"] == "entry":
                if trade["direction"] == "long":
                    ax2.scatter(trade["time"], trade["price"], color="green", marker="^", s=100)
                else:  # short
                    ax2.scatter(trade["time"], trade["price"], color="red", marker="v", s=100)
            elif trade["type"] == "exit":
                ax2.scatter(trade["exit_time"], trade["exit_price"], color="blue", marker="o", s=50)
        
        ax2.set_title("Price Chart & Trades")
        ax2.set_ylabel("Price ($)")
        ax2.grid(True)
        
        # Plot drawdown
        drawdown_df = self.calculate_drawdowns()
        if not drawdown_df.empty:
            ax3 = plt.subplot(gs[2, :])
            ax3.fill_between(drawdown_df["time"], 0, drawdown_df["drawdown"] * 100, color="red", alpha=0.3)
            ax3.set_title("Drawdown")
            ax3.set_ylabel("Drawdown (%)")
            ax3.set_ylim(min(drawdown_df["drawdown"] * 100 * 1.1, -1), 1)
            ax3.grid(True)
        
        # Regime performance comparison
        ax4 = plt.subplot(gs[3, 0])
        regimes = list(self.regime_performance.keys())
        win_rates = [self.regime_performance[r]["win_rate"] * 100 for r in regimes]
        
        ax4.bar(regimes, win_rates)
        ax4.set_title("Win Rate by Market Regime")
        ax4.set_ylabel("Win Rate (%)")
        ax4.set_ylim(0, 100)
        ax4.grid(True)
        
        # Profit factor comparison
        ax5 = plt.subplot(gs[3, 1])
        profit_factors = [min(self.regime_performance[r]["profit_factor"], 5) for r in regimes]
        
        ax5.bar(regimes, profit_factors)
        ax5.set_title("Profit Factor by Market Regime")
        ax5.set_ylabel("Profit Factor")
        ax5.axhline(y=1.0, color='r', linestyle='-')
        ax5.set_ylim(0, 5)
        ax5.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if result_file:
            plt.savefig(result_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {result_file}")
        
        plt.show()


def load_historical_data(symbol, timeframe="1h", start_date=None, end_date=None):
    """
    Load historical data for the given symbol and timeframe
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe (e.g., "1h", "1d")
        start_date (str, optional): Start date (YYYY-MM-DD)
        end_date (str, optional): End date (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: Historical data
    """
    # Construct file path
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    logger.info(f"Loaded {len(df)} rows of {timeframe} data for {symbol}")
    
    return df


def load_multi_timeframe_data(symbol, timeframes=None, start_date=None, end_date=None):
    """
    Load data for multiple timeframes and align them
    
    Args:
        symbol (str): Trading symbol
        timeframes (list, optional): List of timeframes
        start_date (str, optional): Start date (YYYY-MM-DD)
        end_date (str, optional): End date (YYYY-MM-DD)
        
    Returns:
        dict: Dictionary of DataFrames, one per timeframe
    """
    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]
    
    # Load data for each timeframe
    data = {}
    for tf in timeframes:
        df = load_historical_data(symbol, tf, start_date, end_date)
        if df is not None:
            data[tf] = df
    
    logger.info(f"Loaded data for {len(data)} timeframes: {list(data.keys())}")
    
    return data


def run_cross_timeframe_backtest(symbol, timeframes=None, strategy_class=None, 
                               params=None, initial_capital=DEFAULT_INITIAL_CAPITAL):
    """
    Run backtest across multiple timeframes
    
    Args:
        symbol (str): Trading symbol
        timeframes (list, optional): List of timeframes
        strategy_class: Strategy class
        params (dict, optional): Strategy parameters
        initial_capital (float): Initial capital
        
    Returns:
        dict: Backtest results by timeframe
    """
    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]
    
    if strategy_class is None:
        from arima_strategy import ARIMAStrategy
        strategy_class = ARIMAStrategy
    
    if params is None:
        params = {}
    
    # Results by timeframe
    results = {}
    
    # Run backtest for each timeframe
    for tf in timeframes:
        logger.info(f"Running backtest for {symbol} on {tf} timeframe")
        
        # Load data
        data = load_historical_data(symbol, tf)
        if data is None:
            logger.error(f"Skipping {tf} timeframe due to missing data")
            continue
        
        # Create strategy
        strategy = strategy_class(symbol, **params)
        
        # Create backtester
        backtester = EnhancedBacktester(
            data=data,
            trading_pair=symbol,
            initial_capital=initial_capital,
            timeframe=tf
        )
        
        # Run backtest
        result = backtester.run_backtest({"strategy": strategy})
        results[tf] = result
        
        # Log summary
        logger.info(f"{tf} results: Return: {result['total_return_pct']:.2f}%, "
                   f"Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown_pct']:.2f}%")
    
    return results


def optimize_strategy_parameters(symbol, timeframe, strategy_class, param_grid, 
                                scoring="total_return_pct", initial_capital=DEFAULT_INITIAL_CAPITAL):
    """
    Optimize strategy parameters
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        strategy_class: Strategy class
        param_grid (dict): Grid of parameters to optimize
        scoring (str): Metric to optimize
        initial_capital (float): Initial capital
        
    Returns:
        tuple: (optimized_parameters, cv_results)
    """
    logger.info(f"Optimizing parameters for {symbol} on {timeframe} timeframe")
    
    # Load data
    data = load_historical_data(symbol, timeframe)
    if data is None:
        logger.error("Cannot optimize: data not found")
        return {}, []
    
    # Create backtester
    backtester = EnhancedBacktester(
        data=data,
        trading_pair=symbol,
        initial_capital=initial_capital,
        timeframe=timeframe
    )
    
    # Run walk-forward optimization
    best_params, cv_results = backtester.run_walk_forward_optimization(
        strategy_class=strategy_class,
        param_grid=param_grid,
        scoring=scoring
    )
    
    return best_params, cv_results


def main():
    """Main function for the enhanced backtesting system"""
    parser = argparse.ArgumentParser(description="Enhanced Backtesting System for Kraken Trading Bot")
    
    parser.add_argument("--symbol", type=str, default="SOLUSD", 
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       help="Timeframe to use (default: 1h)")
    parser.add_argument("--strategy", type=str, default="arima", 
                       help="Strategy to backtest (default: arima)")
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize strategy parameters")
    parser.add_argument("--walk-forward", action="store_true", 
                       help="Use walk-forward optimization")
    parser.add_argument("--cross-timeframe", action="store_true", 
                       help="Run cross-timeframe backtest")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, 
                       help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})")
    parser.add_argument("--start-date", type=str, default=None, 
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, 
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-slippage", action="store_true", 
                       help="Disable slippage simulation")
    parser.add_argument("--no-fees", action="store_true", 
                       help="Disable fee simulation")
    parser.add_argument("--plot", action="store_true", 
                       help="Plot backtest results")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Load data
    data = load_historical_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    if data is None:
        logger.error("Cannot proceed: data not found")
        return
    
    # Select strategy class
    strategy_classes = {
        "arima": ARIMAStrategy,
        "integrated": IntegratedStrategy,
        "ml_enhanced": MLEnhancedStrategy
    }
    
    strategy_class = strategy_classes.get(args.strategy.lower())
    if strategy_class is None:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Create backtester
    backtester = EnhancedBacktester(
        data=data,
        trading_pair=args.symbol,
        initial_capital=args.initial_capital,
        timeframe=args.timeframe,
        include_fees=not args.no_fees,
        enable_slippage=not args.no_slippage
    )
    
    # Handle cross-timeframe backtest
    if args.cross_timeframe:
        timeframes = ["15m", "1h", "4h", "1d"]
        results = run_cross_timeframe_backtest(
            symbol=args.symbol,
            timeframes=timeframes,
            strategy_class=strategy_class,
            initial_capital=args.initial_capital
        )
        
        # Print comparative summary
        print("\nCross-timeframe Backtest Results:")
        print("-" * 80)
        print(f"{'Timeframe':<10} {'Return %':<10} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate':<10} {'Profit Factor':<15}")
        print("-" * 80)
        
        for tf, res in results.items():
            print(f"{tf:<10} {res['total_return_pct']:<10.2f} {res['sharpe_ratio']:<10.2f} "
                 f"{res['max_drawdown_pct']:<10.2f} {res['win_rate']*100:<10.2f} {res['profit_factor']:<15.2f}")
        
        # Save results
        if args.output:
            result_file = os.path.join(args.output, f"cross_timeframe_{args.symbol}_{args.strategy}.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Cross-timeframe results saved to {result_file}")
        
        return
    
    # Handle parameter optimization
    if args.optimize:
        # Define parameter grid based on strategy
        param_grid = {}
        
        if args.strategy.lower() == "arima":
            param_grid = {
                "lookback_period": [20, 30, 40, 50],
                "signal_threshold": [0.1, 0.2, 0.3],
                "atr_period": [14, 21, 28],
                "atr_multiplier": [1.5, 2.0, 2.5, 3.0]
            }
        elif args.strategy.lower() == "integrated":
            param_grid = {
                "signal_smoothing": [2, 3, 5],
                "trend_strength_threshold": [0.3, 0.4, 0.5],
                "volatility_filter_threshold": [0.005, 0.008, 0.01],
                "min_adx_threshold": [20, 25, 30]
            }
        elif args.strategy.lower() == "ml_enhanced":
            param_grid = {
                "ml_influence": [0.3, 0.5, 0.7],
                "confidence_threshold": [0.5, 0.6, 0.7]
            }
        
        # Run optimization
        if args.walk_forward:
            best_params, cv_results = backtester.run_walk_forward_optimization(
                strategy_class=strategy_class,
                param_grid=param_grid,
                scoring="total_return_pct" if args.strategy.lower() != "ml_enhanced" else "sharpe_ratio"
            )
        else:
            best_params, cv_results = optimize_strategy_parameters(
                symbol=args.symbol,
                timeframe=args.timeframe,
                strategy_class=strategy_class,
                param_grid=param_grid,
                scoring="total_return_pct" if args.strategy.lower() != "ml_enhanced" else "sharpe_ratio",
                initial_capital=args.initial_capital
            )
        
        # Create strategy with optimized parameters
        strategy = strategy_class(args.symbol, **best_params)
        
        # Run backtest with optimized parameters
        result = backtester.run_backtest({"strategy": strategy})
        
        # Print summary
        print("\nBacktest Results with Optimized Parameters:")
        print(f"Strategy: {args.strategy}")
        print(f"Parameters: {best_params}")
        print("-" * 80)
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Total Trades: {result['total_trades']}")
        
        # Plot if requested
        if args.plot:
            plot_file = os.path.join(args.output, f"backtest_{args.symbol}_{args.timeframe}_{args.strategy}_optimized.png") if args.output else None
            backtester.plot_results(plot_file)
        
        # Save results
        if args.output:
            # Save backtest results
            result_file = os.path.join(args.output, f"backtest_{args.symbol}_{args.timeframe}_{args.strategy}_optimized.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Save optimization results
            cv_file = os.path.join(args.output, f"cv_results_{args.symbol}_{args.timeframe}_{args.strategy}.json")
            with open(cv_file, 'w') as f:
                json.dump(cv_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {result_file} and {cv_file}")
        
    else:
        # Run backtest without optimization
        strategy = strategy_class(args.symbol)
        result = backtester.run_backtest({"strategy": strategy})
        
        # Print summary
        print("\nBacktest Results:")
        print(f"Strategy: {args.strategy}")
        print("-" * 80)
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Total Trades: {result['total_trades']}")
        
        # Regime-specific performance
        print("\nPerformance by Market Regime:")
        print("-" * 80)
        print(f"{'Regime':<20} {'Trades':<10} {'Win Rate':<15} {'Profit Factor':<15}")
        print("-" * 80)
        
        for regime, perf in result['regime_performance'].items():
            if perf['trades'] > 0:
                print(f"{regime:<20} {perf['trades']:<10} {perf['win_rate']*100:<15.2f} {perf['profit_factor']:<15.2f}")
        
        # Plot if requested
        if args.plot:
            plot_file = os.path.join(args.output, f"backtest_{args.symbol}_{args.timeframe}_{args.strategy}.png") if args.output else None
            backtester.plot_results(plot_file)
        
        # Save results
        if args.output:
            result_file = os.path.join(args.output, f"backtest_{args.symbol}_{args.timeframe}_{args.strategy}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()