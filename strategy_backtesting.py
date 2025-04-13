#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Kraken Trading Bot

This script implements a sophisticated backtesting framework that:
1. Tests the entire trading system using historical data
2. Optimizes strategy parameters based on backtesting results
3. Integrates ML models for prediction-based strategy testing
4. Provides detailed performance metrics and visualizations
5. Allows for ensemble strategy evaluation and comparison
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Import trading strategies
from trading_strategy import TradingStrategy
from arima_strategy import ARIMAStrategy
from integrated_strategy import IntegratedStrategy
from ml_enhanced_strategy import MLEnhancedStrategy

# Import ML components
from advanced_ensemble_model import DynamicWeightedEnsemble

# Setup logging
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
DEFAULT_TRADING_PAIR = "SOL/USD"
DEFAULT_TIMEFRAME = "1h"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


class MockKrakenExchange:
    """
    Mock implementation of Kraken Exchange API for backtesting.
    Simulates order execution, market data feeds, etc.
    """
    def __init__(self, historical_data: pd.DataFrame, trading_pair: str = "SOL/USD"):
        """
        Initialize the mock exchange with historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            trading_pair (str): Trading pair symbol
        """
        self.data = historical_data.copy()
        self.current_index = 0
        self.trading_pair = trading_pair
        self.current_price = self.data.iloc[0]['close']
        self.orders = []
        self.balances = {}
        self.filled_orders = []
        
        # Initialize exchange timestamp
        self.current_timestamp = self.data.iloc[0]['timestamp']
        if isinstance(self.current_timestamp, str):
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        
        # Extract base and quote currencies from trading pair
        parts = trading_pair.split('/')
        self.base_currency = parts[0]
        self.quote_currency = parts[1]
        
        # Initialize balances
        self.balances[self.quote_currency] = DEFAULT_INITIAL_CAPITAL
        self.balances[self.base_currency] = 0.0
    
    def advance_time(self) -> bool:
        """
        Advance to the next time period in the historical data.
        
        Returns:
            bool: True if successfully advanced, False if at the end of data
        """
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.current_price = self.data.iloc[self.current_index]['close']
            self.current_timestamp = self.data.iloc[self.current_index]['timestamp']
            if isinstance(self.current_timestamp, str):
                self.current_timestamp = pd.to_datetime(self.current_timestamp)
                
            # Process any pending orders
            self._process_orders()
            return True
        return False
    
    def get_current_price(self) -> float:
        """
        Get the current market price.
        
        Returns:
            float: Current price
        """
        return self.current_price
    
    def get_current_timestamp(self) -> datetime:
        """
        Get the current timestamp.
        
        Returns:
            datetime: Current timestamp
        """
        return self.current_timestamp
    
    def get_ohlc_data(self, lookback: int = 100) -> pd.DataFrame:
        """
        Get historical OHLC data up to the current time.
        
        Args:
            lookback (int): Number of periods to look back
            
        Returns:
            pd.DataFrame: OHLC data
        """
        start_idx = max(0, self.current_index - lookback + 1)
        return self.data.iloc[start_idx:self.current_index + 1].copy()
    
    def place_order(self, order_type: str, side: str, volume: float, 
                   price: Optional[float] = None, reduce_only: bool = False) -> Dict:
        """
        Place a new order in the mock exchange.
        
        Args:
            order_type (str): Type of order (market, limit)
            side (str): Buy or sell
            volume (float): Order quantity
            price (float, optional): Limit price (for limit orders)
            reduce_only (bool): Whether the order should only reduce position
            
        Returns:
            dict: Order details
        """
        order_id = f"order_{len(self.orders) + 1}"
        
        # For market orders, use current price
        if order_type.lower() == "market" or price is None:
            price = self.current_price
        
        order = {
            "id": order_id,
            "type": order_type.lower(),
            "side": side.lower(),
            "volume": volume,
            "price": price,
            "status": "pending",
            "timestamp": self.current_timestamp,
            "reduce_only": reduce_only
        }
        
        self.orders.append(order)
        
        # For market orders, execute immediately
        if order_type.lower() == "market":
            self._execute_order(order)
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id (str): ID of the order to cancel
            
        Returns:
            bool: True if cancelled, False if not found or already filled
        """
        for order in self.orders:
            if order["id"] == order_id and order["status"] == "pending":
                order["status"] = "cancelled"
                return True
        return False
    
    def get_balances(self) -> Dict:
        """
        Get current account balances.
        
        Returns:
            dict: Currency balances
        """
        return self.balances.copy()
    
    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for a symbol.
        
        Args:
            symbol (str): Symbol to check position for
            
        Returns:
            dict: Position details
        """
        base_currency = symbol.split('/')[0]
        return {
            "symbol": symbol,
            "size": self.balances.get(base_currency, 0),
            "entry_price": self._calculate_entry_price(symbol),
            "unrealized_pnl": self._calculate_unrealized_pnl(symbol)
        }
    
    def get_filled_orders(self) -> List[Dict]:
        """
        Get history of filled orders.
        
        Returns:
            list: Filled orders
        """
        return self.filled_orders.copy()
    
    def _process_orders(self) -> None:
        """Process any pending orders with the new price"""
        for order in self.orders:
            if order["status"] == "pending":
                # For limit orders, check if price conditions are met
                if order["type"] == "limit":
                    if (order["side"] == "buy" and self.current_price <= order["price"]) or \
                       (order["side"] == "sell" and self.current_price >= order["price"]):
                        self._execute_order(order)
    
    def _execute_order(self, order: Dict) -> None:
        """
        Execute an order (internal function).
        
        Args:
            order (dict): Order to execute
        """
        # Mark order as filled
        order["status"] = "filled"
        order["filled_price"] = self.current_price
        order["filled_timestamp"] = self.current_timestamp
        
        # Update balances
        if order["side"] == "buy":
            # Calculate the cost with a small fee (0.26% maker/taker fee)
            fee_rate = 0.0026
            cost = order["volume"] * self.current_price
            fee = cost * fee_rate
            total_cost = cost + fee
            
            # Update balances
            self.balances[self.quote_currency] -= total_cost
            self.balances[self.base_currency] += order["volume"]
            
        elif order["side"] == "sell":
            # Calculate the proceeds with fee
            fee_rate = 0.0026
            proceeds = order["volume"] * self.current_price
            fee = proceeds * fee_rate
            net_proceeds = proceeds - fee
            
            # Update balances
            self.balances[self.base_currency] -= order["volume"]
            self.balances[self.quote_currency] += net_proceeds
        
        # Add to filled orders
        self.filled_orders.append(order.copy())
    
    def _calculate_entry_price(self, symbol: str) -> float:
        """
        Calculate average entry price for a position.
        
        Args:
            symbol (str): Symbol to calculate for
            
        Returns:
            float: Average entry price, or 0 if no position
        """
        base_currency = symbol.split('/')[0]
        position_size = self.balances.get(base_currency, 0)
        
        if position_size <= 0:
            return 0
        
        # Calculate average entry from buy orders
        buy_orders = [o for o in self.filled_orders 
                      if o["side"] == "buy" and symbol in o["id"]]
        
        if not buy_orders:
            return 0
        
        total_cost = sum(o["volume"] * o["filled_price"] for o in buy_orders)
        total_volume = sum(o["volume"] for o in buy_orders)
        
        return total_cost / total_volume if total_volume > 0 else 0
    
    def _calculate_unrealized_pnl(self, symbol: str) -> float:
        """
        Calculate unrealized profit/loss for a position.
        
        Args:
            symbol (str): Symbol to calculate for
            
        Returns:
            float: Unrealized P&L
        """
        base_currency = symbol.split('/')[0]
        position_size = self.balances.get(base_currency, 0)
        
        if position_size <= 0:
            return 0
        
        entry_price = self._calculate_entry_price(symbol)
        if entry_price == 0:
            return 0
        
        return position_size * (self.current_price - entry_price)


class BacktestTrader:
    """
    Trader implementation for backtesting that works with strategies.
    """
    def __init__(self, exchange: MockKrakenExchange, trading_pair: str = "SOL/USD"):
        """
        Initialize the backtest trader.
        
        Args:
            exchange (MockKrakenExchange): Mock exchange instance
            trading_pair (str): Trading pair to trade
        """
        self.exchange = exchange
        self.trading_pair = trading_pair
        self.strategies = {}
        self.active_positions = {}
        self.trade_history = []
        self.portfolio_history = []
        self.initial_capital = DEFAULT_INITIAL_CAPITAL
        
        # Split the trading pair
        parts = trading_pair.split('/')
        self.base_currency = parts[0]
        self.quote_currency = parts[1]
        
        # Record initial state
        self._record_portfolio_state()
    
    def add_strategy(self, strategy: TradingStrategy, strategy_id: str) -> None:
        """
        Add a trading strategy to the backtester.
        
        Args:
            strategy (TradingStrategy): Strategy instance
            strategy_id (str): Unique ID for the strategy
        """
        self.strategies[strategy_id] = strategy
        self.active_positions[strategy_id] = None
    
    def update(self) -> None:
        """
        Run a single update cycle for the backtester.
        This processes all strategies and executes any trading signals.
        """
        current_price = self.exchange.get_current_price()
        current_time = self.exchange.get_current_timestamp()
        
        # Get the latest market data for the strategies
        ohlc_data = self.exchange.get_ohlc_data()
        
        # Get the latest candle data
        if len(ohlc_data) > 0:
            latest = ohlc_data.iloc[-1]
            open_price = latest['open']
            high_price = latest['high']
            low_price = latest['low']
            close_price = latest['close']
        else:
            # Fallback values if no data
            open_price = current_price
            high_price = current_price
            low_price = current_price
            close_price = current_price
        
        # Process each strategy
        for strategy_id, strategy in self.strategies.items():
            # Update the strategy with new data
            # Different strategies might have different update methods
            if hasattr(strategy, 'update'):
                strategy.update(ohlc_data)
            elif hasattr(strategy, 'update_ohlc'):
                strategy.update_ohlc(open_price, high_price, low_price, close_price)
            
            # Check for entry/exit signals
            if self.active_positions.get(strategy_id) is None:
                # No position, check for entry
                if strategy.should_buy():
                    size = strategy.calculate_position_size() if hasattr(strategy, 'calculate_position_size') else 1.0
                    self._execute_entry(strategy_id, "buy", size)
                elif strategy.should_sell():
                    size = strategy.calculate_position_size() if hasattr(strategy, 'calculate_position_size') else 1.0
                    self._execute_entry(strategy_id, "sell", size)
            else:
                # Have position, check for exit
                position = self.active_positions[strategy_id]
                
                # Different strategies might have different exit methods
                should_exit = False
                if position["side"] == "buy":
                    if hasattr(strategy, 'should_exit_long'):
                        should_exit = strategy.should_exit_long()
                    else:
                        # Fallback to standard should_exit if it exists
                        should_exit = strategy.should_exit() if hasattr(strategy, 'should_exit') else False
                elif position["side"] == "sell":
                    if hasattr(strategy, 'should_exit_short'):
                        should_exit = strategy.should_exit_short()
                    else:
                        # Fallback to standard should_exit if it exists
                        should_exit = strategy.should_exit() if hasattr(strategy, 'should_exit') else False
                
                if should_exit:
                    self._execute_exit(strategy_id)
        
        # Record the portfolio state after processing
        self._record_portfolio_state()
    
    def _execute_entry(self, strategy_id: str, side: str, size: float) -> None:
        """
        Execute an entry order for a strategy.
        
        Args:
            strategy_id (str): Strategy ID
            side (str): Buy or sell
            size (float): Position size
        """
        # Calculate the actual order size based on account balance
        balances = self.exchange.get_balances()
        available_capital = balances[self.quote_currency]
        
        # Use a maximum of 20% of capital per trade
        max_capital = self.initial_capital * 0.2
        capital_to_use = min(max_capital, available_capital)
        
        # Calculate order volume
        current_price = self.exchange.get_current_price()
        volume = capital_to_use / current_price
        
        # Place the order
        order = self.exchange.place_order(
            order_type="market",
            side=side,
            volume=volume,
            price=None
        )
        
        # Record the position
        self.active_positions[strategy_id] = {
            "id": order["id"],
            "side": side,
            "entry_price": current_price,
            "volume": volume,
            "entry_time": self.exchange.get_current_timestamp(),
            "strategy_id": strategy_id
        }
        
        # Log the trade
        logger.info(f"[BACKTEST] {strategy_id}: Entered {side.upper()} position of {volume:.6f} {self.base_currency} @ ${current_price:.2f}")
    
    def _execute_exit(self, strategy_id: str) -> None:
        """
        Execute an exit order for a strategy.
        
        Args:
            strategy_id (str): Strategy ID
        """
        position = self.active_positions[strategy_id]
        if position is None:
            return
        
        # Determine the exit side (opposite of entry)
        exit_side = "sell" if position["side"] == "buy" else "buy"
        
        # Place the exit order
        current_price = self.exchange.get_current_price()
        order = self.exchange.place_order(
            order_type="market",
            side=exit_side,
            volume=position["volume"],
            price=None
        )
        
        # Calculate profit/loss
        if position["side"] == "buy":
            pnl_percentage = (current_price - position["entry_price"]) / position["entry_price"] * 100
        else:
            pnl_percentage = (position["entry_price"] - current_price) / position["entry_price"] * 100
        
        pnl_amount = position["volume"] * abs(current_price - position["entry_price"])
        
        # Record the trade in history
        trade_record = {
            "strategy_id": strategy_id,
            "entry_side": position["side"],
            "entry_price": position["entry_price"],
            "entry_time": position["entry_time"],
            "exit_price": current_price,
            "exit_time": self.exchange.get_current_timestamp(),
            "volume": position["volume"],
            "pnl_percentage": pnl_percentage,
            "pnl_amount": pnl_amount
        }
        self.trade_history.append(trade_record)
        
        # Clear the position
        self.active_positions[strategy_id] = None
        
        # Log the trade
        logger.info(f"[BACKTEST] {strategy_id}: Exited position @ ${current_price:.2f}, PnL: {pnl_percentage:.2f}%")
    
    def _record_portfolio_state(self) -> None:
        """Record the current portfolio state for historical tracking"""
        balances = self.exchange.get_balances()
        current_price = self.exchange.get_current_price()
        current_time = self.exchange.get_current_timestamp()
        
        # Calculate total value (quote currency + base currency value)
        base_value = balances.get(self.base_currency, 0) * current_price
        total_value = balances.get(self.quote_currency, 0) + base_value
        
        # Record the state
        self.portfolio_history.append({
            "timestamp": current_time,
            "total_value": total_value,
            "quote_balance": balances.get(self.quote_currency, 0),
            "base_balance": balances.get(self.base_currency, 0),
            "price": current_price
        })
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            dict: Performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]["total_value"]
        
        # Create a DataFrame from portfolio history
        df = pd.DataFrame(self.portfolio_history)
        df['return'] = df['total_value'].pct_change().fillna(0)
        
        # Calculate metrics
        total_return = (final_value / initial_value - 1) * 100
        daily_returns = df.set_index('timestamp').resample('D')['return'].sum()
        annual_return = total_return / (len(daily_returns) / 365)
        sharpe_ratio = np.sqrt(365) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        # Calculate drawdowns
        df['peak'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['peak']) / df['peak'] * 100
        max_drawdown = df['drawdown'].min()
        
        # Calculate win rate
        wins = sum(1 for trade in self.trade_history if trade['pnl_percentage'] > 0)
        total_trades = len(self.trade_history)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade['pnl_amount'] for trade in self.trade_history if trade['pnl_percentage'] > 0)
        gross_loss = sum(trade['pnl_amount'] for trade in self.trade_history if trade['pnl_percentage'] <= 0)
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        return {
            "initial_capital": initial_value,
            "final_capital": final_value,
            "total_return_pct": total_return,
            "annualized_return_pct": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }
    
    def get_trade_history(self) -> List[Dict]:
        """
        Get the complete trade history.
        
        Returns:
            list: Trade history
        """
        return self.trade_history.copy()
    
    def get_portfolio_history(self) -> List[Dict]:
        """
        Get the portfolio value history.
        
        Returns:
            list: Portfolio history
        """
        return self.portfolio_history.copy()
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """
        Get performance metrics broken down by strategy.
        
        Returns:
            dict: Strategy performance metrics
        """
        strategy_metrics = {}
        
        for strategy_id in self.strategies.keys():
            # Filter trades for this strategy
            strategy_trades = [t for t in self.trade_history if t['strategy_id'] == strategy_id]
            
            if not strategy_trades:
                strategy_metrics[strategy_id] = {
                    "total_trades": 0,
                    "win_rate_pct": 0,
                    "avg_profit_pct": 0,
                    "total_profit_loss": 0
                }
                continue
            
            # Calculate metrics
            wins = sum(1 for trade in strategy_trades if trade['pnl_percentage'] > 0)
            total_trades = len(strategy_trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            avg_profit = sum(trade['pnl_percentage'] for trade in strategy_trades) / total_trades
            total_profit = sum(trade['pnl_amount'] for trade in strategy_trades)
            
            # Maximum consecutive wins and losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for trade in strategy_trades:
                if trade['pnl_percentage'] > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            strategy_metrics[strategy_id] = {
                "total_trades": total_trades,
                "win_rate_pct": win_rate,
                "avg_profit_pct": avg_profit,
                "total_profit_loss": total_profit,
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses
            }
        
        return strategy_metrics


def load_data(symbol: str = "SOLUSD", timeframe: str = "1h") -> pd.DataFrame:
    """
    Load historical data from CSV files.
    
    Args:
        symbol (str): Symbol to load data for
        timeframe (str): Timeframe to load
        
    Returns:
        pd.DataFrame: Historical data
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean data for backtesting.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Prepared data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def run_backtest(data: pd.DataFrame, strategies: Dict[str, TradingStrategy], 
                trading_pair: str = "SOL/USD") -> Tuple[Dict, BacktestTrader]:
    """
    Run a backtest with the provided data and strategies.
    
    Args:
        data (pd.DataFrame): Historical price data
        strategies (dict): Dictionary of strategy instances
        trading_pair (str): Trading pair symbol
        
    Returns:
        tuple: (Performance metrics dict, BacktestTrader instance)
    """
    # Create the mock exchange with historical data
    exchange = MockKrakenExchange(data, trading_pair)
    
    # Create the backtest trader
    trader = BacktestTrader(exchange, trading_pair)
    
    # Add strategies to the trader
    for strategy_id, strategy in strategies.items():
        trader.add_strategy(strategy, strategy_id)
    
    # Run the backtest
    start_time = time.time()
    total_bars = len(data)
    progress_step = max(1, total_bars // 100)  # Update progress every 1%
    
    logger.info(f"Starting backtest with {total_bars} bars of data...")
    
    # Initialize strategies with initial lookback data
    # Advance some periods to allow indicators to initialize
    for _ in range(min(100, total_bars // 10)):
        exchange.advance_time()
    
    # Run the main backtest loop
    bar_count = 0
    while exchange.advance_time():
        bar_count += 1
        trader.update()
        
        # Show progress
        if bar_count % progress_step == 0:
            progress = bar_count / total_bars * 100
            elapsed = time.time() - start_time
            est_total = elapsed / progress * 100 if progress > 0 else 0
            est_remaining = est_total - elapsed
            
            logger.info(f"Progress: {progress:.1f}% | "
                       f"Elapsed: {elapsed:.1f}s | "
                       f"Est. remaining: {est_remaining:.1f}s")
    
    # Calculate final performance metrics
    performance = trader.get_performance_metrics()
    strategy_performance = trader.get_strategy_performance()
    
    # Combine metrics
    performance['strategy_performance'] = strategy_performance
    
    # Log summary results
    logger.info("Backtest completed successfully.")
    logger.info(f"Total Return: {performance['total_return_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {performance['win_rate_pct']:.2f}%")
    logger.info(f"Total Trades: {performance['total_trades']}")
    
    for strategy_id, metrics in strategy_performance.items():
        if metrics['total_trades'] > 0:
            logger.info(f"Strategy {strategy_id}: {metrics['win_rate_pct']:.2f}% win rate, "
                       f"{metrics['avg_profit_pct']:.2f}% avg profit")
    
    return performance, trader


def generate_charts(trader: BacktestTrader, symbol: str, timeframe: str, 
                   performance: Dict) -> None:
    """
    Generate performance charts from backtest results.
    
    Args:
        trader (BacktestTrader): Backtest trader instance
        symbol (str): Symbol that was tested
        timeframe (str): Timeframe that was tested
        performance (dict): Performance metrics
    """
    portfolio_history = trader.get_portfolio_history()
    trade_history = trader.get_trade_history()
    
    # Create DataFrame from history
    df_portfolio = pd.DataFrame(portfolio_history)
    df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])
    df_portfolio.set_index('timestamp', inplace=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. Portfolio Value Chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_portfolio.index, df_portfolio['total_value'], label='Portfolio Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Value (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # 2. Drawdown Chart
    ax2 = fig.add_subplot(gs[1, :])
    df_portfolio['peak'] = df_portfolio['total_value'].cummax()
    df_portfolio['drawdown'] = (df_portfolio['total_value'] - df_portfolio['peak']) / df_portfolio['peak'] * 100
    ax2.fill_between(df_portfolio.index, df_portfolio['drawdown'], 0, color='red', alpha=0.3)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True)
    
    # 3. Trade Win/Loss Chart
    ax3 = fig.add_subplot(gs[2, 0])
    
    if trade_history:
        df_trades = pd.DataFrame(trade_history)
        df_trades['profitable'] = df_trades['pnl_percentage'] > 0
        
        # Count by strategy
        strategy_counts = {
            strategy: {
                'win': sum(1 for _, row in df_trades[df_trades['strategy_id'] == strategy].iterrows() if row['profitable']),
                'loss': sum(1 for _, row in df_trades[df_trades['strategy_id'] == strategy].iterrows() if not row['profitable'])
            }
            for strategy in df_trades['strategy_id'].unique()
        }
        
        # Prepare data for grouped bar chart
        strategies = list(strategy_counts.keys())
        wins = [strategy_counts[s]['win'] for s in strategies]
        losses = [strategy_counts[s]['loss'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax3.bar(x - width/2, wins, width, label='Wins', color='green', alpha=0.6)
        ax3.bar(x + width/2, losses, width, label='Losses', color='red', alpha=0.6)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.set_title('Win/Loss by Strategy')
        ax3.set_ylabel('Number of Trades')
        ax3.legend()
        ax3.grid(True, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No trade data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes)
    
    # 4. Performance Metrics Table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    metrics_to_display = [
        ('Total Return', f"{performance['total_return_pct']:.2f}%"),
        ('Annual Return', f"{performance['annualized_return_pct']:.2f}%"),
        ('Sharpe Ratio', f"{performance['sharpe_ratio']:.2f}"),
        ('Max Drawdown', f"{performance['max_drawdown_pct']:.2f}%"),
        ('Win Rate', f"{performance['win_rate_pct']:.2f}%"),
        ('Total Trades', f"{performance['total_trades']}"),
        ('Profit Factor', f"{performance['profit_factor']:.2f}")
    ]
    
    table = ax4.table(cellText=metrics_to_display, loc='center', cellLoc='left')
    table.auto_set_column_width([0, 1])
    table.scale(1, 1.5)
    ax4.set_title('Performance Metrics')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(RESULTS_DIR, f'backtest_{symbol}_{timeframe}_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    logger.info(f"Performance chart saved to {output_path}")
    plt.close(fig)


def optimize_strategy_parameters(data: pd.DataFrame, strategy_class, 
                               param_grid: Dict[str, List], trading_pair: str = "SOL/USD",
                               metric: str = "total_return_pct") -> Dict:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        data (pd.DataFrame): Historical price data
        strategy_class: Strategy class to optimize
        param_grid (dict): Parameter grid to search
        trading_pair (str): Trading pair symbol
        metric (str): Metric to optimize for
        
    Returns:
        dict: Best parameters
    """
    # Generate all parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Optimizing parameters with {len(param_combinations)} combinations...")
    
    best_performance = -float('inf') if metric == 'total_return_pct' else float('inf')
    best_params = None
    
    # Iterate through parameter combinations
    for i, params in enumerate(param_combinations):
        param_dict = {name: value for name, value in zip(param_names, params)}
        
        # Create strategy with these parameters
        strategy = strategy_class(**param_dict)
        strategies = {"optimizer": strategy}
        
        # Run backtest
        performance, _ = run_backtest(data, strategies, trading_pair)
        
        # Check if this is better
        current_performance = performance[metric]
        is_better = (current_performance > best_performance) if metric == 'total_return_pct' else (current_performance < best_performance)
        
        if is_better:
            best_performance = current_performance
            best_params = param_dict
            
            logger.info(f"New best parameters found: {best_params}")
            logger.info(f"New best {metric}: {best_performance}")
    
    logger.info(f"Optimization complete.")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best {metric}: {best_performance}")
    
    return best_params


def main(args):
    """
    Main function to run the backtest system.
    
    Args:
        args: Command-line arguments
    """
    # Load data
    data = load_data(args.symbol, args.timeframe)
    if data is None:
        return
    
    # Prepare data
    data = prepare_data(data)
    
    # Initialize strategies
    strategies = {}
    
    if args.strategy.lower() == "arima":
        strategies["arima"] = ARIMAStrategy(args.pair)
        
    elif args.strategy.lower() == "integrated":
        strategies["integrated"] = IntegratedStrategy(args.pair)
        
    elif args.strategy.lower() == "ml":
        # Initialize ML model
        model = DynamicWeightedEnsemble(trading_pair=args.pair, timeframe=args.timeframe)
        strategies["ml_enhanced"] = MLEnhancedStrategy(args.pair, model)
        
    elif args.strategy.lower() == "ensemble":
        # Use all strategies
        strategies["arima"] = ARIMAStrategy(args.pair)
        strategies["integrated"] = IntegratedStrategy(args.pair)
        model = DynamicWeightedEnsemble(trading_pair=args.pair, timeframe=args.timeframe)
        strategies["ml_enhanced"] = MLEnhancedStrategy(args.pair, model)
    
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Check if we're optimizing
    if args.optimize:
        # Define parameter grid for the selected strategy
        if args.strategy.lower() == "arima":
            param_grid = {
                "lookback_period": [24, 32, 48],
                "atr_trailing_multiplier": [1.5, 2.0, 2.5, 3.0],
                "entry_atr_multiplier": [0.005, 0.01, 0.015, 0.02]
            }
            best_params = optimize_strategy_parameters(
                data, ARIMAStrategy, param_grid, args.pair, "total_return_pct"
            )
            # Recreate strategy with best parameters
            strategies = {"arima": ARIMAStrategy(args.pair, **best_params)}
            
        elif args.strategy.lower() == "integrated":
            param_grid = {
                "leverage": [20.0, 25.0, 30.0],
                "atr_period": [14, 21, 28],
                "fixed_stop_multiplier": [1.5, 2.0, 2.5],
                "entry_offset_multiplier": [0.005, 0.01, 0.015]
            }
            best_params = optimize_strategy_parameters(
                data, IntegratedStrategy, param_grid, args.pair, "total_return_pct"
            )
            # Recreate strategy with best parameters
            strategies = {"integrated": IntegratedStrategy(args.pair, **best_params)}
    
    # Run backtest
    performance, trader = run_backtest(data, strategies, args.pair)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f'backtest_{args.symbol}_{args.timeframe}_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(performance, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate charts
    generate_charts(trader, args.symbol, args.timeframe, performance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting System for Kraken Trading Bot")
    parser.add_argument("--symbol", type=str, default="SOLUSD", help="Symbol to backtest")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe to use")
    parser.add_argument("--pair", type=str, default="SOL/USD", help="Trading pair")
    parser.add_argument("--strategy", type=str, default="ensemble", help="Strategy to use (arima, integrated, ml, ensemble)")
    parser.add_argument("--optimize", action="store_true", help="Optimize strategy parameters")
    
    args = parser.parse_args()
    main(args)