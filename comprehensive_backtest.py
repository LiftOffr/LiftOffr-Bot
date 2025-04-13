#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Kraken Trading Bot

This script implements a sophisticated backtesting framework that:
1. Tests multiple cryptocurrencies using historical data
2. Optimizes strategy parameters based on backtesting results
3. Integrates ML models for prediction-based strategy testing
4. Provides detailed performance metrics and visualizations
5. Implements strategy correlation analysis to optimize portfolio
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# Import trading strategies
from trading_strategy import TradingStrategy
from arima_strategy import ARIMAStrategy
from integrated_strategy import IntegratedStrategy
from ml_enhanced_strategy import MLEnhancedStrategy

# Import ML components if available
try:
    from advanced_ensemble_model import DynamicWeightedEnsemble
except ImportError:
    pass

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
CRYPTO_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
TIMEFRAMES = ["1h", "4h", "1d"]
MAX_THREADS = 4  # Maximum number of parallel optimizations

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)


class MockKrakenExchange:
    """
    Mock implementation of Kraken Exchange API for backtesting.
    Simulates order execution, market data feeds, etc.
    """
    def __init__(self, historical_data: pd.DataFrame, trading_pair: str = "SOL/USD", 
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        """
        Initialize the mock exchange with historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            trading_pair (str): Trading pair symbol
            initial_capital (float): Initial capital for backtesting
        """
        self.data = historical_data.copy()
        self.current_index = 0
        self.trading_pair = trading_pair
        self.current_price = float(self.data.iloc[0]['close'])
        self.orders = []
        self.balances = {}
        self.filled_orders = []
        self.initial_capital = initial_capital
        
        # Initialize exchange timestamp
        self.current_timestamp = self.data.iloc[0]['timestamp']
        if isinstance(self.current_timestamp, str):
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        
        # Extract base and quote currencies from trading pair
        parts = trading_pair.split('/')
        self.base_currency = parts[0]
        self.quote_currency = parts[1]
        
        # Initialize balances
        self.balances[self.quote_currency] = initial_capital
        self.balances[self.base_currency] = 0.0
    
    def reset(self):
        """Reset the exchange to initial state"""
        self.current_index = 0
        self.current_price = float(self.data.iloc[0]['close'])
        self.orders = []
        self.filled_orders = []
        self.balances = {
            self.quote_currency: self.initial_capital,
            self.base_currency: 0.0
        }
        self.current_timestamp = self.data.iloc[0]['timestamp']
        if isinstance(self.current_timestamp, str):
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
    
    def advance_time(self) -> bool:
        """
        Advance to the next time period in the historical data.
        
        Returns:
            bool: True if successfully advanced, False if at the end of data
        """
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.current_price = float(self.data.iloc[self.current_index]['close'])
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
                      if o["side"] == "buy" and symbol.replace("/", "") in o["id"]]
        
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
    Trader implementation for backtesting that works with multiple strategies.
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
        self.initial_capital = exchange.initial_capital
        
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
            open_price = float(latest['open'])
            high_price = float(latest['high'])
            low_price = float(latest['low'])
            close_price = float(latest['close'])
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
                if hasattr(strategy, 'should_buy') and strategy.should_buy():
                    size = strategy.calculate_position_size() if hasattr(strategy, 'calculate_position_size') else 1.0
                    self._execute_entry(strategy_id, "buy", size)
                elif hasattr(strategy, 'should_sell') and strategy.should_sell():
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
        
        # Apply size adjustment from strategy (as a percentage)
        volume = volume * size
        
        # Place the order
        order = self.exchange.place_order("market", side, volume)
        
        # Record the position
        self.active_positions[strategy_id] = {
            "order_id": order["id"],
            "side": side,
            "entry_price": current_price,
            "volume": volume,
            "timestamp": self.exchange.get_current_timestamp()
        }
        
        # Add to trade history
        self.trade_history.append({
            "strategy": strategy_id,
            "type": "entry",
            "side": side,
            "price": current_price,
            "volume": volume,
            "timestamp": self.exchange.get_current_timestamp(),
            "order_id": order["id"]
        })
        
        # Log the entry
        logger.debug(f"[BACKTEST] {strategy_id}: Entered {side.upper()} position of {volume:.6f} {self.base_currency} @ ${current_price:.2f}")
    
    def _execute_exit(self, strategy_id: str) -> None:
        """
        Execute an exit order for a strategy.
        
        Args:
            strategy_id (str): Strategy ID
        """
        position = self.active_positions[strategy_id]
        if not position:
            return
        
        # Determine the exit side (opposite of entry)
        exit_side = "sell" if position["side"] == "buy" else "buy"
        
        # Place the exit order
        current_price = self.exchange.get_current_price()
        order = self.exchange.place_order("market", exit_side, position["volume"])
        
        # Calculate profit/loss
        if position["side"] == "buy":
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
        
        # Add to trade history
        self.trade_history.append({
            "strategy": strategy_id,
            "type": "exit",
            "side": exit_side,
            "price": current_price,
            "volume": position["volume"],
            "timestamp": self.exchange.get_current_timestamp(),
            "order_id": order["id"],
            "entry_price": position["entry_price"],
            "pnl_pct": pnl_pct
        })
        
        # Clear the position
        self.active_positions[strategy_id] = None
        
        # Log the exit
        logger.debug(f"[BACKTEST] {strategy_id}: Exited position at ${current_price:.2f} with P&L: {pnl_pct:.2%}")
    
    def _record_portfolio_state(self) -> None:
        """Record current portfolio state for later analysis"""
        balances = self.exchange.get_balances()
        filled_orders = self.exchange.get_filled_orders()
        
        # Calculate total portfolio value in quote currency
        quote_balance = balances.get(self.quote_currency, 0)
        base_balance = balances.get(self.base_currency, 0)
        base_value = base_balance * self.exchange.get_current_price()
        total_value = quote_balance + base_value
        
        # Record the state
        self.portfolio_history.append({
            "timestamp": self.exchange.get_current_timestamp(),
            "quote_balance": quote_balance,
            "base_balance": base_balance,
            "base_price": self.exchange.get_current_price(),
            "total_value": total_value
        })
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get the trade history as a DataFrame.
        
        Returns:
            pd.DataFrame: Trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio value history.
        
        Returns:
            pd.DataFrame: Portfolio history
        """
        if not self.portfolio_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.portfolio_history)
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            dict: Performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        portfolio_df = self.get_portfolio_history()
        trade_df = self.get_trade_history()
        
        # Calculate basic metrics
        initial_value = portfolio_df["total_value"].iloc[0]
        final_value = portfolio_df["total_value"].iloc[-1]
        pnl = final_value - initial_value
        pnl_pct = pnl / initial_value
        
        # Calculate trade metrics
        if not trade_df.empty:
            exit_trades = trade_df[trade_df["type"] == "exit"]
            win_trades = exit_trades[exit_trades["pnl_pct"] > 0]
            lose_trades = exit_trades[exit_trades["pnl_pct"] <= 0]
            
            total_trades = len(exit_trades)
            win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
            
            avg_win = win_trades["pnl_pct"].mean() if not win_trades.empty else 0
            avg_loss = lose_trades["pnl_pct"].mean() if not lose_trades.empty else 0
            
            # Calculate profit factor
            gross_profit = win_trades["pnl_pct"].sum() if not win_trades.empty else 0
            gross_loss = lose_trades["pnl_pct"].sum() if not lose_trades.empty else 0
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate drawdown
        portfolio_df["previous_peak"] = portfolio_df["total_value"].cummax()
        portfolio_df["drawdown"] = (portfolio_df["total_value"] - portfolio_df["previous_peak"]) / portfolio_df["previous_peak"]
        max_drawdown = portfolio_df["drawdown"].min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(portfolio_df) > 1:
            portfolio_df["daily_return"] = portfolio_df["total_value"].pct_change()
            sharpe_ratio = portfolio_df["daily_return"].mean() / portfolio_df["daily_return"].std() * np.sqrt(252) if portfolio_df["daily_return"].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "initial_capital": initial_value,
            "final_capital": final_value,
            "profit_loss": pnl,
            "profit_loss_pct": pnl_pct,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def reset(self) -> None:
        """Reset the trader to initial state"""
        self.exchange.reset()
        self.active_positions = {k: None for k in self.strategies}
        self.trade_history = []
        self.portfolio_history = []
        self._record_portfolio_state()


def load_historical_data(pair: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical data for a trading pair and timeframe.
    
    Args:
        pair (str): Trading pair (e.g., "SOL/USD")
        timeframe (str): Timeframe (e.g., "1h", "4h", "1d")
        
    Returns:
        pd.DataFrame: Historical OHLCV data
    """
    # Convert pair name to directory format (e.g., "SOL/USD" -> "SOLUSD")
    dir_name = pair.replace("/", "")
    
    # Determine file path
    file_path = os.path.join(DATA_DIR, dir_name, f"{dir_name}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Historical data file not found: {file_path}")
        return pd.DataFrame()
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def initialize_strategy(strategy_type: str, params: Dict, symbol: str) -> TradingStrategy:
    """
    Initialize a strategy with given parameters.
    
    Args:
        strategy_type (str): Type of strategy
        params (dict): Strategy parameters
        symbol (str): Trading symbol
        
    Returns:
        TradingStrategy: Initialized strategy
    """
    if strategy_type.lower() == "arima":
        # Set default parameters if not provided
        lookback = params.get("lookback_period", 30)
        trailing_mult = params.get("atr_trailing_multiplier", 2.0)
        
        return ARIMAStrategy(symbol=symbol, lookback_period=lookback, 
                           atr_trailing_multiplier=trailing_mult)
    
    elif strategy_type.lower() == "integrated":
        return IntegratedStrategy(symbol=symbol, **params)
    
    elif strategy_type.lower() == "mlenhanced":
        # Check if ML components are available
        if "DynamicWeightedEnsemble" in globals():
            model = params.get("model", None)
            if model is None:
                # Initialize the ML model if not provided
                model = DynamicWeightedEnsemble(trading_pair=symbol)
            
            return MLEnhancedStrategy(symbol=symbol, ml_model=model, **params)
        else:
            logger.error("ML components not available, cannot initialize MLEnhancedStrategy")
            return None
    
    else:
        logger.error(f"Unknown strategy type: {strategy_type}")
        return None


def run_backtest(exchange: MockKrakenExchange, strategy_configs: List[Dict], 
                verbose: bool = False) -> Dict:
    """
    Run a backtest with given exchange and strategies.
    
    Args:
        exchange (MockKrakenExchange): Exchange instance
        strategy_configs (list): List of strategy configurations
        verbose (bool): Whether to print detailed logs
        
    Returns:
        dict: Backtest results
    """
    # Initialize the backtester
    trader = BacktestTrader(exchange, trading_pair=exchange.trading_pair)
    
    # Initialize and add strategies
    for config in strategy_configs:
        strategy_type = config["type"]
        strategy_params = config.get("params", {})
        strategy_id = config.get("id", f"{strategy_type.lower()}-{exchange.trading_pair.replace('/', '')}")
        
        strategy = initialize_strategy(strategy_type, strategy_params, exchange.trading_pair)
        if strategy:
            trader.add_strategy(strategy, strategy_id)
    
    # Run the backtest
    bar_count = 0
    total_bars = len(exchange.data)
    update_interval = max(1, total_bars // 100)  # Update progress every 1%
    
    while exchange.advance_time():
        bar_count += 1
        trader.update()
        
        if verbose and bar_count % update_interval == 0:
            progress = bar_count / total_bars * 100
            logger.info(f"Progress: {progress:.1f}% | Processed {bar_count}/{total_bars} bars")
    
    # Calculate performance metrics
    metrics = trader.get_performance_metrics()
    
    # Add trade and portfolio history to results
    results = {
        "metrics": metrics,
        "trade_history": trader.get_trade_history(),
        "portfolio_history": trader.get_portfolio_history()
    }
    
    if verbose:
        # Print summary
        logger.info(f"Backtest completed: {metrics['total_trades']} trades")
        logger.info(f"Final capital: ${metrics['final_capital']:.2f} ({metrics['profit_loss_pct']:.2%})")
        logger.info(f"Win rate: {metrics['win_rate']:.2%} | Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    
    return results


def optimize_strategy_parameters(pair: str, strategy_type: str, param_grid: Dict, 
                                timeframe: str = "1h", initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                                optimization_target: str = "profit_loss_pct"):
    """
    Optimize strategy parameters using grid search.
    
    Args:
        pair (str): Trading pair
        strategy_type (str): Type of strategy
        param_grid (dict): Parameter grid to search
        timeframe (str): Timeframe for backtesting
        initial_capital (float): Initial capital
        optimization_target (str): Metric to optimize
        
    Returns:
        tuple: (best_params, best_metrics, all_results)
    """
    logger.info(f"Optimizing {strategy_type} strategy for {pair} on {timeframe} timeframe")
    
    # Load historical data
    data = load_historical_data(pair, timeframe)
    if data.empty:
        logger.error(f"No historical data available for {pair} on {timeframe} timeframe")
        return None, None, []
    
    # Create parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Initialize results storage
    results = []
    best_result = None
    best_metrics = None
    best_value = float('-inf')  # For maximizing
    
    # If optimization target is drawdown or other metric to minimize
    if optimization_target in ["max_drawdown"]:
        best_value = float('inf')  # For minimizing
    
    # Process each parameter combination
    for i, params in enumerate(param_combinations):
        # Initialize exchange and strategy
        exchange = MockKrakenExchange(data, trading_pair=pair, initial_capital=initial_capital)
        strategy_config = {
            "type": strategy_type,
            "params": params,
            "id": f"{strategy_type.lower()}-{pair.replace('/', '')}"
        }
        
        # Run backtest
        backtest_results = run_backtest(exchange, [strategy_config], verbose=False)
        metrics = backtest_results["metrics"]
        
        # Record results
        result = {
            "params": params,
            "metrics": metrics
        }
        results.append(result)
        
        # Check if this is the best result
        current_value = metrics.get(optimization_target, 0)
        
        if optimization_target in ["max_drawdown"]:
            # For drawdown, smaller is better
            if current_value > best_value:
                continue
        else:
            # For other metrics, larger is better
            if current_value < best_value:
                continue
        
        best_value = current_value
        best_result = params
        best_metrics = metrics
        
        logger.info(f"Found new best parameters ({i+1}/{len(param_combinations)}): {best_result}")
        logger.info(f"  {optimization_target}: {best_value:.4f}")
    
    # Return best parameters and all results
    return best_result, best_metrics, results


def optimize_multiple_strategies(pair: str, strategies: List[Dict], 
                              timeframe: str = "1h", initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                              max_threads: int = MAX_THREADS):
    """
    Optimize multiple strategies in parallel.
    
    Args:
        pair (str): Trading pair
        strategies (list): List of strategy configurations
        timeframe (str): Timeframe for backtesting
        initial_capital (float): Initial capital
        max_threads (int): Maximum number of parallel optimizations
        
    Returns:
        dict: Optimization results for each strategy
    """
    logger.info(f"Optimizing {len(strategies)} strategies for {pair} on {timeframe} timeframe")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel optimization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit optimization tasks
        futures = {}
        for strategy in strategies:
            strategy_type = strategy["type"]
            param_grid = strategy["param_grid"]
            optimization_target = strategy.get("optimization_target", "profit_loss_pct")
            
            future = executor.submit(
                optimize_strategy_parameters,
                pair, strategy_type, param_grid, timeframe, initial_capital, optimization_target
            )
            futures[future] = strategy_type
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            strategy_type = futures[future]
            try:
                best_params, best_metrics, all_results = future.result()
                results[strategy_type] = {
                    "best_params": best_params,
                    "best_metrics": best_metrics,
                    "all_results": all_results
                }
                
                logger.info(f"Optimization complete for {strategy_type}")
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best metrics: {best_metrics}")
                
            except Exception as e:
                logger.error(f"Error optimizing {strategy_type}: {e}")
    
    return results


def run_optimized_backtest(pair: str, optimized_strategies: Dict, 
                         timeframe: str = "1h", initial_capital: float = DEFAULT_INITIAL_CAPITAL):
    """
    Run a backtest with optimized strategies.
    
    Args:
        pair (str): Trading pair
        optimized_strategies (dict): Optimized strategy configurations
        timeframe (str): Timeframe for backtesting
        initial_capital (float): Initial capital
        
    Returns:
        dict: Backtest results
    """
    # Load historical data
    data = load_historical_data(pair, timeframe)
    if data.empty:
        logger.error(f"No historical data available for {pair} on {timeframe} timeframe")
        return None
    
    # Initialize exchange
    exchange = MockKrakenExchange(data, trading_pair=pair, initial_capital=initial_capital)
    
    # Prepare strategy configurations
    strategy_configs = []
    for strategy_type, strategy_results in optimized_strategies.items():
        best_params = strategy_results["best_params"]
        if best_params:
            strategy_configs.append({
                "type": strategy_type,
                "params": best_params,
                "id": f"{strategy_type.lower()}-{pair.replace('/', '')}"
            })
    
    # Run backtest
    logger.info(f"Running backtest with {len(strategy_configs)} optimized strategies")
    backtest_results = run_backtest(exchange, strategy_configs, verbose=True)
    
    return backtest_results


def plot_backtest_results(backtest_results: Dict, pair: str, timeframe: str, 
                        strategies: List[str], save_path: Optional[str] = None):
    """
    Plot backtest results.
    
    Args:
        backtest_results (dict): Backtest results
        pair (str): Trading pair
        timeframe (str): Timeframe
        strategies (list): Strategy names
        save_path (str, optional): Path to save the plot
    """
    if not backtest_results:
        logger.error("No backtest results to plot")
        return
    
    # Extract data
    portfolio_history = backtest_results["portfolio_history"]
    trade_history = backtest_results["trade_history"]
    metrics = backtest_results["metrics"]
    
    # Convert to DataFrame if needed
    if not isinstance(portfolio_history, pd.DataFrame):
        portfolio_history = pd.DataFrame(portfolio_history)
    
    if not isinstance(trade_history, pd.DataFrame):
        trade_history = pd.DataFrame(trade_history)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Portfolio value plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(portfolio_history["timestamp"], portfolio_history["total_value"], label="Portfolio Value")
    ax1.set_title(f"Backtest Results - {pair} ({timeframe})")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True)
    
    # Mark buy and sell points
    if not trade_history.empty:
        buy_trades = trade_history[(trade_history["type"] == "entry") & (trade_history["side"] == "buy")]
        sell_trades = trade_history[(trade_history["type"] == "exit") & (trade_history["side"] == "sell")]
        
        if not buy_trades.empty:
            ax1.scatter(buy_trades["timestamp"], buy_trades["price"], 
                       marker="^", color="green", s=100, label="Buy")
        
        if not sell_trades.empty:
            ax1.scatter(sell_trades["timestamp"], sell_trades["price"], 
                       marker="v", color="red", s=100, label="Sell")
    
    ax1.legend()
    
    # Drawdown plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    portfolio_history["drawdown"] = (portfolio_history["total_value"] / portfolio_history["total_value"].cummax() - 1)
    ax2.fill_between(portfolio_history["timestamp"], 0, portfolio_history["drawdown"], color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)
    
    # Metrics text
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    
    # Prepare metrics text
    metrics_text = f"Initial Capital: ${metrics['initial_capital']:.2f}\n"
    metrics_text += f"Final Capital: ${metrics['final_capital']:.2f}\n"
    metrics_text += f"Profit/Loss: ${metrics['profit_loss']:.2f} ({metrics['profit_loss_pct']:.2%})\n"
    metrics_text += f"Total Trades: {metrics['total_trades']}\n"
    metrics_text += f"Win Rate: {metrics['win_rate']:.2%}\n"
    metrics_text += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
    metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
    metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
    metrics_text += f"Strategies: {', '.join(strategies)}"
    
    ax3.text(0.1, 0.5, metrics_text, fontsize=12, va="center")
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    
    # Save or show
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved backtest plot to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def main():
    """Main function for the comprehensive backtesting system"""
    parser = argparse.ArgumentParser(description="Comprehensive Backtesting System for Kraken Trading Bot")
    
    parser.add_argument("--pair", type=str, default="SOL/USD", 
                       help="Trading pair to backtest (default: SOL/USD)")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       help="Timeframe to use (default: 1h)")
    parser.add_argument("--strategy", type=str, default="arima", 
                       help="Strategy to backtest (default: arima)")
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize strategy parameters")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, 
                       help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})")
    parser.add_argument("--days", type=int, default=None, 
                       help="Number of days to backtest (default: all available data)")
    parser.add_argument("--plot", action="store_true", 
                       help="Plot backtest results")
    parser.add_argument("--multi-strategy", action="store_true", 
                       help="Use multiple strategies")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory for results")
    parser.add_argument("--ml", action="store_true", 
                       help="Include ML-enhanced strategies")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Define strategy parameter grids for optimization
    param_grids = {
        "arima": {
            "lookback_period": [24, 28, 32, 36, 40],
            "atr_trailing_multiplier": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8],
            "entry_atr_multiplier": [0.01, 0.015, 0.02],
            "risk_buffer_multiplier": [1.0, 1.25, 1.5]
        },
        "integrated": {
            "signal_smoothing": [2, 3, 4, 5],
            "trend_strength_threshold": [0.3, 0.4, 0.5, 0.6],
            "volatility_filter_threshold": [0.005, 0.008, 0.01, 0.012],
            "min_adx_threshold": [20, 22, 25, 28, 30]
        }
    }
    
    # Add ML-enhanced strategy if requested
    if args.ml:
        param_grids["mlenhanced"] = {
            "confidence_threshold": [0.65, 0.7, 0.75, 0.8],
            "trend_alignment_required": [True, False],
            "min_volatility": [0.005, 0.008, 0.01],
            "max_volatility": [0.03, 0.04, 0.05]
        }
    
    # Load historical data
    data = load_historical_data(args.pair, args.timeframe)
    if data.empty:
        logger.error(f"No historical data available for {args.pair} on {args.timeframe} timeframe")
        return
    
    # Limit data to specified days if requested
    if args.days:
        end_date = data["timestamp"].max()
        start_date = end_date - timedelta(days=args.days)
        data = data[data["timestamp"] >= start_date]
        logger.info(f"Limited data to last {args.days} days ({len(data)} bars)")
    
    if args.optimize:
        # Prepare list of strategies to optimize
        strategies_to_optimize = []
        
        if args.multi_strategy:
            # Optimize multiple strategies
            for strategy_type, param_grid in param_grids.items():
                if args.ml and strategy_type == "mlenhanced":
                    # Skip ML strategy if not requested
                    if not args.ml:
                        continue
                
                strategies_to_optimize.append({
                    "type": strategy_type,
                    "param_grid": param_grid,
                    "optimization_target": "profit_loss_pct"
                })
        else:
            # Optimize single strategy
            strategy_type = args.strategy.lower()
            if strategy_type in param_grids:
                strategies_to_optimize.append({
                    "type": strategy_type,
                    "param_grid": param_grids[strategy_type],
                    "optimization_target": "profit_loss_pct"
                })
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return
        
        # Run optimization
        optimization_results = optimize_multiple_strategies(
            args.pair, strategies_to_optimize, args.timeframe, args.capital
        )
        
        # Save optimization results
        if args.output:
            optimization_file = os.path.join(args.output, f"optimization_results_{args.pair.replace('/', '')}_{args.timeframe}.json")
            
            # Convert numpy/pandas types to JSON serializable
            serializable_results = {}
            for strategy_type, results in optimization_results.items():
                best_params = {k: float(v) if isinstance(v, np.float32) else v 
                              for k, v in results["best_params"].items()}
                
                best_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in results["best_metrics"].items()}
                
                serializable_results[strategy_type] = {
                    "best_params": best_params,
                    "best_metrics": best_metrics
                }
            
            with open(optimization_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved optimization results to {optimization_file}")
        
        # Run backtest with optimized parameters
        backtest_results = run_optimized_backtest(
            args.pair, optimization_results, args.timeframe, args.capital
        )
        
        # Plot results if requested
        if args.plot and backtest_results:
            strategy_names = list(optimization_results.keys())
            plot_file = os.path.join(args.output, f"backtest_plot_{args.pair.replace('/', '')}_{args.timeframe}.png") if args.output else None
            
            plot_backtest_results(
                backtest_results, args.pair, args.timeframe, strategy_names, plot_file
            )
    
    else:
        # Run backtest without optimization
        exchange = MockKrakenExchange(data, trading_pair=args.pair, initial_capital=args.capital)
        
        strategy_configs = []
        
        if args.multi_strategy:
            # Use multiple strategies
            strategy_types = list(param_grids.keys())
            if not args.ml:
                # Remove ML strategy if not requested
                strategy_types = [s for s in strategy_types if s != "mlenhanced"]
            
            for strategy_type in strategy_types:
                # Use default parameters
                default_params = {}
                if strategy_type == "arima":
                    default_params = {
                        "lookback_period": 30,
                        "atr_trailing_multiplier": 2.0
                    }
                elif strategy_type == "integrated":
                    default_params = {
                        "signal_smoothing": 3,
                        "trend_strength_threshold": 0.4,
                        "volatility_filter_threshold": 0.008,
                        "min_adx_threshold": 25
                    }
                elif strategy_type == "mlenhanced":
                    default_params = {
                        "confidence_threshold": 0.7,
                        "trend_alignment_required": True,
                        "min_volatility": 0.005,
                        "max_volatility": 0.04
                    }
                
                strategy_configs.append({
                    "type": strategy_type,
                    "params": default_params,
                    "id": f"{strategy_type.lower()}-{args.pair.replace('/', '')}"
                })
        else:
            # Use single strategy
            strategy_type = args.strategy.lower()
            default_params = {}
            
            if strategy_type == "arima":
                default_params = {
                    "lookback_period": 30,
                    "atr_trailing_multiplier": 2.0
                }
            elif strategy_type == "integrated":
                default_params = {
                    "signal_smoothing": 3,
                    "trend_strength_threshold": 0.4,
                    "volatility_filter_threshold": 0.008,
                    "min_adx_threshold": 25
                }
            elif strategy_type == "mlenhanced":
                default_params = {
                    "confidence_threshold": 0.7,
                    "trend_alignment_required": True,
                    "min_volatility": 0.005,
                    "max_volatility": 0.04
                }
            
            strategy_configs.append({
                "type": strategy_type,
                "params": default_params,
                "id": f"{strategy_type.lower()}-{args.pair.replace('/', '')}"
            })
        
        # Run backtest
        backtest_results = run_backtest(exchange, strategy_configs, verbose=True)
        
        # Plot results if requested
        if args.plot and backtest_results:
            strategy_names = [config["type"] for config in strategy_configs]
            plot_file = os.path.join(args.output, f"backtest_plot_{args.pair.replace('/', '')}_{args.timeframe}.png") if args.output else None
            
            plot_backtest_results(
                backtest_results, args.pair, args.timeframe, strategy_names, plot_file
            )


if __name__ == "__main__":
    main()