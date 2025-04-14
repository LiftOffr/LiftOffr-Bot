#!/usr/bin/env python3
"""
Improved Comprehensive Backtesting System with ML Optimization

This enhanced backtesting framework solves the empty results issue and adds:
1. Deep machine learning optimization for all strategy parameters
2. Cross-asset correlation analysis to improve signal generation
3. Enhanced position sizing based on market conditions
4. Multi-timeframe analysis for more accurate entry/exit points
5. Advanced reporting with detailed metrics and visualizations

It produces complete backtest results for all configured trading pairs
and strategies, saving detailed reports for analysis.
"""

import os
import sys
import json
import time
import logging
import argparse
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Configure Plotting
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Local imports
from historical_data_fetcher import fetch_historical_data
from dynamic_position_sizing_ml import get_config, calculate_dynamic_leverage
from market_context import detect_market_regime, MarketRegime, analyze_market_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("improved_backtest.log")
    ]
)

logger = logging.getLogger(__name__)

# Create backtest results directory
BACKTEST_DIR = "backtest_results"
os.makedirs(BACKTEST_DIR, exist_ok=True)

# Constants
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
INITIAL_CAPITAL = 20000.0  # $20,000 starting capital
DEFAULT_FEE_RATE = 0.0026  # 0.26% taker fee on Kraken
DEFAULT_SLIPPAGE = 0.001   # 0.1% average slippage
MAX_POSITIONS_PER_ASSET = 1  # Maximum of 1 position per asset at a time

# Position settings
DEFAULT_STOP_LOSS_PCT = 0.04  # Default 4% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.12  # Default 12% take profit

# Strategies to backtest
STRATEGIES = ["arima", "adaptive", "ensemble", "ml", "integrated"]

class Position:
    """Class representing a trading position during backtesting"""
    
    def __init__(self, asset: str, entry_price: float, entry_time: datetime,
                direction: str, quantity: float, leverage: float,
                strategy: str, stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
                take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
                use_trailing_stop: bool = True):
        """
        Initialize a new position
        
        Args:
            asset: Trading pair symbol
            entry_price: Entry price
            entry_time: Entry timestamp
            direction: 'long' or 'short'
            quantity: Position size in base currency
            leverage: Leverage used
            strategy: Strategy name that opened this position
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_trailing_stop: Whether to use trailing stop
        """
        self.asset = asset
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction.lower()
        self.quantity = quantity
        self.leverage = leverage
        self.strategy = strategy
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_trailing_stop = use_trailing_stop
        
        # Calculate stop loss and take profit levels
        if self.direction == 'long':
            self.stop_loss = entry_price * (1 - stop_loss_pct)
            self.take_profit = entry_price * (1 + take_profit_pct)
        else:  # short
            self.stop_loss = entry_price * (1 + stop_loss_pct)
            self.take_profit = entry_price * (1 - take_profit_pct)
        
        # Trailing stop tracking
        self.trailing_stop = self.stop_loss
        self.max_profit_price = entry_price
        
        # Exit tracking
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.fees_paid = 0.0
    
    def update_trailing_stop(self, current_price: float) -> None:
        """
        Update trailing stop based on current price
        
        Args:
            current_price: Current price
        """
        if not self.use_trailing_stop:
            return
        
        if self.direction == 'long':
            if current_price > self.max_profit_price:
                # Update max profit price
                self.max_profit_price = current_price
                # Update trailing stop (maintain same distance)
                price_diff = self.max_profit_price - self.entry_price
                # Move stop loss up only after price moves up at least 1%
                if price_diff / self.entry_price > 0.01:
                    # Move stop to break even + 10% of profits
                    self.trailing_stop = max(
                        self.trailing_stop,
                        self.entry_price + (price_diff * 0.1)
                    )
        else:  # short
            if current_price < self.max_profit_price:
                # Update max profit price
                self.max_profit_price = current_price
                # Update trailing stop (maintain same distance)
                price_diff = self.entry_price - self.max_profit_price
                # Move stop loss down only after price moves down at least 1%
                if price_diff / self.entry_price > 0.01:
                    # Move stop to break even - 10% of profits
                    self.trailing_stop = min(
                        self.trailing_stop,
                        self.entry_price - (price_diff * 0.1)
                    )
    
    def check_exit_conditions(self, current_price: float, current_time: datetime,
                            fee_rate: float = DEFAULT_FEE_RATE) -> bool:
        """
        Check if position should be closed based on current price
        
        Args:
            current_price: Current price
            current_time: Current timestamp
            fee_rate: Trading fee rate
            
        Returns:
            bool: True if position should be closed, False otherwise
        """
        # Skip if already closed
        if self.exit_price is not None:
            return False
        
        exit_triggered = False
        exit_reason = None
        
        # Check stop loss (use trailing stop if enabled)
        if self.direction == 'long' and current_price <= self.trailing_stop:
            exit_triggered = True
            exit_reason = 'stop_loss'
        elif self.direction == 'short' and current_price >= self.trailing_stop:
            exit_triggered = True
            exit_reason = 'stop_loss'
        
        # Check take profit
        elif self.direction == 'long' and current_price >= self.take_profit:
            exit_triggered = True
            exit_reason = 'take_profit'
        elif self.direction == 'short' and current_price <= self.take_profit:
            exit_triggered = True
            exit_reason = 'take_profit'
        
        if exit_triggered:
            # Update position with exit details
            self.exit_price = current_price
            self.exit_time = current_time
            self.exit_reason = exit_reason
            
            # Calculate PnL
            entry_value = self.entry_price * self.quantity
            exit_value = self.exit_price * self.quantity
            
            # Calculate fees (entry and exit)
            self.fees_paid = (entry_value + exit_value) * fee_rate
            
            if self.direction == 'long':
                self.pnl = (self.exit_price - self.entry_price) * self.quantity * self.leverage
                self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100 * self.leverage
            else:  # short
                self.pnl = (self.entry_price - self.exit_price) * self.quantity * self.leverage
                self.pnl_pct = (1 - self.exit_price / self.entry_price) * 100 * self.leverage
            
            # Subtract fees
            self.pnl -= self.fees_paid
            
            return True
        
        return False
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price
        
        Args:
            current_price: Current price
            
        Returns:
            float: Unrealized P&L in quote currency
        """
        if self.exit_price is not None:
            return self.pnl
        
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.quantity * self.leverage
        else:  # short
            return (self.entry_price - current_price) * self.quantity * self.leverage
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P&L percentage at current price
        
        Args:
            current_price: Current price
            
        Returns:
            float: Unrealized P&L percentage
        """
        if self.exit_price is not None:
            return self.pnl_pct
        
        if self.direction == 'long':
            return (current_price / self.entry_price - 1) * 100 * self.leverage
        else:  # short
            return (1 - current_price / self.entry_price) * 100 * self.leverage
    
    def __str__(self) -> str:
        status = "CLOSED" if self.exit_price is not None else "OPEN"
        direction = "LONG" if self.direction == 'long' else "SHORT"
        
        result = f"{status} {direction} {self.asset} @ {self.entry_price:.4f}"
        result += f" | Strategy: {self.strategy}"
        result += f" | Leverage: {self.leverage}x"
        
        if self.exit_price is not None:
            result += f" | Exit: {self.exit_price:.4f}"
            result += f" | PnL: {self.pnl:.2f} ({self.pnl_pct:.2f}%)"
            result += f" | Reason: {self.exit_reason}"
        
        return result

class BacktestPortfolio:
    """Class for managing a portfolio during backtesting"""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
               fee_rate: float = DEFAULT_FEE_RATE,
               slippage: float = DEFAULT_SLIPPAGE):
        """
        Initialize a new backtest portfolio
        
        Args:
            initial_capital: Initial capital in USD
            fee_rate: Trading fee rate
            slippage: Average slippage
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Open positions by asset and strategy
        self.positions = {}  # {asset: {strategy: position}}
        
        # Closed positions history
        self.closed_positions = []
        
        # Capital allocation by asset (default: equal allocation)
        self.allocation = {}
        
        # Performance tracking
        self.equity_curve = []
        self.drawdowns = []
        self.trade_history = []
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.peak_capital = initial_capital
    
    def set_allocation(self, allocation: Dict[str, float]) -> None:
        """
        Set capital allocation by asset
        
        Args:
            allocation: Dictionary mapping assets to allocation percentages
        """
        # Validate allocation (must sum to 1.0)
        total = sum(allocation.values())
        if abs(total - 1.0) > 0.001:
            # Normalize to 1.0
            allocation = {asset: pct / total for asset, pct in allocation.items()}
        
        self.allocation = allocation
    
    def get_available_capital(self, asset: str) -> float:
        """
        Get available capital for an asset
        
        Args:
            asset: Asset symbol
            
        Returns:
            float: Available capital in USD
        """
        allocation_pct = self.allocation.get(asset, 1.0 / len(SUPPORTED_ASSETS))
        allocated_capital = self.current_capital * allocation_pct
        
        # Subtract capital used by open positions for this asset
        if asset in self.positions:
            for strategy, position in self.positions[asset].items():
                if position.exit_price is None:  # Open position
                    allocated_capital -= (position.entry_price * position.quantity)
        
        return max(0, allocated_capital)
    
    def get_position(self, asset: str, strategy: str) -> Optional[Position]:
        """
        Get open position for an asset and strategy
        
        Args:
            asset: Asset symbol
            strategy: Strategy name
            
        Returns:
            Optional[Position]: Position or None if no open position
        """
        if asset in self.positions and strategy in self.positions[asset]:
            position = self.positions[asset][strategy]
            if position.exit_price is None:  # Only return if still open
                return position
        return None
    
    def open_position(self, asset: str, strategy: str, direction: str,
                    price: float, time: datetime, signal_strength: float,
                    ml_confidence: float, market_regime: str,
                    stop_loss_pct: Optional[float] = None,
                    take_profit_pct: Optional[float] = None,
                    leverage: Optional[float] = None,
                    risk_pct: Optional[float] = None) -> Optional[Position]:
        """
        Open a new position
        
        Args:
            asset: Asset symbol
            strategy: Strategy name
            direction: 'long' or 'short'
            price: Entry price
            time: Entry timestamp
            signal_strength: Signal strength (0.0-1.0)
            ml_confidence: ML model confidence (0.0-1.0)
            market_regime: Current market regime
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            leverage: Leverage to use (optional)
            risk_pct: Risk percentage to use (optional)
            
        Returns:
            Optional[Position]: New position or None if couldn't open
        """
        # Check if already have an open position for this asset and strategy
        if self.get_position(asset, strategy) is not None:
            logger.debug(f"Already have an open position for {asset}/{strategy}")
            return None
        
        # Check if we can have another position for this asset
        asset_positions = sum(1 for strat, pos in self.positions.get(asset, {}).items() 
                            if pos.exit_price is None)
        if asset_positions >= MAX_POSITIONS_PER_ASSET:
            logger.debug(f"Already have maximum positions ({MAX_POSITIONS_PER_ASSET}) for {asset}")
            return None
        
        # Get available capital for this asset
        available_capital = self.get_available_capital(asset)
        
        # Calculate dynamic leverage if not provided
        if leverage is None:
            # Get the config for position sizing
            config = get_config()
            
            # Calculate dynamic leverage based on signal strength and market regime
            leverage = calculate_dynamic_leverage(
                base_price=price,
                signal_strength=signal_strength,
                ml_confidence=ml_confidence,
                market_regime=market_regime,
                config=config
            )
        
        # Use default stop loss/take profit if not provided
        if stop_loss_pct is None:
            stop_loss_pct = DEFAULT_STOP_LOSS_PCT
        
        if take_profit_pct is None:
            take_profit_pct = DEFAULT_TAKE_PROFIT_PCT
        
        # Calculate position size
        if risk_pct is None:
            risk_pct = 0.04  # 4% of available capital at risk per trade
        
        # Risk-based position sizing
        risk_amount = available_capital * risk_pct
        
        # Calculate quantity
        if direction == 'long':
            # Risk = Quantity * Entry Price * (1 - Stop Loss %)
            quantity = risk_amount / (price * stop_loss_pct)
        else:  # short
            # Risk = Quantity * Entry Price * Stop Loss %
            quantity = risk_amount / (price * stop_loss_pct)
        
        # Apply slippage to entry price
        if direction == 'long':
            adjusted_price = price * (1 + self.slippage)
        else:  # short
            adjusted_price = price * (1 - self.slippage)
        
        # Create position
        position = Position(
            asset=asset,
            entry_price=adjusted_price,
            entry_time=time,
            direction=direction,
            quantity=quantity,
            leverage=leverage,
            strategy=strategy,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_trailing_stop=True
        )
        
        # Add to portfolio
        if asset not in self.positions:
            self.positions[asset] = {}
        
        self.positions[asset][strategy] = position
        
        # Update capital
        position_value = position.entry_price * position.quantity
        
        # Add entry fee
        entry_fee = position_value * self.fee_rate
        position_cost = position_value + entry_fee
        
        # Adjust current capital
        self.current_capital -= position_cost
        
        # Update metrics
        self.total_trades += 1
        
        # Log position
        logger.info(f"Opened position: {position}")
        
        # Add to trade history
        self.trade_history.append({
            'type': 'entry',
            'asset': asset,
            'strategy': strategy,
            'direction': direction,
            'price': adjusted_price,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quantity': quantity,
            'leverage': leverage,
            'fee': entry_fee,
            'signal_strength': signal_strength,
            'ml_confidence': ml_confidence,
            'market_regime': market_regime
        })
        
        return position
    
    def close_position(self, asset: str, strategy: str, current_price: float, 
                    current_time: datetime, reason: str = 'manual') -> Optional[Position]:
        """
        Close an open position
        
        Args:
            asset: Asset symbol
            strategy: Strategy name
            current_price: Current price for closing
            current_time: Current timestamp
            reason: Reason for closing
            
        Returns:
            Optional[Position]: Closed position or None if no position found
        """
        position = self.get_position(asset, strategy)
        if position is None:
            return None
        
        # Apply slippage to exit price
        if position.direction == 'long':
            adjusted_price = current_price * (1 - self.slippage)
        else:  # short
            adjusted_price = current_price * (1 + self.slippage)
        
        # Update position with exit details
        position.exit_price = adjusted_price
        position.exit_time = current_time
        position.exit_reason = reason
        
        # Calculate position value
        entry_value = position.entry_price * position.quantity
        exit_value = position.exit_price * position.quantity
        
        # Calculate fees (entry and exit)
        position.fees_paid = (entry_value + exit_value) * self.fee_rate
        
        # Calculate PnL
        if position.direction == 'long':
            position.pnl = (position.exit_price - position.entry_price) * position.quantity * position.leverage
            position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100 * position.leverage
        else:  # short
            position.pnl = (position.entry_price - position.exit_price) * position.quantity * position.leverage
            position.pnl_pct = (1 - position.exit_price / position.entry_price) * 100 * position.leverage
        
        # Subtract fees
        position.pnl -= position.fees_paid
        
        # Update capital
        self.current_capital += exit_value - position.fees_paid
        
        # Update metrics
        if position.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update peak capital if new high
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown
        drawdown = self.peak_capital - self.current_capital
        drawdown_pct = drawdown / self.peak_capital * 100
        
        # Update max drawdown if new high
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct
        
        # Add to drawdowns tracking
        self.drawdowns.append({
            'time': current_time,
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct
        })
        
        # Add to closed positions
        self.closed_positions.append(position)
        
        # Log position
        logger.info(f"Closed position: {position}")
        
        # Add to trade history
        self.trade_history.append({
            'type': 'exit',
            'asset': asset,
            'strategy': strategy,
            'direction': position.direction,
            'price': adjusted_price,
            'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'quantity': position.quantity,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct,
            'fee': position.fees_paid,
            'reason': reason
        })
        
        return position
    
    def update_positions(self, current_prices: Dict[str, float], current_time: datetime) -> List[Position]:
        """
        Update all open positions and check exit conditions
        
        Args:
            current_prices: Dictionary mapping assets to current prices
            current_time: Current timestamp
            
        Returns:
            List[Position]: List of closed positions
        """
        closed_positions = []
        
        # Update each position
        for asset, strategies in list(self.positions.items()):
            if asset not in current_prices:
                continue
            
            current_price = current_prices[asset]
            
            for strategy, position in list(strategies.items()):
                if position.exit_price is not None:
                    continue  # Skip closed positions
                
                # Update trailing stop
                position.update_trailing_stop(current_price)
                
                # Check exit conditions
                if position.check_exit_conditions(current_price, current_time, self.fee_rate):
                    closed_positions.append(position)
                    # Note: Position remains in self.positions but with exit_price set
        
        return closed_positions
    
    def update_equity_curve(self, current_time: datetime, current_prices: Dict[str, float]) -> None:
        """
        Update equity curve with current portfolio value
        
        Args:
            current_time: Current timestamp
            current_prices: Dictionary mapping assets to current prices
        """
        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        
        for asset, strategies in self.positions.items():
            if asset not in current_prices:
                continue
            
            current_price = current_prices[asset]
            
            for strategy, position in strategies.items():
                if position.exit_price is None:  # Open position
                    unrealized_pnl += position.get_unrealized_pnl(current_price)
        
        # Calculate total equity (capital + unrealized P&L)
        total_equity = self.current_capital + unrealized_pnl
        
        # Add to equity curve
        self.equity_curve.append({
            'time': current_time,
            'equity': total_equity,
            'unrealized_pnl': unrealized_pnl
        })
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Calculate returns from equity curve
        if len(self.equity_curve) > 1:
            equity_values = [point['equity'] for point in self.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Calculate Sortino ratio (downside deviation only)
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 1 and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)  # Annualized
            else:
                sortino_ratio = 0.0
            
            # Calculate Calmar ratio (return / max drawdown)
            if self.max_drawdown_pct > 0:
                calmar_ratio = (self.current_capital / self.initial_capital - 1) * 100 / self.max_drawdown_pct
            else:
                calmar_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
        
        # Calculate profit statistics
        total_profit = sum(position.pnl for position in self.closed_positions if position.pnl > 0)
        total_loss = sum(position.pnl for position in self.closed_positions if position.pnl <= 0)
        
        winning_trades = sum(1 for position in self.closed_positions if position.pnl > 0)
        losing_trades = sum(1 for position in self.closed_positions if position.pnl <= 0)
        
        win_rate = winning_trades / len(self.closed_positions) if self.closed_positions else 0.0
        
        if winning_trades > 0:
            avg_win = total_profit / winning_trades
        else:
            avg_win = 0.0
        
        if losing_trades > 0:
            avg_loss = abs(total_loss) / losing_trades
        else:
            avg_loss = 0.0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate total return
        total_return = self.current_capital - self.initial_capital
        total_return_pct = (self.current_capital / self.initial_capital - 1) * 100
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

class ImprovedBacktestEngine:
    """
    Enhanced backtesting engine that supports:
    1. Multi-asset, multi-strategy backtesting
    2. ML signal generation and optimization
    3. Technical and fundamental signal integration
    4. Ensemble model backtesting
    5. Advanced metrics and visualizations
    """
    
    def __init__(self, start_date=None, end_date=None, initial_capital=INITIAL_CAPITAL,
                fee_rate=DEFAULT_FEE_RATE, slippage=DEFAULT_SLIPPAGE):
        """
        Initialize backtesting engine
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital in USD
            fee_rate: Trading fee rate
            slippage: Average slippage
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Data storage
        self.data = {}  # {asset: {timeframe: pd.DataFrame}}
        self.models = {}  # {asset: {model_type: model}}
        self.ensemble_configs = {}  # {asset: {config}}
        self.signal_cache = {}  # Cache for ML signals
        
        # Initialize portfolio
        self.portfolio = BacktestPortfolio(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage=slippage
        )
    
    def load_historical_data(self, assets, timeframes=None):
        """
        Load historical data for backtesting
        
        Args:
            assets: List of assets to load data for
            timeframes: List of timeframes to load
        """
        if timeframes is None:
            timeframes = TIMEFRAMES
        
        logger.info(f"Loading historical data for {len(assets)} assets and {len(timeframes)} timeframes")
        
        for asset in assets:
            self.data[asset] = {}
            
            for timeframe in timeframes:
                # Check if file exists
                file_path = f"historical_data/{asset.replace('/', '')}" \
                           f"_{timeframe}.csv"
                
                if not os.path.exists(file_path):
                    logger.warning(f"Historical data file not found: {file_path}")
                    continue
                
                # Load data from CSV
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                
                # Set index to timestamp
                df.set_index('timestamp', inplace=True)
                
                # Filter by date range if specified
                if self.start_date is not None:
                    df = df[df.index >= self.start_date]
                
                if self.end_date is not None:
                    df = df[df.index <= self.end_date]
                
                # Store data
                self.data[asset][timeframe] = df
                
                logger.info(f"Loaded {len(df)} rows of {timeframe} data for {asset}")
    
    def load_ml_models(self, assets):
        """
        Load ML models for backtesting
        
        Args:
            assets: List of assets to load models for
        """
        logger.info(f"Loading ML models for {len(assets)} assets")
        
        for asset in assets:
            asset_code = asset.replace('/', '')
            self.models[asset] = {}
            
            # Load ensemble configuration
            ensemble_config_path = f"models/ensemble/{asset_code}_ensemble.json"
            weights_path = f"models/ensemble/{asset_code}_weights.json"
            position_sizing_path = f"models/ensemble/{asset_code}_position_sizing.json"
            
            if os.path.exists(ensemble_config_path) and \
               os.path.exists(weights_path) and \
               os.path.exists(position_sizing_path):
                
                with open(ensemble_config_path, 'r') as f:
                    ensemble_config = json.load(f)
                
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                
                with open(position_sizing_path, 'r') as f:
                    position_sizing = json.load(f)
                
                # Store ensemble configuration
                self.ensemble_configs[asset] = {
                    'config': ensemble_config,
                    'weights': weights,
                    'position_sizing': position_sizing
                }
                
                logger.info(f"Loaded ensemble configuration for {asset}")
    
    def get_arima_signal(self, asset, timestamp, data):
        """
        Generate ARIMA strategy signal
        
        Args:
            asset: Asset symbol
            timestamp: Current timestamp
            data: Price data
            
        Returns:
            Tuple: (direction, strength, confidence, take_profit_pct, stop_loss_pct)
        """
        # Based on the current ARIMA strategy
        window_size = 24  # 24 hours of data
        forecast_horizon = 3  # 3 hours ahead
        
        # Get the recent window of data
        recent_data = data[data.index <= timestamp].tail(window_size)
        
        if len(recent_data) < window_size:
            return None, 0.0, 0.0, 0.12, 0.04
        
        # Simple trend detection based on recent price action
        current_price = recent_data['close'].iloc[-1]
        prev_price = recent_data['close'].iloc[-4]  # 3 hours ago
        
        # Direction based on price movement
        if current_price > prev_price:
            direction = 'long'
            strength = min(0.8, (current_price / prev_price - 1) * 50)  # Scale the strength
        else:
            direction = 'short'
            strength = min(0.8, (prev_price / current_price - 1) * 50)
        
        # Confidence based on volatility (lower volatility = higher confidence)
        volatility = recent_data['close'].pct_change().std()
        confidence = max(0.1, min(0.9, 1.0 - (volatility * 50)))
        
        return direction, strength, confidence, 0.12, 0.04
    
    def get_adaptive_signal(self, asset, timestamp, data):
        """
        Generate Adaptive strategy signal
        
        Args:
            asset: Asset symbol
            timestamp: Current timestamp
            data: Price data
            
        Returns:
            Tuple: (direction, strength, confidence, take_profit_pct, stop_loss_pct)
        """
        # Based on the current Adaptive strategy
        ema_short = 9
        ema_long = 21
        rsi_period = 14
        adx_period = 14
        volatility_window = 20
        volatility_threshold = 0.006
        
        # Get the recent data
        recent_data = data[data.index <= timestamp].tail(50)  # Enough for EMA calculations
        
        if len(recent_data) < 50:
            return None, 0.0, 0.0, 0.12, 0.04
        
        # Calculate indicators
        recent_data['ema_short'] = recent_data['close'].ewm(span=ema_short, adjust=False).mean()
        recent_data['ema_long'] = recent_data['close'].ewm(span=ema_long, adjust=False).mean()
        
        # RSI calculation
        delta = recent_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        recent_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility calculation (standard deviation of returns)
        recent_data['volatility'] = recent_data['close'].pct_change().rolling(window=volatility_window).std()
        
        # Get latest values
        current_ema_short = recent_data['ema_short'].iloc[-1]
        current_ema_long = recent_data['ema_long'].iloc[-1]
        current_rsi = recent_data['rsi'].iloc[-1]
        current_volatility = recent_data['volatility'].iloc[-1]
        
        # Direction based on EMA crossover
        if current_ema_short > current_ema_long:
            direction = 'long'
            strength = min(0.8, (current_ema_short / current_ema_long - 1) * 50)
        else:
            direction = 'short'
            strength = min(0.8, (current_ema_long / current_ema_short - 1) * 50)
        
        # Adjust strength based on RSI
        if direction == 'long' and current_rsi > 70:
            strength *= 0.5  # Reduce strength in overbought conditions
        elif direction == 'short' and current_rsi < 30:
            strength *= 0.5  # Reduce strength in oversold conditions
        
        # Confidence based on volatility
        if current_volatility < volatility_threshold:
            confidence = max(0.3, min(0.9, 1.0 - (current_volatility / volatility_threshold)))
        else:
            confidence = 0.3  # Low confidence in high volatility
        
        return direction, strength, confidence, 0.12, 0.04
    
    def get_ensemble_signal(self, asset, timestamp, data_dict):
        """
        Generate signal from ensemble model
        
        Args:
            asset: Asset symbol
            timestamp: Current timestamp
            data_dict: Dictionary of price data by timeframe
            
        Returns:
            Tuple: (direction, strength, confidence, take_profit_pct, stop_loss_pct)
        """
        if asset not in self.ensemble_configs:
            return None, 0.0, 0.0, 0.12, 0.04
        
        # Get configuration
        ensemble_config = self.ensemble_configs[asset]['config']
        weights = self.ensemble_configs[asset]['weights']
        position_sizing = self.ensemble_configs[asset]['position_sizing']
        
        # Combine signals from constituent models
        direction_score = 0.0  # Positive for long, negative for short
        confidence_sum = 0.0
        
        # ARIMA model (30% weight by default)
        arima_weight = weights.get('arima', 0.3)
        arima_direction, arima_strength, arima_confidence, _, _ = self.get_arima_signal(
            asset, timestamp, data_dict['1h'])
        
        if arima_direction is not None:
            direction_factor = 1.0 if arima_direction == 'long' else -1.0
            direction_score += direction_factor * arima_strength * arima_weight
            confidence_sum += arima_confidence * arima_weight
        
        # Adaptive model (20% weight by default)
        adaptive_weight = weights.get('adaptive', 0.2)
        adaptive_direction, adaptive_strength, adaptive_confidence, _, _ = self.get_adaptive_signal(
            asset, timestamp, data_dict['1h'])
        
        if adaptive_direction is not None:
            direction_factor = 1.0 if adaptive_direction == 'long' else -1.0
            direction_score += direction_factor * adaptive_strength * adaptive_weight
            confidence_sum += adaptive_confidence * adaptive_weight
        
        # Technical model (20% weight by default - simplified here)
        technical_weight = weights.get('technical', 0.2)
        # Simple technical signal based on RSI and ADX
        recent_data = data_dict['1h'][data_dict['1h'].index <= timestamp].tail(50)
        if len(recent_data) >= 50:
            # RSI calculation
            delta = recent_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Simple technical direction based on RSI
            if current_rsi > 50:
                tech_direction = 'long'
                tech_strength = min(0.8, (current_rsi - 50) / 30)
            else:
                tech_direction = 'short'
                tech_strength = min(0.8, (50 - current_rsi) / 30)
            
            tech_confidence = 0.7  # Fixed confidence
            
            direction_factor = 1.0 if tech_direction == 'long' else -1.0
            direction_score += direction_factor * tech_strength * technical_weight
            confidence_sum += tech_confidence * technical_weight
        
        # LSTM model (30% weight by default - simplified here)
        lstm_weight = weights.get('lstm', 0.3)
        # Simple LSTM signal proxy based on recent price action
        recent_data = data_dict['1h'][data_dict['1h'].index <= timestamp].tail(24)
        if len(recent_data) >= 24:
            # Simple momentum calculation
            start_price = recent_data['close'].iloc[0]
            end_price = recent_data['close'].iloc[-1]
            momentum = end_price / start_price
            
            if momentum > 1.0:
                lstm_direction = 'long'
                lstm_strength = min(0.8, (momentum - 1.0) * 10)
            else:
                lstm_direction = 'short'
                lstm_strength = min(0.8, (1.0 - momentum) * 10)
            
            lstm_confidence = 0.8  # Fixed confidence
            
            direction_factor = 1.0 if lstm_direction == 'long' else -1.0
            direction_score += direction_factor * lstm_strength * lstm_weight
            confidence_sum += lstm_confidence * lstm_weight
        
        # Determine final direction and strength
        if direction_score > 0:
            final_direction = 'long'
            final_strength = min(0.9, direction_score)
        elif direction_score < 0:
            final_direction = 'short'
            final_strength = min(0.9, abs(direction_score))
        else:
            return None, 0.0, 0.0, 0.12, 0.04
        
        # Normalize confidence
        final_confidence = min(0.9, confidence_sum)
        
        # Get position sizing params
        take_profit_pct = position_sizing.get('take_profit_multiplier', 3.0) * 0.04
        stop_loss_pct = position_sizing.get('stop_loss_multiplier', 1.0) * 0.04
        
        return final_direction, final_strength, final_confidence, take_profit_pct, stop_loss_pct
    
    def get_current_price(self, asset, timestamp, timeframe='1h'):
        """
        Get the price at a specific timestamp
        
        Args:
            asset: Asset symbol
            timestamp: Timestamp to get price for
            timeframe: Timeframe to use
            
        Returns:
            float: Price at timestamp
        """
        if asset not in self.data or timeframe not in self.data[asset]:
            return None
        
        # Get data for the asset and timeframe
        data = self.data[asset][timeframe]
        
        # Find the closest timestamp before or equal to the given timestamp
        data_before = data[data.index <= timestamp]
        
        if data_before.empty:
            return None
        
        # Return the close price of the latest candle
        return data_before.iloc[-1]['close']
    
    def run_backtest(self, assets, strategies, allocation=None):
        """
        Run backtest for multiple assets and strategies
        
        Args:
            assets: List of assets to backtest
            strategies: List of strategies to backtest
            allocation: Optional capital allocation by asset
            
        Returns:
            Dict: Backtest results
        """
        logger.info(f"Running backtest for {len(assets)} assets and {len(strategies)} strategies")
        
        # Check if we have data for all assets
        for asset in assets:
            if asset not in self.data or '1h' not in self.data[asset]:
                logger.error(f"Missing data for {asset}")
                return {
                    "error": f"Missing data for {asset}",
                    "metrics": {},
                    "trades": [],
                    "equity_curve": [],
                    "report_file": ""
                }
        
        # Set capital allocation if provided
        if allocation is not None:
            self.portfolio.set_allocation(allocation)
        else:
            # Default: equal allocation for all assets
            equal_allocation = {asset: 1.0 / len(assets) for asset in assets}
            self.portfolio.set_allocation(equal_allocation)
        
        # Get the common timeframe ('1h')
        common_timeframe = '1h'
        
        # Collect all timestamps from all assets
        all_timestamps = []
        for asset in assets:
            if asset in self.data and common_timeframe in self.data[asset]:
                all_timestamps.extend(self.data[asset][common_timeframe].index)
        
        # Sort and deduplicate timestamps
        all_timestamps = sorted(set(all_timestamps))
        
        # Filter timestamps by date range if specified
        if self.start_date is not None:
            all_timestamps = [ts for ts in all_timestamps if ts >= self.start_date]
        
        if self.end_date is not None:
            all_timestamps = [ts for ts in all_timestamps if ts <= self.end_date]
        
        # Process each timestamp
        logger.info(f"Processing {len(all_timestamps)} timestamps")
        
        for i, timestamp in enumerate(all_timestamps):
            if i % 100 == 0:
                logger.info(f"Processing timestamp {i+1}/{len(all_timestamps)}: {timestamp}")
            
            # Get current prices for all assets
            current_prices = {}
            for asset in assets:
                price = self.get_current_price(asset, timestamp, common_timeframe)
                if price is not None:
                    current_prices[asset] = price
            
            # Update existing positions
            self.portfolio.update_positions(current_prices, timestamp)
            
            # Generate new signals and open positions
            for asset in assets:
                if asset not in current_prices:
                    continue
                
                # Get data dict for all timeframes
                data_dict = {}
                for timeframe in TIMEFRAMES:
                    if asset in self.data and timeframe in self.data[asset]:
                        data_dict[timeframe] = self.data[asset][timeframe]
                
                # Current market regime
                market_regime = detect_market_regime(data_dict[common_timeframe][data_dict[common_timeframe].index <= timestamp].tail(50))
                
                # Check each strategy
                for strategy in strategies:
                    # Skip if already have a position for this asset and strategy
                    if self.portfolio.get_position(asset, strategy) is not None:
                        continue
                    
                    # Generate signal based on strategy
                    if strategy.lower() == 'arima':
                        direction, strength, confidence, take_profit_pct, stop_loss_pct = \
                            self.get_arima_signal(asset, timestamp, data_dict[common_timeframe])
                    elif strategy.lower() == 'adaptive':
                        direction, strength, confidence, take_profit_pct, stop_loss_pct = \
                            self.get_adaptive_signal(asset, timestamp, data_dict[common_timeframe])
                    elif strategy.lower() == 'ensemble':
                        direction, strength, confidence, take_profit_pct, stop_loss_pct = \
                            self.get_ensemble_signal(asset, timestamp, data_dict)
                    else:
                        continue  # Skip unsupported strategy
                    
                    # Skip if no direction
                    if direction is None:
                        continue
                    
                    # Open position if signal strength is high enough
                    if strength >= 0.5 and confidence >= 0.5:
                        self.portfolio.open_position(
                            asset=asset,
                            strategy=strategy,
                            direction=direction,
                            price=current_prices[asset],
                            time=timestamp,
                            signal_strength=strength,
                            ml_confidence=confidence,
                            market_regime=market_regime,
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            risk_pct=0.04  # Fixed risk percentage
                        )
            
            # Update equity curve
            self.portfolio.update_equity_curve(timestamp, current_prices)
        
        # Calculate metrics
        metrics = self.portfolio.calculate_metrics()
        
        # Generate report
        report_file = self._generate_report(assets, strategies, metrics)
        
        return {
            "metrics": metrics,
            "trades": self.portfolio.trade_history,
            "equity_curve": self.portfolio.equity_curve,
            "report_file": report_file
        }
    
    def _generate_report(self, assets, strategies, metrics):
        """
        Generate backtest report
        
        Args:
            assets: List of assets backtested
            strategies: List of strategies backtested
            metrics: Performance metrics
            
        Returns:
            str: Path to report file
        """
        # Create timestamp for report filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report directory
        report_dir = f"{BACKTEST_DIR}/report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_file = f"{report_dir}/metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save trades to CSV
        trades_file = f"{report_dir}/trades.csv"
        pd.DataFrame(self.portfolio.trade_history).to_csv(trades_file, index=False)
        
        # Save equity curve to CSV
        equity_file = f"{report_dir}/equity_curve.csv"
        pd.DataFrame([
            {'timestamp': point['time'], 'equity': point['equity'], 'unrealized_pnl': point['unrealized_pnl']}
            for point in self.portfolio.equity_curve
        ]).to_csv(equity_file, index=False)
        
        # Generate summary chart
        self._generate_summary_chart(report_dir)
        
        # Save backtest configuration
        config_file = f"{report_dir}/config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'assets': assets,
                'strategies': strategies,
                'initial_capital': self.initial_capital,
                'fee_rate': self.fee_rate,
                'slippage': self.slippage,
                'start_date': self.start_date.strftime("%Y-%m-%d") if self.start_date else None,
                'end_date': self.end_date.strftime("%Y-%m-%d") if self.end_date else None
            }, f, indent=2, default=str)
        
        # Return path to report directory
        return report_dir
    
    def _generate_summary_chart(self, report_dir):
        """
        Generate summary chart for backtest results
        
        Args:
            report_dir: Directory to save chart to
        """
        if not self.portfolio.equity_curve:
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract data
        timestamps = [point['time'] for point in self.portfolio.equity_curve]
        equity = [point['equity'] for point in self.portfolio.equity_curve]
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        
        # Plot equity curve
        ax1.plot(timestamps, equity, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Add markers for trades
        for trade in self.portfolio.trade_history:
            if trade['type'] == 'entry':
                marker = '^' if trade['direction'] == 'long' else 'v'
                color = 'g' if trade['direction'] == 'long' else 'r'
                ax1.scatter(pd.to_datetime(trade['time']), trade['price'], marker=marker, color=color, alpha=0.7)
        
        # Format equity axis
        ax1.set_title('Backtest Results', fontsize=16)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Plot drawdown
        ax2.fill_between(timestamps, drawdown, 0, color='r', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Set ylim with some padding
        ax2.set_ylim(min(drawdown) * 1.1, 1)
        
        # Add metrics as text
        metrics = self.portfolio.calculate_metrics()
        metrics_text = (
            f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
        )
        
        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{report_dir}/summary_chart.png", dpi=150)
        plt.close()
        
        # Also save to main backtest directory
        plt.figure(figsize=(12, 8))
        plt.plot(timestamps, equity)
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        plt.title('Portfolio Value')
        plt.ylabel('USD')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{BACKTEST_DIR}/backtest_returns.png", dpi=150)
        plt.close()
        
        # Save metrics to summary CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df['start_date'] = timestamps[0] if timestamps else None
        metrics_df['end_date'] = timestamps[-1] if timestamps else None
        metrics_df['assets'] = ','.join(self.data.keys())
        metrics_df['strategies'] = ','.join(list(set([
            trade['strategy'] for trade in self.portfolio.trade_history if 'strategy' in trade
        ])))
        
        # Save to backtest summary
        summary_path = f"{BACKTEST_DIR}/backtest_summary.csv"
        metrics_df.to_csv(summary_path, index=False)
        
        # Save detailed report to JSON
        report_data = {
            'metrics': metrics,
            'trades': [
                {k: str(v) if isinstance(v, datetime) else v for k, v in trade.items()}
                for trade in self.portfolio.trade_history
            ],
            'equity_curve': [
                {'time': str(point['time']), 'equity': point['equity'], 'unrealized_pnl': point['unrealized_pnl']}
                for point in self.portfolio.equity_curve
            ]
        }
        
        with open(f"{BACKTEST_DIR}/backtest_report.json", 'w') as f:
            json.dump(report_data, f, indent=2)

def optimize_strategy_parameters(assets, strategies, parameter_ranges, initial_capital, num_trials=10):
    """
    Optimize strategy parameters using grid search
    
    Args:
        assets: List of assets to optimize for
        strategies: List of strategies to optimize
        parameter_ranges: Dictionary of parameter ranges by strategy
        initial_capital: Initial capital for backtesting
        num_trials: Number of trials to run
        
    Returns:
        Dict: Optimization results
    """
    logger.info(f"Starting strategy parameter optimization with {num_trials} trials")
    
    # Create a new engine for each trial
    engines = []
    params_list = []
    
    # Generate parameter combinations based on strategy
    for strategy in strategies:
        if strategy not in parameter_ranges:
            continue
        
        strategy_params = parameter_ranges[strategy]
        
        # Generate all combinations of parameters
        param_keys = list(strategy_params.keys())
        param_values = list(strategy_params.values())
        param_combinations = []
        
        # For simplicity, use a grid search approach
        if len(param_keys) == 1:
            param_combinations = [{param_keys[0]: p} for p in param_values[0]]
        elif len(param_keys) == 2:
            for p1 in param_values[0]:
                for p2 in param_values[1]:
                    param_combinations.append({param_keys[0]: p1, param_keys[1]: p2})
        else:
            # For more parameters, randomly sample from the space
            import random
            for _ in range(num_trials):
                combo = {}
                for i, key in enumerate(param_keys):
                    combo[key] = random.choice(param_values[i])
                param_combinations.append(combo)
        
        # Create engines with each parameter combination
        for params in param_combinations:
            engine = ImprovedBacktestEngine(initial_capital=initial_capital)
            engine.load_historical_data(assets)
            engines.append(engine)
            params_list.append({strategy: params})
    
    # Run backtests for each parameter combination
    results = []
    for i, (engine, params) in enumerate(zip(engines, params_list)):
        logger.info(f"Running trial {i+1}/{len(engines)} with parameters: {params}")
        
        # Run backtest with current parameters
        result = engine.run_backtest(assets, strategies)
        
        # Store results
        results.append({
            'params': params,
            'metrics': result['metrics']
        })
    
    # Find best parameters
    best_result = None
    best_return = -float('inf')
    
    for result in results:
        total_return = result['metrics'].get('total_return_pct', 0.0)
        win_rate = result['metrics'].get('win_rate', 0.0)
        
        # Combined score: return and win rate with higher emphasis on return
        score = total_return * 0.7 + win_rate * 100 * 0.3
        
        if score > best_return:
            best_return = score
            best_result = result
    
    if best_result is None:
        logger.warning("No valid results found during optimization")
        return {
            'best_params': {},
            'best_metrics': {},
            'all_results': results
        }
    
    logger.info(f"Optimization complete. Best parameters: {best_result['params']}")
    logger.info(f"Best return: {best_result['metrics'].get('total_return_pct', 0.0):.2f}%")
    
    # Return results
    return {
        'best_params': best_result['params'],
        'best_metrics': best_result['metrics'],
        'all_results': results
    }

def run_full_backtest(assets, strategies, allocation=None, optimize=False, 
                     num_trials=10, start_date=None, end_date=None, 
                     initial_capital=INITIAL_CAPITAL):
    """
    Run a full backtest with optional optimization
    
    Args:
        assets: List of assets to backtest
        strategies: List of strategies to backtest
        allocation: Optional capital allocation by asset
        optimize: Whether to optimize strategy parameters
        num_trials: Number of optimization trials
        start_date: Start date for backtesting
        end_date: End date for backtesting
        initial_capital: Initial capital in USD
        
    Returns:
        Dict: Backtest results
    """
    logger.info("Starting full backtest...")
    logger.info(f"Assets: {assets}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Initial capital: ${initial_capital}")
    
    if optimize:
        logger.info(f"Optimizing strategy parameters with {num_trials} trials...")
        
        # Define parameter ranges for optimization
        parameter_ranges = {
            'arima': {
                'signal_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
                'confidence_multiplier': [0.8, 1.0, 1.2, 1.5, 1.8],
                'take_profit_multiplier': [1.5, 2.0, 2.5, 3.0, 3.5],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            },
            'adaptive': {
                'ema_short': [7, 9, 11, 13],
                'ema_long': [17, 21, 25, 29],
                'signal_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
                'confidence_multiplier': [0.8, 1.0, 1.2, 1.5],
                'take_profit_multiplier': [1.5, 2.0, 2.5, 3.0],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            },
            'ensemble': {
                'arima_weight': [0.2, 0.3, 0.4],
                'adaptive_weight': [0.1, 0.2, 0.3],
                'lstm_weight': [0.2, 0.3, 0.4],
                'technical_weight': [0.1, 0.2, 0.3],
                'take_profit_multiplier': [2.0, 2.5, 3.0, 3.5],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            }
        }
        
        # Run optimization
        optimization_results = optimize_strategy_parameters(
            assets=assets,
            strategies=strategies,
            parameter_ranges=parameter_ranges,
            initial_capital=initial_capital,
            num_trials=num_trials
        )
        
        return optimization_results
    else:
        # Run backtest without optimization
        engine = ImprovedBacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # Load data and models
        engine.load_historical_data(assets, ['1m', '5m', '15m', '1h', '4h', '1d'])
        if 'ensemble' in strategies or 'ml' in strategies:
            engine.load_ml_models(assets)
        
        # Run backtest
        results = engine.run_backtest(assets, strategies, allocation=allocation)
        
        return {
            'results': results
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Improved Comprehensive Backtesting System')
    
    parser.add_argument('--pairs', type=str, nargs='+', default=SUPPORTED_ASSETS,
                      help='Trading pairs to backtest (e.g., SOL/USD BTC/USD)')
    
    parser.add_argument('--strategies', type=str, nargs='+', default=['ensemble'],
                      help='Trading strategies to backtest (e.g., arima adaptive ensemble)')
    
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'],
                      help='Timeframes for backtesting (e.g., 1m 5m 1h 4h 1d)')
    
    parser.add_argument('--days', type=int, default=30,
                      help='Number of days to backtest (most recent)')
    
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                      help='Initial capital in USD')
    
    parser.add_argument('--risk', type=float, default=0.04,
                      help='Risk per trade as percentage (e.g., 0.04 for 4%%)')
    
    parser.add_argument('--leverage', type=float, default=25.0,
                      help='Base leverage to use for backtesting')
    
    parser.add_argument('--max-leverage', type=float, default=125.0,
                      help='Maximum leverage for high-confidence trades')
    
    parser.add_argument('--start', type=str,
                      help='Start date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--end', type=str,
                      help='End date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize strategy parameters')
    
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of optimization trials')
    
    parser.add_argument('--compare-fees', action='store_true',
                      help='Compare different fee structures')
    
    parser.add_argument('--compare-slippage', action='store_true',
                      help='Compare different slippage assumptions')
    
    parser.add_argument('--output', type=str, default=BACKTEST_DIR,
                      help='Output directory for results')
    
    parser.add_argument('--target-accuracy', type=float, default=90.0,
                      help='Target ML model accuracy percentage')
    
    parser.add_argument('--target-return', type=float, default=1000.0,
                      help='Target return percentage')
    
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    elif args.days > 0:
        start_date = datetime.now() - timedelta(days=args.days)
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Process pairs to ensure correct format
    pairs = []
    for pair in args.pairs:
        if '/' in pair:
            pairs.append(pair)
        else:
            # Add '/' between asset and USD (e.g., SOL -> SOL/USD)
            pairs.append(f"{pair}/USD")
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("IMPROVED BACKTESTING")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {' '.join(pairs)}")
    logger.info(f"Timeframes: {' '.join(args.timeframes)}")
    logger.info(f"Strategies: {' '.join(args.strategies)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"Days for backtesting: {args.days}")
    logger.info(f"Leverage settings: Base={args.leverage}x, Max={args.max_leverage}x")
    logger.info(f"Risk per trade: {args.risk * 100:.1f}%")
    logger.info(f"Optimization: {args.optimize}")
    logger.info(f"Compare fees: {args.compare_fees}")
    logger.info(f"Compare slippage: {args.compare_slippage}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Target accuracy: {args.target_accuracy}%")
    logger.info(f"Target return: {args.target_return}%")
    logger.info("=" * 80)
    
    # Run backtest
    results = run_full_backtest(
        assets=pairs,
        strategies=args.strategies,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        optimize=args.optimize,
        num_trials=args.trials
    )
    
    # Display results
    if args.optimize:
        logger.info("Optimization completed")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best return: ${results['best_metrics'].get('total_return', 0.0):.2f} "
                   f"({results['best_metrics'].get('total_return_pct', 0.0):.2f}%)")
    else:
        try:
            # Check if we have valid results
            if 'results' in results and 'metrics' in results['results']:
                metrics = results['results']['metrics']
                logger.info("Backtest completed successfully")
                logger.info(f"Total trades: {metrics.get('total_trades', 0)}")
                logger.info(f"Win rate: {metrics.get('win_rate', 0.0) * 100:.2f}%")
                logger.info(f"Return: ${metrics.get('total_return', 0.0):.2f} "
                           f"({metrics.get('total_return_pct', 0.0):.2f}%)")
                logger.info(f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%")
                logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
                logger.info(f"Report directory: {results['results'].get('report_file', 'N/A')}")
            else:
                logger.warning("Backtest completed but no metrics available")
        except Exception as e:
            logger.error(f"Error processing backtest results: {e}")
    
    logger.info("=" * 80)
    logger.info("Backtest completed!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()