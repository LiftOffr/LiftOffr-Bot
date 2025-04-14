#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Hyper-Optimized Trading Bot

This script implements an advanced backtesting framework that:
1. Uses actual historical data at multiple timeframes
2. Simulates real-world trading conditions (slippage, fees, liquidity)
3. Tests multiple assets simultaneously
4. Evaluates extreme leverage settings
5. Generates detailed performance metrics and visualizations
6. Optimizes strategy parameters for maximum profit

The backtester simulates the exact same execution logic used in live trading
to ensure results are as realistic as possible.
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
from hyper_optimized_ml_training import load_model_with_metadata
from market_context import detect_market_regime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_backtest.log")
    ]
)

logger = logging.getLogger(__name__)

# Create backtest results directory
BACKTEST_DIR = "backtest_results"
os.makedirs(BACKTEST_DIR, exist_ok=True)

# Constants
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
INITIAL_CAPITAL = 20000.0  # $20,000 starting capital
DEFAULT_FEE_RATE = 0.0026  # 0.26% taker fee on Kraken
DEFAULT_SLIPPAGE = 0.001   # 0.1% average slippage
MAX_POSITIONS_PER_ASSET = 1  # Maximum of 1 position per asset at a time

# Position settings
DEFAULT_STOP_LOSS_PCT = 0.04  # Default 4% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.12  # Default 12% take profit

# Strategies to backtest
STRATEGIES = ["ARIMA", "Adaptive", "ML", "Integrated"]

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
                    margin_pct: Optional[float] = None) -> Optional[Position]:
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
            margin_pct: Margin percentage to use (optional)
            
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
                asset=asset
            )
        
        # Default margin percentage
        if margin_pct is None:
            margin_pct = 0.22  # Default to 22% of available capital
        
        # Calculate position size
        margin_amount = available_capital * margin_pct
        position_value = margin_amount * leverage
        quantity = position_value / price
        
        # Apply slippage to entry price
        if direction.lower() == 'long':
            adjusted_price = price * (1 + self.slippage)
        else:  # short
            adjusted_price = price * (1 - self.slippage)
        
        # Use provided or default stop loss and take profit
        stop_loss = stop_loss_pct or DEFAULT_STOP_LOSS_PCT
        take_profit = take_profit_pct or DEFAULT_TAKE_PROFIT_PCT
        
        # Create position
        position = Position(
            asset=asset,
            entry_price=adjusted_price,
            entry_time=time,
            direction=direction,
            quantity=quantity,
            leverage=leverage,
            strategy=strategy,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit
        )
        
        # Add to open positions
        if asset not in self.positions:
            self.positions[asset] = {}
        self.positions[asset][strategy] = position
        
        # Record trade in history
        self.trade_history.append({
            'asset': asset,
            'strategy': strategy,
            'action': 'open',
            'direction': direction,
            'price': adjusted_price,
            'time': time,
            'quantity': quantity,
            'leverage': leverage,
            'margin_amount': margin_amount,
            'signal_strength': signal_strength,
            'ml_confidence': ml_confidence,
            'market_regime': market_regime
        })
        
        logger.debug(f"Opened position: {position}")
        
        return position
    
    def close_position(self, asset: str, strategy: str, price: float, time: datetime,
                     reason: str) -> Optional[Position]:
        """
        Close an open position
        
        Args:
            asset: Asset symbol
            strategy: Strategy name
            price: Exit price
            time: Exit timestamp
            reason: Exit reason
            
        Returns:
            Optional[Position]: Closed position or None if no open position
        """
        position = self.get_position(asset, strategy)
        if position is None:
            return None
        
        # Apply slippage to exit price
        if position.direction == 'long':
            adjusted_price = price * (1 - self.slippage)
        else:  # short
            adjusted_price = price * (1 + self.slippage)
        
        # Update position with exit details
        position.exit_price = adjusted_price
        position.exit_time = time
        position.exit_reason = reason
        
        # Calculate PnL
        entry_value = position.entry_price * position.quantity
        exit_value = position.exit_price * position.quantity
        
        # Calculate fees (entry and exit)
        fees = (entry_value + exit_value) * self.fee_rate
        position.fees_paid = fees
        
        # Calculate PnL
        if position.direction == 'long':
            position.pnl = ((position.exit_price - position.entry_price) * 
                          position.quantity * position.leverage) - fees
            position.pnl_pct = ((position.exit_price / position.entry_price - 1) * 
                              100 * position.leverage)
        else:  # short
            position.pnl = ((position.entry_price - position.exit_price) * 
                          position.quantity * position.leverage) - fees
            position.pnl_pct = ((1 - position.exit_price / position.entry_price) * 
                              100 * position.leverage)
        
        # Update portfolio capital
        self.current_capital += position.pnl
        
        # Update metrics
        self.total_trades += 1
        if position.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        drawdown = self.peak_capital - self.current_capital
        drawdown_pct = drawdown / self.peak_capital * 100
        
        if drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct
        
        self.drawdowns.append((time, drawdown_pct))
        
        # Move position to closed positions
        self.closed_positions.append(position)
        del self.positions[asset][strategy]
        
        # Record trade in history
        self.trade_history.append({
            'asset': asset,
            'strategy': strategy,
            'action': 'close',
            'direction': position.direction,
            'price': adjusted_price,
            'time': time,
            'quantity': position.quantity,
            'leverage': position.leverage,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct,
            'fees': fees,
            'reason': reason
        })
        
        logger.debug(f"Closed position: {position}")
        
        return position
    
    def update_positions(self, asset: str, current_price: float, current_time: datetime) -> None:
        """
        Update all open positions for an asset
        
        Args:
            asset: Asset symbol
            current_price: Current price
            current_time: Current timestamp
        """
        if asset not in self.positions:
            return
        
        positions_to_close = []
        
        # Update each open position
        for strategy, position in self.positions[asset].items():
            if position.exit_price is not None:
                continue  # Skip closed positions
            
            # Update trailing stop
            position.update_trailing_stop(current_price)
            
            # Check exit conditions
            if position.check_exit_conditions(current_price, current_time, self.fee_rate):
                positions_to_close.append((asset, strategy, position.exit_price, 
                                         current_time, position.exit_reason))
        
        # Close positions that hit exit conditions
        for asset, strategy, price, time, reason in positions_to_close:
            self.close_position(asset, strategy, price, time, reason)
    
    def update_equity_curve(self, time: datetime) -> None:
        """
        Update equity curve with current portfolio value
        
        Args:
            time: Current timestamp
        """
        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        for asset, strategies in self.positions.items():
            for strategy, position in strategies.items():
                if position.exit_price is None:  # Only include open positions
                    # We would need the current price for each asset here
                    # For now, we'll use the entry price as a placeholder
                    unrealized_pnl += position.get_unrealized_pnl(position.entry_price)
        
        # Add to equity curve
        self.equity_curve.append((time, self.current_capital + unrealized_pnl))
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return portfolio performance metrics
        
        Returns:
            Dict: Dictionary of performance metrics
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        loss_rate = self.losing_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0)
        total_loss = sum(abs(pos.pnl) for pos in self.closed_positions if pos.pnl < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average trade
        avg_trade = sum(pos.pnl for pos in self.closed_positions) / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average win and loss
        avg_win = sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0) / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = sum(pos.pnl for pos in self.closed_positions if pos.pnl < 0) / self.losing_trades if self.losing_trades > 0 else 0
        
        # Calculate total return
        total_return = self.current_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = [curr[1] / prev[1] - 1 for prev, curr in zip(self.equity_curve[:-1], self.equity_curve[1:])]
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate strategy performance
        strategy_metrics = {}
        for strategy in STRATEGIES:
            strategy_trades = [pos for pos in self.closed_positions if pos.strategy == strategy]
            if not strategy_trades:
                continue
            
            strategy_win_rate = sum(1 for pos in strategy_trades if pos.pnl > 0) / len(strategy_trades)
            strategy_profit = sum(pos.pnl for pos in strategy_trades)
            strategy_profit_pct = strategy_profit / self.initial_capital * 100
            
            strategy_metrics[strategy] = {
                'trades': len(strategy_trades),
                'win_rate': strategy_win_rate * 100,
                'profit': strategy_profit,
                'profit_pct': strategy_profit_pct,
                'avg_leverage': sum(pos.leverage for pos in strategy_trades) / len(strategy_trades)
            }
        
        # Calculate asset performance
        asset_metrics = {}
        for asset in SUPPORTED_ASSETS:
            asset_trades = [pos for pos in self.closed_positions if pos.asset == asset]
            if not asset_trades:
                continue
            
            asset_win_rate = sum(1 for pos in asset_trades if pos.pnl > 0) / len(asset_trades)
            asset_profit = sum(pos.pnl for pos in asset_trades)
            asset_profit_pct = asset_profit / self.initial_capital * 100
            
            asset_metrics[asset] = {
                'trades': len(asset_trades),
                'win_rate': asset_win_rate * 100,
                'profit': asset_profit,
                'profit_pct': asset_profit_pct,
                'avg_leverage': sum(pos.leverage for pos in asset_trades) / len(asset_trades)
            }
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate * 100,
            'loss_rate': loss_rate * 100,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'strategy_metrics': strategy_metrics,
            'asset_metrics': asset_metrics
        }
    
    def generate_report(self, title: str = "Backtest Report") -> str:
        """
        Generate a detailed backtest report
        
        Args:
            title: Report title
            
        Returns:
            str: Path to the saved report
        """
        metrics = self.get_metrics()
        
        # Create report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(BACKTEST_DIR, f"backtest_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{title.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial Capital: ${metrics['initial_capital']:.2f}\n")
            f.write(f"Final Capital: ${metrics['final_capital']:.2f}\n")
            f.write(f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)\n")
            f.write(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n")
            
            f.write("TRADE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']:.2f}%)\n")
            f.write(f"Losing Trades: {metrics['losing_trades']} ({metrics['loss_rate']:.2f}%)\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Average Trade: ${metrics['avg_trade']:.2f}\n")
            f.write(f"Average Win: ${metrics['avg_win']:.2f}\n")
            f.write(f"Average Loss: ${metrics['avg_loss']:.2f}\n\n")
            
            f.write("STRATEGY PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for strategy, stats in metrics['strategy_metrics'].items():
                f.write(f"{strategy}:\n")
                f.write(f"  Trades: {stats['trades']}\n")
                f.write(f"  Win Rate: {stats['win_rate']:.2f}%\n")
                f.write(f"  Profit: ${stats['profit']:.2f} ({stats['profit_pct']:.2f}%)\n")
                f.write(f"  Avg Leverage: {stats['avg_leverage']:.2f}x\n\n")
            
            f.write("ASSET PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for asset, stats in metrics['asset_metrics'].items():
                f.write(f"{asset}:\n")
                f.write(f"  Trades: {stats['trades']}\n")
                f.write(f"  Win Rate: {stats['win_rate']:.2f}%\n")
                f.write(f"  Profit: ${stats['profit']:.2f} ({stats['profit_pct']:.2f}%)\n")
                f.write(f"  Avg Leverage: {stats['avg_leverage']:.2f}x\n\n")
            
            f.write("TOP 10 WINNING TRADES\n")
            f.write("-" * 80 + "\n")
            winning_trades = sorted(self.closed_positions, key=lambda x: x.pnl, reverse=True)[:10]
            for i, trade in enumerate(winning_trades, 1):
                f.write(f"{i}. {trade.asset} {trade.direction.upper()} @ {trade.entry_price:.4f} -> {trade.exit_price:.4f}\n")
                f.write(f"   Strategy: {trade.strategy} | Leverage: {trade.leverage}x\n")
                f.write(f"   PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)\n")
                f.write(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"   Exit: {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"   Reason: {trade.exit_reason}\n\n")
            
            f.write("TOP 10 LOSING TRADES\n")
            f.write("-" * 80 + "\n")
            losing_trades = sorted(self.closed_positions, key=lambda x: x.pnl)[:10]
            for i, trade in enumerate(losing_trades, 1):
                f.write(f"{i}. {trade.asset} {trade.direction.upper()} @ {trade.entry_price:.4f} -> {trade.exit_price:.4f}\n")
                f.write(f"   Strategy: {trade.strategy} | Leverage: {trade.leverage}x\n")
                f.write(f"   PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)\n")
                f.write(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"   Exit: {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"   Reason: {trade.exit_reason}\n\n")
        
        logger.info(f"Backtest report saved to {report_file}")
        
        # Generate plots
        self.generate_performance_plots(title, timestamp)
        
        return report_file
    
    def generate_performance_plots(self, title: str, timestamp: str) -> None:
        """
        Generate performance plots
        
        Args:
            title: Plot title
            timestamp: Timestamp for filenames
        """
        # Create plot directory
        plot_dir = os.path.join(BACKTEST_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Equity curve
        plt.figure(figsize=(12, 6))
        dates = [date for date, value in self.equity_curve]
        values = [value for date, value in self.equity_curve]
        
        plt.plot(dates, values, linewidth=2)
        plt.title(f"Equity Curve - {title}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.gcf().autofmt_xdate()
        
        # Format y-axis to show dollar values
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"equity_curve_{timestamp}.png"))
        plt.close()
        
        # Drawdown chart
        plt.figure(figsize=(12, 6))
        drawdown_dates = [date for date, value in self.drawdowns]
        drawdown_values = [value for date, value in self.drawdowns]
        
        plt.plot(drawdown_dates, drawdown_values, linewidth=2, color='red')
        plt.title(f"Drawdown Chart - {title}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(drawdown_dates) // 10)))
        plt.gcf().autofmt_xdate()
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"drawdown_{timestamp}.png"))
        plt.close()
        
        # Strategy performance
        metrics = self.get_metrics()
        if metrics['strategy_metrics']:
            plt.figure(figsize=(14, 6))
            
            strategies = list(metrics['strategy_metrics'].keys())
            profits = [metrics['strategy_metrics'][s]['profit_pct'] for s in strategies]
            win_rates = [metrics['strategy_metrics'][s]['win_rate'] for s in strategies]
            trades = [metrics['strategy_metrics'][s]['trades'] for s in strategies]
            
            # Normalize trades for visualization
            max_trades = max(trades) if trades else 1
            normalized_trades = [t / max_trades * 50 for t in trades]  # Scale for bubble size
            
            # Create scatter plot
            sc = plt.scatter(win_rates, profits, s=normalized_trades, alpha=0.6)
            
            # Add strategy labels
            for i, strategy in enumerate(strategies):
                plt.annotate(strategy, (win_rates[i], profits[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title(f"Strategy Performance - {title}")
            plt.xlabel("Win Rate (%)")
            plt.ylabel("Profit (%)")
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=50, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"strategy_performance_{timestamp}.png"))
            plt.close()
        
        # Asset performance
        if metrics['asset_metrics']:
            plt.figure(figsize=(10, 5))
            
            assets = list(metrics['asset_metrics'].keys())
            profits = [metrics['asset_metrics'][a]['profit_pct'] for a in assets]
            
            colors = ['green' if p > 0 else 'red' for p in profits]
            
            plt.bar(assets, profits, color=colors)
            plt.title(f"Asset Performance - {title}")
            plt.xlabel("Asset")
            plt.ylabel("Profit (%)")
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add profit values on top of bars
            for i, p in enumerate(profits):
                plt.text(i, p + (1 if p > 0 else -1), f"{p:.1f}%", 
                       ha='center', va='bottom' if p > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"asset_performance_{timestamp}.png"))
            plt.close()
        
        # Monthly returns
        if len(self.equity_curve) > 30:
            # Group equity curve by month
            df = pd.DataFrame(self.equity_curve, columns=['date', 'value'])
            df['month'] = df['date'].dt.strftime('%Y-%m')
            monthly_returns = []
            
            months = df['month'].unique()
            for i in range(1, len(months)):
                prev_month = df[df['month'] == months[i-1]]['value'].iloc[-1]
                curr_month = df[df['month'] == months[i]]['value'].iloc[-1]
                monthly_return = (curr_month / prev_month - 1) * 100
                monthly_returns.append((months[i], monthly_return))
            
            if monthly_returns:
                plt.figure(figsize=(12, 6))
                
                months = [m for m, r in monthly_returns]
                returns = [r for m, r in monthly_returns]
                colors = ['green' if r > 0 else 'red' for r in returns]
                
                plt.bar(months, returns, color=colors)
                plt.title(f"Monthly Returns - {title}")
                plt.xlabel("Month")
                plt.ylabel("Return (%)")
                plt.grid(True, alpha=0.3, axis='y')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"monthly_returns_{timestamp}.png"))
                plt.close()

class BacktestEngine:
    """Class for running backtests"""
    
    def __init__(self, start_date: datetime = None, end_date: datetime = None,
               initial_capital: float = INITIAL_CAPITAL,
               fee_rate: float = DEFAULT_FEE_RATE,
               slippage: float = DEFAULT_SLIPPAGE):
        """
        Initialize backtest engine
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital in USD
            fee_rate: Trading fee rate
            slippage: Average slippage
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=180))
        self.end_date = end_date or datetime.now()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Historical data by asset and timeframe
        self.historical_data = {}  # {asset: {timeframe: DataFrame}}
        
        # ML models by asset
        self.ml_models = {}  # {asset: model}
        
        # Strategy parameters
        self.strategy_params = {}
        
        # Portfolio
        self.portfolio = BacktestPortfolio(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage=slippage
        )
    
    def load_historical_data(self, assets: List[str], timeframes: List[str]) -> None:
        """
        Load historical data for backtesting
        
        Args:
            assets: List of assets to load data for
            timeframes: List of timeframes to load data for
        """
        logger.info("Loading historical data...")
        
        for asset in assets:
            self.historical_data[asset] = {}
            
            for timeframe in timeframes:
                logger.info(f"Loading {asset} {timeframe} data...")
                
                # Fetch historical data
                data = fetch_historical_data(asset, timeframe=timeframe)
                
                if data is None or len(data) == 0:
                    logger.warning(f"No data available for {asset} {timeframe}")
                    continue
                
                # Ensure data is within date range
                if isinstance(data.index, pd.DatetimeIndex):
                    data = data[(data.index >= self.start_date) & (data.index <= self.end_date)]
                else:
                    # Convert timestamp to datetime if needed
                    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                    data = data[(data['datetime'] >= self.start_date) & (data['datetime'] <= self.end_date)]
                    data.set_index('datetime', inplace=True)
                
                logger.info(f"Loaded {len(data)} candles for {asset} {timeframe}")
                
                # Save data
                self.historical_data[asset][timeframe] = data
    
    def load_ml_models(self, assets: List[str]) -> None:
        """
        Load ML models for backtesting
        
        Args:
            assets: List of assets to load models for
        """
        logger.info("Loading ML models...")
        
        for asset in assets:
            logger.info(f"Loading model for {asset}...")
            
            # Load model
            model, scaler, features = load_model_with_metadata(asset)
            
            if model is None:
                logger.warning(f"No model available for {asset}")
                continue
            
            logger.info(f"Loaded model for {asset}")
            
            # Save model
            self.ml_models[asset] = {
                'model': model,
                'scaler': scaler,
                'features': features
            }
    
    def set_default_strategy_params(self) -> None:
        """Set default strategy parameters"""
        self.strategy_params = {
            'ARIMA': {
                'window_size': 50,
                'signal_threshold': 0.08,
                'confidence_multiplier': 1.2,
                'take_profit_multiplier': 2.5,
                'stop_loss_multiplier': 1.0
            },
            'Adaptive': {
                'ema_short': 9,
                'ema_long': 21,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'signal_threshold': 0.10,
                'confidence_multiplier': 1.0,
                'take_profit_multiplier': 2.0,
                'stop_loss_multiplier': 1.0
            },
            'ML': {
                'confidence_threshold': 0.70,
                'signal_threshold': 0.15,
                'confidence_multiplier': 1.5,
                'take_profit_multiplier': 3.0,
                'stop_loss_multiplier': 1.0
            },
            'Integrated': {
                'signal_threshold': 0.12,
                'confidence_multiplier': 1.3,
                'take_profit_multiplier': 2.8,
                'stop_loss_multiplier': 1.0
            }
        }
    
    def run_backtest(self, assets: List[str], strategies: List[str],
                  allocation: Optional[Dict[str, float]] = None,
                  parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run backtest
        
        Args:
            assets: List of assets to backtest
            strategies: List of strategies to backtest
            allocation: Optional capital allocation by asset
            parameters: Optional strategy parameters
            
        Returns:
            Dict: Backtest results
        """
        logger.info("Starting backtest...")
        
        # Set allocation
        if allocation:
            self.portfolio.set_allocation(allocation)
        else:
            # Default to equal allocation
            self.portfolio.set_allocation({asset: 1.0 / len(assets) for asset in assets})
        
        # Set strategy parameters
        self.set_default_strategy_params()
        if parameters:
            # Update with provided parameters
            for strategy, params in parameters.items():
                if strategy in self.strategy_params:
                    self.strategy_params[strategy].update(params)
        
        # Load historical data if not already loaded
        if not self.historical_data:
            self.load_historical_data(assets, ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        # Load ML models if not already loaded
        if not self.ml_models and 'ML' in strategies:
            self.load_ml_models(assets)
        
        # Get common timeframe for backtesting (1h is a good balance)
        timeframe = '1h'
        
        # Check if we have data for all assets
        for asset in assets:
            if asset not in self.historical_data or timeframe not in self.historical_data[asset]:
                logger.error(f"No {timeframe} data available for {asset}")
                return {'error': f"No {timeframe} data available for {asset}"}
        
        # Get price data
        price_data = {}
        for asset in assets:
            price_data[asset] = self.historical_data[asset][timeframe]
        
        # Determine common date range
        start_dates = [data.index.min() for asset, data in price_data.items()]
        end_dates = [data.index.max() for asset, data in price_data.items()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        logger.info(f"Common date range: {common_start} to {common_end}")
        
        # Create date range for backtesting
        dates = pd.date_range(start=common_start, end=common_end, freq=timeframe)
        
        # Run backtest
        logger.info(f"Running backtest on {len(dates)} {timeframe} candles...")
        
        for date in dates:
            # Update portfolio with current date
            self.portfolio.update_equity_curve(date)
            
            # Process each asset
            for asset in assets:
                # Get current candle
                try:
                    current_candle = price_data[asset].loc[date]
                except KeyError:
                    # Skip if no data for this date
                    continue
                
                # Extract price data
                if isinstance(current_candle, pd.Series):
                    current_price = current_candle['close']
                else:
                    current_price = current_candle.iloc[0]['close']
                
                # Update open positions
                self.portfolio.update_positions(asset, current_price, date)
                
                # Generate signals for each strategy
                for strategy in strategies:
                    # Skip if we already have an open position for this asset and strategy
                    if self.portfolio.get_position(asset, strategy) is not None:
                        continue
                    
                    # Generate signal
                    signal = self.generate_signal(
                        strategy=strategy,
                        asset=asset,
                        date=date,
                        current_price=current_price,
                        price_data=price_data
                    )
                    
                    if signal:
                        # Process signal
                        self.process_signal(
                            strategy=strategy,
                            asset=asset,
                            date=date,
                            current_price=current_price,
                            signal=signal
                        )
        
        # Calculate final portfolio value
        final_date = dates[-1] if dates.size > 0 else datetime.now()
        self.portfolio.update_equity_curve(final_date)
        
        # Generate backtest report
        title = f"Backtest {', '.join(assets)} ({common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')})"
        report_file = self.portfolio.generate_report(title)
        
        # Get metrics
        metrics = self.portfolio.get_metrics()
        
        return {
            'metrics': metrics,
            'report_file': report_file,
            'portfolio': self.portfolio
        }
    
    def generate_signal(self, strategy: str, asset: str, date: datetime,
                      current_price: float, price_data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """
        Generate trading signal for a strategy
        
        Args:
            strategy: Strategy name
            asset: Asset symbol
            date: Current date
            current_price: Current price
            price_data: Price data by asset
            
        Returns:
            Optional[Dict]: Signal or None if no signal
        """
        # Get parameters for this strategy
        params = self.strategy_params.get(strategy, {})
        
        # Get historical data up to this date
        df = price_data[asset]
        df = df[df.index <= date]
        
        if len(df) < 50:  # Minimum data required
            return None
        
        # Detect market regime
        market_regime = detect_market_regime(df)
        
        if strategy == 'ARIMA':
            # Implement ARIMA strategy logic
            signal_strength = 0.8  # Simplified for example
            direction = 'long' if df['close'].iloc[-1] > df['close'].iloc[-2] else 'short'
            ml_confidence = 0.75  # Placeholder
            
            # Only generate signal if strength is above threshold
            if signal_strength < params.get('signal_threshold', 0.08):
                return None
            
            return {
                'direction': direction,
                'signal_strength': signal_strength,
                'ml_confidence': ml_confidence,
                'market_regime': market_regime,
                'stop_loss_pct': DEFAULT_STOP_LOSS_PCT * params.get('stop_loss_multiplier', 1.0),
                'take_profit_pct': DEFAULT_TAKE_PROFIT_PCT * params.get('take_profit_multiplier', 1.0)
            }
        
        elif strategy == 'Adaptive':
            # Implement Adaptive strategy logic
            ema_short = params.get('ema_short', 9)
            ema_long = params.get('ema_long', 21)
            
            df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
            
            # Generate signal based on EMA crossover
            if df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1] and df['ema_short'].iloc[-2] <= df['ema_long'].iloc[-2]:
                direction = 'long'
                signal_strength = 0.7
            elif df['ema_short'].iloc[-1] < df['ema_long'].iloc[-1] and df['ema_short'].iloc[-2] >= df['ema_long'].iloc[-2]:
                direction = 'short'
                signal_strength = 0.7
            else:
                return None
            
            ml_confidence = 0.65  # Placeholder
            
            # Only generate signal if strength is above threshold
            if signal_strength < params.get('signal_threshold', 0.10):
                return None
            
            return {
                'direction': direction,
                'signal_strength': signal_strength,
                'ml_confidence': ml_confidence,
                'market_regime': market_regime,
                'stop_loss_pct': DEFAULT_STOP_LOSS_PCT * params.get('stop_loss_multiplier', 1.0),
                'take_profit_pct': DEFAULT_TAKE_PROFIT_PCT * params.get('take_profit_multiplier', 1.0)
            }
        
        elif strategy == 'ML':
            # Check if we have a model for this asset
            if asset not in self.ml_models:
                return None
            
            # Get model
            model_info = self.ml_models[asset]
            model = model_info['model']
            
            # Generate prediction
            # In a real implementation, you would prepare the data and make a prediction
            # For now, we'll use a simplified approach
            prediction = 0.75  # Placeholder
            
            # Determine direction and confidence
            direction = 'long' if prediction > 0.5 else 'short'
            ml_confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
            signal_strength = ml_confidence  # Use ML confidence as signal strength
            
            # Only generate signal if confidence is above threshold
            if ml_confidence < params.get('confidence_threshold', 0.70):
                return None
            
            return {
                'direction': direction,
                'signal_strength': signal_strength,
                'ml_confidence': ml_confidence,
                'market_regime': market_regime,
                'stop_loss_pct': DEFAULT_STOP_LOSS_PCT * params.get('stop_loss_multiplier', 1.0),
                'take_profit_pct': DEFAULT_TAKE_PROFIT_PCT * params.get('take_profit_multiplier', 1.0)
            }
        
        elif strategy == 'Integrated':
            # Implement Integrated strategy logic (combining multiple signals)
            # This is a simplified example
            signal_strength = 0.85
            direction = 'long' if df['close'].iloc[-1] > df['close'].iloc[-2] else 'short'
            ml_confidence = 0.80
            
            # Only generate signal if strength is above threshold
            if signal_strength < params.get('signal_threshold', 0.12):
                return None
            
            return {
                'direction': direction,
                'signal_strength': signal_strength,
                'ml_confidence': ml_confidence,
                'market_regime': market_regime,
                'stop_loss_pct': DEFAULT_STOP_LOSS_PCT * params.get('stop_loss_multiplier', 1.0),
                'take_profit_pct': DEFAULT_TAKE_PROFIT_PCT * params.get('take_profit_multiplier', 1.0)
            }
        
        return None
    
    def process_signal(self, strategy: str, asset: str, date: datetime,
                     current_price: float, signal: Dict) -> None:
        """
        Process trading signal
        
        Args:
            strategy: Strategy name
            asset: Asset symbol
            date: Current date
            current_price: Current price
            signal: Signal dictionary
        """
        # Open position
        self.portfolio.open_position(
            asset=asset,
            strategy=strategy,
            direction=signal['direction'],
            price=current_price,
            time=date,
            signal_strength=signal['signal_strength'],
            ml_confidence=signal['ml_confidence'],
            market_regime=signal['market_regime'],
            stop_loss_pct=signal.get('stop_loss_pct'),
            take_profit_pct=signal.get('take_profit_pct')
        )

def optimize_strategy_parameters(assets: List[str], strategies: List[str],
                               parameter_ranges: Dict[str, Dict[str, List]],
                               initial_capital: float = INITIAL_CAPITAL,
                               num_trials: int = 10) -> Dict[str, Any]:
    """
    Optimize strategy parameters using random search
    
    Args:
        assets: List of assets to backtest
        strategies: List of strategies to optimize
        parameter_ranges: Dictionary of parameter ranges by strategy
        initial_capital: Initial capital in USD
        num_trials: Number of optimization trials
        
    Returns:
        Dict: Optimization results
    """
    logger.info("Starting strategy parameter optimization...")
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=initial_capital)
    
    # Load historical data
    engine.load_historical_data(assets, ['1h'])
    
    # Load ML models if needed
    if 'ML' in strategies:
        engine.load_ml_models(assets)
    
    # Set default parameters
    engine.set_default_strategy_params()
    
    # Run optimization trials
    best_params = {}
    best_metrics = {}
    best_return = 0.0
    
    for trial in range(num_trials):
        logger.info(f"Running optimization trial {trial+1}/{num_trials}...")
        
        # Generate random parameters for each strategy
        parameters = {}
        for strategy in strategies:
            if strategy in parameter_ranges:
                strategy_params = {}
                for param, values in parameter_ranges[strategy].items():
                    # Choose random value from range
                    strategy_params[param] = np.random.choice(values)
                parameters[strategy] = strategy_params
        
        # Run backtest with these parameters
        results = engine.run_backtest(assets, strategies, parameters=parameters)
        
        if 'error' in results:
            logger.error(f"Error in trial {trial+1}: {results['error']}")
            continue
        
        metrics = results['metrics']
        total_return = metrics['total_return']
        
        logger.info(f"Trial {trial+1} results: Return: ${total_return:.2f}, "
                   f"Win Rate: {metrics['win_rate']:.2f}%, "
                   f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Update best parameters if this trial has better return
        if total_return > best_return:
            best_return = total_return
            best_params = parameters
            best_metrics = metrics
            
            logger.info(f"New best parameters found in trial {trial+1}")
    
    # Run final backtest with best parameters
    logger.info("Running final backtest with best parameters...")
    final_results = engine.run_backtest(assets, strategies, parameters=best_params)
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'final_results': final_results
    }

def run_full_backtest(assets: List[str], strategies: List[str],
                    allocation: Optional[Dict[str, float]] = None,
                    optimize: bool = False, num_trials: int = 10,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    initial_capital: float = INITIAL_CAPITAL) -> Dict[str, Any]:
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
            'ARIMA': {
                'signal_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
                'confidence_multiplier': [0.8, 1.0, 1.2, 1.5, 1.8],
                'take_profit_multiplier': [1.5, 2.0, 2.5, 3.0, 3.5],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            },
            'Adaptive': {
                'ema_short': [7, 9, 11, 13],
                'ema_long': [17, 21, 25, 29],
                'signal_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
                'confidence_multiplier': [0.8, 1.0, 1.2, 1.5],
                'take_profit_multiplier': [1.5, 2.0, 2.5, 3.0],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            },
            'ML': {
                'confidence_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
                'confidence_multiplier': [1.0, 1.2, 1.5, 1.8, 2.0],
                'take_profit_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],
                'stop_loss_multiplier': [0.8, 1.0, 1.2, 1.5]
            },
            'Integrated': {
                'signal_threshold': [0.08, 0.10, 0.12, 0.15, 0.18],
                'confidence_multiplier': [1.0, 1.2, 1.3, 1.5, 1.8],
                'take_profit_multiplier': [2.0, 2.5, 2.8, 3.0, 3.5],
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
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # Load data and models
        engine.load_historical_data(assets, ['1m', '5m', '15m', '1h', '4h', '1d'])
        if 'ML' in strategies:
            engine.load_ml_models(assets)
        
        # Run backtest
        results = engine.run_backtest(assets, strategies, allocation=allocation)
        
        return {
            'results': results
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive Backtesting System')
    
    parser.add_argument('--assets', nargs='+', default=SUPPORTED_ASSETS,
                      help='Assets to backtest')
    
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES,
                      help='Strategies to backtest')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize strategy parameters')
    
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of optimization trials')
    
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                      help='Initial capital in USD')
    
    parser.add_argument('--start-date', type=str,
                      help='Start date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                      help='End date for backtesting (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Run backtest
    results = run_full_backtest(
        assets=args.assets,
        strategies=args.strategies,
        optimize=args.optimize,
        num_trials=args.trials,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital
    )
    
    if args.optimize:
        logger.info("Optimization completed")
        logger.info(f"Best return: ${results['best_metrics']['total_return']:.2f} "
                   f"({results['best_metrics']['total_return_pct']:.2f}%)")
        logger.info(f"Best parameters: {results['best_params']}")
    else:
        logger.info("Backtest completed")
        logger.info(f"Return: ${results['results']['metrics']['total_return']:.2f} "
                   f"({results['results']['metrics']['total_return_pct']:.2f}%)")
        logger.info(f"Report: {results['results']['report_file']}")

if __name__ == "__main__":
    main()