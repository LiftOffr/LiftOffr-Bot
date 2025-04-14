#!/usr/bin/env python3
"""
Maximize Returns Backtesting

This script performs advanced backtesting with a focus on maximizing returns
through optimal position sizing, entry/exit timing, and strategy parameters.
It builds on the hyper-optimized ML models to implement profit-focused trading.

Key features:
1. Advanced position sizing based on conviction levels
2. Leverage scaling based on prediction confidence
3. Dynamic stop-loss and take-profit levels
4. Market regime-specific parameter sets
5. Comprehensive performance reporting with profit metrics

Usage:
    python maximize_returns_backtest.py [--pairs PAIRS] [--days DAYS] [--capital CAPITAL]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure Plotting
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/maximize_returns.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
DEFAULT_TIMEFRAME = "1h"
HISTORICAL_DATA_DIR = "historical_data"
BACKTEST_DIR = "backtest_results/maximized"
ENSEMBLE_DIR = "models/ensemble"
INITIAL_CAPITAL = 20000.0
DEFAULT_DAYS = 90
DEFAULT_FEE_RATE = 0.0026  # 0.26% taker fee
DEFAULT_SLIPPAGE = 0.001   # 0.1% slippage

# Leverage settings
BASE_LEVERAGE = 20.0
MAX_LEVERAGE = 125.0
MIN_LEVERAGE = 10.0

# Risk management settings
MAX_POSITIONS_PER_PAIR = 1
DEFAULT_RISK_PER_TRADE = 0.20  # 20% of available capital
CONFIDENCE_THRESHOLD = 0.65    # Min confidence for trade entry

class Position:
    """
    Represents a trading position with entry and exit details
    """
    def __init__(self, 
                 asset: str,
                 strategy: str,
                 side: str,
                 entry_time: datetime,
                 entry_price: float,
                 quantity: float,
                 leverage: float = 1.0,
                 confidence: float = 0.5,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        """
        Initialize a position
        
        Args:
            asset: Asset symbol (e.g., "SOL/USD")
            strategy: Strategy that created this position
            side: "long" or "short"
            entry_time: Entry timestamp
            entry_price: Entry price
            quantity: Position quantity
            leverage: Position leverage
            confidence: Confidence score (0.0-1.0)
            stop_loss: Stop-loss price (optional)
            take_profit: Take-profit price (optional)
        """
        self.asset = asset
        self.strategy = strategy
        self.side = side.lower()
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.leverage = leverage
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Exit details (None until position is closed)
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = None
        self.pnl_pct = None
        self.duration = None
        
    def close(self, 
              exit_time: datetime,
              exit_price: float,
              exit_reason: str,
              fee_rate: float = DEFAULT_FEE_RATE):
        """
        Close the position
        
        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            fee_rate: Trading fee rate
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate P&L
        entry_value = self.entry_price * self.quantity
        exit_value = self.exit_price * self.quantity
        
        # Calculate fees
        entry_fee = entry_value * fee_rate
        exit_fee = exit_value * fee_rate
        total_fees = entry_fee + exit_fee
        
        if self.side == "long":
            self.pnl = (exit_value - entry_value) * self.leverage - total_fees
            price_change_pct = (self.exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (entry_value - exit_value) * self.leverage - total_fees
            price_change_pct = (self.entry_price - self.exit_price) / self.entry_price
            
        self.pnl_pct = price_change_pct * self.leverage - (fee_rate * 2)
        self.duration = (self.exit_time - self.entry_time).total_seconds() / 3600  # hours
        
    def get_current_pnl(self, current_price: float, fee_rate: float = DEFAULT_FEE_RATE) -> float:
        """
        Calculate current unrealized P&L
        
        Args:
            current_price: Current market price
            fee_rate: Trading fee rate
            
        Returns:
            float: Current P&L percentage
        """
        if self.exit_price is not None:
            return self.pnl_pct
            
        entry_value = self.entry_price * self.quantity
        current_value = current_price * self.quantity
        
        # Calculate fees (entry fee only, since position still open)
        entry_fee = entry_value * fee_rate
        
        if self.side == "long":
            unrealized_pnl = (current_value - entry_value) * self.leverage - entry_fee
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            unrealized_pnl = (entry_value - current_value) * self.leverage - entry_fee
            price_change_pct = (self.entry_price - current_price) / self.entry_price
            
        return price_change_pct * self.leverage - fee_rate

class MaximizeReturnsBacktester:
    """
    Backtesting engine focused on maximizing returns through
    optimized position sizing, entry/exit timing, and strategy parameters
    """
    
    def __init__(self,
                 assets: List[str],
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 timeframe: str = DEFAULT_TIMEFRAME,
                 initial_capital: float = INITIAL_CAPITAL,
                 fee_rate: float = DEFAULT_FEE_RATE,
                 slippage: float = DEFAULT_SLIPPAGE,
                 base_leverage: float = BASE_LEVERAGE,
                 max_leverage: float = MAX_LEVERAGE,
                 risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize the backtester
        
        Args:
            assets: List of assets to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            timeframe: Timeframe for data
            initial_capital: Initial capital in USD
            fee_rate: Trading fee rate
            slippage: Average slippage
            base_leverage: Base leverage to use
            max_leverage: Maximum leverage for high-confidence trades
            risk_per_trade: Risk percentage per trade (of available capital)
            confidence_threshold: Minimum confidence for trade entry
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.confidence_threshold = confidence_threshold
        
        # Initialize member variables
        self.historical_data = {}
        self.ensemble_configs = {}
        self.ensemble_weights = {}
        self.market_regime_weights = {}
        self.predictions = {}
        
        # Portfolio state
        self.current_capital = initial_capital
        self.positions = {}  # asset -> strategy -> position
        self.closed_positions = []
        
        # Capital allocation (default: equal allocation)
        self.allocation = {}
        
        # Performance tracking
        self.equity_curve = []
        
        # Results
        self.results = {}
        
        # Output directory
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        
    def load_historical_data(self) -> bool:
        """
        Load historical data for all assets
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading historical data...")
        
        for asset in self.assets:
            pair_code = asset.replace("/", "")
            data_path = os.path.join(HISTORICAL_DATA_DIR, f"{pair_code}_{self.timeframe}.csv")
            
            if not os.path.exists(data_path):
                logger.error(f"Historical data not found: {data_path}")
                return False
                
            try:
                # Load data
                data = pd.read_csv(data_path)
                
                # Parse timestamps
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
                elif 'datetime' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['datetime'])
                    
                # Sort by timestamp
                data = data.sort_values('timestamp')
                
                # Filter by date range if specified
                if self.start_date:
                    data = data[data['timestamp'] >= self.start_date]
                if self.end_date:
                    data = data[data['timestamp'] <= self.end_date]
                    
                # Ensure we have required columns
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in data.columns:
                        logger.error(f"Required column '{col}' not found in {data_path}")
                        return False
                        
                # Store data
                self.historical_data[asset] = data
                logger.info(f"Loaded {len(data)} rows for {asset}")
                
            except Exception as e:
                logger.error(f"Error loading data for {asset}: {e}")
                return False
                
        return True
        
    def load_ensemble_configs(self) -> bool:
        """
        Load ensemble configurations and weights for all assets
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading ensemble configurations...")
        
        for asset in self.assets:
            pair_code = asset.replace("/", "")
            
            # Load ensemble configuration
            config_path = os.path.join(ENSEMBLE_DIR, f"{pair_code}_ensemble.json")
            if not os.path.exists(config_path):
                logger.error(f"Ensemble config not found: {config_path}")
                return False
                
            # Load weights
            weights_path = os.path.join(ENSEMBLE_DIR, f"{pair_code}_weights.json")
            if not os.path.exists(weights_path):
                logger.error(f"Ensemble weights not found: {weights_path}")
                return False
                
            try:
                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.ensemble_configs[asset] = config
                
                # Load weights
                with open(weights_path, 'r') as f:
                    weights_data = json.load(f)
                self.ensemble_weights[asset] = weights_data["base_weights"]
                self.market_regime_weights[asset] = weights_data["market_regime_weights"]
                
                logger.info(f"Loaded ensemble config and weights for {asset}")
                
            except Exception as e:
                logger.error(f"Error loading ensemble data for {asset}: {e}")
                return False
                
        return True
        
    def generate_predictions(self) -> bool:
        """
        Generate predictions for all assets
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Generating predictions...")
        
        for asset in self.assets:
            try:
                # Get historical data for this asset
                data = self.historical_data[asset]
                
                # Detect market regimes
                regimes = []
                for i in range(len(data)):
                    if i < 100:  # Need enough data for regime detection
                        regimes.append('ranging')  # Default to ranging for early data
                        continue
                        
                    # Get data window for regime detection
                    window = data.iloc[max(0, i-100):i+1]
                    
                    # Simple regime detection based on volatility
                    if 'close' in window.columns:
                        returns = window['close'].pct_change().dropna()
                        volatility = returns.std()
                        
                        if volatility > 0.015:  # High volatility
                            regime = 'volatile'
                        elif abs(window['close'].iloc[-1] / window['close'].iloc[0] - 1) > 0.1:  # Trending
                            regime = 'trending'
                        else:
                            regime = 'ranging'
                    else:
                        regime = 'ranging'  # Default
                        
                    regimes.append(regime)
                    
                # Add regime to data
                data['market_regime'] = regimes
                
                # Generate prediction confidence (simulating ML model outputs)
                # In a real implementation, this would load and run the actual models
                
                # Create prediction and confidence columns
                data['prediction'] = 0.5  # Default neutral
                data['confidence'] = 0.5  # Default medium confidence
                
                # Simulate some predictions based on indicators
                data['ma_fast'] = data['close'].rolling(window=10).mean()
                data['ma_slow'] = data['close'].rolling(window=30).mean()
                
                # Simple moving average crossover as prediction signal
                data.loc[data['ma_fast'] > data['ma_slow'], 'prediction'] = 0.7  # Bullish
                data.loc[data['ma_fast'] < data['ma_slow'], 'prediction'] = 0.3  # Bearish
                
                # Generate confidence based on strength of signal and regime
                for i in range(len(data)):
                    if i < 30:  # Skip rows without enough history
                        continue
                        
                    row = data.iloc[i]
                    regime = row['market_regime']
                    
                    # Base confidence on signal strength
                    ma_diff = abs(row['ma_fast'] / row['ma_slow'] - 1)
                    
                    # Scale confidence based on regime
                    if regime == 'trending':
                        confidence = 0.6 + min(0.35, ma_diff * 10)  # Higher in trending markets
                    elif regime == 'volatile':
                        confidence = 0.5 + min(0.4, ma_diff * 15)   # Even higher in volatile markets
                    else:  # ranging
                        confidence = 0.5 + min(0.25, ma_diff * 5)   # Lower in ranging markets
                        
                    data.iloc[i, data.columns.get_loc('confidence')] = confidence
                    
                # Store predictions
                self.predictions[asset] = data
                logger.info(f"Generated predictions for {asset}")
                
            except Exception as e:
                logger.error(f"Error generating predictions for {asset}: {e}")
                return False
                
        return True
        
    def calculate_dynamic_leverage(self, confidence: float) -> float:
        """
        Calculate dynamic leverage based on confidence
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            
        Returns:
            float: Calculated leverage
        """
        if confidence < self.confidence_threshold:
            return 0.0  # No trade if below threshold
            
        # Scale leverage from base to max based on confidence
        confidence_above_threshold = confidence - self.confidence_threshold
        confidence_scale = confidence_above_threshold / (1.0 - self.confidence_threshold)
        
        # Calculate leverage
        leverage = self.base_leverage + (self.max_leverage - self.base_leverage) * confidence_scale
        
        # Ensure within bounds
        leverage = max(MIN_LEVERAGE, min(self.max_leverage, leverage))
        
        return leverage
        
    def get_available_capital(self, asset: str) -> float:
        """
        Get available capital for a specific asset
        
        Args:
            asset: Asset to get capital for
            
        Returns:
            float: Available capital in USD
        """
        allocation_pct = self.allocation.get(asset, 1.0 / len(self.assets))
        allocated_capital = self.current_capital * allocation_pct
        
        # Subtract capital used by open positions for this asset
        if asset in self.positions:
            for strategy, position in self.positions[asset].items():
                if position.exit_price is None:  # Open position
                    allocated_capital -= (position.entry_price * position.quantity)
                    
        return max(0, allocated_capital)
        
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtest across all assets
        
        Returns:
            Dict: Backtest results
        """
        logger.info("Starting backtest...")
        
        # Load data
        if not self.load_historical_data():
            return {"error": "Failed to load historical data"}
            
        # Load ensemble configurations
        if not self.load_ensemble_configs():
            return {"error": "Failed to load ensemble configurations"}
            
        # Generate predictions
        if not self.generate_predictions():
            return {"error": "Failed to generate predictions"}
            
        # Set equal allocation by default
        self.allocation = {asset: 1.0 / len(self.assets) for asset in self.assets}
        
        # Initialize positions dictionary
        self.positions = {asset: {} for asset in self.assets}
        
        # Initialize equity curve with starting capital
        self.equity_curve = [{'timestamp': self.start_date or datetime.now(), 'equity': self.initial_capital}]
        
        # Combine all asset timeframes into one timeline
        all_timestamps = set()
        for asset, data in self.predictions.items():
            all_timestamps.update(data['timestamp'].tolist())
            
        timeline = sorted(all_timestamps)
        
        # Run simulation
        for timestamp in tqdm(timeline):
            # Process each asset at this timestamp
            for asset in self.assets:
                if asset not in self.predictions:
                    continue
                    
                data = self.predictions[asset]
                
                # Skip if no data for this timestamp
                if timestamp not in data['timestamp'].values:
                    continue
                    
                # Get current row
                row = data[data['timestamp'] == timestamp].iloc[0]
                
                # Check for position exits
                self._check_exits(asset, row)
                
                # Check for new entries
                self._check_entries(asset, row)
                
            # Update equity curve
            self._update_equity(timestamp)
            
        # Close any remaining open positions at the end
        for asset in self.assets:
            if asset in self.positions:
                for strategy, position in list(self.positions[asset].items()):
                    if position.exit_price is None:
                        # Get last price for this asset
                        last_row = self.predictions[asset].iloc[-1]
                        last_price = last_row['close']
                        
                        # Close position
                        position.close(
                            exit_time=last_row['timestamp'],
                            exit_price=last_price,
                            exit_reason="end_of_backtest"
                        )
                        
                        # Add to closed positions
                        self.closed_positions.append(position)
                        
                        # Remove from open positions
                        del self.positions[asset][strategy]
                        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        # Generate equity curve dataframe
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Save backtest results
        results_path = os.path.join(BACKTEST_DIR, "maximized_returns_backtest.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save equity curve
        equity_path = os.path.join(BACKTEST_DIR, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False)
        
        # Save trade history
        trades_df = pd.DataFrame([vars(pos) for pos in self.closed_positions])
        trades_path = os.path.join(BACKTEST_DIR, "trade_history.csv")
        trades_df.to_csv(trades_path, index=False)
        
        # Plot equity curve
        self._plot_equity_curve(equity_df)
        
        return metrics
        
    def _check_exits(self, asset: str, row: pd.Series) -> None:
        """
        Check if any positions should be exited
        
        Args:
            asset: Asset to check
            row: Current data row
        """
        if asset not in self.positions:
            return
            
        current_price = row['close']
        timestamp = row['timestamp']
        
        # Check each open position
        for strategy, position in list(self.positions[asset].items()):
            if position.exit_price is not None:
                continue  # Already closed
                
            # Check stop-loss
            if position.stop_loss is not None:
                if (position.side == "long" and current_price <= position.stop_loss) or \
                   (position.side == "short" and current_price >= position.stop_loss):
                    # Close position at stop-loss
                    position.close(
                        exit_time=timestamp,
                        exit_price=position.stop_loss,
                        exit_reason="stop_loss"
                    )
                    
                    # Add to closed positions
                    self.closed_positions.append(position)
                    
                    # Remove from open positions
                    del self.positions[asset][strategy]
                    
                    # Update capital
                    self.current_capital += position.pnl
                    
                    continue
                    
            # Check take-profit
            if position.take_profit is not None:
                if (position.side == "long" and current_price >= position.take_profit) or \
                   (position.side == "short" and current_price <= position.take_profit):
                    # Close position at take-profit
                    position.close(
                        exit_time=timestamp,
                        exit_price=position.take_profit,
                        exit_reason="take_profit"
                    )
                    
                    # Add to closed positions
                    self.closed_positions.append(position)
                    
                    # Remove from open positions
                    del self.positions[asset][strategy]
                    
                    # Update capital
                    self.current_capital += position.pnl
                    
                    continue
                    
            # Check for signal reversal
            prediction = row['prediction']
            confidence = row['confidence']
            
            # Exit long if bearish with high confidence
            if position.side == "long" and prediction < 0.5 and confidence >= self.confidence_threshold:
                # Close position
                position.close(
                    exit_time=timestamp,
                    exit_price=current_price,
                    exit_reason="signal_reversal"
                )
                
                # Add to closed positions
                self.closed_positions.append(position)
                
                # Remove from open positions
                del self.positions[asset][strategy]
                
                # Update capital
                self.current_capital += position.pnl
                
            # Exit short if bullish with high confidence
            elif position.side == "short" and prediction > 0.5 and confidence >= self.confidence_threshold:
                # Close position
                position.close(
                    exit_time=timestamp,
                    exit_price=current_price,
                    exit_reason="signal_reversal"
                )
                
                # Add to closed positions
                self.closed_positions.append(position)
                
                # Remove from open positions
                del self.positions[asset][strategy]
                
                # Update capital
                self.current_capital += position.pnl
                
    def _check_entries(self, asset: str, row: pd.Series) -> None:
        """
        Check if a new position should be entered
        
        Args:
            asset: Asset to check
            row: Current data row
        """
        # Don't enter if maximum positions reached
        if asset in self.positions and len(self.positions[asset]) >= MAX_POSITIONS_PER_PAIR:
            return
            
        # Get prediction and confidence
        prediction = row['prediction']
        confidence = row['confidence']
        
        # Only enter if confidence is above threshold
        if confidence < self.confidence_threshold:
            return
            
        # Determine trade direction
        if prediction > 0.5:
            side = "long"
            strategy = "ensemble_ml"  # Using ML ensemble strategy
        elif prediction < 0.5:
            side = "short"
            strategy = "ensemble_ml"  # Using ML ensemble strategy
        else:
            return  # No clear signal
            
        # Skip if already have a position with this strategy
        if asset in self.positions and strategy in self.positions[asset]:
            return
            
        # Calculate dynamic leverage based on confidence
        leverage = self.calculate_dynamic_leverage(confidence)
        
        # Skip if no leverage (confidence too low)
        if leverage <= 0:
            return
            
        # Calculate position size
        available_capital = self.get_available_capital(asset)
        position_capital = available_capital * self.risk_per_trade
        current_price = row['close']
        
        # Calculate quantity
        quantity = position_capital / current_price
        
        # Skip if quantity too small
        if quantity * current_price < 10.0:  # Minimum $10 position
            return
            
        # Calculate stop-loss and take-profit based on ATR if available
        stop_loss = None
        take_profit = None
        
        if 'atr' in row:
            atr = row['atr']
            
            if side == "long":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 5)
            else:  # short
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 5)
        else:
            # Use percentage-based if ATR not available
            if side == "long":
                stop_loss = current_price * 0.96  # 4% stop-loss
                take_profit = current_price * 1.12  # 12% take-profit
            else:  # short
                stop_loss = current_price * 1.04  # 4% stop-loss
                take_profit = current_price * 0.88  # 12% take-profit
                
        # Create position
        position = Position(
            asset=asset,
            strategy=strategy,
            side=side,
            entry_time=row['timestamp'],
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add to positions
        if asset not in self.positions:
            self.positions[asset] = {}
        self.positions[asset][strategy] = position
        
        # Update capital
        self.current_capital -= position_capital
        
    def _update_equity(self, timestamp: datetime) -> None:
        """
        Update equity curve at a given timestamp
        
        Args:
            timestamp: Current timestamp
        """
        # Start with current capital
        total_equity = self.current_capital
        
        # Add value of open positions
        for asset, strategies in self.positions.items():
            for strategy, position in strategies.items():
                if position.exit_price is None:  # Open position
                    # Get current price for this asset
                    asset_data = self.predictions[asset]
                    asset_data_at_time = asset_data[asset_data['timestamp'] <= timestamp]
                    
                    if len(asset_data_at_time) == 0:
                        continue
                        
                    current_price = asset_data_at_time.iloc[-1]['close']
                    
                    # Calculate position value
                    position_value = position.entry_price * position.quantity
                    
                    # Calculate unrealized P&L
                    unrealized_pnl = position.get_current_pnl(current_price) * position_value
                    
                    # Add to equity
                    total_equity += position_value + unrealized_pnl
                    
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity
        })
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        # Calculate basic metrics
        total_trades = len(self.closed_positions)
        if total_trades == 0:
            return {
                "error": "No trades executed in backtest"
            }
            
        winning_trades = sum(1 for pos in self.closed_positions if pos.pnl > 0)
        losing_trades = sum(1 for pos in self.closed_positions if pos.pnl <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate returns
        final_equity = self.equity_curve[-1]['equity']
        total_return = final_equity - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Calculate profit factor
        gross_profit = sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0)
        gross_loss = sum(abs(pos.pnl) for pos in self.closed_positions if pos.pnl <= 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = self.initial_capital
        drawdowns = []
        
        for equity in equity_values:
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak
            drawdowns.append(drawdown)
            
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Annualized Sharpe ratio (assuming daily returns)
        annual_factor = 252  # Trading days per year
        if len(equity_df) > 1:
            sharpe_ratio = (equity_df['return'].mean() / equity_df['return'].std()) * np.sqrt(annual_factor)
        else:
            sharpe_ratio = 0.0
            
        # Summarize by asset
        asset_summary = {}
        for asset in self.assets:
            asset_positions = [pos for pos in self.closed_positions if pos.asset == asset]
            if not asset_positions:
                continue
                
            asset_trades = len(asset_positions)
            asset_winners = sum(1 for pos in asset_positions if pos.pnl > 0)
            asset_win_rate = asset_winners / asset_trades if asset_trades > 0 else 0.0
            asset_profit = sum(pos.pnl for pos in asset_positions)
            asset_profit_pct = asset_profit / self.initial_capital
            
            asset_summary[asset] = {
                "trades": asset_trades,
                "win_rate": asset_win_rate,
                "profit": asset_profit,
                "profit_pct": asset_profit_pct
            }
            
        # Return metrics
        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0,
            "asset_summary": asset_summary,
            "backtest_config": {
                "assets": self.assets,
                "timeframe": self.timeframe,
                "fee_rate": self.fee_rate,
                "slippage": self.slippage,
                "base_leverage": self.base_leverage,
                "max_leverage": self.max_leverage,
                "risk_per_trade": self.risk_per_trade,
                "confidence_threshold": self.confidence_threshold
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def _plot_equity_curve(self, equity_df: pd.DataFrame) -> None:
        """
        Plot and save equity curve
        
        Args:
            equity_df: DataFrame with equity curve data
        """
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['timestamp'], equity_df['equity'])
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(BACKTEST_DIR, "equity_curve.png"))
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Maximize Returns Backtesting")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=SUPPORTED_PAIRS,
                        help="Trading pairs to backtest")
    
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help="Timeframe for data")
    
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="Number of days to backtest (most recent)")
    
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help="Initial capital in USD")
    
    parser.add_argument("--leverage", type=float, default=BASE_LEVERAGE,
                        help="Base leverage to use")
    
    parser.add_argument("--max-leverage", type=float, default=MAX_LEVERAGE,
                        help="Maximum leverage for high-confidence trades")
    
    parser.add_argument("--risk", type=float, default=DEFAULT_RISK_PER_TRADE,
                        help="Risk per trade as percentage (e.g., 0.20 for 20%%)")
    
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Minimum confidence for trade entry")
    
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE,
                        help="Trading fee rate")
    
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE,
                        help="Average slippage")
    
    parser.add_argument("--start", type=str,
                        help="Start date for backtesting (YYYY-MM-DD)")
    
    parser.add_argument("--end", type=str,
                        help="End date for backtesting (YYYY-MM-DD)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    elif args.days > 0:
        start_date = datetime.now() - timedelta(days=args.days)
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Print configuration
    print("=" * 80)
    print("MAXIMIZE RETURNS BACKTESTING")
    print("=" * 80)
    print(f"Trading pairs: {' '.join(args.pairs)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Starting capital: ${args.capital:.2f}")
    if start_date:
        print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    if end_date:
        print(f"End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"Leverage settings: Base={args.leverage}x, Max={args.max_leverage}x")
    print(f"Risk per trade: {args.risk * 100:.1f}%")
    print(f"Confidence threshold: {args.confidence * 100:.1f}%")
    print(f"Fee rate: {args.fee_rate * 100:.2f}%")
    print(f"Slippage: {args.slippage * 100:.2f}%")
    print("=" * 80)
    
    # Create backtester
    backtester = MaximizeReturnsBacktester(
        assets=args.pairs,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        base_leverage=args.leverage,
        max_leverage=args.max_leverage,
        risk_per_trade=args.risk,
        confidence_threshold=args.confidence
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print results
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct'] * 100:.2f}%)")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate'] * 100:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        print("\nAsset Summary:")
        for asset, stats in results['asset_summary'].items():
            print(f"  {asset}: {stats['trades']} trades, {stats['win_rate'] * 100:.1f}% win rate, ${stats['profit']:.2f} profit")
            
    print("=" * 80)
    print(f"Detailed results saved to {BACKTEST_DIR}/")
    print("=" * 80)
    
if __name__ == "__main__":
    main()