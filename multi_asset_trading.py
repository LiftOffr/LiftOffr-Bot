"""
Multi-Asset Trading Module for Kraken Trading Bot

This module enables the bot to trade multiple assets simultaneously
with the extreme leverage settings and enhanced ML models.
Supported assets: SOL/USD, ETH/USD, BTC/USD

Key features:
1. Parallel trading of multiple assets
2. Asset-specific model training and optimization
3. Cross-asset correlation analysis
4. Portfolio-level risk management
5. Dynamic resource allocation based on ML confidence
"""

import logging
import os
import threading
import time
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict

from bot_manager import BotManager
from kraken_trading_bot import KrakenTradingBot
from config import (
    INITIAL_CAPITAL, MARGIN_PERCENT, TRADE_QUANTITY,
    LOOP_INTERVAL, STATUS_UPDATE_INTERVAL,
    LEVERAGE, ENABLE_CROSS_STRATEGY_EXITS, 
    SIGNAL_STRENGTH_ADVANTAGE, MIN_SIGNAL_STRENGTH
)
from dynamic_position_sizing import (
    calculate_dynamic_leverage, calculate_dynamic_margin_percent,
    adjust_limit_order_price, PositionSizingConfig
)
from ml_models import train_model, evaluate_model, prepare_data
import market_context

logger = logging.getLogger(__name__)

# Supported trading pairs
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]

# Asset-specific extreme leverage settings
ASSET_LEVERAGE_CONFIG = {
    "SOL/USD": {"base": 35.0, "max": 125.0, "min": 20.0},
    "ETH/USD": {"base": 30.0, "max": 100.0, "min": 15.0},
    "BTC/USD": {"base": 25.0, "max": 85.0, "min": 12.0}
}

# Model performance tracking
MODEL_PERFORMANCE = {
    asset: {
        "accuracy": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "trades": 0,
        "last_training": None
    } for asset in SUPPORTED_ASSETS
}

class MultiAssetManager:
    """
    Manager for running trading bots on multiple assets simultaneously
    """
    def __init__(self, capital_allocation: Dict[str, float] = None):
        """
        Initialize the multi-asset manager
        
        Args:
            capital_allocation: Dictionary mapping asset names to allocation percentages
                               (e.g., {"SOL/USD": 0.4, "ETH/USD": 0.3, "BTC/USD": 0.3})
        """
        self.bot_managers = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Default capital allocation (equal distribution)
        self.capital_allocation = capital_allocation or {
            "SOL/USD": 0.34,
            "ETH/USD": 0.33,
            "BTC/USD": 0.33
        }
        
        # Ensure allocations sum to 1.0
        total_allocation = sum(self.capital_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:  # Allow small rounding errors
            logger.warning(f"Capital allocations sum to {total_allocation}, normalizing...")
            for asset in self.capital_allocation:
                self.capital_allocation[asset] /= total_allocation
        
        # Calculate actual capital amounts
        self.asset_capital = {
            asset: INITIAL_CAPITAL * allocation
            for asset, allocation in self.capital_allocation.items()
        }
        
        # Initialize model data
        self.model_data = {asset: {} for asset in SUPPORTED_ASSETS}
        
        # Cross-asset correlation data
        self.correlation_matrix = None
        self.last_correlation_update = 0
        
        # Cache for historical data
        self.historical_data_cache = {}
        
        # Portfolio performance tracking
        self.portfolio_value = INITIAL_CAPITAL
        self.portfolio_peak = INITIAL_CAPITAL
        self.portfolio_drawdown = 0.0
        self.total_profit = 0.0
        self.total_trades = 0
        
        logger.info(f"Initialized multi-asset manager with allocations: {self.capital_allocation}")
    
    def initialize_bot_managers(self):
        """
        Initialize a separate bot manager for each asset
        """
        for asset in SUPPORTED_ASSETS:
            logger.info(f"Initializing bot manager for {asset}")
            
            # Create bot manager with asset-specific capital
            bot_manager = BotManager()
            bot_manager.portfolio_value = self.asset_capital[asset]
            bot_manager.available_funds = self.asset_capital[asset]
            
            # Configure leverage settings for this asset
            leverage_config = ASSET_LEVERAGE_CONFIG.get(asset, ASSET_LEVERAGE_CONFIG["SOL/USD"])
            
            # Add the primary strategies
            self._add_strategies_for_asset(bot_manager, asset, leverage_config)
            
            self.bot_managers[asset] = bot_manager
    
    def _add_strategies_for_asset(self, bot_manager: BotManager, asset: str, leverage_config: Dict[str, float]):
        """
        Add specialized strategies for a specific asset
        
        Args:
            bot_manager: The bot manager to add strategies to
            asset: The trading pair symbol
            leverage_config: Asset-specific leverage configuration
        """
        # Calculate trade quantity based on asset price level
        quantity = self._calculate_appropriate_quantity(asset)
        
        # Add adaptive strategy
        bot_manager.add_bot(
            strategy_type="adaptive",
            trading_pair=asset,
            trade_quantity=quantity,
            margin_percent=0.22,  # Enhanced margin percent (previously 0.20)
            leverage=leverage_config["base"]
        )
        
        # Add ARIMA strategy
        bot_manager.add_bot(
            strategy_type="arima",
            trading_pair=asset,
            trade_quantity=quantity,
            margin_percent=0.22,  # Enhanced margin percent
            leverage=leverage_config["base"]
        )
        
        # Add integrated strategy when appropriate
        if asset in ["SOL/USD", "ETH/USD"]:  # More volatile assets get integrated strategy
            bot_manager.add_bot(
                strategy_type="integrated",
                trading_pair=asset,
                trade_quantity=quantity,
                margin_percent=0.22,
                leverage=leverage_config["base"]
            )
        
        logger.info(f"Added strategies for {asset} with base leverage {leverage_config['base']}x")
    
    def _calculate_appropriate_quantity(self, asset: str) -> float:
        """
        Calculate an appropriate trade quantity based on asset price level
        
        Args:
            asset: The trading pair symbol
            
        Returns:
            float: Calculated trade quantity
        """
        # Default quantities based on typical price ranges
        default_quantities = {
            "SOL/USD": 1.0,
            "ETH/USD": 0.1,
            "BTC/USD": 0.01
        }
        
        # Start with default and adjust later based on current price
        return default_quantities.get(asset, TRADE_QUANTITY)
    
    def update_correlation_matrix(self, force: bool = False):
        """
        Update the cross-asset correlation matrix
        
        Args:
            force: Force update even if the last update was recent
        """
        current_time = time.time()
        
        # Update at most once per hour unless forced
        if not force and self.correlation_matrix is not None and \
           current_time - self.last_correlation_update < 3600:
            return
        
        logger.info("Updating cross-asset correlation matrix")
        
        # Gather price data for all assets
        price_data = {}
        for asset in SUPPORTED_ASSETS:
            if asset not in self.historical_data_cache:
                # Fetch historical data if not cached
                self._fetch_historical_data(asset)
            
            if asset in self.historical_data_cache:
                # Extract close prices
                price_data[asset] = self.historical_data_cache[asset]['close'].values
        
        # Ensure all price series have the same length
        min_length = min(len(prices) for prices in price_data.values())
        aligned_data = {}
        for asset, prices in price_data.items():
            aligned_data[asset] = prices[-min_length:]
        
        # Create DataFrame and calculate correlation matrix
        df = pd.DataFrame(aligned_data)
        self.correlation_matrix = df.corr()
        
        self.last_correlation_update = current_time
        logger.info(f"Updated correlation matrix: {self.correlation_matrix}")
        
        # Use correlation data to adjust risk parameters
        self._apply_correlation_adjustments()
    
    def _apply_correlation_adjustments(self):
        """
        Apply adjustments to position sizing based on correlation data
        """
        if self.correlation_matrix is None:
            return
        
        for asset1 in SUPPORTED_ASSETS:
            # Calculate average correlation with other assets
            correlations = [
                self.correlation_matrix.loc[asset1, asset2]
                for asset2 in SUPPORTED_ASSETS if asset2 != asset1
            ]
            avg_correlation = sum(correlations) / len(correlations)
            
            # Adjust position sizing parameters
            if avg_correlation > 0.8:  # High correlation - reduce overall exposure
                logger.info(f"High correlation detected for {asset1}, reducing exposure")
                self._adjust_position_parameters(asset1, multiplier=0.85)
            elif avg_correlation < 0.3:  # Low correlation - slight increase in exposure
                logger.info(f"Low correlation detected for {asset1}, slightly increasing exposure")
                self._adjust_position_parameters(asset1, multiplier=1.1)
    
    def _adjust_position_parameters(self, asset: str, multiplier: float):
        """
        Adjust position sizing parameters for a specific asset
        
        Args:
            asset: The trading pair symbol
            multiplier: Adjustment multiplier (>1.0 increases, <1.0 decreases)
        """
        if asset not in self.bot_managers:
            return
        
        # Adjust margin percent for all bots of this asset
        bot_manager = self.bot_managers[asset]
        for bot_id, bot in bot_manager.bots.items():
            # Only adjust if bot is for the specified asset
            if bot.trading_pair == asset:
                # Limit adjustments to keep within reasonable ranges
                current_margin = bot.margin_percent
                new_margin = current_margin * multiplier
                new_margin = max(0.15, min(0.30, new_margin))  # Keep between 15-30%
                
                bot.margin_percent = new_margin
                logger.info(f"Adjusted margin for {bot_id} from {current_margin:.2f} to {new_margin:.2f}")
    
    def _fetch_historical_data(self, asset: str, timeframe: str = "1h", limit: int = 1000):
        """
        Fetch and cache historical data for an asset
        
        Args:
            asset: The trading pair symbol
            timeframe: Timeframe for data (1m, 5m, 1h, etc.)
            limit: Number of candles to fetch
        """
        for bot_manager in self.bot_managers.values():
            for bot in bot_manager.bots.values():
                if bot.trading_pair == asset:
                    try:
                        logger.info(f"Fetching historical data for {asset} ({timeframe})")
                        historical_data = bot.exchange.fetch_ohlcv(
                            asset, timeframe=timeframe, limit=limit
                        )
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(
                            historical_data,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        self.historical_data_cache[asset] = df
                        logger.info(f"Cached {len(df)} candles for {asset}")
                        return
                    except Exception as e:
                        logger.error(f"Error fetching historical data for {asset}: {e}")
    
    def train_ml_models(self, assets: List[str] = None):
        """
        Train ML models for specified assets (or all if none specified)
        
        Args:
            assets: List of assets to train models for
        """
        assets = assets or SUPPORTED_ASSETS
        
        for asset in assets:
            logger.info(f"Training ML models for {asset}")
            
            # Ensure we have historical data
            if asset not in self.historical_data_cache:
                self._fetch_historical_data(asset)
            
            if asset not in self.historical_data_cache:
                logger.error(f"Cannot train models for {asset}: no historical data")
                continue
            
            try:
                # Prepare data
                df = self.historical_data_cache[asset]
                X_train, X_test, y_train, y_test = prepare_data(df)
                
                # Train model
                model, scaler = train_model(X_train, y_train)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test)
                
                # Store model and performance metrics
                self.model_data[asset] = {
                    "model": model,
                    "scaler": scaler,
                    "metrics": metrics,
                    "trained_at": time.time()
                }
                
                # Update performance tracking
                MODEL_PERFORMANCE[asset]["accuracy"] = metrics["accuracy"]
                MODEL_PERFORMANCE[asset]["last_training"] = time.time()
                
                logger.info(f"ML model for {asset} trained with accuracy: {metrics['accuracy']:.2f}")
                
            except Exception as e:
                logger.error(f"Error training ML model for {asset}: {e}")
    
    def start_all(self):
        """
        Start trading on all assets
        """
        self.running = True
        
        # Initialize bot managers if not already done
        if not self.bot_managers:
            self.initialize_bot_managers()
        
        # Update correlation matrix
        self.update_correlation_matrix(force=True)
        
        # Train ML models for all assets
        self.train_ml_models()
        
        # Start bot managers for each asset
        for asset, bot_manager in self.bot_managers.items():
            logger.info(f"Starting trading for {asset}")
            bot_manager.start_all()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Multi-asset trading started successfully")
    
    def stop_all(self):
        """
        Stop trading on all assets
        """
        self.running = False
        
        # Stop bot managers for each asset
        for asset, bot_manager in self.bot_managers.items():
            logger.info(f"Stopping trading for {asset}")
            bot_manager.stop_all()
        
        logger.info("Multi-asset trading stopped successfully")
    
    def _monitoring_loop(self):
        """
        Background thread for monitoring and maintaining the multi-asset system
        """
        logger.info("Starting multi-asset monitoring loop")
        
        while self.running:
            try:
                # Update total portfolio value and other metrics
                self._update_portfolio_metrics()
                
                # Update correlation matrix periodically
                self.update_correlation_matrix()
                
                # Retrain models periodically (every 24 hours)
                self._check_for_model_retraining()
                
                # Log overall status
                self._log_trading_status()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep before next iteration
            time.sleep(300)  # Check every 5 minutes
    
    def _update_portfolio_metrics(self):
        """
        Update portfolio-wide metrics
        """
        # Calculate total portfolio value
        total_value = 0
        for asset, bot_manager in self.bot_managers.items():
            total_value += bot_manager.portfolio_value
        
        previous_value = self.portfolio_value
        self.portfolio_value = total_value
        
        # Update peak value and drawdown
        if total_value > self.portfolio_peak:
            self.portfolio_peak = total_value
        
        if self.portfolio_peak > 0:
            self.portfolio_drawdown = max(
                self.portfolio_drawdown,
                (self.portfolio_peak - total_value) / self.portfolio_peak
            )
        
        # Calculate period return
        period_return = (total_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # Aggregate total trades
        self.total_trades = sum(
            bot_manager.trade_count for bot_manager in self.bot_managers.values()
        )
        
        # Calculate total profit
        self.total_profit = self.portfolio_value - INITIAL_CAPITAL
        total_profit_percent = self.total_profit / INITIAL_CAPITAL * 100
        
        logger.info(f"Portfolio value: ${total_value:.2f} (${self.total_profit:+.2f}, {total_profit_percent:+.2f}%)")
        logger.info(f"Max drawdown: {self.portfolio_drawdown:.2%}, Total trades: {self.total_trades}")
    
    def _check_for_model_retraining(self):
        """
        Check if any models need retraining
        """
        current_time = time.time()
        retraining_assets = []
        
        for asset in SUPPORTED_ASSETS:
            last_training = MODEL_PERFORMANCE[asset]["last_training"]
            
            # Retrain if never trained or more than 24 hours old
            if last_training is None or current_time - last_training > 86400:
                retraining_assets.append(asset)
        
        if retraining_assets:
            logger.info(f"Retraining models for: {', '.join(retraining_assets)}")
            self.train_ml_models(retraining_assets)
    
    def _log_trading_status(self):
        """
        Log detailed trading status across all assets
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"MULTI-ASSET TRADING STATUS [Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}]")
        logger.info("=" * 80)
        
        for asset in SUPPORTED_ASSETS:
            if asset in self.bot_managers:
                bot_manager = self.bot_managers[asset]
                
                # Count positions by direction
                long_positions = 0
                short_positions = 0
                no_positions = 0
                
                for bot_id, bot in bot_manager.bots.items():
                    if bot.position_side == "long":
                        long_positions += 1
                    elif bot.position_side == "short":
                        short_positions += 1
                    else:
                        no_positions += 1
                
                # Calculate asset P&L
                asset_initial = self.asset_capital.get(asset, 0)
                asset_current = bot_manager.portfolio_value
                asset_profit = asset_current - asset_initial
                asset_profit_percent = asset_profit / asset_initial * 100 if asset_initial > 0 else 0
                
                logger.info(f"{asset}:")
                logger.info(f"  Capital: ${asset_current:.2f} (${asset_profit:+.2f}, {asset_profit_percent:+.2f}%)")
                logger.info(f"  Positions: {long_positions} long, {short_positions} short, {no_positions} neutral")
                logger.info(f"  ML Model: Accuracy={MODEL_PERFORMANCE[asset]['accuracy']:.2f}")
                logger.info("-" * 40)
        
        logger.info(f"Portfolio Total: ${self.portfolio_value:.2f} (${self.total_profit:+.2f})")
        logger.info("=" * 80)
    
    def get_status(self):
        """
        Get dictionary of status information
        
        Returns:
            dict: Status information
        """
        status = {
            "portfolio_value": self.portfolio_value,
            "total_profit": self.total_profit,
            "total_profit_percent": self.total_profit / INITIAL_CAPITAL * 100 if INITIAL_CAPITAL > 0 else 0,
            "max_drawdown": self.portfolio_drawdown * 100,
            "total_trades": self.total_trades,
            "assets": {}
        }
        
        for asset in SUPPORTED_ASSETS:
            if asset in self.bot_managers:
                bot_manager = self.bot_managers[asset]
                
                asset_initial = self.asset_capital.get(asset, 0)
                asset_current = bot_manager.portfolio_value
                asset_profit = asset_current - asset_initial
                
                status["assets"][asset] = {
                    "portfolio_value": asset_current,
                    "profit": asset_profit,
                    "profit_percent": asset_profit / asset_initial * 100 if asset_initial > 0 else 0,
                    "accuracy": MODEL_PERFORMANCE[asset]["accuracy"],
                    "positions": {}
                }
                
                for bot_id, bot in bot_manager.bots.items():
                    status["assets"][asset]["positions"][bot_id] = {
                        "strategy": bot.strategy_type,
                        "side": bot.position_side,
                        "entry_price": bot.entry_price,
                        "current_price": bot.current_price
                    }
        
        return status

# Helper function to start multi-asset trading
def start_multi_asset_trading(allocation=None):
    """
    Start trading on multiple assets with the specified allocation
    
    Args:
        allocation: Dictionary mapping asset names to allocation percentages
    
    Returns:
        MultiAssetManager: The initialized manager instance
    """
    manager = MultiAssetManager(capital_allocation=allocation)
    manager.start_all()
    return manager

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Example allocation
    allocation = {
        "SOL/USD": 0.40,  # 40% to SOL (higher volatility, higher potential returns)
        "ETH/USD": 0.35,  # 35% to ETH 
        "BTC/USD": 0.25   # 25% to BTC (lower volatility, more stable)
    }
    
    manager = start_multi_asset_trading(allocation)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping multi-asset trading")
        manager.stop_all()