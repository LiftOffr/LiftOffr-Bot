#!/usr/bin/env python3
"""
Realtime ML Trader for Kraken Trading Bot

This module implements a realtime machine learning trading system that:
1. Uses ensemble ML models to make trading decisions
2. Integrates with Kraken websocket for real-time data
3. Dynamically adjusts trading parameters based on predictions
4. Implements advanced risk management and position sizing
5. Runs in a continuous loop for automated trading
"""
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import threading

# Import our modules
import kraken_api_client as kraken
from kraken_websocket_client import KrakenWebsocketClient
import advanced_ml_integration as ami
from trade_optimizer import TradeOptimizer
from trade_entry_manager import TradeEntryManager
import risk_management as risk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
MODEL_DIR = "ml_models"
CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
CACHE_DIR = f"{DATA_DIR}/realtime_cache"

# Default trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Risk parameters
DEFAULT_RISK_RATE = 0.05  # 5% of portfolio per trade
MIN_CONFIDENCE = 0.65     # Minimum confidence for a trade
SAFE_MODE = True          # More conservative trading

class RealtimeDataManager:
    """
    Manager for real-time market data acquisition and processing
    """
    def __init__(self, trading_pairs: List[str] = DEFAULT_PAIRS):
        """
        Initialize the real-time data manager
        
        Args:
            trading_pairs: List of trading pairs to track
        """
        self.trading_pairs = trading_pairs
        self.data_frames = {}  # Store dataframes for each pair
        self.latest_prices = {}  # Store latest prices for each pair
        self.websocket_client = None
        self.rest_client = kraken.KrakenAPIClient()
        
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize data frames
        self._initialize_data()
        
        logger.info(f"Initialized RealtimeDataManager for {len(trading_pairs)} pairs")
    
    def _initialize_data(self):
        """Initialize data frames with historical data"""
        for pair in self.trading_pairs:
            # Load historical data (start with last 100 candles)
            try:
                df = self._fetch_historical_candles(pair, 100)
                
                if not df.empty:
                    # Save to cache
                    self.data_frames[pair] = df
                    
                    # Extract latest price
                    if 'close' in df.columns and not df['close'].empty:
                        self.latest_prices[pair] = float(df['close'].iloc[-1])
                        logger.info(f"Initial price for {pair}: {self.latest_prices[pair]}")
            
            except Exception as e:
                logger.error(f"Error initializing data for {pair}: {e}")
    
    def _fetch_historical_candles(self, pair: str, count: int = 100) -> pd.DataFrame:
        """
        Fetch historical candles from Kraken API
        
        Args:
            pair: Trading pair
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert pair to Kraken format
            kraken_pair = pair.replace("/", "")
            
            # Get OHLC data
            response = self.rest_client._request(
                kraken.PUBLIC_ENDPOINTS["ohlc"],
                {"pair": kraken_pair, "interval": 1}  # 1-minute candles
            )
            
            # Find the result key in the response
            result_key = None
            for key in response.keys():
                if key != "last":
                    result_key = key
                    break
            
            if not result_key:
                logger.error(f"No data found for {pair}")
                return pd.DataFrame()
            
            # Extract OHLC data
            candles = response[result_key]
            if not candles:
                logger.error(f"No candles found for {pair}")
                return pd.DataFrame()
            
            # Convert to DataFrame (limit to requested count)
            df = pd.DataFrame(
                candles[-count:],
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                ]
            )
            
            # Convert types
            df['timestamp'] = df['timestamp'].astype(float)
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                df[col] = df[col].astype(float)
            
            # Convert timestamp to datetime and set as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            logger.info(f"Fetched {len(df)} historical candles for {pair}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical candles for {pair}: {e}")
            return pd.DataFrame()
    
    def start_websocket(self):
        """Start the websocket client for real-time updates"""
        try:
            # Create websocket client
            self.websocket_client = KrakenWebsocketClient()
            
            # Define callback for ticker updates
            def on_ticker_update(data):
                try:
                    pair = data.get('pair')
                    if pair and pair in self.trading_pairs:
                        # Extract price data
                        ask = float(data.get('a', [0])[0])
                        bid = float(data.get('b', [0])[0])
                        close = float(data.get('c', [0])[0])
                        
                        # Update latest price
                        self.latest_prices[pair] = close
                        
                        # Create new row for dataframe
                        timestamp = time.time()
                        dt = pd.to_datetime(timestamp, unit='s')
                        
                        # If we already have a row for this minute, update it
                        # Otherwise, create a new row
                        minute_dt = dt.floor('1min')
                        if pair in self.data_frames and not self.data_frames[pair].empty:
                            last_dt = self.data_frames[pair].index[-1]
                            
                            if last_dt.floor('1min') == minute_dt:
                                # Update existing row
                                self.data_frames[pair].at[last_dt, 'close'] = close
                                self.data_frames[pair].at[last_dt, 'high'] = max(
                                    self.data_frames[pair].at[last_dt, 'high'], close
                                )
                                self.data_frames[pair].at[last_dt, 'low'] = min(
                                    self.data_frames[pair].at[last_dt, 'low'], close
                                )
                            else:
                                # Create new row
                                new_row = pd.DataFrame([{
                                    'timestamp': timestamp,
                                    'open': close,
                                    'high': close,
                                    'low': close,
                                    'close': close,
                                    'volume': 0.0,
                                    'vwap': close,
                                    'count': 1
                                }], index=[dt])
                                
                                self.data_frames[pair] = pd.concat([self.data_frames[pair], new_row])
                        else:
                            # Initialize dataframe with this row
                            self.data_frames[pair] = pd.DataFrame([{
                                'timestamp': timestamp,
                                'open': close,
                                'high': close,
                                'low': close,
                                'close': close,
                                'volume': 0.0,
                                'vwap': close,
                                'count': 1
                            }], index=[dt])
                        
                        logger.debug(f"Updated price for {pair}: {close}")
                except Exception as e:
                    logger.error(f"Error processing ticker update: {e}")
            
            # Subscribe to tickers for all pairs
            for pair in self.trading_pairs:
                kraken_pair = pair.replace("/", "")
                self.websocket_client.subscribe_ticker(kraken_pair, on_ticker_update)
            
            # Start websocket client in a separate thread
            threading.Thread(target=self.websocket_client.start, daemon=True).start()
            
            logger.info("Started websocket client for real-time updates")
        
        except Exception as e:
            logger.error(f"Error starting websocket client: {e}")
    
    def get_latest_price(self, pair: str) -> float:
        """
        Get the latest price for a pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Latest price (or None if not available)
        """
        # If we have real-time data, use that
        if pair in self.latest_prices and self.latest_prices[pair] > 0:
            return self.latest_prices[pair]
        
        # Otherwise, try to get from REST API
        try:
            # Convert pair to Kraken format
            kraken_pair = pair.replace("/", "")
            
            # Get ticker data
            response = self.rest_client._request(
                kraken.PUBLIC_ENDPOINTS["ticker"],
                {"pair": kraken_pair}
            )
            
            # Extract price
            for key, data in response.items():
                if key != "last":
                    price = float(data.get('c', [0])[0])
                    
                    # Update our cache
                    self.latest_prices[pair] = price
                    
                    return price
        
        except Exception as e:
            logger.error(f"Error getting latest price for {pair}: {e}")
        
        return None
    
    def get_latest_data(self, pair: str, count: int = 60) -> pd.DataFrame:
        """
        Get the latest data for a pair
        
        Args:
            pair: Trading pair
            count: Number of rows to return
            
        Returns:
            DataFrame with latest data
        """
        if pair not in self.data_frames or self.data_frames[pair].empty:
            logger.warning(f"No data available for {pair}")
            return pd.DataFrame()
        
        # Get last 'count' rows
        return self.data_frames[pair].tail(count)
    
    def save_data_to_cache(self):
        """Save data to cache files"""
        try:
            # Save latest prices
            with open(f"{CACHE_DIR}/latest_prices.json", 'w') as f:
                json.dump(self.latest_prices, f, indent=2)
            
            # Save dataframes
            for pair, df in self.data_frames.items():
                # Convert pair to safe filename
                safe_pair = pair.replace("/", "_")
                
                # Save to CSV
                df.to_csv(f"{CACHE_DIR}/{safe_pair}_realtime.csv")
            
            logger.info("Saved data to cache")
        
        except Exception as e:
            logger.error(f"Error saving data to cache: {e}")
    
    def load_data_from_cache(self):
        """Load data from cache files"""
        try:
            # Load latest prices
            prices_file = f"{CACHE_DIR}/latest_prices.json"
            if os.path.exists(prices_file):
                with open(prices_file, 'r') as f:
                    self.latest_prices = json.load(f)
                
                logger.info(f"Loaded {len(self.latest_prices)} prices from cache")
            
            # Load dataframes
            for pair in self.trading_pairs:
                # Convert pair to safe filename
                safe_pair = pair.replace("/", "_")
                
                # Load from CSV
                csv_file = f"{CACHE_DIR}/{safe_pair}_realtime.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    self.data_frames[pair] = df
                    
                    logger.info(f"Loaded {len(df)} rows for {pair} from cache")
        
        except Exception as e:
            logger.error(f"Error loading data from cache: {e}")

class RealtimeMLTrader:
    """
    Realtime ML trader that uses ensemble ML models to make trading decisions
    """
    def __init__(self, trading_pairs: List[str] = DEFAULT_PAIRS, sandbox: bool = True):
        """
        Initialize the realtime ML trader
        
        Args:
            trading_pairs: List of trading pairs to trade
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs
        self.sandbox = sandbox
        
        # Initialize components
        self.data_manager = RealtimeDataManager(trading_pairs)
        self.ml_trader = ami.AdvancedMLTrader(trading_pairs)
        self.optimizer = TradeOptimizer(trading_pairs)
        self.rest_client = kraken.KrakenAPIClient(sandbox=sandbox)
        self.entry_manager = TradeEntryManager()
        
        # Current positions and portfolio
        self.positions = {}
        self.portfolio = {'total': 20000.0, 'available': 20000.0}  # Default starting portfolio
        
        # Load configuration
        self._load_config()
        
        # Load positions and portfolio
        self._load_positions_and_portfolio()
        
        logger.info(f"Initialized RealtimeMLTrader for {len(trading_pairs)} pairs")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {CONFIG_FILE}")
            else:
                self.config = {
                    'pairs': self.trading_pairs,
                    'models': {},
                    'parameters': {
                        'risk_rate': DEFAULT_RISK_RATE,
                        'min_confidence': MIN_CONFIDENCE,
                        'safe_mode': SAFE_MODE
                    }
                }
                logger.warning(f"Configuration file {CONFIG_FILE} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = {
                'pairs': self.trading_pairs,
                'models': {},
                'parameters': {
                    'risk_rate': DEFAULT_RISK_RATE,
                    'min_confidence': MIN_CONFIDENCE,
                    'safe_mode': SAFE_MODE
                }
            }
    
    def _load_positions_and_portfolio(self):
        """Load positions and portfolio from files"""
        try:
            # Load positions
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} positions from {POSITIONS_FILE}")
            
            # Load portfolio
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    self.portfolio = json.load(f)
                logger.info(f"Loaded portfolio from {PORTFOLIO_FILE}: ${self.portfolio.get('total', 0):.2f}")
        except Exception as e:
            logger.error(f"Error loading positions and portfolio: {e}")
    
    def _save_positions_and_portfolio(self):
        """Save positions and portfolio to files"""
        try:
            # Save positions
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(self.positions, f, indent=2)
            
            # Save portfolio
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(self.portfolio, f, indent=2)
            
            # Update portfolio history
            self._update_portfolio_history()
            
            logger.info(f"Saved positions and portfolio")
        except Exception as e:
            logger.error(f"Error saving positions and portfolio: {e}")
    
    def _update_portfolio_history(self):
        """Update portfolio history file"""
        try:
            # Load existing history
            history = []
            if os.path.exists(PORTFOLIO_HISTORY_FILE):
                with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            
            # Add current portfolio
            history.append({
                'timestamp': datetime.now().isoformat(),
                'total': self.portfolio.get('total', 0),
                'available': self.portfolio.get('available', 0)
            })
            
            # Keep only last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save history
            with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def _add_trade_to_history(self, trade: Dict[str, Any]):
        """
        Add a trade to the trade history
        
        Args:
            trade: Trade data
        """
        try:
            # Load existing trades
            trades = []
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            
            # Add trade
            trades.append(trade)
            
            # Save trades
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"Added trade to history")
        except Exception as e:
            logger.error(f"Error adding trade to history: {e}")
    
    def start_data_collection(self):
        """Start data collection"""
        # Start websocket for real-time data
        self.data_manager.start_websocket()
        
        # Load cached data
        self.data_manager.load_data_from_cache()
        
        logger.info("Started data collection")
    
    def update_portfolio(self):
        """Update portfolio with current position values"""
        try:
            # Reset portfolio available amount
            self.portfolio['available'] = self.portfolio['total']
            
            # Calculate total position values
            for pair, position in self.positions.items():
                # Get latest price
                current_price = self.data_manager.get_latest_price(pair)
                
                if current_price is None:
                    logger.warning(f"No price available for {pair}, skipping portfolio update")
                    continue
                
                # Extract position data
                entry_price = float(position.get('entry_price', 0))
                amount = float(position.get('amount', 0))
                direction = position.get('direction', 'long')
                leverage = float(position.get('leverage', 1.0))
                
                # Calculate position value
                position_value = amount * entry_price
                
                # Update available capital
                self.portfolio['available'] -= position_value
                
                # Calculate unrealized PnL
                if direction == 'long':
                    pnl_pct = ((current_price / entry_price) - 1) * 100 * leverage
                else:
                    pnl_pct = ((entry_price / current_price) - 1) * 100 * leverage
                
                pnl_usd = position_value * (pnl_pct / 100)
                
                # Update position data
                position['current_price'] = current_price
                position['pnl_pct'] = pnl_pct
                position['pnl_usd'] = pnl_usd
                position['updated_at'] = datetime.now().isoformat()
                
                # Update position liquidation status
                # For long positions: liquidation at entry_price * (1 - (1/leverage))
                # For short positions: liquidation at entry_price * (1 + (1/leverage))
                if direction == 'long':
                    liquidation_price = entry_price * (1 - (95.0 / leverage) / 100)
                    is_liquidated = current_price <= liquidation_price
                else:
                    liquidation_price = entry_price * (1 + (95.0 / leverage) / 100)
                    is_liquidated = current_price >= liquidation_price
                
                position['liquidation_price'] = liquidation_price
                
                # If liquidated, close the position
                if is_liquidated:
                    logger.warning(f"Position liquidated: {pair} at ${current_price}")
                    
                    # Record the trade with liquidation
                    trade = {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'amount': amount,
                        'leverage': leverage,
                        'entry_time': position.get('entry_time'),
                        'exit_time': datetime.now().isoformat(),
                        'pnl': -100.0,  # 100% loss
                        'pnl_usd': -position_value,
                        'reason': 'liquidation'
                    }
                    
                    self._add_trade_to_history(trade)
                    
                    # Update portfolio
                    self.portfolio['total'] -= position_value
                    
                    # Remove position
                    del self.positions[pair]
            
            # Calculate total unrealized PnL
            total_unrealized_pnl = sum(
                position.get('pnl_usd', 0)
                for position in self.positions.values()
            )
            
            # Add to portfolio data
            self.portfolio['unrealized_pnl'] = total_unrealized_pnl
            self.portfolio['unrealized_pnl_pct'] = (
                (total_unrealized_pnl / self.portfolio['total']) * 100
                if self.portfolio['total'] > 0 else 0
            )
            
            # Save updated positions and portfolio
            self._save_positions_and_portfolio()
            
            logger.info(
                f"Updated portfolio: ${self.portfolio['total']:.2f} "
                f"(Available: ${self.portfolio['available']:.2f}, "
                f"Unrealized PnL: ${total_unrealized_pnl:.2f})"
            )
        
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def check_exit_signals(self):
        """Check for exit signals for current positions"""
        try:
            # Loop through current positions
            positions_to_close = []
            
            for pair, position in self.positions.items():
                # Get latest price
                current_price = self.data_manager.get_latest_price(pair)
                
                if current_price is None:
                    logger.warning(f"No price available for {pair}, skipping exit check")
                    continue
                
                # Extract position data
                entry_price = float(position.get('entry_price', 0))
                direction = position.get('direction', 'long')
                leverage = float(position.get('leverage', 1.0))
                amount = float(position.get('amount', 0))
                
                # Calculate PnL
                if direction == 'long':
                    pnl_pct = ((current_price / entry_price) - 1) * 100 * leverage
                else:
                    pnl_pct = ((entry_price / current_price) - 1) * 100 * leverage
                
                # Check stop loss
                stop_loss_pct = float(position.get('stop_loss_pct', 4.0))
                if pnl_pct <= -stop_loss_pct:
                    logger.info(f"Stop loss triggered for {pair} at ${current_price} (PnL: {pnl_pct:.2f}%)")
                    positions_to_close.append((pair, "stop_loss", pnl_pct))
                    continue
                
                # Check take profit
                take_profit_pct = float(position.get('take_profit_pct', 15.0))
                if pnl_pct >= take_profit_pct:
                    logger.info(f"Take profit triggered for {pair} at ${current_price} (PnL: {pnl_pct:.2f}%)")
                    positions_to_close.append((pair, "take_profit", pnl_pct))
                    continue
                
                # Check trailing stop
                trail_activation = float(position.get('trailing_stop_activation', 5.0))
                trail_distance = float(position.get('trailing_stop_distance', 2.5))
                
                # If trailing stop is activated and price retraces
                if pnl_pct >= trail_activation and position.get('trailing_stop_active', False):
                    highest_pct = float(position.get('highest_pnl_pct', pnl_pct))
                    
                    if (highest_pct - pnl_pct) >= trail_distance:
                        logger.info(f"Trailing stop triggered for {pair} at ${current_price} (PnL: {pnl_pct:.2f}%)")
                        positions_to_close.append((pair, "trailing_stop", pnl_pct))
                        continue
                
                # Update highest PnL for trailing stop
                if pnl_pct >= trail_activation:
                    highest_pct = float(position.get('highest_pnl_pct', 0))
                    if pnl_pct > highest_pct:
                        position['highest_pnl_pct'] = pnl_pct
                        position['trailing_stop_active'] = True
                
                # Get ML predictions for exit
                try:
                    # Get latest data for prediction
                    df = self.data_manager.get_latest_data(pair)
                    
                    if not df.empty:
                        # Get trading decision
                        decision = self.ml_trader.get_trading_decision(pair, df, current_price)
                        
                        # Check for exit signal
                        if decision.get('action') == 'exit' and decision.get('confidence', 0) >= MIN_CONFIDENCE:
                            logger.info(
                                f"ML exit signal for {pair} at ${current_price} "
                                f"(confidence: {decision.get('confidence', 0):.2f})"
                            )
                            positions_to_close.append((pair, "ml_signal", pnl_pct))
                
                except Exception as e:
                    logger.error(f"Error getting ML predictions for {pair}: {e}")
            
            # Close positions that need to be closed
            for pair, reason, pnl_pct in positions_to_close:
                self._close_position(pair, reason, pnl_pct)
        
        except Exception as e:
            logger.error(f"Error checking exit signals: {e}")
    
    def _close_position(self, pair: str, reason: str, pnl_pct: float):
        """
        Close a position
        
        Args:
            pair: Trading pair
            reason: Reason for closing
            pnl_pct: PnL percentage
        """
        try:
            if pair not in self.positions:
                logger.warning(f"Position {pair} not found, cannot close")
                return
            
            # Get position data
            position = self.positions[pair]
            entry_price = float(position.get('entry_price', 0))
            amount = float(position.get('amount', 0))
            direction = position.get('direction', 'long')
            leverage = float(position.get('leverage', 1.0))
            
            # Get current price
            current_price = self.data_manager.get_latest_price(pair)
            
            if current_price is None:
                logger.warning(f"No price available for {pair}, cannot close position")
                return
            
            # Calculate PnL
            pnl_usd = amount * entry_price * (pnl_pct / 100)
            
            # Update portfolio
            self.portfolio['total'] += pnl_usd
            self.portfolio['available'] += (amount * entry_price + pnl_usd)
            
            # Record the trade
            trade = {
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': current_price,
                'amount': amount,
                'leverage': leverage,
                'entry_time': position.get('entry_time'),
                'exit_time': datetime.now().isoformat(),
                'pnl': pnl_pct,
                'pnl_usd': pnl_usd,
                'reason': reason
            }
            
            self._add_trade_to_history(trade)
            
            # Remove position
            del self.positions[pair]
            
            # Save positions and portfolio
            self._save_positions_and_portfolio()
            
            logger.info(
                f"Closed position: {pair} at ${current_price}, "
                f"PnL: {pnl_pct:.2f}% (${pnl_usd:.2f}), "
                f"Reason: {reason}"
            )
        
        except Exception as e:
            logger.error(f"Error closing position {pair}: {e}")
    
    def check_entry_signals(self):
        """Check for entry signals"""
        try:
            # Update available capital first
            self.update_portfolio()
            
            # Check if we have enough available capital
            if self.portfolio['available'] <= 0:
                logger.warning("No available capital for new trades")
                return
            
            # Loop through all trading pairs
            for pair in self.trading_pairs:
                # Skip if we already have a position
                if pair in self.positions:
                    continue
                
                # Get latest price
                current_price = self.data_manager.get_latest_price(pair)
                
                if current_price is None:
                    logger.warning(f"No price available for {pair}, skipping entry check")
                    continue
                
                # Get latest data for prediction
                df = self.data_manager.get_latest_data(pair)
                
                if df.empty:
                    logger.warning(f"No data available for {pair}, skipping entry check")
                    continue
                
                # Get trading decision
                decision = self.ml_trader.get_trading_decision(pair, df, current_price)
                
                # Check for entry signal
                action = decision.get('action', '')
                confidence = decision.get('confidence', 0)
                
                if action.startswith('enter_') and confidence >= MIN_CONFIDENCE:
                    direction = decision.get('direction', 'long')
                    
                    # Get parameters from decision
                    params = decision.get('parameters', {})
                    
                    # Calculate optimal entry price
                    entry_price = params.get('entry_price', current_price)
                    
                    # Compare current price to optimal entry price
                    price_diff_pct = ((current_price / entry_price) - 1) * 100
                    
                    # Skip if current price is too far from optimal entry price
                    max_price_diff = 0.5  # Max 0.5% difference
                    if abs(price_diff_pct) > max_price_diff:
                        logger.info(
                            f"Skipping entry for {pair}: Current price ${current_price} "
                            f"too far from optimal entry ${entry_price} ({price_diff_pct:.2f}%)"
                        )
                        continue
                    
                    # Get leverage based on confidence
                    leverage = params.get('leverage', 5.0)  # Default to 5x leverage
                    
                    # Validate leverage
                    leverage = min(max(5.0, leverage), 125.0)  # Keep between 5x and 125x
                    
                    # Get other parameters
                    stop_loss_pct = params.get('stop_loss_pct', 4.0)
                    take_profit_pct = params.get('take_profit_pct', 15.0)
                    trailing_activation = params.get('trailing_stop_activation', 5.0)
                    trailing_distance = params.get('trailing_stop_distance', 2.5)
                    
                    # Calculate position size based on risk
                    risk_pct = DEFAULT_RISK_RATE * (confidence ** 2)  # Scale by confidence squared
                    
                    # Calculate max position size based on risk
                    max_risk_usd = self.portfolio['total'] * risk_pct
                    max_position_usd = max_risk_usd / (stop_loss_pct / 100) * leverage
                    
                    # Limit to available capital
                    max_position_usd = min(max_position_usd, self.portfolio['available'])
                    
                    # Determine if this trade should be taken based on capital constraints
                    allocation_needed = max_position_usd / self.portfolio['total']
                    
                    # Skip if allocation is too small
                    if allocation_needed < 0.01:  # Min 1% allocation
                        logger.info(
                            f"Skipping entry for {pair}: Allocation too small "
                            f"({allocation_needed*100:.2f}%)"
                        )
                        continue
                    
                    # Use trade entry manager to determine position size
                    position_size = self.entry_manager.calculate_position_size(
                        self.portfolio['total'],
                        self.portfolio['available'],
                        max_position_usd
                    )
                    
                    # Calculate amount of asset
                    amount = position_size / current_price
                    
                    # Create position
                    position = {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': current_price,
                        'amount': amount,
                        'leverage': leverage,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'trailing_stop_activation': trailing_activation,
                        'trailing_stop_distance': trailing_distance,
                        'entry_time': datetime.now().isoformat(),
                        'confidence': confidence,
                        'trailing_stop_active': False,
                        'highest_pnl_pct': 0.0
                    }
                    
                    # Add position
                    self.positions[pair] = position
                    
                    # Update portfolio
                    self.portfolio['available'] -= position_size
                    
                    # Save positions and portfolio
                    self._save_positions_and_portfolio()
                    
                    logger.info(
                        f"Entered position: {pair} {direction} at ${current_price}, "
                        f"Amount: ${position_size:.2f}, "
                        f"Leverage: {leverage:.1f}x, "
                        f"Confidence: {confidence:.2f}"
                    )
        
        except Exception as e:
            logger.error(f"Error checking entry signals: {e}")
    
    def run_trading_loop(self, interval: int = 60):
        """
        Run the trading loop
        
        Args:
            interval: Interval between iterations in seconds
        """
        try:
            logger.info(f"Starting trading loop with interval {interval} seconds")
            
            # Initial portfolio update
            self.update_portfolio()
            
            while True:
                try:
                    # Update portfolio
                    self.update_portfolio()
                    
                    # Check exit signals
                    self.check_exit_signals()
                    
                    # Check entry signals
                    self.check_entry_signals()
                    
                    # Save data to cache
                    self.data_manager.save_data_to_cache()
                    
                    # Sleep for the interval
                    time.sleep(interval)
                
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}")
                    time.sleep(10)  # Sleep for a shorter time on error
        
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

def main():
    """Main function"""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Realtime ML Trader")
    parser.add_argument("--sandbox", action="store_true", default=True, help="Run in sandbox mode")
    parser.add_argument("--interval", type=int, default=60, help="Trading loop interval in seconds")
    args = parser.parse_args()
    
    # Create realtime ML trader
    trader = RealtimeMLTrader(DEFAULT_PAIRS, args.sandbox)
    
    # Start data collection
    trader.start_data_collection()
    
    # Wait for initial data
    logger.info("Waiting for initial data...")
    time.sleep(10)
    
    # Run trading loop
    trader.run_trading_loop(args.interval)

if __name__ == "__main__":
    main()